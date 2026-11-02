"""SelfContact's public behavior, checked against small independent CPU calculations.

Run from falcor2-inverse with:
    PYTHONPATH=. ../falcor2/.venv/bin/python -m pytest tests/test_contact.py -q
"""

from types import SimpleNamespace

import numpy as np
import pytest
import slangpy as spy
import torch

from falcor2_inverse.contact import SelfContact
from falcor2_inverse.device import create_device


@pytest.fixture(scope="module")
def device():
    result = create_device(shader_cache=False, device_type=spy.DeviceType.vulkan)
    if not result.has_feature(spy.Feature.ray_tracing):
        pytest.skip("These regressions require a Vulkan ray-tracing pipeline")
    print(f"SelfContact device: {result.info.type.name}, {result.info.adapter_name}, "
          f"ray_query={result.has_feature(spy.Feature.ray_query)}, "
          f"ray_tracing={result.has_feature(spy.Feature.ray_tracing)}")
    yield result
    result.close()


def contact_for(device, vertices, faces, dhat=0.5, **kwargs):
    return SelfContact(faces, len(vertices), dhat, device=device, traversal="trace_ray", **kwargs)


def point_mesh(gap=0.25):
    return (np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0], [0, 0, gap]], np.float64),
            np.array([[0, 1, 2]], np.uint32))


def tilted_point_mesh():
    triangle = np.array([[-1, -1, 0], [1, -1, 0.2], [0, 1, -0.1]], np.float64)
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal /= np.linalg.norm(normal)
    point = np.array([0.25, 0.35, 0.4]) @ triangle + 0.125 * normal
    return np.vstack((triangle, point)), np.array([[0, 1, 2]], np.uint32)


def triangle_distance(point, triangle):
    """Euclidean projection onto a triangle, using a float64 linear solve and boundary segments."""
    a, b, c = triangle
    basis = np.column_stack((b - a, c - a))
    uv, _, rank, _ = np.linalg.lstsq(basis, point - a, rcond=None)
    candidates = []
    if rank == 2 and np.all(uv >= 0) and uv.sum() <= 1:
        candidates.append(a + basis @ uv)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        length_squared = np.dot(edge, edge)
        fraction = np.clip(np.dot(point - start, edge) / length_squared, 0, 1) if length_squared else 0
        candidates.append(start + fraction * edge)
    return min(np.linalg.norm(point - candidate) for candidate in candidates)


def barrier_energy(vertices, faces, dhat, kappa):
    energy, pairs = 0.0, 0
    for vertex, point in enumerate(vertices):
        for face in faces:
            if vertex in face:
                continue
            distance = triangle_distance(point, vertices[face])
            if distance < dhat:
                assert distance > 0, "The finite-difference fixtures must stay away from exact contact"
                energy -= kappa * (distance - dhat) ** 2 * np.log(distance / dhat)
                pairs += 1
    return energy, pairs


def barrier_reference(vertices, faces, dhat, image_gradient, strength=1.0):
    # Keep this normalization fixed while differentiating the geometric barrier energy.
    strong_gradient = np.quantile(np.linalg.norm(np.asarray(image_gradient, np.float32), axis=1), 0.99)
    kappa = strength * strong_gradient / (dhat * (np.log(2) + 0.5))
    result = np.zeros_like(vertices, dtype=np.float64)
    epsilon = 1e-5
    for index in np.ndindex(vertices.shape):
        before, after = vertices.copy(), vertices.copy()
        before[index] -= epsilon
        after[index] += epsilon
        result[index] = (barrier_energy(after, faces, dhat, kappa)[0]
                         - barrier_energy(before, faces, dhat, kappa)[0]) / (2 * epsilon)
    return result, barrier_energy(vertices, faces, dhat, kappa)[1]


def segment_touches_box(origin, displacement, lower, upper):
    direction = displacement if np.linalg.norm(displacement) else np.array([0.0, 0.0, 1e-9])
    start, end = 0.0, 1.0
    for axis in range(3):
        if direction[axis] == 0:
            if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                return False
        else:
            first, last = sorted(((lower[axis] - origin[axis]) / direction[axis],
                                  (upper[axis] - origin[axis]) / direction[axis]))
            start, end = max(start, first), min(end, last)
    return start <= end


def step_reference(vertices, faces, displacement, dhat, safety):
    alpha = 1.0
    for vertex, point in enumerate(vertices):
        for face in faces:
            if vertex in face:
                continue
            swept = np.vstack((vertices[face], vertices[face] + displacement[face]))
            if not segment_touches_box(point, displacement[vertex], swept.min(axis=0) - dhat,
                                       swept.max(axis=0) + dhat):
                continue
            motion = np.linalg.norm(displacement[vertex] - displacement[face], axis=1).max()
            if motion:
                alpha = min(alpha, safety * triangle_distance(point, vertices[face]) / motion)
    return alpha


def crossing_reference(vertices, faces):
    """Count pierced edges once, using plane intersection followed by barycentric containment."""
    edges = {tuple(sorted((int(face[i]), int(face[(i + 1) % 3])))) for face in faces for i in range(3)}
    count = 0
    for start, end in edges:
        origin, direction = vertices[start], vertices[end] - vertices[start]
        if not np.linalg.norm(direction):
            continue
        for face in faces:
            if start in face or end in face:
                continue
            a, b, c = vertices[face]
            normal = np.cross(b - a, c - a)
            denominator = np.dot(direction, normal)
            if abs(denominator) < 1e-12:
                continue
            fraction = np.dot(a - origin, normal) / denominator
            if not 0 <= fraction <= 1:
                continue
            uv = np.linalg.lstsq(np.column_stack((b - a, c - a)), origin + fraction * direction - a,
                                 rcond=None)[0]
            if np.all(uv >= -1e-10) and uv.sum() <= 1 + 1e-10:
                count += 1
                break
    return count


def crossing_mesh(endpoint=False, two_planes=False):
    triangle = np.array([[-2, -2, 0], [2, -2, 0], [0, 2, 0]], np.float64)
    other = (np.array([[0, 0, 0], [0.25, 0, 1], [0.5, 0, 1]], np.float64) if endpoint else
             np.array([[-0.25, 0, -1], [0.25, 0, 1], [0.5, 0, -1]], np.float64))
    vertices = np.vstack((triangle, triangle + [0, 0, 0.5], other)) if two_planes else np.vstack((triangle, other))
    return vertices, np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)


@pytest.mark.parametrize("tensor_device", ["cpu", "cuda"])
def test_barrier_autograd_matches_energy_finite_difference(device, tensor_device):
    if tensor_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA PyTorch tensors are unavailable")
    vertices, faces = tilted_point_mesh()
    incoming = np.tile([1.0, 0.0, 0.0], (len(vertices), 1))
    contact = contact_for(device, vertices, faces, strength=0.7)
    expected, pairs = barrier_reference(vertices, faces, contact.dhat, incoming, contact.strength)
    dtype = torch.float64 if tensor_device == "cpu" else torch.float32
    x = torch.tensor(vertices, dtype=dtype, device=tensor_device, requires_grad=True)
    image_gradient = torch.tensor(incoming, dtype=dtype, device=tensor_device)
    output = contact.barrier(x)
    torch.testing.assert_close(output, x, rtol=0, atol=0)
    (output * image_gradient).sum().backward()
    assert x.grad is not None and x.grad.device == x.device and x.grad.dtype == x.dtype
    np.testing.assert_allclose(x.grad.detach().cpu().numpy(), incoming + expected, rtol=3e-4, atol=3e-5)
    assert contact.stats["barrier_pairs"] == pairs == 1
    np.testing.assert_allclose(expected.sum(axis=0), 0, atol=1e-7)


def test_barrier_gathers_all_pairs_and_resets(device):
    triangle = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float64)
    vertices = np.vstack((triangle, triangle + [0, 0, 0.3], [0, 0, 0.1]))
    faces = np.arange(6, dtype=np.uint32).reshape(2, 3)
    contact = contact_for(device, vertices, faces, dhat=0.25)
    incoming = np.tile([1.0, 0.0, 0.0], (len(vertices), 1))
    expected, pairs = barrier_reference(vertices, faces, contact.dhat, incoming)
    actual = contact.barrier_gradient(vertices, torch.tensor(incoming)).numpy()
    assert contact.stats["barrier_pairs"] == pairs == 2
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(actual.sum(axis=0), 0, atol=2e-6)
    far = vertices.copy()
    far[-1, 2] = 2
    np.testing.assert_array_equal(contact.barrier_gradient(far, torch.tensor(incoming)).numpy(), 0)
    assert contact.stats["barrier_pairs"] == 0
    np.testing.assert_allclose(contact.barrier_gradient(vertices, torch.tensor(incoming)).numpy(), actual,
                               rtol=1e-6, atol=1e-6)
    assert contact.stats["barrier_pairs"] == 2
    # Zero incoming gradient disables the force but still reports the geometric pairs.
    np.testing.assert_array_equal(contact.barrier_gradient(vertices, torch.zeros_like(torch.tensor(incoming))).numpy(), 0)
    assert contact.stats["barrier_pairs"] == 2


@pytest.mark.parametrize("motion", ["vertex", "triangle", "deforming_triangle", "rigid", "stationary"])
def test_step_bound_relative_motion_and_reset(device, motion):
    vertices, faces = point_mesh()
    displacement = np.zeros_like(vertices)
    if motion == "vertex":
        displacement[3, 2] = -0.5
    elif motion == "triangle":
        displacement[:3, 2] = 0.5  # The query vertex is stationary inside the triangle's swept box.
    elif motion == "deforming_triangle":
        displacement[:3, 2] = [0.25, 0.5, 0.375]
    elif motion == "rigid":
        displacement[:] = [0.25, -0.5, 0.5]  # Swept boxes overlap the query, but relative motion is zero.
    contact = contact_for(device, vertices, faces, dhat=0.1)
    expected = step_reference(vertices, faces, displacement, contact.dhat, safety=0.8)
    actual = contact.step_bound(vertices, displacement)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert contact.stats["alpha"] == actual
    if motion in ("vertex", "triangle", "deforming_triangle"):
        np.testing.assert_allclose(actual, 0.4, rtol=1e-6)
        moved = vertices + actual * displacement
        assert triangle_distance(moved[3], moved[faces[0]]) > 0
    else:
        assert actual == 1
    assert contact.step_bound(vertices, np.zeros_like(displacement)) == 1
    assert contact.stats["alpha"] == 1


@pytest.mark.parametrize("endpoint,two_planes", [(False, False), (True, False), (False, True)])
def test_intersections_count_edges_once_and_reset(device, endpoint, two_planes):
    vertices, faces = crossing_mesh(endpoint, two_planes)
    contact = contact_for(device, vertices, faces)
    expected = crossing_reference(vertices, faces)
    assert expected == 2
    assert contact.intersections(vertices) == expected
    separated = vertices.copy()
    separated[-3:, 2] += 3
    assert contact.intersections(separated) == 0
    assert contact.intersections(vertices) == expected


def test_zero_length_edge_and_incident_exclusion(device):
    vertices, _ = point_mesh()
    vertices[3] = [0, 0, -1]
    vertices = np.vstack((vertices, [0, 0, 1]))
    faces = np.array([[0, 1, 2], [3, 3, 4]], np.uint32)
    contact = contact_for(device, vertices, faces)
    assert contact.intersections(vertices) == crossing_reference(vertices, faces) == 1

    vertices, faces = point_mesh()
    vertices = vertices[:3]
    contact = contact_for(device, vertices, faces)
    np.testing.assert_array_equal(contact.barrier_gradient(vertices, torch.ones((3, 3))).numpy(), 0)
    assert contact.stats["barrier_pairs"] == 0
    assert contact.intersections(vertices) == 0
    assert contact.step_bound(vertices, np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0]], np.float32)) == 1


@pytest.mark.parametrize("vertex_count", [0, 4])
def test_empty_mesh_operations_are_noops(device, vertex_count):
    vertices = np.zeros((vertex_count, 3), np.float32)
    contact = contact_for(device, vertices, np.empty((0, 3), np.uint32))
    x = torch.tensor(vertices, requires_grad=True)
    contact.barrier(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    assert contact.stats["barrier_pairs"] == 0
    assert contact.intersections(vertices) == 0
    assert contact.step_bound(vertices, np.ones_like(vertices)) == 1


def test_traversal_selection_and_capability_errors(device):
    vertices, faces = point_mesh()
    automatic = SelfContact(faces, len(vertices), 0.5, device=device)
    expected = "ray_query" if device.has_feature(spy.Feature.ray_query) else "trace_ray"
    assert automatic.traversal == expected
    assert contact_for(device, vertices, faces).traversal == "trace_ray"
    with pytest.raises(ValueError):
        SelfContact(faces, len(vertices), 0.5, device=device, traversal="invalid")
    if not device.has_feature(spy.Feature.ray_query):
        with pytest.raises(RuntimeError, match="ray.query|ray quer"):
            SelfContact(faces, len(vertices), 0.5, device=device, traversal="ray_query")
    unavailable = SimpleNamespace(has_feature=lambda feature: False,
                                  info=SimpleNamespace(adapter_name="unsupported-test-device", type=spy.DeviceType.vulkan))
    with pytest.raises(RuntimeError):
        SelfContact(faces, len(vertices), 0.5, device=unavailable)


def test_inline_query_and_pipeline_parity_when_available(device):
    if not device.has_feature(spy.Feature.ray_query):
        pytest.skip("This Vulkan device does not support inline ray queries")
    vertices, faces = tilted_point_mesh()
    query = SelfContact(faces, len(vertices), 0.5, device=device, traversal="ray_query")
    pipeline = contact_for(device, vertices, faces)
    incoming = torch.ones(vertices.shape)
    np.testing.assert_allclose(query.barrier_gradient(vertices, incoming), pipeline.barrier_gradient(vertices, incoming),
                               rtol=1e-5, atol=1e-6)
    assert query.stats["barrier_pairs"] == pipeline.stats["barrier_pairs"]
    displacement = np.zeros_like(vertices)
    displacement[:3, 2] = 0.5
    np.testing.assert_allclose(query.step_bound(vertices, displacement), pipeline.step_bound(vertices, displacement),
                               rtol=1e-6)
    vertices, faces = crossing_mesh(two_planes=True)
    query = SelfContact(faces, len(vertices), 0.5, device=device, traversal="ray_query")
    pipeline = contact_for(device, vertices, faces)
    assert query.intersections(vertices) == pipeline.intersections(vertices) == 2
