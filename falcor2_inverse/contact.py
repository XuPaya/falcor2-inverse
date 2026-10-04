"""Self-contact handling for mesh optimization in the spirit of incremental potential contact (IPC, Li et al. 2020).

- ``barrier(vertices)``: the identity, whose backward pass adds the gradient of a log barrier on the distances
  between vertices and non-incident triangles closer than dhat, which pushes surface sheets apart before they
  touch. Its stiffness is set relative to the incoming (image) gradient.
- ``step_bound(vertices, displacement)``: conservative advancement (as in additive CCD, Li et al. 2021), the
  largest fraction of a step that moves no vertex through a triangle it approaches, which keeps an
  intersection-free mesh intersection-free (for vertex-triangle contacts; edge-edge contacts are not handled).
- ``intersections(vertices)``: the number of edges that pierce non-incident triangles.

With large steps (falcor2_inverse.optim), whose vertices are linear in the optimized variable u:

    contact = SelfContact(faces, len(vertices), dhat=0.25 * mean_edge_length(vertices, faces))
    for ...:
        x = contact.barrier(steps.from_differential(u))
        loss = ...render with x...
        optimizer.zero_grad()
        loss.backward()
        before = u.detach().clone()
        optimizer.step()
        with torch.no_grad():  # shorten the step
            u.copy_(before + contact.step_bound(x, steps.from_differential(u) - x) * (u - before))

The proximity queries run on the GPU (shaders/contact.slang): the triangles' bounding boxes, grown by dhat, go
into a ray-tracing acceleration structure, and a ray from each vertex visits the boxes it touches.
"""

import numpy as np
import slangpy as spy
import torch

from falcor2_inverse.device import default_device
from falcor2_inverse.geometry import mesh_edges

KERNELS = ("triangle_bounds", "barrier", "step_bound", "edge_crossings")


def _numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def mean_edge_length(positions, faces) -> float:
    positions, edges = _numpy(positions).astype(np.float64), mesh_edges(_numpy(faces))
    return float(np.linalg.norm(positions[edges[:, 1]] - positions[edges[:, 0]], axis=1).mean())


class SelfContact:
    """IPC-style barrier and step bound for a triangle mesh with fixed triangles (faces) and vertex_count vertices."""

    def __init__(self, faces, vertex_count, dhat, strength=1.0, device=None):
        self.device = device = device or default_device()
        self.dhat, self.strength = dhat, strength
        self.faces = _numpy(faces).astype(np.uint32)
        self.edges = mesh_edges(self.faces).astype(np.uint32)
        self.vertex_count = vertex_count
        self.kernels = {k: device.create_compute_kernel(device.load_program("contact.slang", [k])) for k in KERNELS}
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access

        def buffer(data, struct_size):
            return device.create_buffer(data=np.ascontiguousarray(data).reshape(-1), struct_size=struct_size,
                                        usage=usage)

        zeros = np.zeros((vertex_count, 3), np.float32)
        self.buffers = {"positions": buffer(zeros, 12), "displacements": buffer(zeros, 12),
                        "gradient": buffer(zeros, 4), "faces": buffer(self.faces, 12), "edges": buffer(self.edges, 8),
                        "bounds": buffer(np.zeros((len(self.faces), 6), np.float32), 12),
                        "count": buffer(np.zeros(1, np.uint32), 4)}
        self._accel = self._acceleration_structures()
        self.stats = {"barrier_pairs": 0, "alpha": 1.0}

    def _acceleration_structures(self):
        """A BLAS of the triangles' boxes in one TLAS instance (as falcor2's PSDR photon BVH)."""
        builds = []
        inputs = [spy.AccelerationStructureBuildInputProceduralPrimitives({
            "aabb_buffers": [self.buffers["bounds"]], "aabb_stride": 24, "primitive_count": len(self.faces)})]
        for kind in (spy.AccelerationStructureKind.bottom_level, spy.AccelerationStructureKind.top_level):
            desc = spy.AccelerationStructureBuildDesc(
                {"inputs": inputs, "flags": spy.AccelerationStructureBuildFlags.prefer_fast_build})
            sizes = self.device.get_acceleration_structure_sizes(desc)
            accel = self.device.create_acceleration_structure(kind=kind, size=sizes.acceleration_structure_size)
            scratch = self.device.create_buffer(size=sizes.scratch_size, usage=spy.BufferUsage.unordered_access)
            builds.append((desc, accel, scratch))
            if kind == spy.AccelerationStructureKind.bottom_level:
                instances = self.device.create_acceleration_structure_instance_list(1)
                instances.write(0, {"transform": spy.float3x4.identity(), "instance_mask": 0xFF,
                                    "acceleration_structure": accel.handle})
                inputs = [instances.build_input_instances()]
        return builds, instances

    def _build(self, x, margin, dx=None):
        """Upload the vertices (and displacements) and build the boxes of the triangles, swept by the
        displacements and grown by margin."""
        self.buffers["positions"].copy_from_numpy(np.ascontiguousarray(_numpy(x), np.float32))
        if dx is not None:
            self.buffers["displacements"].copy_from_numpy(np.ascontiguousarray(_numpy(dx), np.float32))
        self.kernels["triangle_bounds"].dispatch(
            thread_count=[len(self.faces), 1, 1], triangle_count=len(self.faces), margin=margin,
            scale=0.0 if dx is None else 1.0, positions=self.buffers["positions"],
            displacements=self.buffers["displacements"], faces=self.buffers["faces"], bounds=self.buffers["bounds"])
        encoder = self.device.create_command_encoder()
        for desc, accel, scratch in self._accel[0]:
            encoder.build_acceleration_structure(desc=desc, dst=accel, src=None, scratch_buffer=scratch)
        self.device.submit_command_buffer(encoder.finish())

    def _dispatch(self, kernel, threads, start=0, **kwargs):
        """Run a query kernel on the boxes; returns count[0], which starts at start."""
        self.buffers["count"].copy_from_numpy(np.array([start], np.uint32))
        self.kernels[kernel].dispatch(thread_count=[threads, 1, 1], boxes=self._accel[0][-1][1],
                                      positions=self.buffers["positions"], faces=self.buffers["faces"],
                                      count=self.buffers["count"], **kwargs)
        return int(self.buffers["count"].to_numpy().view(np.uint32)[0])

    def barrier(self, vertices: torch.Tensor) -> torch.Tensor:
        """The vertices (V, 3); backpropagating through them adds barrier_gradient() to their gradient."""
        return _Barrier.apply(vertices, self)

    def barrier_gradient(self, vertices, image_gradient) -> torch.Tensor:
        """Gradient of kappa * sum b(d) over the pairs closer than dhat, with kappa such that the barrier force at
        d = dhat / 2 matches the strong (99th percentile) image gradients."""
        g = torch.as_tensor(image_gradient)
        force = self.dhat * (np.log(2) + 0.5)  # |b'(dhat / 2)|
        kappa = self.strength * float(torch.quantile(torch.linalg.norm(g.detach().float(), dim=1), 0.99)) / force
        self._build(vertices, self.dhat)
        self.buffers["gradient"].copy_from_numpy(np.zeros((self.vertex_count, 3), np.float32))
        self.stats["barrier_pairs"] = self._dispatch("barrier", self.vertex_count, vertex_count=self.vertex_count,
                                                     dhat=self.dhat, kappa=kappa, gradient=self.buffers["gradient"])
        gradient = self.buffers["gradient"].to_numpy().view(np.float32).reshape(-1, 3)
        return torch.as_tensor(gradient, dtype=g.dtype, device=g.device)

    def step_bound(self, vertices, displacement, safety=0.8) -> float:
        """Largest alpha <= 1 such that vertices + alpha displacement moves no vertex by more than safety times its
        distance to a triangle, relative to the triangle's motion (conservative advancement)."""
        self._build(vertices, self.dhat, displacement)
        bits = self._dispatch("step_bound", self.vertex_count, start=np.float32(1).view(np.uint32),
                              vertex_count=self.vertex_count, safety=safety,
                              displacements=self.buffers["displacements"])
        self.stats["alpha"] = alpha = float(np.uint32(bits).view(np.float32))
        return alpha

    def intersections(self, vertices) -> int:
        """Number of mesh edges that pierce non-incident triangles (self-intersections)."""
        x = _numpy(vertices)
        self._build(x, 1e-6 * float(np.ptp(x, axis=0).max()))
        return self._dispatch("edge_crossings", len(self.edges), edge_count=len(self.edges),
                              edges=self.buffers["edges"])


class _Barrier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vertices, contact):
        ctx.contact = contact
        ctx.save_for_backward(vertices)
        return vertices.clone()

    @staticmethod
    def backward(ctx, grad):
        (vertices,) = ctx.saved_tensors
        return grad + ctx.contact.barrier_gradient(vertices, grad), None
