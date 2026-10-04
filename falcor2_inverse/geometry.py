"""Triangle mesh utilities: normals, edges, Laplacians, subdivision, procedural shapes and a surface distance."""

import numpy as np
import torch


def vertex_normals(positions, faces) -> np.ndarray:
    """Area-weighted vertex normals (V, 3)."""
    positions, faces = np.asarray(positions, np.float64), np.asarray(faces, np.int64)
    p = positions[faces]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    normals = np.zeros_like(positions)
    for k in range(3):
        np.add.at(normals, faces[:, k], n)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def tangents(normals) -> np.ndarray:
    """Unit tangents orthogonal to the normals (a fixed tangent would be parallel to some normals)."""
    normals = np.asarray(normals, np.float64)
    up = np.where(np.abs(normals[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    t = np.cross(up, normals)
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)


def mesh_edges(faces) -> np.ndarray:
    """Unique undirected edges (E, 2)."""
    faces = np.asarray(faces, np.int64)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(edges, axis=1), axis=0)


def laplacian(faces, vertex_count, device=None) -> torch.Tensor:
    """Combinatorial graph Laplacian L = D - A of a mesh (sparse, float32)."""
    edges = torch.as_tensor(mesh_edges(faces), device=device)
    rows = torch.cat([edges[:, 0], edges[:, 1], torch.arange(vertex_count, device=device)])
    cols = torch.cat([edges[:, 1], edges[:, 0], torch.arange(vertex_count, device=device)])
    degree = torch.bincount(edges.reshape(-1), minlength=vertex_count).float()
    values = torch.cat([-torch.ones(2 * len(edges), device=device), degree])
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (vertex_count, vertex_count)).coalesce()


def subdivide(positions, faces):
    """Midpoint subdivision: every triangle becomes four. Returns the new positions (float32) and triangles."""
    positions, faces = np.asarray(positions, np.float64), np.asarray(faces, np.int64)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, inverse = np.unique(edges, axis=0, return_inverse=True)
    ab, bc, ca = (len(positions) + inverse.reshape(3, -1)).astype(np.int64)
    a, b, c = faces.T
    triangles = np.concatenate([np.stack(f, 1) for f in ((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca))])
    return np.concatenate([positions, positions[unique].mean(1)]).astype(np.float32), triangles


def surface_samples(positions, faces, count, rng):
    p = np.asarray(positions, np.float64)[np.asarray(faces)]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    tri = rng.choice(len(p), count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return p[tri, 0] + u[:, None] * (p[tri, 1] - p[tri, 0]) + v[:, None] * (p[tri, 2] - p[tri, 0])


def chamfer(a, b, count=8000, seed=0) -> float:
    """Symmetric mean distance between the surfaces of meshes a and b, each (positions, faces)."""
    rng = np.random.default_rng(seed)
    pa, pb = (torch.as_tensor(surface_samples(*mesh, count, rng)) for mesh in (a, b))
    return float(0.5 * (torch.cdist(pa, pb).min(1).values.mean() + torch.cdist(pb, pa).min(1).values.mean()))


# ----------------------------------------------------------------------------
# Procedural shapes: positions (float32) and triangles (uint32)


def sphere(center=(0, 0, 0), radius=1.0, subdivisions=3):
    """Icosphere (outward-facing triangles)."""
    t = (1 + 5 ** 0.5) / 2
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
         [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7],
         [9, 8, 1]]
    v = [np.asarray(x, np.float64) / np.linalg.norm(x) for x in v]
    for _ in range(subdivisions):
        mid, faces = {}, []

        def middle(a, b):
            key = (min(a, b), max(a, b))
            if key not in mid:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                mid[key] = len(v) - 1
            return mid[key]

        for a, b, c in f:
            ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
            faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = faces
    return (np.asarray(v) * radius + np.asarray(center)).astype(np.float32), np.asarray(f, np.uint32)


def grid(center, u, v, n):
    """n x n quad grid in the plane spanned by u and v (facing cross(u, v)), with texture coordinates."""
    s, t = np.meshgrid(np.linspace(-1, 1, n + 1), np.linspace(-1, 1, n + 1))
    positions = np.asarray(center) + s[..., None] * np.asarray(u) + t[..., None] * np.asarray(v)
    ids = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b, c, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]
    faces = np.stack([np.stack([a, b, c], -1), np.stack([c, b, d], -1)], 2).reshape(-1, 3)
    uv = np.stack([(s + 1) / 2, (t + 1) / 2], -1)
    return positions.reshape(-1, 3).astype(np.float32), faces.astype(np.uint32), uv.reshape(-1, 2).astype(np.float32)


def quad(center, u, v):
    """Quad facing along cross(u, v), with texture coordinates."""
    return grid(center, u, v, 1)


def box(center, half):
    """Closed box with outward-facing triangles and flat faces (four vertices per face)."""
    positions, faces = [], []
    c, half = np.asarray(center, np.float64), np.broadcast_to(np.asarray(half, np.float64), (3,))
    for axis in range(3):
        for sign in (-1.0, 1.0):
            n = np.zeros(3)
            n[axis] = sign
            u, v = np.roll(n, 1), np.roll(n, 2)
            if sign < 0:
                u, v = v, u
            k = len(positions)
            positions += [c + half * (n - u - v), c + half * (n + u - v), c + half * (n - u + v), c + half * (n + u + v)]
            faces += [[k, k + 1, k + 2], [k + 2, k + 1, k + 3]]
    return np.asarray(positions, np.float32), np.asarray(faces, np.uint32)


def disk(center, radius, segments):
    """Triangle fan facing +z: vertex 0 is the center, 1..segments the rim."""
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    rim = np.stack([np.cos(angles), np.sin(angles), np.zeros(segments)], 1) * radius
    positions = np.concatenate([[[0, 0, 0]], rim]) + np.asarray(center)
    faces = [[0, 1 + i, 1 + (i + 1) % segments] for i in range(segments)]
    return positions.astype(np.float32), np.asarray(faces, np.uint32)
