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
    """IPC-style barrier and step bound for a triangle mesh with fixed triangles (faces) and vertex_count vertices.

    ``traversal`` selects inline ``ray_query`` or a ``trace_ray`` pipeline; ``auto`` prefers inline queries when
    supported and otherwise uses a ray-tracing pipeline (including Vulkan devices without ray queries).
    """

    def __init__(self, faces, vertex_count, dhat, strength=1.0, device=None, *, traversal="auto"):
        self.device = device = device or default_device()
        if traversal not in ("auto", "ray_query", "trace_ray"):
            raise ValueError("traversal must be auto, ray_query or trace_ray")
        if traversal == "auto":
            traversal = "ray_query" if device.has_feature(spy.Feature.ray_query) else "trace_ray"
        feature = spy.Feature.ray_query if traversal == "ray_query" else spy.Feature.ray_tracing
        if not device.has_feature(feature):
            raise RuntimeError(f"SelfContact traversal={traversal!r} is unsupported by {device.info.adapter_name}.")
        self.traversal = traversal
        self.dhat, self.strength = dhat, strength
        self.faces = _numpy(faces).astype(np.uint32).reshape(-1, 3)
        self.edges = mesh_edges(self.faces).astype(np.uint32)
        self.vertex_count = vertex_count
        names = KERNELS if traversal == "ray_query" else ("triangle_bounds",)
        self.kernels = {k: device.create_compute_kernel(device.load_program("contact.slang", [k])) for k in names}
        self._pipelines = {}
        if traversal == "trace_ray":
            for name in KERNELS[1:]:
                raygen, any_hit, miss = f"{name}_trace", f"{name}_any_hit", f"{name}_miss"
                program = device.load_program("contact.slang", [raygen, "contact_intersection", any_hit, miss])
                pipeline = device.create_ray_tracing_pipeline(
                    program=program, hit_groups=[{"hit_group_name": "contact_hit_group",
                        "intersection_entry_point": "contact_intersection", "any_hit_entry_point": any_hit}],
                    max_recursion=1, max_ray_payload_size=32, max_attribute_size=4)
                table = device.create_shader_table(program=program, ray_gen_entry_points=[raygen],
                                                   miss_entry_points=[miss], hit_group_names=["contact_hit_group"])
                self._pipelines[name] = pipeline, table
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access

        def buffer(data, struct_size):
            data = np.ascontiguousarray(data).reshape(-1)
            if not data.size:
                data = np.zeros(struct_size // data.dtype.itemsize, dtype=data.dtype)
            return device.create_buffer(data=data, struct_size=struct_size, usage=usage)

        zeros = np.zeros((vertex_count, 3), np.float32)
        self.buffers = {"positions": buffer(zeros, 12), "displacements": buffer(zeros, 12),
                        "gradient": buffer(zeros, 4), "faces": buffer(self.faces, 12), "edges": buffer(self.edges, 8),
                        "bounds": buffer(np.zeros((len(self.faces), 6), np.float32), 12),
                        "count": buffer(np.zeros(1, np.uint32), 4)}
        self._accel = self._acceleration_structures() if len(self.faces) else None
        self.stats = {"barrier_pairs": 0, "alpha": 1.0}

    def _acceleration_structures(self):
        """A BLAS of the triangles' boxes in one TLAS instance (as falcor2's PSDR photon BVH)."""
        builds = []
        inputs = [spy.AccelerationStructureBuildInputProceduralPrimitives({
            "aabb_buffers": [self.buffers["bounds"]], "aabb_stride": 24, "primitive_count": len(self.faces),
            "flags": spy.AccelerationStructureGeometryFlags.no_duplicate_any_hit_invocation})]
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
        if self._accel is None:
            return
        self.buffers["positions"].copy_from_numpy(np.ascontiguousarray(_numpy(x), np.float32))
        if dx is not None:
            self.buffers["displacements"].copy_from_numpy(np.ascontiguousarray(_numpy(dx), np.float32))
        self.kernels["triangle_bounds"].dispatch(
            thread_count=[len(self.faces), 1, 1], vars={"triangle_count": len(self.faces), "margin": margin,
                "scale": 0.0 if dx is None else 1.0, "positions": self.buffers["positions"],
                "displacements": self.buffers["displacements"], "faces": self.buffers["faces"],
                "bounds": self.buffers["bounds"]})
        encoder = self.device.create_command_encoder()
        for desc, accel, scratch in self._accel[0]:
            encoder.build_acceleration_structure(desc=desc, dst=accel, src=None, scratch_buffer=scratch)
        self.device.submit_command_buffer(encoder.finish())

    def _dispatch(self, kernel, threads, start=0, **kwargs):
        """Run a query kernel on the boxes; returns count[0], which starts at start."""
        if self._accel is None or threads == 0:
            return int(start)
        self.buffers["count"].copy_from_numpy(np.array([start], np.uint32))
        parameters = {"boxes": self._accel[0][-1][1], "positions": self.buffers["positions"],
                      "faces": self.buffers["faces"], "bounds": self.buffers["bounds"],
                      "count": self.buffers["count"], **kwargs}
        if self.traversal == "ray_query":
            self.kernels[kernel].dispatch(thread_count=[threads, 1, 1], vars=parameters)
        else:
            pipeline, table = self._pipelines[kernel]
            encoder = self.device.create_command_encoder()
            with encoder.begin_ray_tracing_pass() as ray_pass:
                cursor = spy.ShaderCursor(ray_pass.bind_pipeline(pipeline, table))
                for key, value in parameters.items():
                    cursor[key] = value
                ray_pass.dispatch_rays(0, [threads, 1, 1])
            self.device.submit_command_buffer(encoder.finish())
        return int(self.buffers["count"].to_numpy().view(np.uint32)[0])

    def barrier(self, vertices: torch.Tensor) -> torch.Tensor:
        """The vertices (V, 3); backpropagating through them adds barrier_gradient() to their gradient."""
        return _Barrier.apply(vertices, self)

    def barrier_gradient(self, vertices, image_gradient) -> torch.Tensor:
        """Gradient of kappa * sum b(d) over the pairs closer than dhat, with kappa such that the barrier force at
        d = dhat / 2 matches the strong (99th percentile) image gradients."""
        g = torch.as_tensor(image_gradient)
        if self._accel is None or self.vertex_count == 0:
            self.stats["barrier_pairs"] = 0
            return torch.zeros_like(g)
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
        if self._accel is None or self.vertex_count == 0:
            self.stats["alpha"] = 1.0
            return 1.0
        self._build(vertices, self.dhat, displacement)
        bits = self._dispatch("step_bound", self.vertex_count, start=np.float32(1).view(np.uint32),
                              vertex_count=self.vertex_count, safety=safety,
                              displacements=self.buffers["displacements"])
        self.stats["alpha"] = alpha = float(np.uint32(bits).view(np.float32))
        return alpha

    def intersections(self, vertices) -> int:
        """Number of mesh edges that pierce non-incident triangles (self-intersections)."""
        if self._accel is None or not len(self.edges):
            return 0
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
