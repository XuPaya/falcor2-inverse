"""Differentiable rendering of falcor2 scenes as a PyTorch operation.

``Renderer(scene, camera)(params)`` returns a linear-radiance image (H, W, 3) on the torch device. ``params`` maps
parameter keys (see Scene.parameters()) to tensors; their values are written into the scene, and tensors that
require gradients receive dL/dvalue from falcor2's path-space differentiable renderer (falcor2.psdr): the image and
its derivative are estimated with independent samples, so that gradients of losses like mean((I - target)^2)
are unbiased.
"""

import contextlib

import numpy as np
import torch

import falcor2 as f2
from falcor2.psdr import PSDR, TERMS

from falcor2_inverse.device import torch_device
from falcor2_inverse.params import EnvMapTexture, LightEmission, MaterialFactor, MaterialTexture


class Renderer:
    """Renders a Scene with PSDR from one camera (or one of several, chosen per call).

    spp: samples per pixel of the image; grad_spp: of its derivative (default: spp). max_bounces: path length.
    terms: the PSDR terms of gradients (default: all four when geometry or lights move, else only "interior").
    photon_radius: PSDR's photon radius for shadow and reflected-silhouette edges (default: scene-relative).
    photon_count: PSDR's edge photons per pass of the shadow-edge term (default: PSDR's); more photons, less variance.
    light_tracing: None (off) or PSDR.set_light_tracing() arguments (paths_per_pixel, target, partition_scale): light
    paths from the analytic lights, so that light reaching surfaces only through delta or peaked transmission (the sun
    through glass or sheer curtains) is in the image and its derivatives (part of the interior term).
    check_visibility: let PSDR drop edge photons hidden from the receiver or the source (default off; large radii in
    cluttered scenes otherwise gather hidden silhouette edges and overestimate shadow-edge derivatives).
    vertex_grad_clamp: bound of each path sample's interior-term vertex gradient, relative to its contribution per
    scene extent (heavy-tailed near sharp glossy lobes; default: PSDR's, 0: off).
    light_sampler: the falcor2 light sampler of next-event estimation (default: PSDR's f2.PowerLightSampler, which
    weights lights by emitted power and so starves small lights next to a distant sun disk; f2.UniformLightSampler
    picks every light, and every emissive triangle, alike).
    primal: False returns a zero image instead of rendering it, for callers that only backpropagate an adjoint they
    computed elsewhere (e.g. at a converged forward render); the gradients are unchanged (default True).
    accumulate: see accumulating().
    """

    def __init__(self, scene, camera=None, spp=4, grad_spp=None, max_bounces=2, terms=None, photon_radius=None,
                 seed=0, light_sampler=None, vertex_grad_clamp=None, check_visibility=False, photon_count=None):
        self.scene, self.spp, self.grad_spp, self.max_bounces = scene, spp, grad_spp, max_bounces
        self.terms, self.photon_radius, self.seed, self.light_sampler = terms, photon_radius, seed, light_sampler
        self.vertex_grad_clamp, self.check_visibility = vertex_grad_clamp, check_visibility
        self.photon_count = photon_count
        self.light_tracing = None
        self.primal, self.accumulate = True, False
        self.camera = self._camera(camera) if camera is not None else None
        self._psdr, self._version = {}, None
        self._pending = {}  # id(PSDR) -> (PSDR, texel owners, {key: (leaf tensor, parameter slots)})

    def _camera(self, camera) -> f2.Camera:
        if camera is None:
            if self.camera is not None:
                return self.camera
            cameras = self.scene.cameras
            if not cameras:
                raise ValueError("The scene has no camera; add one with Scene.add_camera()")
            return next(iter(cameras.values()))
        return self.scene.cameras[camera] if isinstance(camera, str) else camera

    def psdr(self, camera=None) -> PSDR:
        """The PSDR object rendering from a camera (one per resolution, rebuilt when the scene's structure
        changes), updated to the scene's current state."""
        camera = self._camera(camera)
        scene = self.scene
        if self._version != scene.version:
            self.flush_gradients()
            self._psdr, self._version = {}, scene.version
            scene.update()
        size = (camera.width, camera.height)
        psdr = self._psdr.get(size)
        if psdr is None:
            psdr = self._psdr[size] = PSDR(scene.device, scene.f2, camera, max_bounces=self.max_bounces,
                                           photon_radius=self.photon_radius, light_sampler=self.light_sampler,
                                           vertex_grad_clamp=self.vertex_grad_clamp,
                                           check_visibility=self.check_visibility,
                                           **({} if self.photon_count is None else {"photon_count": self.photon_count}))
        else:
            state, geometry_state, current = psdr.refreshed
            if geometry_state != scene.geometry_state:
                self.flush_gradients()  # the vertex layout may change
            if current is not camera:
                psdr.set_camera(camera)  # refreshes edges and light values
            elif state != scene.state:
                psdr.update(geometry=geometry_state != scene.geometry_state)
        settings = dict(self.light_tracing) if self.light_tracing else None
        if getattr(psdr, "light_tracing_settings", None) != settings:
            psdr.set_light_tracing(**settings) if settings else psdr.set_light_tracing(0)
            psdr.light_tracing_settings = settings
        psdr.refreshed = (scene.state, scene.geometry_state, camera)
        return psdr

    def __call__(self, params=None, camera=None, spp=None, seed=None, primal=None) -> torch.Tensor:
        """Render; params: tensors (or arrays) by parameter key, written into the scene. primal: overrides
        self.primal (False: zero image, gradients only)."""
        keys = list(params or {})
        values = [v if isinstance(v, torch.Tensor) else torch.as_tensor(np.asarray(v, np.float32))
                  for v in (params[k] for k in keys)]
        if seed is None:
            seed, self.seed = self.seed, self.seed + 1
        call = _Call(self, keys, self._camera(camera), spp or self.spp, seed,
                     self.primal if primal is None else primal, values)
        if torch.is_grad_enabled() and any(v.requires_grad for v in values):
            return _Render.apply(call, *values)
        return call.forward(values)

    @contextlib.contextmanager
    def accumulating(self):
        """Within the block, the gradients of material factors, material and environment-map textures and light
        emission that are leaf tensors (requires_grad, not computed from other tensors) add up on the GPU over the
        backward passes, and reach the tensors' .grad when the block ends (or at flush_gradients()): one readback of
        PSDR's gradient buffer instead of one per backward pass, e.g. per multi-view iteration. Read .grad only after
        the block. Backward passes with other parameters (geometry, transforms) or non-leaf tensors flush first and
        run as outside the block."""
        previous, self.accumulate = self.accumulate, True
        try:
            yield self
        finally:
            self.accumulate = previous
            self.flush_gradients()

    def flush_gradients(self):
        """Adds the gradients accumulated by accumulate-mode backward passes to their leaf tensors' .grad."""
        pending, self._pending = self._pending, {}
        for psdr, _owners, entries in pending.values():
            _, grad_params = psdr.gradients()
            _add_grads(grad_params, entries.values())


SLOT_PARAMS = (MaterialFactor, MaterialTexture, LightEmission, EnvMapTexture)


def _slots(psdr, p) -> slice:
    """PSDR parameter slots of a factor, texture or light-emission parameter."""
    if isinstance(p, MaterialFactor):
        first = psdr.material_slot(p.material, p.name)
        return slice(first, first + p.size)
    if isinstance(p, MaterialTexture):
        return psdr.texture_slot(p.material, p.name)
    if isinstance(p, LightEmission):
        first = psdr.light_slot(p.light, "intensity" if isinstance(p.light, f2.EnvMapLight) else "emission")
        return slice(first, first + 3)
    return psdr.texture_slot(p.light)


def _add_grads(grad_params: np.ndarray, entries):
    """Adds slots of grad_params (uploaded once per torch device) to the .grad of each (tensor, slots) entry."""
    uploaded = {}
    with torch.no_grad():
        for tensor, slots in entries:
            if tensor.device not in uploaded:
                uploaded[tensor.device] = torch.from_numpy(grad_params).to(tensor.device)
            g = uploaded[tensor.device][slots].reshape(tensor.shape).to(tensor.dtype)
            if tensor.grad is None:
                tensor.grad = g.clone()
            else:
                tensor.grad += g


class _Call:
    """One rendering: what forward() and backward() share."""

    def __init__(self, renderer, keys, camera, spp, seed, primal=True, inputs=()):
        self.renderer, self.keys, self.camera, self.spp, self.seed = renderer, keys, camera, spp, seed
        self.primal, self.inputs = primal, list(inputs)

    def _apply(self, values):
        self.renderer.scene.apply(dict(zip(self.keys, values)))
        return self.renderer.psdr(self.camera)

    def forward(self, values) -> torch.Tensor:
        psdr = self._apply(values)
        if not self.primal:
            return torch.zeros((psdr.height, psdr.width, 3), dtype=torch.float32, device=torch_device())
        image = psdr.forward(spp=self.spp, seed=2 * self.seed, terms=())["primal"]
        return torch.from_numpy(image).to(torch_device())

    def backward(self, values, grad_image, needs_grad):
        renderer = self.renderer
        psdr = self._apply(values)  # another rendering may have changed the scene since forward()
        params = renderer.scene.params
        wanted = [k for k, need in zip(self.keys, needs_grad) if need]
        terms = renderer.terms or (TERMS if any(params[k].geometric for k in wanted) else ("interior",))
        # Texel parameters only for the textures whose gradients are wanted (none: no texel derivatives at all).
        owners = [(params[k].material, params[k].name) if isinstance(params[k], MaterialTexture) else params[k].light
                  for k in wanted if isinstance(params[k], (MaterialTexture, EnvMapTexture))]
        # Accumulate mode: the passes of one PSDR object add up on the GPU while they want the same keys of the same
        # leaf tensors (the sum can't be split between tensors or keys); anything else flushes first.
        leaves = dict(zip(wanted, [v for v, need in zip(self.inputs, needs_grad) if need]))
        defer = (renderer.accumulate and wanted and len(leaves) == len(wanted)
                 and all(isinstance(params[k], SLOT_PARAMS) for k in wanted)
                 and all(v.is_leaf and v.requires_grad for v in leaves.values()))
        pending = renderer._pending.get(id(psdr))
        if pending is not None and not (defer and pending[1] == owners and pending[2].keys() == leaves.keys()
                                        and all(pending[2][k][0] is v for k, v in leaves.items())):
            renderer.flush_gradients()
            pending = None
        psdr.set_texel_gradients(owners)
        dl_di = grad_image.detach().to(torch.float32).cpu().numpy()
        spp = renderer.grad_spp or self.spp
        if defer:
            psdr.backward(dl_di, spp=spp, seed=2 * self.seed + 1, terms=terms, accumulate=True)
            if pending is None:
                renderer._pending[id(psdr)] = (psdr, owners, {k: (v, _slots(psdr, params[k]))
                                                              for k, v in leaves.items()})
            return [None] * len(self.keys)
        grad_vertex, grad_params = psdr.backward(dl_di, spp=spp, seed=2 * self.seed + 1, terms=terms)
        # Slot-based gradients go to the values' device in one transfer, each key taking its slice there.
        grads, uploaded = [], {}
        for k, v, need in zip(self.keys, values, needs_grad):
            if not need:
                grads.append(None)
            elif not isinstance(params[k], SLOT_PARAMS):
                grads.append(params[k].grad(psdr, grad_vertex, grad_params, v))
            else:
                if v.device not in uploaded:
                    uploaded[v.device] = torch.from_numpy(grad_params).to(v.device)
                grads.append(uploaded[v.device][_slots(psdr, params[k])].reshape(v.shape).clone())
        return grads


class _Render(torch.autograd.Function):
    @staticmethod
    def forward(ctx, call, *values):
        ctx.call = call
        ctx.save_for_backward(*values)
        return call.forward(list(values))

    @staticmethod
    def backward(ctx, grad_image):
        grads = ctx.call.backward(list(ctx.saved_tensors), grad_image, ctx.needs_input_grad[1:])
        return (None, *grads)


def render(scene, params=None, camera=None, spp=4, max_bounces=2, seed=0, **options) -> torch.Tensor:
    """One-off rendering (builds a Renderer; keep one for repeated renderings)."""
    return Renderer(scene, camera, spp=spp, max_bounces=max_bounces, seed=seed, **options)(params)
