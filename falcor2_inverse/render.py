"""Differentiable rendering of falcor2 scenes as a PyTorch operation.

``Renderer(scene, camera)(params)`` returns a linear-radiance image (H, W, 3) on the torch device. ``params`` maps
parameter keys (see Scene.parameters()) to tensors; their values are written into the scene, and tensors that
require gradients receive dL/dvalue from falcor2's path-space differentiable renderer (falcor2.psdr): the image and
its derivative are estimated with independent samples, so that gradients of losses like mean((I - target)^2)
are unbiased.
"""

import numpy as np
import torch

import falcor2 as f2
from falcor2.psdr import PSDR, TERMS

from falcor2_inverse.device import torch_device


class Renderer:
    """Renders a Scene with PSDR from one camera (or one of several, chosen per call).

    spp: samples per pixel of the image; grad_spp: of its derivative (default: spp). max_bounces: path length.
    terms: the PSDR terms of gradients (default: all four when geometry or lights move, else only "interior").
    photon_radius: PSDR's photon radius for shadow and reflected-silhouette edges (default: scene-relative).
    """

    def __init__(self, scene, camera=None, spp=4, grad_spp=None, max_bounces=2, terms=None, photon_radius=None,
                 seed=0):
        self.scene, self.spp, self.grad_spp, self.max_bounces = scene, spp, grad_spp, max_bounces
        self.terms, self.photon_radius, self.seed = terms, photon_radius, seed
        self.camera = self._camera(camera) if camera is not None else None
        self._psdr, self._version = {}, None

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
            self._psdr, self._version = {}, scene.version
            scene.update()
        size = (camera.width, camera.height)
        psdr = self._psdr.get(size)
        if psdr is None:
            psdr = self._psdr[size] = PSDR(scene.device, scene.f2, camera, max_bounces=self.max_bounces,
                                           photon_radius=self.photon_radius)
        else:
            state, geometry_state, current = psdr.refreshed
            if current is not camera:
                psdr.set_camera(camera)  # refreshes edges and light values
            elif state != scene.state:
                psdr.update(geometry=geometry_state != scene.geometry_state)
        psdr.refreshed = (scene.state, scene.geometry_state, camera)
        return psdr

    def __call__(self, params=None, camera=None, spp=None, seed=None) -> torch.Tensor:
        """Render; params: tensors (or arrays) by parameter key, written into the scene."""
        keys = list(params or {})
        values = [v if isinstance(v, torch.Tensor) else torch.as_tensor(np.asarray(v, np.float32))
                  for v in (params[k] for k in keys)]
        if seed is None:
            seed, self.seed = self.seed, self.seed + 1
        call = _Call(self, keys, self._camera(camera), spp or self.spp, seed)
        if torch.is_grad_enabled() and any(v.requires_grad for v in values):
            return _Render.apply(call, *values)
        return call.forward(values)


class _Call:
    """One rendering: what forward() and backward() share."""

    def __init__(self, renderer, keys, camera, spp, seed):
        self.renderer, self.keys, self.camera, self.spp, self.seed = renderer, keys, camera, spp, seed

    def _apply(self, values):
        self.renderer.scene.apply(dict(zip(self.keys, values)))
        return self.renderer.psdr(self.camera)

    def forward(self, values) -> torch.Tensor:
        psdr = self._apply(values)
        image = psdr.forward(spp=self.spp, seed=2 * self.seed, terms=())["primal"]
        return torch.from_numpy(image).to(torch_device())

    def backward(self, values, grad_image, needs_grad):
        psdr = self._apply(values)  # another rendering may have changed the scene since forward()
        params = self.renderer.scene.params
        wanted = [k for k, need in zip(self.keys, needs_grad) if need]
        terms = self.renderer.terms or (TERMS if any(params[k].geometric for k in wanted) else ("interior",))
        dl_di = grad_image.detach().to(torch.float32).cpu().numpy()
        grad_vertex, grad_params = psdr.backward(dl_di, spp=self.renderer.grad_spp or self.spp,
                                                 seed=2 * self.seed + 1, terms=terms)
        return [params[k].grad(psdr, grad_vertex, grad_params, v) if need else None
                for k, v, need in zip(self.keys, values, needs_grad)]


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
