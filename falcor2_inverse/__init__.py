"""Inverse rendering with falcor2's path-space differentiable renderer (falcor2.psdr) from PyTorch.

    import torch
    import falcor2_inverse as fi

    scene = fi.Scene.load("scene.xml")              # Mitsuba, USD, glTF, OBJ, PLY... or build one with add_*()
    renderer = fi.Renderer(scene, spp=4)            # scene.cameras[0]
    target = fi.load_image("target.exr")
    params = scene.parameters("red.*")              # {"red.base_color_factor": tensor, ...}
    params.requires_grad_("red.base_color_factor")
    optimizer = torch.optim.Adam(params.trainable(), lr=0.01)
    for _ in range(100):
        loss = fi.mse(renderer(params), target)    # renders with the parameter values
        optimizer.zero_grad()
        loss.backward()                             # gradients of the tensors that require them
        optimizer.step()

Parameter values may be any tensors, e.g. computed by a network from other parameters.
"""

from falcor2_inverse.contact import SelfContact, mean_edge_length
from falcor2_inverse.device import create_device, default_device, torch_device
from falcor2_inverse.geometry import box, chamfer, disk, grid, laplacian, quad, sphere, subdivide
from falcor2_inverse.images import load_image, save_image, tonemap
from falcor2_inverse.loaders import read_mesh
from falcor2_inverse.losses import l1, mse, relative_mse
from falcor2_inverse.optim import AdamUniform, LargeSteps
from falcor2_inverse.params import ParameterDict
from falcor2_inverse.render import Renderer, render
from falcor2_inverse.scene import Scene

__all__ = ["AdamUniform", "LargeSteps", "ParameterDict", "Renderer", "Scene", "SelfContact", "box", "chamfer",
           "create_device", "default_device", "disk", "grid", "l1", "laplacian", "load_image", "mean_edge_length",
           "mse", "quad", "read_mesh", "relative_mse", "render", "save_image", "sphere", "subdivide", "tonemap",
           "torch_device"]
