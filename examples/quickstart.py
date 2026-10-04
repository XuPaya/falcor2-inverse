"""Quickstart: build a scene, render a target image of it, and recover a material's color and roughness, a light's
position and an object's position from the target in a PyTorch loop.

    python examples/quickstart.py
"""

import torch
from PIL import Image

import falcor2_inverse as fi
from falcor2_inverse.images import frame
from common import RESULTS, output_dir


def make_scene(size=128):
    scene = fi.Scene()
    floor = scene.add_material("floor", base_color_factor=(0.7, 0.7, 0.7), roughness_factor=0.8)
    red = scene.add_material("red", base_color_factor=(0.8, 0.2, 0.1), roughness_factor=0.4)
    positions, faces, uv = fi.quad((0, 0, 0), (2, 0, 0), (0, 2, 0))
    scene.add_mesh("floor", positions, faces, floor, texcoords=uv)
    scene.add_mesh("ball", *fi.sphere((0, 0, 0), 0.3, 3), red, translation=(0.1, 0.0, 0.35))
    scene.add_point_light("lamp", (0.6, -0.5, 1.2), (1.5, 1.5, 1.5))
    scene.add_rect_light("panel", (-0.5, 0.4, 1.4), (4.0, 4.0, 4.0), 0.4)
    scene.add_camera("camera", eye=(0.0, -2.4, 2.0), target=(0, 0, 0.2), resolution=(size, size))
    return scene


def run(iterations=200, spp=8, out_dir=RESULTS):
    scene = make_scene()
    renderer = fi.Renderer(scene, spp=spp)
    with torch.no_grad():
        target = renderer(spp=1024)  # the scene as built

    # Parameters by key; the values are tensors. Start from wrong values and optimize the ones that require grad.
    params = scene.parameters("red.base_color_factor", "red.roughness_factor", "lamp.translation", "ball.translation")
    truth = {k: v.clone() for k, v in params.items()}
    params["red.base_color_factor"][:] = 0.5
    params["red.roughness_factor"][:] = 0.8
    params["lamp.translation"] += torch.tensor([0.3, 0.2, -0.2], device=target.device)
    params["ball.translation"] += torch.tensor([-0.15, 0.1, 0.0], device=target.device)
    params.requires_grad_("*")
    optimizer = torch.optim.Adam(params.trainable(), lr=0.02)
    # Gradients are Monte Carlo estimates: decaying the learning rate averages out their noise.
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 ** (1 / iterations))
    with torch.no_grad():
        initial = renderer(params, spp=1024)
    for it in range(iterations):
        image = renderer(params)
        loss = fi.mse(image, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():  # keep the material parameters valid
            params["red.base_color_factor"].clamp_(0.0, 1.0)
            params["red.roughness_factor"].clamp_(0.05, 1.0)
        if it % 20 == 0:
            print(f"iteration {it:3d}: loss {loss.item():.6f}", flush=True)
    with torch.no_grad():
        final = renderer(params, spp=1024)
    out = output_dir("quickstart", out_dir)
    Image.fromarray(frame([initial, final, target], ["initial", "optimized", "target"], scale=2)).save(
        out / "images.png")
    errors = {k: float(torch.linalg.norm(params[k].detach() - truth[k])) for k in params}
    for k in params:
        print(f"{k}: {params[k].detach().cpu().numpy().round(3)} (true {truth[k].cpu().numpy().round(3)})")
    return {"errors": errors, "initial_loss": fi.mse(initial, target).item(), "final_loss": fi.mse(final, target).item()}


if __name__ == "__main__":
    print(run())
