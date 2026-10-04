"""Texture recovery: the base color texture (256 x 256) of a floor, from three 512 x 512 views.

A textured floor carries a bunny that casts a sun shadow (distant light) under a dim environment map. The
texels are optimized from gray with Adam so that the renderings of all views match their targets; each
iteration renders one view. The gradient is the interior term's texel derivative: every pixel sample
scatters dL/dI into the four texels of its bilinear footprint.

    python examples/texture.py
"""

import time

import numpy as np
import torch
from PIL import Image, ImageDraw

import falcor2_inverse as fi
from falcor2_inverse.images import frame, save_videos, stack, tonemap
from common import ASSETS, ENVMAP, RESULTS, load_mesh, output_dir, save_curves, save_history

TEXELS = 256
VIEWS = ((0.0, -1.5, 1.6), (1.3, -0.6, 1.6), (-1.2, -0.9, 1.4))


def pattern(n=TEXELS):
    """A colorful test texture: stripes, disks and text, (n, n, 3) in [0, 1]."""
    image = Image.new("RGB", (n, n), (230, 220, 200))
    draw = ImageDraw.Draw(image)
    colors = [(200, 60, 50), (40, 120, 190), (240, 180, 40), (60, 160, 90), (130, 70, 160)]
    for i in range(10):
        draw.rectangle((0, i * n // 10, n, i * n // 10 + n // 40), fill=colors[i % 5])
    for i, (x, y, r) in enumerate([(0.25, 0.3, 0.15), (0.72, 0.28, 0.12), (0.5, 0.7, 0.2), (0.15, 0.8, 0.08)]):
        draw.ellipse(((x - r) * n, (y - r) * n, (x + r) * n, (y + r) * n), fill=colors[(i + 2) % 5],
                     outline=(20, 20, 20), width=3)
    draw.text((0.08 * n, 0.47 * n), "falcor2 PSDR", fill=(10, 10, 10), font_size=n // 8)
    return np.asarray(image, np.float32) / 255


def make_scene(size):
    scene = fi.Scene()
    floor = scene.add_material("floor", base_color_texture=np.ones((TEXELS, TEXELS, 3)), roughness_factor=0.8)
    body = scene.add_material("body", base_color_factor=(0.8, 0.8, 0.8), roughness_factor=0.6)
    positions, faces, uv = fi.grid([0, 0, 0], [0.8, 0, 0], [0, 0.8, 0], 1)
    # Texture rows from the far side of the floor down to the near side.
    scene.add_mesh("floor", positions, faces, floor, texcoords=np.stack([uv[:, 0], 1 - uv[:, 1]], 1))
    scene.add_mesh("bunny", *load_mesh(ASSETS / "meshes" / "bunny.obj", center=(0.1, 0.15, 0.2), size=0.4, y_up=True),
                   body)
    scene.add_distant_light("sun", (-0.4, 0.5, -1.0), (3000.0, 2850.0, 2700.0), cutoff_angle=1.0)
    scene.add_env_map("sky", ENVMAP, intensity=(0.3, 0.3, 0.3))
    for i, eye in enumerate(VIEWS):
        scene.add_camera(f"view{i}", eye, (0.0, 0.0, 0.0), fov_y=42, resolution=(size, size))
    return scene


def run(iterations=300, size=512, spp=1, target_spp=256, lr=0.03, bounces=2, out_dir=RESULTS, video_every=6):
    scene = make_scene(size)
    renderer = fi.Renderer(scene, spp=spp, max_bounces=bounces)
    cameras = list(scene.cameras)
    truth = torch.as_tensor(pattern(), device=fi.torch_device())
    alpha = torch.ones_like(truth[..., :1])

    def texels(rgb):  # the texture parameter holds RGBA texels
        return torch.cat([rgb, alpha], -1)

    with torch.no_grad():
        targets = [renderer({"floor.base_color_texture": texels(truth)}, camera=c, spp=target_spp, seed=12345)
                   for c in cameras]
    rgb = torch.full_like(truth, 0.5).requires_grad_()
    optimizer = torch.optim.Adam([rgb], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 ** (1 / max(1, iterations - 1)))
    history = {"loss": [], "texture_rmse": [], "seconds": []}

    def preview(label):
        with torch.no_grad():
            image = renderer({"floor.base_color_texture": texels(rgb)}, camera=cameras[0], spp=64, seed=7)
        return stack([frame([targets[0], image], ["target, view 0", label]),
                      frame([rgb, truth], ["texture", "true texture"])])

    frames = [preview("initial")]
    start = time.perf_counter()
    for it in range(iterations):
        view = it % len(cameras)
        image = renderer({"floor.base_color_texture": texels(rgb)}, camera=cameras[view])
        loss = fi.mse(image, targets[view])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            rgb.clamp_(0.0, 1.0)
        history["loss"].append(loss.item())
        history["texture_rmse"].append(float(torch.sqrt(torch.mean((rgb.detach() - truth) ** 2))))
        history["seconds"].append(time.perf_counter() - start)
        if (it + 1) % video_every == 0:
            elapsed = time.perf_counter()
            frames.append(preview(f"iteration {it + 1}"))
            start += time.perf_counter() - elapsed  # previews are not timed
    out = output_dir("texture", out_dir)
    save_videos(out / "optimization", frames, fps=10)
    Image.fromarray(frames[-1]).save(out / "final.png")
    Image.fromarray(tonemap(rgb)).save(out / "texture.png")
    save_curves(out / "curves.png", {"image loss": history["loss"], "texture RMSE": history["texture_rmse"]},
                "texture: loss and texel RMSE")
    metrics = {"initial_rmse": float(torch.sqrt(torch.mean((0.5 - truth) ** 2))),
               "final_rmse": history["texture_rmse"][-1],
               "initial_loss": float(np.mean(history["loss"][:3])), "final_loss": float(np.mean(history["loss"][-3:])),
               "seconds_per_iteration": history["seconds"][-1] / iterations}
    save_history(out / "history.json", history, **metrics)
    return metrics


if __name__ == "__main__":
    print(run())
