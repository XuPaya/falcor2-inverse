"""Light position recovery: a point light and a rect light move until shading and shadows match the target.

A bunny and a box stand on a floor, lit by a point light and a rect light (and a dim environment map). Both
lights start at wrong positions; their world-space translations are optimized with Adam. The gradient
comes from the interior term (falloff and incident directions of next-event estimation) and from the
secondary term (the shadow edges move with the lights).

    python examples/moving_light.py
"""

import time

import numpy as np
import torch
from PIL import Image

import falcor2_inverse as fi
from falcor2_inverse.images import frame, save_videos, stack
from common import ASSETS, ENVMAP, RESULTS, load_mesh, output_dir, save_curves, save_history

TRUE_POSITIONS = [(0.55, -0.45, 1.1), (-0.5, 0.25, 1.3)]
START_POSITIONS = [(-0.3, -0.7, 0.8), (0.45, 0.45, 1.0)]
KEYS = ["point.translation", "rect.translation"]


def make_scene(size):
    scene = fi.Scene()
    floor = scene.add_material("floor", base_color_factor=(0.65, 0.62, 0.58), roughness_factor=0.8)
    body = scene.add_material("body", base_color_factor=(0.75, 0.45, 0.3), roughness_factor=0.5)
    crate = scene.add_material("crate", base_color_factor=(0.3, 0.45, 0.7), roughness_factor=0.4)
    scene.add_mesh("floor", *fi.grid([0, 0, 0], [1.4, 0, 0], [0, 1.4, 0], 1)[:2], floor)
    positions, faces = load_mesh(ASSETS / "meshes" / "bunny.obj", center=(-0.1, 0.0, 0.0), size=0.5, y_up=True)
    positions[:, 2] -= positions[:, 2].min()
    scene.add_mesh("bunny", positions, faces, body)
    scene.add_mesh("crate", *fi.box([0.35, 0.3, 0.12], 0.12), crate)
    scene.add_point_light("point", TRUE_POSITIONS[0], (1.2, 1.1, 1.0))
    scene.add_rect_light("rect", TRUE_POSITIONS[1], (5.0, 5.5, 6.5), 0.35)
    scene.add_env_map("sky", ENVMAP, intensity=(0.1, 0.1, 0.1))
    scene.add_camera("camera", (0.0, -2.0, 1.5), (0.0, 0.0, 0.15), fov_y=40, resolution=(size, size))
    return scene


def run(iterations=300, size=512, spp=4, target_spp=256, lr=0.04, bounces=2, out_dir=RESULTS, video_every=4):
    scene = make_scene(size)
    # Moving lights change shading and shadows, not what the camera sees directly: two of the four terms.
    renderer = fi.Renderer(scene, spp=spp, max_bounces=bounces, terms=("interior", "secondary"))
    with torch.no_grad():
        target = renderer(spp=target_spp, seed=12345)
    params = {k: torch.tensor(p, device=target.device, requires_grad=True) for k, p in zip(KEYS, START_POSITIONS)}
    truth = torch.tensor(TRUE_POSITIONS, device=target.device)
    optimizer = torch.optim.Adam(params.values(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.25 ** (1 / max(1, iterations - 1)))
    history = {"loss": [], "distance": [], "positions": [], "seconds": []}

    def preview(label):
        with torch.no_grad():
            return frame([target, renderer(params, spp=64, seed=7)], ["target", label])

    frames = [preview("initial")]
    start = time.perf_counter()
    for it in range(iterations):
        loss = fi.mse(renderer(params), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        positions = torch.stack([params[k].detach() for k in KEYS])
        history["loss"].append(loss.item())
        history["distance"].append(torch.linalg.norm(positions - truth, dim=1).tolist())
        history["positions"].append(positions.tolist())
        history["seconds"].append(time.perf_counter() - start)
        if (it + 1) % video_every == 0:
            elapsed = time.perf_counter()
            frames.append(preview(f"iteration {it + 1}"))
            start += time.perf_counter() - elapsed  # previews are not timed
    out = output_dir("moving_light", out_dir)
    save_videos(out / "optimization", frames, fps=12)
    Image.fromarray(stack([frames[0], frames[-1]])).save(out / "images.png")
    distance = np.asarray(history["distance"])
    save_curves(out / "curves.png", {"image loss": history["loss"], "point light distance": distance[:, 0],
                                     "rect light distance": distance[:, 1]}, "moving light: loss and distances")
    metrics = {"initial_distance": np.linalg.norm(np.subtract(START_POSITIONS, TRUE_POSITIONS), axis=1).tolist(),
               "final_distance": history["distance"][-1], "seconds_per_iteration": history["seconds"][-1] / iterations}
    save_history(out / "history.json", history, **metrics)
    return metrics


if __name__ == "__main__":
    print(run())
