"""Planar shape optimization (cf. Falcor's planar star-shape example; PSDR-Enzyme / Mitsuba mesh optimization).

A flat triangle fan floats above a dark floor under a rect light and is seen from above. Its rim
vertices (x, y) are optimized so that its outline and cast shadow match a five-pointed star, with a
smoothness regularizer on the rim's second differences. The gradient comes from the boundary terms:
the rim is a primary silhouette, the pixel boundaries cut it, and its shadow edge is a secondary edge.

    python examples/shape.py
"""

import numpy as np
import torch
from PIL import Image

import falcor2_inverse as fi
from falcor2_inverse.images import frame, save_videos
from common import RESULTS, output_dir, save_curves, save_history

SEGMENTS = 40
RADIUS = 0.4
SMOOTHNESS = 1e-4


def star_rim():
    angles = np.linspace(0, 2 * np.pi, SEGMENTS, endpoint=False)
    radius = RADIUS * (1 + 0.3 * np.cos(5 * angles))
    return np.stack([np.cos(angles), np.sin(angles)], 1) * radius[:, None]


def make_scene(size=64):
    scene = fi.Scene()
    floor = scene.add_material("floor", base_color_factor=(0.25, 0.25, 0.3), roughness_factor=1.0)
    plate = scene.add_material("plate", base_color_factor=(0.9, 0.8, 0.5), roughness_factor=1.0)
    scene.add_mesh("floor", *fi.quad([0, 0, 0], [2, 0, 0], [0, 2, 0])[:2], floor)
    scene.add_mesh("plate", *fi.disk([0, 0, 0], RADIUS, SEGMENTS), plate, translation=(0, 0, 0.3))
    scene.add_rect_light("rect", (0.25, 0.2, 1.8), (8.0, 8.0, 8.0), 0.3)
    scene.add_camera("camera", (0.0, -0.4, 2.2), (0.0, 0.0, 0.0), fov_y=40, resolution=(size, size))
    return scene


def run(iterations=200, spp=8, target_spp=512, out_dir=RESULTS, smoothness=SMOOTHNESS):
    scene = make_scene()
    renderer = fi.Renderer(scene, spp=spp, max_bounces=1)
    # The rim vertices by angle, in the geometry's vertex order (the fan's center is at the origin).
    positions = scene.parameters()["plate.vertex_positions"]
    rim = torch.nonzero(torch.linalg.norm(positions[:, :2], dim=1) > 1e-6).squeeze(1)
    rim = rim[torch.argsort(torch.remainder(torch.atan2(positions[rim, 1], positions[rim, 0]), 2 * np.pi))]

    def vertices(xy):
        """The plate's vertices with the rim at xy (differentiable)."""
        v = positions.clone()
        v[rim, :2] = xy
        return v

    star = torch.as_tensor(star_rim(), dtype=torch.float32, device=positions.device)
    with torch.no_grad():
        target = renderer({"plate.vertex_positions": vertices(star)}, spp=target_spp, seed=12345)
        start_image = renderer({"plate.vertex_positions": positions}, spp=target_spp, seed=12345)
    xy = positions[rim, :2].clone().requires_grad_()
    optimizer = torch.optim.Adam([xy], lr=0.004)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 ** (1 / max(1, iterations - 1)))

    def rms():
        return float(torch.sqrt(((xy.detach() - star) ** 2).sum(1).mean()))

    history = {"loss": [], "image_loss": [], "rms": [rms()]}
    frames = []
    for it in range(iterations):
        image = renderer({"plate.vertex_positions": vertices(xy)})
        image_loss = fi.mse(image, target)
        # Second differences along the rim (the mesh Laplacian would also pull the rim towards the fan center).
        second = 2 * xy - xy.roll(1, 0) - xy.roll(-1, 0)
        loss = image_loss + smoothness * (second ** 2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        history["loss"].append(loss.item())
        history["image_loss"].append(image_loss.item())
        history["rms"].append(rms())
        if it % 2 == 0:
            frames.append(frame([target, image], ["target", f"iteration {it}"], scale=256 // target.shape[1]))
    with torch.no_grad():
        final_image = renderer({"plate.vertex_positions": vertices(xy)}, spp=target_spp, seed=12345)
    out = output_dir("shape", out_dir)
    save_videos(out / "optimization", frames)
    Image.fromarray(frame([start_image, target, final_image, (final_image - target).abs() * 4],
                          ["initial", "target", "optimized", "|optimized - target| x4"], scale=3)).save(out / "images.png")
    save_curves(out / "curves.png", {"image loss": history["image_loss"], "rim RMS error": history["rms"]},
                "shape: loss and RMS distance of rim vertices to the star")
    metrics = {"initial_rms": history["rms"][0], "final_rms": history["rms"][-1],
               "initial_loss": history["image_loss"][0], "final_loss": float(np.mean(history["image_loss"][-10:]))}
    save_history(out / "history.json", history, **metrics)
    return metrics


if __name__ == "__main__":
    print(run())
