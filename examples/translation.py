"""Object-translation recovery (cf. Falcor's Cornell-box transform example and PSDR-Enzyme's translation tests).

A box above a floor, lit by an emissive quad, a point light and a rect light. The box's (x, y) offset is
recovered from one image, which needs all four PSDR terms: the box's silhouettes move (primary and
pixel boundary), its shading changes (interior) and its shadows move (photon-mapping secondary edges).
The box floats slightly above the floor: as in PSDR-Enzyme, edge sampling misses the surface behind an
edge that touches it (the ray from the edge starts an epsilon away from the edge).

    python examples/translation.py
"""

import torch
from PIL import Image

import falcor2_inverse as fi
from falcor2_inverse.images import frame, save_videos
from common import RESULTS, output_dir, save_curves, save_history


def make_scene(size=48):
    scene = fi.Scene()
    floor = scene.add_material("floor", base_color_factor=(0.7, 0.7, 0.7), roughness_factor=0.8)
    red = scene.add_material("red", base_color_factor=(0.8, 0.3, 0.2), roughness_factor=0.5)
    emitter = scene.add_material("emitter", base_color_factor=(0.0, 0.0, 0.0), emissive_factor=(5.0, 5.0, 5.0))
    scene.add_mesh("floor", *fi.quad([0, 0, 0], [2, 0, 0], [0, 2, 0])[:2], floor)
    scene.add_mesh("box", *fi.box([0, 0, 0], 0.25), red, translation=(0.0, 0.0, 0.3))
    scene.add_mesh("emitter", *fi.quad([-0.6, 0.4, 1.5], [0, 0.3, 0], [0.3, 0, 0])[:2], emitter)
    scene.add_point_light("point", (0.8, -0.7, 1.3), (1.5, 1.5, 1.5))
    scene.add_rect_light("rect", (0.7, 0.7, 1.6), (4.0, 4.0, 4.0), 0.4)
    scene.add_camera("camera", (0.0, -2.4, 2.0), (0.0, 0.0, 0.2), resolution=(size, size))
    return scene


def run(iterations=120, spp=16, target_spp=1024, out_dir=RESULTS, initial=(0.2, -0.15)):
    scene = make_scene()
    renderer = fi.Renderer(scene, spp=spp)
    with torch.no_grad():
        target = renderer(spp=target_spp, seed=12345)
    base = scene.parameters()["box.translation"]
    # Only x and y are optimized: the translation is computed from them.
    offset = torch.tensor(initial, device=base.device, requires_grad=True)

    def translation():
        return base + torch.cat([offset, offset.new_zeros(1)])

    optimizer = torch.optim.Adam([offset], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 ** (1 / max(1, iterations - 1)))
    with torch.no_grad():
        start_image = renderer({"box.translation": translation()}, spp=target_spp, seed=12345)
    history = {"loss": [], "distance": [float(torch.linalg.norm(offset))], "offset": []}
    frames = []
    for it in range(iterations):
        image = renderer({"box.translation": translation()})
        loss = fi.mse(image, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        history["loss"].append(loss.item())
        history["distance"].append(float(torch.linalg.norm(offset)))
        history["offset"].append(offset.tolist())
        if it % 2 == 0:
            frames.append(frame([target, image], ["target", f"iteration {it}"], scale=256 // target.shape[1]))
    with torch.no_grad():
        final_image = renderer({"box.translation": translation()}, spp=target_spp, seed=12345)
    out = output_dir("translation", out_dir)
    save_videos(out / "optimization", frames)
    Image.fromarray(frame([start_image, target, final_image, (final_image - target).abs() * 4],
                          ["initial", "target", "optimized", "|optimized - target| x4"], scale=4)).save(out / "images.png")
    save_curves(out / "curves.png", {"image loss": history["loss"], "distance to target": history["distance"]},
                "translation: loss and |offset - target offset|")
    metrics = {"initial_distance": history["distance"][0], "final_distance": history["distance"][-1],
               "initial_loss": fi.mse(start_image, target).item(), "final_loss": fi.mse(final_image, target).item(),
               "recovered_offset": offset.tolist()}
    save_history(out / "history.json", history, **metrics)
    return metrics


if __name__ == "__main__":
    print(run())
