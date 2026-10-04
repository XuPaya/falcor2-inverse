"""Material and light recovery (cf. Falcor's sphere-materials example and Mitsuba's Cornell-box color tutorial).

A StandardMaterial floor, an OpenPBRMaterial box, an emissive quad, a point light and a rect light.
``materials`` recovers the floor base color and the box's OpenPBR base color and specular roughness;
``lights`` recovers the point-light intensity, rect-light radiance and the quad's emissive factor.

    python examples/appearance.py [materials] [lights]
"""

import sys

import torch
from PIL import Image

import falcor2_inverse as fi
from falcor2_inverse.images import frame, save_videos
from common import RESULTS, output_dir, save_curves, save_history

# The parameters of each experiment: initial value, learning rate and bounds.
SETTINGS = {
    "materials": {"floor.base_color_factor": ([0.5, 0.5, 0.5], 0.02, None),
                  "box.base_color": ([0.5, 0.5, 0.5], 0.02, None),
                  "box.specular_roughness": ([0.6], 0.02, (0.05, 1.0))},
    # The rect light is dim: its radiance has the weakest gradient and converges slowest.
    "lights": {"point.intensity": ([0.5, 0.5, 0.5], 0.05, None),
               "rect.radiance": ([1.0, 1.0, 1.0], 0.1, None),
               "emitter.emissive_factor": ([2.0, 2.0, 2.0], 0.1, (0.0, None))},
}


def make_scene(size=48):
    scene = fi.Scene()
    floor = scene.add_material("floor", base_color_factor=(0.7, 0.45, 0.3), roughness_factor=0.8)
    box = scene.add_material("box", "openpbr", base_color=(0.2, 0.5, 0.8), specular_roughness=0.25)
    emitter = scene.add_material("emitter", base_color_factor=(0.0, 0.0, 0.0), emissive_factor=(5.0, 4.0, 3.0))
    scene.add_mesh("floor", *fi.quad([0, 0, 0], [2, 0, 0], [0, 2, 0])[:2], floor)
    scene.add_mesh("box", *fi.box([0, 0, 0.3], 0.3), box)
    scene.add_mesh("emitter", *fi.quad([-0.6, 0.4, 1.5], [0, 0.3, 0], [0.3, 0, 0])[:2], emitter)  # facing down
    scene.add_point_light("point", (0.8, -0.7, 1.3), (1.5, 1.5, 1.5))
    scene.add_rect_light("rect", (0.7, 0.7, 1.6), (4.0, 4.0, 5.0), 0.4)
    scene.add_camera("camera", (0.0, -2.4, 2.0), (0.0, 0.0, 0.2), resolution=(size, size))
    return scene


def run(kind="materials", iterations=None, spp=16, target_spp=1024, out_dir=RESULTS):
    iterations = iterations or (150 if kind == "materials" else 300)
    settings = SETTINGS[kind]
    scene = make_scene()
    renderer = fi.Renderer(scene, spp=spp)
    truth = scene.parameters(*settings)
    with torch.no_grad():
        target = renderer(spp=target_spp, seed=12345)
    params = {k: torch.tensor(v, device=target.device, requires_grad=True) for k, (v, _, _) in settings.items()}
    optimizer = torch.optim.Adam([{"params": [params[k]], "lr": lr} for k, (_, lr, _) in settings.items()])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 ** (1 / max(1, iterations - 1)))

    def error():
        value, true = (torch.cat([p[k].detach() for k in settings]) for p in (params, truth))
        return float(torch.linalg.norm(value - true) / torch.linalg.norm(true))

    with torch.no_grad():  # with the target's samples: without the Monte Carlo noise of the iterations' images
        start_image = renderer(params, spp=target_spp, seed=12345)
    history = {"loss": [], "error": [error()]}
    frames = []
    for it in range(iterations):
        image = renderer(params)
        loss = fi.mse(image, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            for key, (_, _, bounds) in settings.items():
                if bounds is not None:
                    params[key].clamp_(*bounds)
        history["loss"].append(loss.item())
        history["error"].append(error())
        if it % 2 == 0:
            frames.append(frame([target, image], ["target", f"iteration {it}"], scale=256 // target.shape[1]))
    with torch.no_grad():
        final_image = renderer(params, spp=target_spp, seed=12345)
    out = output_dir(kind, out_dir)
    save_videos(out / "optimization", frames)
    Image.fromarray(frame([start_image, target, final_image, (final_image - target).abs() * 4],
                          ["initial", "target", "optimized", "|optimized - target| x4"], scale=4)).save(out / "images.png")
    save_curves(out / "curves.png", {"image loss": history["loss"], "parameter error": history["error"]},
                f"{kind}: loss and relative parameter error")
    metrics = {"initial_error": history["error"][0], "final_error": history["error"][-1],
               "initial_loss": fi.mse(start_image, target).item(), "final_loss": fi.mse(final_image, target).item(),
               "truth": {k: v.tolist() for k, v in truth.items()},
               "recovered": {k: v.tolist() for k, v in params.items()}}
    save_history(out / "history.json", history, **metrics)
    return metrics


if __name__ == "__main__":
    for kind in sys.argv[1:] or SETTINGS:
        print(kind, run(kind))
