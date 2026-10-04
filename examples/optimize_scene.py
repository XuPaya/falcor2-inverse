"""Optimize parameters of a scene file (Mitsuba 3 XML, USD, glTF, OBJ, PLY...), as in Mitsuba's Cornell-box
tutorial: render a target with the scene's own values, set the chosen parameters to 0.5, and recover them.

    python examples/optimize_scene.py                                      # assets/scenes/cornell_box.xml
    python examples/optimize_scene.py scene.usda --params "*.base_color_factor"
    python examples/optimize_scene.py --list                               # the parameters of a scene

Parameters are chosen by key or glob pattern (see Scene.parameters()).
"""

import argparse

import torch
from PIL import Image

import falcor2_inverse as fi
from falcor2_inverse.images import frame, save_videos
from common import ASSETS, RESULTS, output_dir, save_curves, save_history

CORNELL_BOX = ASSETS / "scenes" / "cornell_box.xml"


def run(path=CORNELL_BOX, patterns=("red.base_color_factor", "green.base_color_factor"), iterations=200, spp=16,
        target_spp=1024, size=256, lr=0.02, out_dir=RESULTS):
    scene = fi.Scene.load(path)
    camera = next(iter(scene.cameras.values()))
    scale = size / max(camera.width, camera.height)  # render at most size x size pixels
    camera.width, camera.height = round(camera.width * scale), round(camera.height * scale)
    renderer = fi.Renderer(scene, camera, spp=spp, max_bounces=scene.metadata.get("max_bounces", 3))
    truth = scene.parameters(*patterns)
    print("optimizing", ", ".join(truth))
    with torch.no_grad():
        target = renderer(spp=target_spp, seed=12345)
    params = fi.ParameterDict({k: torch.full_like(v, 0.5).requires_grad_() for k, v in truth.items()})
    optimizer = torch.optim.Adam(params.trainable(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 ** (1 / max(1, iterations - 1)))

    def error():
        return {k: float(torch.linalg.norm(params[k].detach() - truth[k])) for k in params}

    with torch.no_grad():
        start_image = renderer(params, spp=target_spp, seed=12345)
    history = {"loss": [], "errors": [error()]}
    frames = []
    for it in range(iterations):
        image = renderer(params)
        loss = fi.mse(image, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        history["loss"].append(loss.item())
        history["errors"].append(error())
        if it % 2 == 0:
            frames.append(frame([target, image], ["target", f"iteration {it}"]))
    with torch.no_grad():
        final_image = renderer(params, spp=target_spp, seed=12345)
    out = output_dir("optimize_scene", out_dir)
    save_videos(out / "optimization", frames)
    Image.fromarray(frame([start_image, target, final_image], ["initial", "target", "optimized"])).save(
        out / "images.png")
    curves = {"image loss": history["loss"]}
    curves.update({f"{k} error": [e[k] for e in history["errors"]] for k in params})
    save_curves(out / "curves.png", curves, "scene parameters: loss and errors")
    metrics = {"initial_errors": history["errors"][0], "final_errors": history["errors"][-1],
               "initial_loss": fi.mse(start_image, target).item(), "final_loss": fi.mse(final_image, target).item(),
               "truth": {k: v.tolist() for k, v in truth.items()},
               "recovered": {k: v.tolist() for k, v in params.items()}}
    save_history(out / "history.json", history, **metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", nargs="?", default=str(CORNELL_BOX))
    parser.add_argument("--params", nargs="+", default=["red.base_color_factor", "green.base_color_factor"],
                        help="keys or glob patterns of the parameters to optimize")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--list", action="store_true", help="list the scene's parameters and exit")
    args = parser.parse_args()
    if args.list:
        for key, value in fi.Scene.load(args.scene).parameters().items():
            print(f"{key:48s} {tuple(value.shape)}")
    else:
        print(run(args.scene, args.params, args.iterations))
