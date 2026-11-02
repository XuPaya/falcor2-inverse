"""Validate rendering, every gradient term and a short PyTorch inverse-rendering fit.

From this repository's root, with falcor2's environment:
    PYTHONPATH=. ../falcor2/.venv/bin/python examples/validate_backend.py

The first run compiles several shaders. Results are printed as JSON and saved to
results/validate_backend/diagnostics.json; --output selects another directory.
Use --mixed-materials to fit all three material families together, saving to
results/validate_backend_mixed/ by default.
"""

import argparse
import json
from pathlib import Path
import time

import torch
from PIL import Image

import falcor2_inverse as fi
from falcor2_inverse.images import frame
from falcor2.psdr import TERMS


def make_scene(size, mixed_materials=False):
    scene = fi.Scene()
    floor = scene.add_material("floor", base_color_factor=(0.6, 0.6, 0.6), roughness_factor=0.8)
    if mixed_materials:
        red = scene.add_material("red", "openpbr", base_color=(0.8, 0.2, 0.1), specular_roughness=0.8)
        blue = scene.add_material("blue", "SpongeCakeMaterial", albedo=(0.1, 0.3, 0.8),
                                  roughness=0.8, thickness=1.0, base_weight=0.0)
        scene.add_mesh("cloth", *fi.sphere((0, 0, 0), 0.3, 2), blue, translation=(-0.4, 0, 0.45))
    else:
        red = scene.add_material("red", base_color_factor=(0.8, 0.2, 0.1), roughness_factor=0.8)
    positions, faces, uv = fi.quad((0, 0, 0), (2, 0, 0), (0, 2, 0))
    scene.add_mesh("floor", positions, faces, floor, texcoords=uv)
    scene.add_mesh("ball", *fi.sphere((0, 0, 0), 0.3, 2), red,
                   translation=(0.4 if mixed_materials else 0.1, 0, 0.45))
    scene.add_point_light("lamp", (0.6, -0.5, 1.2), (1.5, 1.5, 1.5))
    scene.add_camera("camera", eye=(0, -2.4, 2), target=(0, 0, 0.2), resolution=(size, size))
    return scene


def validate(size=24, iterations=20, output=Path("results/validate_backend"), mixed_materials=False):
    started = time.monotonic()
    scene = make_scene(size, mixed_materials)
    color_keys = (("floor.base_color_factor", "red.base_color", "blue.albedo") if mixed_materials
                  else ("red.base_color_factor",))
    renderer = fi.Renderer(scene, spp=8, grad_spp=8, max_bounces=2, photon_radius=0.05,
                           photon_count=4096, check_visibility=True)
    print("Rendering target (the first run compiles shaders)...", flush=True)
    with torch.no_grad():
        target = renderer(spp=64, seed=101)
    assert torch.isfinite(target).all() and target.max() > 0
    report = {"backend": scene.device.info.type.name, "adapter": scene.device.info.adapter_name,
              "traversal": renderer.psdr().traversal, "image_mean": target.mean().item(), "gradient_norms": {},
              "material_families": sorted({m.slang_type_name for m in scene.f2.materials})}
    params = scene.parameters(*color_keys, "ball.translation", "lamp.translation")
    params.requires_grad_("*")
    # Differentiate each estimator independently so one working term cannot hide a broken one.
    for term in TERMS:
        print(f"Checking {term} gradients...", flush=True)
        renderer.terms = (term,)
        for value in params.values():
            value.grad = None
        image = renderer(params, seed=5)
        weights = torch.linspace(0.2, 1, size, device=image.device)[None, :, None]
        (image * weights).mean().backward()
        norms = {}
        for key, value in params.items():
            assert value.grad is not None and torch.isfinite(value.grad).all(), (term, key)
            norms[key] = value.grad.norm().item()
        assert norms["ball.translation"] > 0, (term, norms)
        if term == "interior":
            assert all(norms[key] > 0 for key in color_keys), norms
        report["gradient_norms"][term] = norms
        print(f"Checked {term}: {norms}", flush=True)

    renderer.terms = ("interior",)
    fit = scene.parameters(*color_keys)
    truth = {key: value.clone() for key, value in fit.items()}
    for key, value in fit.items():
        value[:] = torch.tensor([0.2, 0.5, 0.6], device=value.device)
    fit.requires_grad_("*")
    optimizer = torch.optim.Adam(fit.trainable(), lr=0.04)
    with torch.no_grad():
        initial_image = renderer(fit, spp=64, seed=101)
        initial = fi.mse(initial_image, target).item()
    print(f"Optimizing material color for {iterations} iterations...", flush=True)
    for iteration in range(iterations):
        optimizer.zero_grad()
        loss = fi.mse(renderer(fit), target)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fit.trainable())
        optimizer.step()
        with torch.no_grad():
            for value in fit.values():
                value.clamp_(0, 1)
    with torch.no_grad():
        final_image = renderer(fit, spp=64, seed=101)
        final = fi.mse(final_image, target).item()
    assert final < 0.4 * initial, (initial, final)
    report.update(initial_loss=initial, final_loss=final, iterations=iterations,
                  optimized_colors={key: value.detach().cpu().tolist() for key, value in fit.items()},
                  target_colors={key: value.cpu().tolist() for key, value in truth.items()},
                  elapsed_seconds=time.monotonic() - started)
    if not mixed_materials:
        report.update(optimized_color=report["optimized_colors"][color_keys[0]],
                      target_color=report["target_colors"][color_keys[0]])
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame([initial_image, final_image, target], ["initial", "optimized", "target"], scale=4)).save(
        output / "comparison.png")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--mixed-materials", action="store_true",
                        help="Fit Standard, OpenPBR and SpongeCake material colors together.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.size <= 0 or args.iterations <= 0:
        parser.error("--size and --iterations must be positive")
    if args.output is None:
        args.output = Path("results/validate_backend_mixed" if args.mixed_materials else "results/validate_backend")
    result = validate(args.size, args.iterations, args.output, args.mixed_materials)
    args.output.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2)
    (args.output / "diagnostics.json").write_text(text + "\n")
    print(text)
