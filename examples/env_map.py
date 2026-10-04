"""Environment map recovery: the lighting of a scene from renderings of a light probe.

A rough metal sphere, which reflects the environment (blurred), and a diffuse sphere, which shows its irradiance,
are seen from eight directions against the environment. The environment map's 64 x 32 texels are optimized from a
constant map, in log space (radiance spans orders of magnitude, and stays positive), with a loss on relative errors
and a smoothness prior that fills in directions no view constrains: any differentiable function of tensors can
give a parameter its value, and any loss can use it.

    python examples/env_map.py
"""

import numpy as np
import torch
from PIL import Image

import falcor2_inverse as fi
from falcor2_inverse.images import frame, save_videos, stack
from common import ENVMAP, RESULTS, output_dir, save_curves, save_history

HEIGHT, WIDTH = 32, 64


def make_scene(size, env_map, count=8):
    scene = fi.Scene()
    metal = scene.add_material("metal", base_color_factor=(0.95, 0.95, 0.95), metallic_factor=1.0,
                               roughness_factor=0.2)
    plaster = scene.add_material("plaster", base_color_factor=(0.8, 0.8, 0.8), roughness_factor=1.0)
    scene.add_mesh("probe", *fi.sphere((0, 0, 0), 0.3, 4), metal, translation=(-0.35, 0.0, 0.0))
    scene.add_mesh("ball", *fi.sphere((0, 0, 0), 0.3, 4), plaster, translation=(0.35, 0.0, 0.0))
    scene.add_env_map("sky", env_map, rotation=(np.pi / 2, 0, 0))  # the map's up (+y) to +z
    for i in range(count):  # around the spheres, alternately above and below them
        azimuth, elevation = 2 * np.pi * i / count, np.radians(30 if i % 2 == 0 else -15)
        eye = 2.5 * np.array([np.sin(azimuth) * np.cos(elevation), -np.cos(azimuth) * np.cos(elevation),
                              np.sin(elevation)])
        scene.add_camera(f"view{i}", eye, (0, 0, 0), fov_y=40, resolution=(size, size))
    return scene


def run(iterations=400, size=128, spp=16, target_spp=1024, lr=0.2, smoothness=0.05, out_dir=RESULTS,
        video_every=5):
    device = fi.torch_device()
    full = fi.load_image(ENVMAP, device)  # latitude-longitude map
    truth = torch.nn.functional.avg_pool2d(full.permute(2, 0, 1)[None], full.shape[0] // HEIGHT)[0].permute(1, 2, 0)
    alpha = torch.ones_like(truth[..., :1])

    def texels(rgb):  # environment map texels are RGBA
        return torch.cat([rgb, alpha], -1)

    # The scene starts with a constant map (whose sampling distribution is kept): no hint of the true lighting.
    scene = make_scene(size, np.full((HEIGHT, WIDTH, 3), 0.5, np.float32))
    renderer = fi.Renderer(scene, spp=spp)
    cameras = list(scene.cameras)
    with torch.no_grad():
        targets = [renderer({"sky.texture": texels(truth)}, camera=c, spp=target_spp, seed=12345) for c in cameras]
    log_rgb = torch.full_like(truth, float(torch.stack(targets).mean().log())).requires_grad_()
    optimizer = torch.optim.Adam([log_rgb], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 ** (1 / max(1, iterations - 1)))

    def log_error():
        return float(torch.sqrt(torch.mean((log_rgb.detach() - truth.clamp_min(1e-4).log()) ** 2)))

    def preview(label):
        with torch.no_grad():
            image = renderer({"sky.texture": texels(log_rgb.exp())}, camera=cameras[0], spp=64, seed=7)
        return stack([frame([targets[0], image], ["target, view 0", label]),
                      frame([log_rgb.exp(), truth], ["environment map", "true map"], scale=4)])

    history = {"loss": [], "log_rmse": [log_error()]}
    frames = [preview("initial")]
    for it in range(iterations):
        view = it % len(cameras)
        image = renderer({"sky.texture": texels(log_rgb.exp())}, camera=cameras[view])
        # Squared differences of neighboring texels (wrapping around in longitude).
        roughness = ((log_rgb - log_rgb.roll(1, 1)) ** 2).mean() + (log_rgb.diff(dim=0) ** 2).mean()
        loss = fi.relative_mse(image, targets[view]) + smoothness * roughness
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        history["loss"].append(loss.item())
        history["log_rmse"].append(log_error())
        if (it + 1) % video_every == 0:
            frames.append(preview(f"iteration {it + 1}"))
    out = output_dir("env_map", out_dir)
    save_videos(out / "optimization", frames, fps=10)
    Image.fromarray(frames[-1]).save(out / "final.png")
    fi.save_image(out / "env_map.exr", log_rgb.exp())
    save_curves(out / "curves.png", {"relative image loss": history["loss"], "log texel RMSE": history["log_rmse"]},
                "environment map: loss and texel error")
    metrics = {"initial_log_rmse": history["log_rmse"][0], "final_log_rmse": history["log_rmse"][-1],
               "initial_loss": float(np.mean(history["loss"][:8])), "final_loss": float(np.mean(history["loss"][-8:]))}
    save_history(out / "history.json", history, **metrics)
    return metrics


if __name__ == "__main__":
    print(run())
