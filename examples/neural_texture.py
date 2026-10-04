"""A texture represented by a neural network, trained through the renderer.

The floor texture of the `texture` example is a function of texture coordinates: random Fourier features and a
small MLP (Tancik et al. 2020). Its values on the texel grid are the scene's texture parameter, so the renderer's
texel gradients flow into the network's weights: any tensor computed with PyTorch can be a scene parameter.

    python examples/neural_texture.py
"""

import numpy as np
import torch
from PIL import Image

import falcor2_inverse as fi
from falcor2_inverse.images import frame, save_videos, stack, tonemap
from common import RESULTS, output_dir, save_curves, save_history
from texture import TEXELS, make_scene, pattern


class NeuralTexture(torch.nn.Module):
    """RGB in [0, 1] of texture coordinates (..., 2)."""

    def __init__(self, features=128, width=128, scale=16.0, seed=0):
        super().__init__()
        frequencies = torch.randn(2, features, generator=torch.Generator().manual_seed(seed)) * scale
        self.register_buffer("frequencies", frequencies)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(2 * features, width), torch.nn.ReLU(),
                                       torch.nn.Linear(width, width), torch.nn.ReLU(), torch.nn.Linear(width, 3))

    def forward(self, uv):
        x = 2 * np.pi * uv @ self.frequencies
        return torch.sigmoid(self.mlp(torch.cat([torch.sin(x), torch.cos(x)], -1)))


def run(iterations=600, size=512, spp=1, target_spp=256, lr=3e-3, out_dir=RESULTS, video_every=10):
    device = fi.torch_device()
    scene = make_scene(size)
    renderer = fi.Renderer(scene, spp=spp)
    cameras = list(scene.cameras)
    truth = torch.as_tensor(pattern(), device=device)
    alpha = torch.ones_like(truth[..., :1])
    with torch.no_grad():
        targets = [renderer({"floor.base_color_texture": torch.cat([truth, alpha], -1)}, camera=c, spp=target_spp,
                            seed=12345) for c in cameras]
    # Texel centers (u to the right, v down: rows of the texture).
    v, u = torch.meshgrid(*[(torch.arange(TEXELS, device=device) + 0.5) / TEXELS] * 2, indexing="ij")
    uv = torch.stack([u, v], -1)
    network = NeuralTexture().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 ** (1 / max(1, iterations - 1)))

    def preview(label, rgb):
        with torch.no_grad():
            image = renderer({"floor.base_color_texture": torch.cat([rgb, alpha], -1)}, camera=cameras[0], spp=64,
                             seed=7)
        return stack([frame([targets[0], image], ["target, view 0", label]),
                      frame([rgb, truth], ["network texture", "true texture"])])

    history = {"loss": [], "texture_rmse": []}
    frames = []
    for it in range(iterations):
        rgb = network(uv)
        view = it % len(cameras)
        loss = fi.mse(renderer({"floor.base_color_texture": torch.cat([rgb, alpha], -1)}, camera=cameras[view]),
                      targets[view])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        history["loss"].append(loss.item())
        history["texture_rmse"].append(float(torch.sqrt(torch.mean((rgb.detach() - truth) ** 2))))
        if it % video_every == 0:
            frames.append(preview(f"iteration {it}", rgb.detach()))
    with torch.no_grad():
        rgb = network(uv)
    frames.append(preview("final", rgb))
    out = output_dir("neural_texture", out_dir)
    save_videos(out / "optimization", frames, fps=10)
    Image.fromarray(frames[-1]).save(out / "final.png")
    Image.fromarray(tonemap(rgb)).save(out / "texture.png")
    save_curves(out / "curves.png", {"image loss": history["loss"], "texture RMSE": history["texture_rmse"]},
                "neural texture: loss and texel RMSE")
    metrics = {"initial_rmse": history["texture_rmse"][0],
               "final_rmse": float(torch.sqrt(torch.mean((rgb - truth) ** 2))),
               "initial_loss": float(np.mean(history["loss"][:3])), "final_loss": float(np.mean(history["loss"][-3:]))}
    save_history(out / "history.json", history, **metrics)
    return metrics


if __name__ == "__main__":
    print(run())
