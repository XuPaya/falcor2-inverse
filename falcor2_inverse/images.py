"""Images: display tonemapping, image files and optimization videos."""

from pathlib import Path

import numpy as np
import slangpy as spy
import torch
from PIL import Image, ImageDraw

from falcor2_inverse.device import torch_device
from falcor2_inverse.params import _rgba


def _numpy(image) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    return np.asarray(image, np.float32)


def tonemap(image, exposure=1.0) -> np.ndarray:
    """8-bit display values (gamma 2.2) of a linear image (H, W, 3)."""
    return (np.clip(np.clip(_numpy(image) * exposure, 0, 1) ** (1 / 2.2), 0, 1) * 255).astype(np.uint8)


def save_image(path, image, exposure=1.0):
    """Write a linear image (H, W, 3): as is to .exr and .hdr files, tonemapped to others (.png, .jpg)."""
    if Path(path).suffix.lower() in (".exr", ".hdr"):
        spy.Bitmap(np.ascontiguousarray(_numpy(image))).write(str(path))
    else:
        Image.fromarray(tonemap(image, exposure)).save(path)


def load_image(path, device=None) -> torch.Tensor:
    """Linear RGB (H, W, 3, float32) of an image file, e.g. a target photograph (8-bit images are sRGB)."""
    data = np.asarray(spy.Bitmap(str(path)))
    if data.ndim == 2 or data.shape[-1] == 1:
        data = np.repeat(data.reshape(data.shape[0], data.shape[1], 1), 3, -1)
    return torch.as_tensor(_rgba(data, srgb=True)[..., :3], device=device or torch_device())


def frame(images, labels, exposure=1.0, scale=1) -> np.ndarray:
    """Side-by-side tonemapped images (H, W, 3) with labels, as an RGB uint8 frame."""
    tiles = []
    for image, label in zip(images, labels):
        tile = Image.fromarray(tonemap(image, exposure))
        if scale != 1:
            tile = tile.resize((tile.width * scale, tile.height * scale), Image.NEAREST)
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 7 * len(label) + 6, 13), fill=(0, 0, 0))
        draw.text((3, 1), label, fill=(255, 255, 255))
        tiles.append(np.pad(np.asarray(tile), ((2, 2), (2, 2), (0, 0)), constant_values=255))
    return np.concatenate(tiles, 1)


def stack(rows) -> np.ndarray:
    """Frames of possibly different widths stacked vertically (padded with white)."""
    width = max(r.shape[1] for r in rows)
    return np.concatenate([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows], 0)


def save_video(path, frames, fps=10, max_width=1024, max_frames=80):
    """Write frames as an animated GIF (or WebP for a .webp path), downscaled to max_width and subsampled to at
    most max_frames frames (always keeping the last one)."""
    path = Path(path)
    if len(frames) > max_frames:
        keep = np.unique(np.round(np.linspace(0, len(frames) - 1, max_frames)).astype(int))
        frames = [frames[i] for i in keep]
    images = [Image.fromarray(np.asarray(f)) for f in frames]
    if images[0].width > max_width:
        size = (max_width, round(images[0].height * max_width / images[0].width))
        images = [im.resize(size, Image.LANCZOS) for im in images]
    duration = int(1000 / fps)
    if path.suffix == ".webp":
        images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0, quality=80)
    else:
        palette = [im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in images]
        palette[0].save(path, save_all=True, append_images=palette[1:], duration=duration, loop=0, disposal=1)
    return path


def save_videos(stem, frames, fps=10):
    """Write <stem>.webp (up to 1024 pixels wide) and a smaller <stem>.gif for viewers without WebP."""
    save_video(Path(stem).with_suffix(".webp"), frames, fps)
    save_video(Path(stem).with_suffix(".gif"), frames, fps, max_width=640)
