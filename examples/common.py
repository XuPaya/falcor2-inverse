"""Helpers shared by the examples: paths, target shapes, a mesh viewer, plots and result files."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import falcor2 as f2
import falcor2_inverse as fi

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"
ENVMAP = Path(f2.__file__).resolve().parents[1] / "data" / "assets" / "envmaps" / "aerodynamics_workshop_512.hdr"
SHAPES = ("cube", "bumpy", "bunny")


def output_dir(name, out_dir=RESULTS) -> Path:
    path = Path(out_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_mesh(path, center=(0.0, 0.0, 0.0), size=None, y_up=False):
    """Positions and triangles of a mesh file. y_up rotates a y-up mesh (like the Stanford models) to z-up; with
    size, the mesh is centered at center and scaled uniformly to fit a box of that size."""
    mesh = fi.read_mesh(path)
    positions = mesh["positions"].astype(np.float64)
    if y_up:
        positions = np.stack([positions[:, 0], -positions[:, 2], positions[:, 1]], 1)
    if size is not None:
        lo, hi = positions.min(0), positions.max(0)
        positions = (positions - (lo + hi) / 2) * (size / (hi - lo).max()) + np.asarray(center)
    return positions.astype(np.float32), mesh["faces"]


def bumpy(radius=0.28, amplitude=0.18, subdivisions=5, seed=4):
    """Sphere with smooth bumps: the radius is modulated by eight Gaussian bumps of random signs."""
    positions, faces = fi.sphere((0, 0, 0), 1.0, subdivisions)
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(8, 3))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    signs = np.where(np.arange(8) % 2 == 0, 1.0, -0.6)
    bumps = (signs * np.exp((positions @ centers.T - 1) / 0.12)).sum(1)
    return (positions * (radius * (1 + amplitude * bumps))[:, None]).astype(np.float32), faces


def target_shape(name):
    """Positions and triangles of a target shape centered at the origin, about 0.5-0.6 across. The simple targets
    mirror PSDR-Enzyme's sphere_to_cube and sphere_to_bumpy experiments."""
    if name == "cube":
        return fi.box((0, 0, 0), 0.25)
    if name == "bumpy":
        return bumpy()
    return load_mesh(ASSETS / "meshes" / "bunny.obj", size=0.6, y_up=True)


class MeshViewer:
    """Renders a mesh in object space from a fixed direction in a studio scene (for videos and figures)."""

    def __init__(self, size=256, direction=(1.0, -1.6, 0.9)):
        self.scene = scene = fi.Scene()
        body = scene.add_material("body", base_color_factor=(0.8, 0.55, 0.35), roughness_factor=0.8)
        scene.add_mesh("mesh", *fi.box((0, 0, 0), 0.1), body)  # replaced by render()
        scene.add_distant_light("sun", (-0.4, 0.3, -1.0), (2000.0, 2000.0, 2000.0), cutoff_angle=1.0)
        scene.add_constant_light("ambient", (0.35, 0.35, 0.35))
        eye = 2.0 * np.asarray(direction) / np.linalg.norm(direction)
        scene.add_camera("camera", eye, (0, 0, 0), fov_y=25, resolution=(size, size))
        self.renderer = fi.Renderer(scene, spp=32, max_bounces=1)

    def render(self, positions, faces):
        self.scene.set_mesh("mesh", positions, faces)
        return self.renderer(seed=0)


def save_curves(path, curves, title, size=(480, 300)):
    """Log-scale line plot of named curves (dict name -> list of values)."""
    width, height = size
    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    colors = [(33, 102, 172), (214, 96, 77), (27, 158, 119), (117, 112, 179), (230, 171, 2)]
    values = np.concatenate([np.asarray(c, np.float64).reshape(-1) for c in curves.values()])
    lo, hi = np.log10(max(values[values > 0].min(), 1e-12)), np.log10(values.max())
    hi = hi if hi > lo else lo + 1
    left, top, right, bottom = 50, 24, width - 10, height - 30
    draw.rectangle((left, top, right, bottom), outline=(0, 0, 0))
    draw.text((left, 4), title, fill=(0, 0, 0))
    for e in range(int(np.floor(lo)), int(np.ceil(hi)) + 1):
        y = bottom - (e - lo) / (hi - lo) * (bottom - top)
        if top <= y <= bottom:
            draw.line((left, y, right, y), fill=(225, 225, 225))
            draw.text((4, y - 6), f"1e{e}", fill=(0, 0, 0))
    for i, (name, curve) in enumerate(curves.items()):
        curve = np.maximum(np.asarray(curve, np.float64), 10**lo)
        xs = left + np.arange(len(curve)) / max(1, len(curve) - 1) * (right - left)
        ys = bottom - (np.log10(curve) - lo) / (hi - lo) * (bottom - top)
        draw.line(list(zip(xs, ys)), fill=colors[i % len(colors)], width=2)
        draw.text((right - 150, top + 6 + 14 * i), name, fill=colors[i % len(colors)])
    draw.text((left, bottom + 8), f"iteration (0 - {max(len(c) for c in curves.values()) - 1})", fill=(0, 0, 0))
    image.save(path)


def save_history(path, history, **extra):
    Path(path).write_text(json.dumps(dict(history, **extra), indent=1))
