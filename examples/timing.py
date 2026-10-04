"""Timing: how long an optimization iteration (render, loss, backward, Adam step) takes in a 512 x 512 scene: the
Stanford bunny (OpenPBR, 7k triangles) on a textured floor, lit by an environment map, a rect light and a point
light. Optimizing the bunny's material uses only PSDR's interior term; its vertices use all four terms, which are
also timed one at a time. The first call of each kind compiles its kernels (minutes on the first run; later runs
load them from the shader cache).

    python examples/timing.py [--size 512] [--spp 1] [--bounces 3]
"""

import argparse
import time

import numpy as np
import torch

import falcor2_inverse as fi
from falcor2.psdr import TERMS
from common import ASSETS, ENVMAP, RESULTS, load_mesh, output_dir


def checker(n=256, tiles=8):
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    on = ((i * tiles // n + j * tiles // n) % 2)[..., None]
    return np.where(on, [0.75, 0.6, 0.45], [0.25, 0.3, 0.35]).astype(np.float32)


def make_scene(size):
    scene = fi.Scene()
    floor = scene.add_material("floor", base_color_texture=checker(), roughness_factor=0.7)
    body = scene.add_material("body", "openpbr", base_color=(0.8, 0.45, 0.25), specular_roughness=0.3,
                              base_metalness=0.2)
    positions, faces, uv = fi.quad((0, 0, 0), (1.5, 0, 0), (0, 1.5, 0))
    scene.add_mesh("floor", positions, faces, floor, texcoords=uv)
    positions, faces = load_mesh(ASSETS / "meshes" / "bunny.obj", center=(0, 0, 0.45), size=0.9, y_up=True)
    scene.add_mesh("bunny", positions, faces, body)
    scene.add_env_map("sky", ENVMAP, intensity=(0.5, 0.5, 0.5))
    scene.add_rect_light("rect", (0.8, -0.6, 1.6), (6.0, 6.0, 6.0), 0.5)
    scene.add_point_light("point", (-0.9, -0.5, 1.2), (1.5, 1.5, 1.5))
    scene.add_camera("camera", (0.0, -2.2, 1.4), (0.0, 0.0, 0.35), fov_y=40, resolution=(size, size))
    return scene


def timeit(fn, calls, warmup=5):
    """Seconds of the first call (which compiles kernels) and milliseconds per call after warm-up calls."""
    start = time.perf_counter()
    fn()
    first = time.perf_counter() - start
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(calls):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return first, (time.perf_counter() - start) / calls * 1000


def optimization_step(renderer, params, target):
    optimizer = torch.optim.Adam(params.trainable(), lr=1e-4)

    def step():
        loss = fi.mse(renderer(params), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return step


def run(size=512, spp=1, bounces=3, calls=20, out_dir=RESULTS):
    scene = make_scene(size)
    renderer = fi.Renderer(scene, spp=spp, max_bounces=bounces)
    material = scene.parameters("body.base_color", "body.specular_roughness")
    material.requires_grad_("*")
    vertices = scene.parameters("bunny.vertex_positions")
    vertices.requires_grad_("*")
    times = {}
    with torch.no_grad():
        times["rendering without gradients"] = timeit(lambda: renderer(vertices), calls)
        target = renderer(spp=64)
    fi.save_image(output_dir("timing", out_dir) / "image.png", target)
    times["iteration: material (interior)"] = timeit(optimization_step(renderer, material, target), calls)
    times["iteration: vertices (all four terms)"] = timeit(optimization_step(renderer, vertices, target), calls)
    for term in TERMS:
        renderer.terms = (term,)
        times[f"iteration: vertices ({term} only)"] = timeit(optimization_step(renderer, vertices, target), calls)
    print(f"{size} x {size}, {spp} spp, {bounces} bounces:")
    for name, (first, ms) in times.items():
        print(f"  {name:38s} {ms:7.1f} ms ({1000 / ms:5.1f} per second), first call {first:5.1f} s")
    return {name: ms for name, (_, ms) in times.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--spp", type=int, default=1)
    parser.add_argument("--bounces", type=int, default=3)
    parser.add_argument("--calls", type=int, default=20, help="timed calls of each kind")
    args = parser.parse_args()
    run(args.size, args.spp, args.bounces, args.calls)
