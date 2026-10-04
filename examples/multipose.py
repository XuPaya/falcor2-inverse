"""Multi-pose shape reconstruction from shading, shadow and reflection (cf. PSDR-Enzyme's multi-view experiments).

A diffuse object (a cube, a bumpy sphere or the bunny, see common.target_shape) is shown in front of a glossy mirror
above a floor, lit by the sun (a distant light) from the side and an environment map. With `poses="object"`
(default) the camera is fixed and the object floats in random orientations: each pose shows it directly, in the
mirror, which shows its back, and by its shadow on the floor. With `poses="camera"` the object stands on the floor
and the cameras are on an arc in front of it.

Starting from a sphere around the object, the object-space vertices are optimized with large steps (Nicolet et al.
2021, as in PSDR-Enzyme) so that the renderings of all poses match their targets; each iteration renders one pose,
cycling through them. The mesh is refined once (midpoint subdivision), as PSDR-Enzyme remeshes between stages.

The sun shines along the mirror's back side: sunlight reflected by the mirror onto the floor would be a caustic,
which path tracing renders with fireflies. The mirror is glossy (roughness 0.25): falcor2 treats roughness below
0.08 as a delta lobe, whose paths PSDR does not differentiate, and the derivatives of paths through very sharp
lobes are heavy-tailed.

    python examples/multipose.py [--shape cube|bumpy|bunny] [--poses object|camera] [--contact]
"""

import argparse
import time

import numpy as np
import torch
from PIL import Image

import falcor2_inverse as fi
from falcor2_inverse.images import frame, save_videos, stack
from falcor2_inverse.transforms import quaternion_to_axis_angle
from common import ENVMAP, RESULTS, SHAPES, MeshViewer, output_dir, save_curves, save_history, target_shape

CENTER = np.array([0.0, 0.1, 0.5])  # object poses rotate about this point
STANDING = np.array([0.0, 0.1, 0.32])  # camera poses look at the standing object


def camera_positions(count, target, distance=2.4):
    """Positions on an arc in front of the mirror: azimuth -55..55 degrees, elevation 15..40 degrees."""
    azimuth = np.radians(np.linspace(-55, 55, count))
    elevation = np.radians(15 + 25 * (np.arange(count) % 3) / 2)
    directions = np.stack([np.sin(azimuth) * np.cos(elevation), -np.cos(azimuth) * np.cos(elevation),
                           np.sin(elevation)], 1)
    return np.asarray(target) + distance * directions


def rotations(count, seed=3):
    """Uniformly random rotations (axis-angle vectors); the first is the identity."""
    q = np.random.default_rng(seed).normal(size=(count - 1, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return [torch.zeros(3)] + [torch.as_tensor(quaternion_to_axis_angle(x), dtype=torch.float32) for x in q]


def make_scene(size, mesh, eyes, target, translation):
    scene = fi.Scene()
    floor = scene.add_material("floor", base_color_factor=(0.35, 0.35, 0.37), roughness_factor=0.9)
    mirror = scene.add_material("mirror", base_color_factor=(0.9, 0.9, 0.9), metallic_factor=1.0, roughness_factor=0.25)
    body = scene.add_material("body", base_color_factor=(0.8, 0.55, 0.35), roughness_factor=0.8)
    scene.add_mesh("floor", *fi.grid([0, 0.2, 0], [1.6, 0, 0], [0, 1.2, 0], 1)[:2], floor)
    scene.add_mesh("mirror", *fi.quad([0, 0.9, 0.7], [1.2, 0, 0], [0, 0, 0.7])[:2], mirror)  # facing -y
    scene.add_mesh("body", *mesh, body, translation=translation)
    scene.add_distant_light("sun", (0.8, 0.1, -1.0), (3000.0, 2850.0, 2700.0), cutoff_angle=1.0)
    scene.add_env_map("sky", ENVMAP, intensity=(0.3, 0.3, 0.3))
    for i, eye in enumerate(eyes):
        scene.add_camera(f"camera{i}", eye, target, fov_y=34, resolution=(size, size))
    return scene


def reconstruct(scene, renderer, views, target, initial, iterations, lr, lmbda, contact=False, target_spp=256,
                video_every=40, size=512):
    """Reconstruct the mesh "body" of a scene from renderings of views ((pose parameters, camera) pairs).

    The targets are rendered with the target mesh (positions, faces). The optimization starts from the initial
    mesh and runs in stages: stage s optimizes it subdivided s times, for iterations[s] steps with learning rate
    lr[s] and smoothing lmbda[s] (large steps). Each iteration renders one view, cycling through them. With
    contact, self-contact handling (falcor2_inverse.contact) keeps the mesh free of self-intersections.
    Returns metrics, history, video frames and the final mesh."""
    viewer = MeshViewer(size=size)
    true_view = viewer.render(*target)
    diagonal = float(np.linalg.norm(np.ptp(target[0], axis=0)))
    scene.set_mesh("body", *target)
    with torch.no_grad():
        targets = [renderer(pose, camera=camera, spp=target_spp, seed=12345) for pose, camera in views]
    scene.set_mesh("body", *initial)
    shown = [0, len(views) // 3, 2 * len(views) // 3]

    def render(view, x, **options):
        pose, camera = views[view]
        return renderer({**pose, "body.vertex_positions": x}, camera=camera, **options)

    def preview(label, x, faces):
        with torch.no_grad():
            rendered = [render(v, x, spp=64, seed=7) for v in shown]
        truth = frame([targets[v] for v in shown] + [true_view], [f"target, pose {v}" for v in shown] + ["true shape"])
        current = frame(rendered + [viewer.render(x, faces)], [label] * len(shown) + ["current shape"])
        return stack([truth, current])

    def evaluate(x):
        with torch.no_grad():
            return float(np.mean([fi.mse(render(v, x, spp=64, seed=7), targets[v]).item() for v in range(len(views))]))

    positions, faces = scene.mesh_arrays("body")
    x = torch.as_tensor(positions, device=fi.torch_device())
    initial_loss = evaluate(x)
    initial_chamfer = fi.chamfer((positions, faces), target) / diagonal
    frames = [preview("initial", x, faces)]
    history = {"loss": [], "seconds": [], "vertices": [], "alpha": [], "barrier_pairs": []}
    start, it = time.perf_counter(), 0
    for stage, (count, rate, smoothing) in enumerate(zip(iterations, lr, lmbda)):
        if stage > 0:
            scene.set_mesh("body", *fi.subdivide(x.cpu().numpy(), faces))  # a new vertex layout
        positions, faces = scene.mesh_arrays("body")
        x = torch.as_tensor(positions, device=fi.torch_device())
        steps = fi.LargeSteps(faces, len(x), lmbda=smoothing)
        u = steps.to_differential(x).requires_grad_()
        optimizer = fi.AdamUniform([u], lr=rate)
        # Self-contact (optional) and the self-intersection count: dhat is a quarter of the mean edge length.
        self_contact = fi.SelfContact(faces, len(x), 0.25 * fi.mean_edge_length(x, faces))
        for _ in range(count):
            view = it % len(views)
            x = steps.from_differential(u)
            if contact:
                x = self_contact.barrier(x)
            loss = fi.mse(render(view, x), targets[view])
            optimizer.zero_grad()
            loss.backward()
            before = u.detach().clone()
            optimizer.step()
            if contact:  # shorten the step so that no vertex passes through a triangle (x is linear in u)
                with torch.no_grad():
                    u.copy_(before + self_contact.step_bound(x, steps.from_differential(u) - x) * (u - before))
            history["loss"].append(loss.item())
            history["seconds"].append(time.perf_counter() - start)
            history["vertices"].append(len(x))
            history["alpha"].append(self_contact.stats["alpha"])
            history["barrier_pairs"].append(self_contact.stats["barrier_pairs"])
            it += 1
            if it % video_every == 0:
                elapsed = time.perf_counter()
                with torch.no_grad():
                    frames.append(preview(f"iteration {it} ({len(x)} vertices)", steps.from_differential(u), faces))
                start += time.perf_counter() - elapsed  # previews are not timed
        with torch.no_grad():
            x = steps.from_differential(u)
    positions = x.cpu().numpy()
    # Image losses of 64-spp renderings of all views, surface distances relative to the target's diagonal.
    metrics = {"initial_loss": initial_loss, "final_loss": evaluate(x), "initial_chamfer": initial_chamfer,
               "final_chamfer": fi.chamfer((positions, faces), target) / diagonal,
               "intersecting_edges": self_contact.intersections(positions),
               "seconds_per_iteration": history["seconds"][-1] / it}
    frames.append(preview(f"final ({len(x)} vertices)", x, faces))
    return metrics, history, frames, (positions, faces)


def save(out, metrics, history, frames, mesh, title):
    save_videos(out / "optimization", frames, fps=8)
    Image.fromarray(frames[0]).save(out / "initial.png")
    Image.fromarray(frames[-1]).save(out / "final.png")
    save_curves(out / "curves.png", {"image loss (pose of the iteration)": history["loss"]}, title)
    np.savez(out / "mesh.npz", positions=mesh[0], faces=mesh[1])
    save_history(out / "history.json", history, **metrics)


def run(iterations=(800, 2400), size=512, spp=4, target_spp=256, poses="object", count=12, bounces=2,
        lr=(0.02, 0.01), lmbda=(30.0, 15.0), shape="cube", out_dir=RESULTS, video_every=40, contact=False,
        subdivisions=3):
    """`count` is the number of poses (object rotations or cameras)."""
    if isinstance(iterations, int):
        iterations = (iterations // 2, iterations - iterations // 2)
    # Object-space meshes centered at the origin, placed by the instance transform.
    target = target_shape(shape)
    initial = fi.sphere((0, 0, 0), 1.0, subdivisions)
    if poses == "object":
        views = [({"body.rotation": r}, "camera0") for r in rotations(count)]
        translation, look, eyes = CENTER, CENTER, [camera_positions(3, CENTER, 2.6)[1]]
        initial = (initial[0] * np.float32(0.46), initial[1])  # a sphere around the object: silhouettes carve it
    else:
        views = [({}, f"camera{i}") for i in range(count)]
        translation = STANDING + [0, 0, 0.02 - STANDING[2] - target[0][:, 2].min()]
        look, eyes = STANDING, camera_positions(count, STANDING)
        # An ellipsoid around the standing object, above the floor.
        initial = (initial[0] * np.float32([0.34, 0.3, 0.3]) + np.float32([0, 0, 0.33 - translation[2]]), initial[1])
    scene = make_scene(size, target, eyes, look, translation)
    renderer = fi.Renderer(scene, spp=spp, max_bounces=bounces)
    metrics, history, frames, mesh = reconstruct(scene, renderer, views, target, initial, iterations, lr, lmbda,
                                                 contact, target_spp, video_every, size)
    out = output_dir(f"multipose_{poses}_{shape}" + ("_contact" if contact else ""), out_dir)
    save(out, metrics, history, frames, mesh, "multi-pose: loss")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", choices=SHAPES, default="cube")
    parser.add_argument("--poses", choices=("object", "camera"), default="object")
    parser.add_argument("--iterations", type=int, nargs="+", default=[800, 2400], help="per stage")
    parser.add_argument("--contact", action="store_true", help="self-contact handling (falcor2_inverse.contact)")
    parser.add_argument("--subdivisions", type=int, default=3,
                        help="of the initial icosphere (3: 642 vertices, refined to 2562; 4: 2562, to 10242)")
    args = parser.parse_args()
    print(run(iterations=tuple(args.iterations), poses=args.poses, shape=args.shape, contact=args.contact,
              subdivisions=args.subdivisions))
