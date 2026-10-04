"""Multi-pose shape reconstruction from the shadow alone or from the mirror alone.

The object (a cube, a bumpy sphere or the bunny, see common.target_shape) floats in 12 random orientations and is
never visible to the camera:

- `view="shadow"`: the camera looks straight down at the floor from below the object, which casts a sun shadow
  (a distant light from straight above) and occludes the ambient light. With one bounce, the image depends on
  the geometry only through the visibility of the lights: the gradient is entirely the photon-mapping
  (secondary edge) term. The shadows are orthographic silhouettes along the sun, from 12 directions.
- `view="mirror"`: the object floats behind the camera, which sees only a glossy mirror (roughness 0.15) showing
  the object lit by the sun against a uniform sky. Primary edges and the pixel boundary term vanish (the camera
  sees only the static mirror); the gradient is the secondary term on mirror-to-object segments (the reflected
  silhouette, against the sky: escaping BSDF samples) and the interior term of the reflected shading.

Starting from a sphere around the object, the object-space vertices are optimized with large steps and one
refinement (midpoint subdivision), as in the `multipose` example.

    python examples/isolated.py [shadow|mirror] [--shape cube|bumpy|bunny] [--contact]
"""

import argparse

import falcor2_inverse as fi
from common import RESULTS, SHAPES, output_dir, target_shape
from multipose import reconstruct, rotations, save

CENTER = {"shadow": (0.0, 0.0, 1.3), "mirror": (0.0, -0.65, 0.6)}
BOUNCES = {"shadow": 1, "mirror": 2}
# Terms that can be non-zero: the object is never seen directly.
GRADIENT_TERMS = {"shadow": ("secondary",), "mirror": ("interior", "secondary")}


def make_scene(view, size, mesh):
    scene = fi.Scene()
    body = scene.add_material("body", base_color_factor=(0.8, 0.55, 0.35), roughness_factor=0.8)
    if view == "shadow":
        floor = scene.add_material("floor", base_color_factor=(0.7, 0.7, 0.7), roughness_factor=0.9)
        scene.add_mesh("floor", *fi.grid([0, 0, 0], [2, 0, 0], [0, 2, 0], 1)[:2], floor)
        scene.add_distant_light("sun", (0.0, 0.0, -1.0), (3000.0, 3000.0, 3000.0), cutoff_angle=1.0)
        scene.add_constant_light("ambient", (0.15, 0.15, 0.15))
        scene.add_camera("camera", (0.0, 0.0, 0.6), (0.0, 0.0, 0.0), fov_y=90, resolution=(size, size))
    else:
        # Close to the camera and the object: the blur of the glossy reflection grows with the distance.
        mirror = scene.add_material("mirror", base_color_factor=(0.95, 0.95, 0.95), metallic_factor=1.0,
                                    roughness_factor=0.15)
        scene.add_mesh("mirror", *fi.quad([0, 0.6, 0.6], [0.45, 0, 0], [0, 0, 0.45])[:2], mirror)  # facing -y
        # The sun shines past the mirror's back side onto the object's side that faces the mirror.
        scene.add_distant_light("sun", (0.3, -0.6, -0.7), (3000.0, 3000.0, 3000.0), cutoff_angle=1.0)
        scene.add_constant_light("ambient", (0.4, 0.4, 0.4))
        scene.add_camera("camera", (0.0, -0.1, 0.6), (0.0, 0.6, 0.6), fov_y=40, resolution=(size, size))
    scene.add_mesh("body", *mesh, body, translation=CENTER[view])
    return scene


def run(view="shadow", shape="cube", iterations=(800, 1600), size=512, spp=4, target_spp=256, count=12,
        lr=(0.02, 0.01), lmbda=(30.0, 15.0), out_dir=RESULTS, video_every=40, contact=False, subdivisions=3):
    if isinstance(iterations, int):
        iterations = (iterations // 3, iterations - iterations // 3)
    target = target_shape(shape)
    scene = make_scene(view, size, target)
    renderer = fi.Renderer(scene, spp=spp, max_bounces=BOUNCES[view], terms=GRADIENT_TERMS[view])
    views = [({"body.rotation": r}, "camera") for r in rotations(count)]
    initial = fi.sphere((0, 0, 0), 0.45, subdivisions)  # around the target
    metrics, history, frames, mesh = reconstruct(scene, renderer, views, target, initial, iterations, lr, lmbda,
                                                 contact, target_spp, video_every, size)
    metrics["gradient_terms"] = list(GRADIENT_TERMS[view])
    out = output_dir(f"isolated_{view}_{shape}" + ("_contact" if contact else ""), out_dir)
    save(out, metrics, history, frames, mesh, f"{view} only: loss")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("view", nargs="?", choices=("shadow", "mirror"), default="shadow")
    parser.add_argument("--shape", choices=SHAPES, default="cube")
    parser.add_argument("--iterations", type=int, nargs="+", default=[800, 1600], help="per stage")
    parser.add_argument("--contact", action="store_true", help="self-contact handling (falcor2_inverse.contact)")
    parser.add_argument("--subdivisions", type=int, default=3,
                        help="of the initial icosphere (3: 642 vertices, refined to 2562; 4: 2562, to 10242)")
    args = parser.parse_args()
    print(run(args.view, args.shape, iterations=tuple(args.iterations), contact=args.contact,
              subdivisions=args.subdivisions))
