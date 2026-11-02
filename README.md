# falcor2-inverse

Inverse rendering with falcor2's path-space differentiable renderer (`falcor2.psdr`) from
PyTorch. Load a scene (Mitsuba 3, USD, glTF, OBJ, PLY) or build one, choose any of its parameters, and optimize
them in an ordinary PyTorch loop: the renderer is a differentiable PyTorch operation, and every tensor that
requires gradients gets them, including tensors computed by your own code (networks, reparameterizations).

```python
import torch
import falcor2_inverse as fi

scene = fi.Scene.load("assets/scenes/cornell_box.xml")      # a Mitsuba scene
renderer = fi.Renderer(scene, spp=16, max_bounces=3)
with torch.no_grad():
    target = renderer(spp=1024)                             # or a photograph: fi.load_image("target.exr")

params = scene.parameters("red.base_color_factor")         # {key: tensor}
params["red.base_color_factor"] = torch.full((3,), 0.5, device=target.device, requires_grad=True)
optimizer = torch.optim.Adam(params.trainable(), lr=0.02)
for it in range(200):
    loss = fi.mse(renderer(params), target)                # renders with the parameter values
    optimizer.zero_grad()
    loss.backward()                                         # PSDR's gradients for the tensors that require them
    optimizer.step()
```

## Installation

A built falcor2 checkout with PSDR support and its Python environment, with NumPy, Pillow, and a CUDA-enabled
PyTorch installation for the GPU. Then, from this repository's root:

```bash
pip install -e .
```

On Ubuntu with a sibling falcor2 checkout whose environment already contains those dependencies:

```bash
source ../falcor2/.venv/bin/activate
python -m pip install --no-deps --no-build-isolation -e .
```

Run the example commands below in that environment, or set `PYTHONPATH` to the repository root.
Rendering uses Vulkan on Linux and D3D12 on Windows (SlangPy's
platform default); images and parameters are PyTorch tensors on the first CUDA device (the CPU without one).
To select Vulkan explicitly, pass `device=fi.create_device(device_type=slangpy.DeviceType.vulkan)` to `fi.Scene()`
or `fi.Scene.load(...)` after importing `slangpy`. PSDR kernels compile the first time a scene type is
rendered (seconds for StandardMaterial scenes, a few minutes with OpenPBRMaterial) and are cached in
`.shader-cache/`; the cache does not track `#include`d headers, so delete it after editing them.

PSDR and `SelfContact` use inline ray queries when supported and otherwise use a ray-tracing pipeline. On
Vulkan this means the GPU and driver must expose either `VK_KHR_ray_query` or `VK_KHR_ray_tracing_pipeline`.
The PSDR fallback supports the primal image and all four gradient terms, including the secondary term's
photon search. `SelfContact` supports barrier gradients, safe step bounds, and intersection counts on both paths.

Scenes can combine `StandardMaterial`, `OpenPBRMaterial`, and `SpongeCakeMaterial`. With the Vulkan pipeline
fallback, scenes containing all three use separate BSDF backward kernels to avoid a failure of the combined
shader. Shared environment and light-source derivatives are accumulated once. The fallback caches shaders
for the scene's material families; the first mixed-material run can take several minutes to compile.
`fi.Renderer` rebuilds them after structural scene changes. Direct `PSDR` users must recreate it when adding
a new material family.

## Scenes

`fi.Scene.load(path)` loads:

| Format | Loader |
|---|---|
| USD (`.usd`, `.usda`, `.usdc`, `.usdz`), glTF (`.gltf`, `.glb`), falcor2 Python scenes (`.py`) | falcor2's importers |
| Mitsuba 3 (`.xml`) | `falcor2_inverse.loaders.mitsuba`: shapes (obj, ply, serialized, rectangle, cube, sphere, disk), BSDFs as StandardMaterials (diffuse, plastic, conductor, dielectric and their rough variants, principled, twosided, normalmap), bitmap and checkerboard textures, area, point, spot, directional, envmap and constant emitters, perspective cameras; `Scene.metadata` holds the integrator's path length and the sampler's spp. |
| Meshes (`.obj`, `.ply`, Mitsuba `.serialized`) | one mesh with a gray material, without cameras or lights |

Add a camera and lighting before rendering a bare mesh import. `examples/optimize_scene.py` expects a scene
file containing a camera and lighting; use `--list` to inspect the parameters of either a scene or a bare mesh.

Scenes can also be built (or extended) in Python:

```python
from pathlib import Path
import numpy as np
import falcor2 as f2
import falcor2_inverse as fi

scene = fi.Scene()
red = scene.add_material("red", base_color_factor=(0.8, 0.2, 0.1), roughness_factor=0.4)    # StandardMaterial
blue = scene.add_material("blue", "openpbr", base_color=(0.2, 0.4, 0.8), specular_roughness=0.3)
image = np.full((16, 16, 3), 0.6, dtype=np.float32)
floor = scene.add_material("floor", base_color_texture=image)                               # (H, W, 3) array
positions, faces, uv = fi.quad((0, 0, 0), (2, 0, 0), (0, 2, 0))
scene.add_mesh("floor", positions, faces, floor, texcoords=uv)
mesh = fi.read_mesh("assets/meshes/bunny.obj")                                               # or fi.sphere(...)
scene.add_mesh("bunny", mesh["positions"], mesh["faces"], red, translation=(0, 0, 0.3), rotation=(0, 0, 0.5))
scene.add_point_light("lamp", position=(1, -1, 2), intensity=(2, 2, 2))
scene.add_rect_light("panel", position=(0, 0, 2), radiance=(5, 5, 5), size=0.5)             # also disk, sphere,
sky = Path(f2.__file__).resolve().parents[1] / "data/assets/envmaps/aerodynamics_workshop_512.hdr"
scene.add_env_map("sky", sky, rotation=(np.pi / 2, 0, 0))                                  # distant, constant
scene.add_camera("camera", eye=(0, -3, 2), target=(0, 0, 0.3), fov_y=40, resolution=(512, 512))
```

Objects are found by name in `scene.meshes`, `scene.materials`, `scene.lights` and `scene.cameras` (USD prim paths
are shortened to their shortest unique suffix; Mitsuba objects are named by id, or by file name). `scene.f2` is the
falcor2 scene for anything else.

## Parameters

`scene.parameters(*patterns)` returns the differentiable parameters (all, or the keys matching glob patterns) as
a `ParameterDict` of float32 tensors, keyed `"<object>.<parameter>"`:

| Key | Value |
|---|---|
| `<mesh>.vertex_positions` | object-space vertices (V, 3), in the geometry's stored vertex order (falcor2 may reorder vertices: `scene.mesh_arrays(name)` gives the matching triangles) |
| `<mesh>.translation`, `.rotation`, `.scale` | the mesh instance's transform (3): rotations are axis-angle vectors (radians) |
| `<material>.<factor>` (StandardMaterial) | `base_color_factor`, `metallic_factor`, `roughness_factor`, `ior`, `emissive_factor`, transmission factors, `normal_texture_scale` |
| `<material>.<factor>` (OpenPBRMaterial) | attributes such as `base_color`, `specular_roughness`, subsurface factors, and `normal_texture_scale` |
| `<material>.<factor>` (SpongeCakeMaterial) | `albedo`, `thickness`, `roughness`, `top_albedo`, `top_thickness`, `top_roughness`, `base_color`, `base_weight`, `normal_texture_scale` |
| `<material>.<texture>_texture` | linear RGBA texels (H, W, 4): StandardMaterial base color, metallic-roughness, emissive, normal, transmission; OpenPBR attribute and normal maps; SpongeCake albedo, top albedo, base color, and normal maps |
| `<light>.intensity` / `.radiance` | point light intensity, area light radiance (3) |
| `<light>.translation` | an analytic light's position (3) |
| `<env map>.intensity`, `.rotation`, `.texture` | an environment map's intensity (3), rotation (3) and RGBA texels (H, W, 4) |

Give a renderer any subset of them, as tensors (or arrays): `renderer({"red.base_color_factor": color})`. The
values are written into the scene, and the tensors that require gradients receive them. A value can be any
tensor, e.g. a network's output (`examples/neural_texture.py`), `log_texels.exp()` (`examples/env_map.py`) or a
translation built from two optimized coordinates (`examples/translation.py`). `ParameterDict.requires_grad_(*patterns)`
and `trainable()` help select what to optimize; `scene.set_parameters(values)` writes values for good.

## Rendering

`fi.Renderer(scene, camera=None, spp=4, grad_spp=None, max_bounces=2, terms=None, photon_radius=None,
light_sampler=None)` renders
from a camera (the scene's first by default, or one chosen per call: `renderer(params, camera="view1")`) and
returns the linear radiance (H, W, 3). Each call uses new random samples, and the derivative is estimated with
samples independent of the image's, so that gradients of losses like `mse(image, target)` are unbiased.
Gradients come from PSDR's four terms: `interior` (shading), `pixel` (pixel-boundary edges), `primary`
(silhouettes seen by the camera) and `secondary` (shadows and reflected silhouettes, by photon mapping). By
default all four are used when geometry or lights move, and only `interior` otherwise; `terms` selects them.
`light_sampler` picks the lights of next-event estimation: the default `f2.PowerLightSampler` weights lights by
emitted power, so a distant sun disk starves small area lights (their light and gradients become a few fireflies);
`f2.UniformLightSampler()` picks every light alike (and every emissive triangle, so not with emissive meshes).
`vertex_grad_clamp` (default 0 = off) bounds each path sample's interior-term vertex gradient relative to its
contribution per scene extent: sharp glossy lobes make geometry derivatives heavy-tailed in multi-bounce scenes, and
values of 3-10 tame them, but the clamp also cuts legitimate near-field signal (surfaces a few cm apart), so check a
clamped gradient against finite differences for the scene.
PSDR samples antithetically: each pixel-boundary sample comes with its three mirror images on the pixel's perimeter,
and a camera path whose first vertex samples a glossy reflection is paired with a path whose sample there is turned
by half a turn about the lobe, which cuts the variance of geometry derivatives of glossy objects several-fold
(`PSDR.forward/backward(antithetic=False)` turns it off).
Changes of the scene's structure (`scene.set_mesh()`, new objects) are picked up automatically.

## Optimization

- Losses: `fi.mse`, `fi.l1`, `fi.relative_mse` (for high dynamic range). Any PyTorch loss works.
- Meshes: `fi.LargeSteps(faces, vertex_count, lmbda)` and `fi.AdamUniform` (Nicolet et al. 2021, "Large Steps in
  Inverse Rendering of Geometry"): optimize `u = steps.to_differential(vertices)` and render
  `steps.from_differential(u)`. `fi.subdivide` refines a mesh between stages; `fi.laplacian` and `fi.chamfer`
  are there for regularizers and evaluation.
- Self-contact: `fi.SelfContact(faces, vertex_count, dhat)` handles vertex-triangle contact during mesh
  optimization, in the spirit of incremental potential contact (IPC): `contact.barrier(vertices)` adds a log
  barrier's gradient for vertex-triangle pairs closer than dhat, and `contact.step_bound(vertices, step)` shortens
  steps that would pass a vertex through a triangle (`examples/multipose.py --contact`). Its proximity queries
  run on the GPU through inline ray queries or a ray-tracing pipeline, selected automatically. Edge-edge
  contacts are not handled.
- Images: `fi.load_image`, `fi.save_image` (EXR/HDR linear, PNG/JPEG tonemapped), `falcor2_inverse.images` for
  labeled frames and videos of optimizations.

## Examples

The optimization examples render targets with the true parameters, start from wrong ones, and optimize them.
Most write videos, images, loss curves, and `history.json` to `results/<name>/`. `quickstart` writes `images.png`
and prints parameter errors; `validate_backend` writes a comparison image and diagnostics JSON.

```bash
python examples/quickstart.py
python examples/validate_backend.py             # small forward/backward and optimization check
python examples/validate_backend.py --mixed-materials  # fit all three material families together
python examples/isolated.py shadow --shape bunny --subdivisions 4 --iterations 1000 2000 --contact
python examples/run_all.py                      # all optimization examples, with default settings
```

`validate_backend.py` checks finite images and gradients, exercises the four gradient terms, and fits material
colors to a rendered target. `--mixed-materials` fits Standard, OpenPBR, and SpongeCake colors in one scene.
Reports and comparison images go to `results/validate_backend/` or `results/validate_backend_mixed/`; use
`--output` to choose another directory. On a sibling falcor2 checkout's Linux environment, run it from this
repository with `PYTHONPATH=. ../falcor2/.venv/bin/python examples/validate_backend.py`.

| Example | Recovers | PSDR terms |
|---|---|---|
| `quickstart` | a sphere's color, roughness and position and a point light's position | all four |
| `appearance` | `materials`: StandardMaterial and OpenPBRMaterial colors and roughness; `lights`: point-light intensity, rect-light radiance, emissive factor | interior |
| `translation` | the (x, y) position of a box on a floor | all four |
| `shape` | the rim of a planar mesh (circle to star), with a smoothness regularizer | all four |
| `multipose` | a cube, bumpy sphere or bunny from a sphere (large steps, one refinement) in 12 poses: shading, sun shadow, glossy mirror, environment map | all four |
| `isolated` | the same shapes from their sun shadows alone (`shadow`) or their reflection in a mirror alone (`mirror`) | secondary (and interior) |
| `texture` | a 256 x 256 base color texture from three views | interior (texels) |
| `neural_texture` | the same texture as a network (Fourier features and an MLP) trained through the renderer | interior |
| `env_map` | a 64 x 32 environment map (in log space, with a smoothness prior) from a metal and a diffuse sphere | interior (texels) |
| `moving_light` | the positions of a point light and a rect light from shading and shadows | interior, secondary |
| `optimize_scene` | parameters of a scene file (Mitsuba's Cornell-box tutorial: the wall colors), chosen by pattern | as needed |

`examples/render_meshes.py` renders saved meshes from `multipose` and `isolated` next to their targets; it expects
their `mesh.npz` and `history.json` files. `examples/timing.py`
times an optimization iteration (render, loss, backward, step) at 512 x 512, of the bunny's material or of its
vertices with all four PSDR terms and with each term alone.

Ubuntu validation (2026-10-30): all 13 optimization configurations registered by `run_all.py` completed with
their default iteration counts, image sizes, and sampling settings on Ubuntu 24.04.4 LTS (aarch64), an NVIDIA
GB300, Vulkan/TraceRay, and CUDA PyTorch tensors. A shortened run of all 13 entries also passed together in one
process. Both backend validators, the timing example, mesh rendering,
and the README Python snippets also ran successfully. The documented bunny/contact command completed all
3,000 updates with zero crossing edges. Detailed local reports and outputs are in `results/ubuntu_validation/`.

Reference measurements from earlier runs (surface distances: Chamfer distance relative to the target's bounding-box
diagonal; "crossing edges": edges that pierce the mesh, i.e. self-intersections):

| Example | Initial | Final |
|---|---|---|
| `quickstart` | | each parameter within 0.0015 of the truth |
| `appearance materials` / `lights` | relative parameter error 0.47 / 0.67 | 0.007 / 0.08 (the dim rect light is slowest) |
| `translation` | offset 0.25 | 0.004 |
| `shape` | rim RMS error 0.085 | 0.016 |
| `texture` / `neural_texture` | texel RMSE 0.32 | 0.089 / 0.10 |
| `env_map` | log texel RMSE 2.3 | 0.72 |
| `moving_light` | distances 0.94, 1.02 | 0.007, 0.037 |
| `optimize_scene` (Cornell box) | color errors 0.64, 0.55 | 0.0007, 0.0004 |
| `multipose` (cube) | surface distance 0.158 | 0.012 (146 crossing edges at the cube's edges) |
| `isolated shadow` / `mirror` (cube) | 0.147 | 0.020 / 0.017 |
| `isolated shadow` (bunny, 10,242 vertices: `--subdivisions 4 --iterations 1000 2000`) | 0.204 | 0.011 and 306 crossing edges (the ears fold into each other); with `--contact`, 0.009 and none |
| `multipose` (bunny, 10,242 vertices: `--subdivisions 4 --iterations 1500 1500`) | 0.214 | 0.0135 and 270 crossing edges; with `--contact`, 0.011 and none |

## Notes

- falcor2 stores some material factors as half floats (e.g. base colors): written values are rounded.
- Paths through near-specular lobes (roughness below about 0.2) have heavy-tailed derivatives; falcor2 treats
  roughness below 0.08 as a delta lobe, which PSDR does not differentiate.
- A material texture that is not floating point (or only a file path) is replaced by an RGBA32F copy when its
  texels are first written, which rebuilds the renderers. Environment map texels are updated in place; their
  sampling distribution is not rebuilt (estimates stay unbiased).
- falcor2's environment maps are centered on -z with +y up; `DistantLight.cutoff_angle` is in degrees.
- The Mitsuba loader maps BSDFs to StandardMaterial approximately (e.g. conductors by their F0 color, roughness
  as the square root of alpha) and skips unsupported plugins with a warning.
- Edge sampling misses the surface behind an edge that touches it: objects resting on a floor have slightly
  biased translation gradients (as in PSDR-Enzyme).
- PSDR finds a mesh's edges with its vertices joined by position, so meshes split at seams (flat-shaded, textured)
  are handled, but coincident boundaries of separate meshes (e.g. a box made of one mesh per face, as in
  falcor2's USD Cornell box) are counted twice: merge such meshes before optimizing their geometry.

## Tests

```bash
python -m pytest tests -q
```

`tests/test_contact.py` checks `SelfContact` against CPU references and finite differences, including CUDA
autograd, moving triangles, candidate counts, edge crossings, and empty meshes. Inline-query comparison is
skipped when that feature is unavailable.

The PSDR traversal and mixed-material regressions live in the sibling falcor2 checkout:

```bash
cd ../falcor2
.venv/bin/python -m pytest -q tests/python/render/test_psdr_trace_ray.py \
    tests/python/render/test_psdr_photon_gather.py tests/python/render/test_psdr_edges.py \
    tests/python/render/test_psdr_material_batches.py
```

These exercise the TraceRay fallback, photon search, material and light finite differences, forward/reverse
geometry derivatives, shared environment contributions, light tracing, and gradient accumulation. The first
run compiles several large shader variants and can take tens of minutes.

## Layout

- `falcor2_inverse/`: `scene.py` (Scene), `params.py` (parameters and their gradients), `render.py` (Renderer),
  `loaders/` (Mitsuba, meshes), `optim.py` (large steps, AdamUniform), `contact.py` and `shaders/contact.slang`
  (self-contact), `geometry.py`, `transforms.py`, `losses.py`, `images.py`, `device.py`.
- `examples/`: the examples and their shared helpers (`common.py`).
- `assets/`: the Stanford bunny (from PSDR-Enzyme) and a Mitsuba Cornell box.
