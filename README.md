# falcor2-inverse

Inverse rendering with falcor2's path-space differentiable renderer (`falcor2.psdr`, branch `psdr-falcor2`) from
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
params["red.base_color_factor"] = torch.full((3,), 0.5, device="cuda", requires_grad=True)
optimizer = torch.optim.Adam(params.trainable(), lr=0.02)
for it in range(200):
    loss = fi.mse(renderer(params), target)                # renders with the parameter values
    optimizer.zero_grad()
    loss.backward()                                         # PSDR's gradients for the tensors that require them
    optimizer.step()
```

## Installation

A built falcor2 checkout on branch `psdr-falcor2` and its Python environment, with PyTorch (with CUDA; e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu124`). Then, from this repository's root:

```bash
pip install -e .
```

or set `PYTHONPATH` to the repository root. Rendering runs on a D3D12 device; images and parameters are PyTorch
tensors on the first CUDA device (the CPU without one). PSDR kernels compile the first time a scene type is
rendered (seconds for StandardMaterial scenes, a few minutes with OpenPBRMaterial) and are cached in
`.shader-cache/`; the cache does not track `#include`d headers, so delete it after editing them.

## Scenes

`fi.Scene.load(path)` loads:

| Format | Loader |
|---|---|
| USD (`.usd`, `.usda`, `.usdc`, `.usdz`), glTF (`.gltf`, `.glb`), falcor2 Python scenes (`.py`) | falcor2's importers |
| Mitsuba 3 (`.xml`) | `falcor2_inverse.loaders.mitsuba`: shapes (obj, ply, serialized, rectangle, cube, sphere, disk), BSDFs as StandardMaterials (diffuse, plastic, conductor, dielectric and their rough variants, principled, twosided, normalmap), bitmap and checkerboard textures, area, point, spot, directional, envmap and constant emitters, perspective cameras; `Scene.metadata` holds the integrator's path length and the sampler's spp. |
| Meshes (`.obj`, `.ply`, Mitsuba `.serialized`) | one mesh with a gray material |

Scenes can also be built (or extended) in Python:

```python
scene = fi.Scene()
red = scene.add_material("red", base_color_factor=(0.8, 0.2, 0.1), roughness_factor=0.4)    # StandardMaterial
blue = scene.add_material("blue", "openpbr", base_color=(0.2, 0.4, 0.8), specular_roughness=0.3)
floor = scene.add_material("floor", base_color_texture=image)                               # (H, W, 3) array
positions, faces, uv = fi.quad((0, 0, 0), (2, 0, 0), (0, 2, 0))
scene.add_mesh("floor", positions, faces, floor, texcoords=uv)
mesh = fi.read_mesh("bunny.obj")                                                            # or fi.sphere(...)
scene.add_mesh("bunny", mesh["positions"], mesh["faces"], red, translation=(0, 0, 0.3), rotation=(0, 0, 0.5))
scene.add_point_light("lamp", position=(1, -1, 2), intensity=(2, 2, 2))
scene.add_rect_light("panel", position=(0, 0, 2), radiance=(5, 5, 5), size=0.5)             # also disk, sphere,
scene.add_env_map("sky", "sky.exr", rotation=(np.pi / 2, 0, 0))                             # distant, constant
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
| `<material>.<factor>` | StandardMaterial: `base_color_factor`, `metallic_factor`, `roughness_factor`, `ior`, `emissive_factor`, transmission factors, `normal_texture_scale`; OpenPBRMaterial: its 30 attributes (`base_color`, `specular_roughness`, ...) |
| `<material>.<texture>_texture` | linear RGBA texels (H, W, 4) of a texture: StandardMaterial base color, metallic-roughness, emissive, normal, transmission; any OpenPBR attribute's texture |
| `<light>.intensity` / `.radiance` | point light intensity, area light radiance (3) |
| `<light>.translation` | an analytic light's position (3) |
| `<env map>.intensity`, `.rotation`, `.texture` | an environment map's intensity (3), rotation (3) and RGBA texels (H, W, 4) |

Give a renderer any subset of them, as tensors (or arrays): `renderer({"red.base_color_factor": color})`. The
values are written into the scene, and the tensors that require gradients receive them. A value can be any
tensor, e.g. a network's output (`examples/neural_texture.py`), `log_texels.exp()` (`examples/env_map.py`) or a
translation built from two optimized coordinates (`examples/translation.py`). `ParameterDict.requires_grad_(*patterns)`
and `trainable()` help select what to optimize; `scene.set_parameters(values)` writes values for good.

## Rendering

`fi.Renderer(scene, camera=None, spp=4, grad_spp=None, max_bounces=2, terms=None, photon_radius=None)` renders
from a camera (the scene's first by default, or one chosen per call: `renderer(params, camera="view1")`) and
returns the linear radiance (H, W, 3). Each call uses new random samples, and the derivative is estimated with
samples independent of the image's, so that gradients of losses like `mse(image, target)` are unbiased.
Gradients come from PSDR's four terms: `interior` (shading), `pixel` (pixel-boundary edges), `primary`
(silhouettes seen by the camera) and `secondary` (shadows and reflected silhouettes, by photon mapping). By
default all four are used when geometry or lights move, and only `interior` otherwise; `terms` selects them.
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
- Self-contact: `fi.SelfContact(faces, vertex_count, dhat)` keeps a mesh from intersecting itself while it is
  optimized, in the spirit of incremental potential contact (IPC): `contact.barrier(vertices)` adds a log
  barrier's gradient for vertex-triangle pairs closer than dhat, and `contact.step_bound(vertices, step)` shortens
  steps that would pass a vertex through a triangle (`examples/multipose.py --contact`). Its proximity queries use
  hardware ray tracing: a few ms per iteration at 10k vertices.
- Images: `fi.load_image`, `fi.save_image` (EXR/HDR linear, PNG/JPEG tonemapped), `falcor2_inverse.images` for
  labeled frames and videos of optimizations.

## Examples

Each example renders targets with the true parameters, starts from wrong ones and optimizes them, writing videos,
images, loss curves and `history.json` to `results/<name>/`:

```bash
python examples/quickstart.py
python examples/isolated.py shadow --shape bunny --subdivisions 4 --iterations 1000 2000 --contact
python examples/run_all.py                      # all of them, with their default settings
```

| Example | Recovers | PSDR terms |
|---|---|---|
| `quickstart` | a sphere's color, roughness and position and a point light's position | all four |
| `appearance` | `materials`: StandardMaterial and OpenPBRMaterial colors and roughness; `lights`: point-light intensity, rect-light radiance, emissive factor | interior |
| `translation` | the (x, y) position of a box on a floor | all four |
| `shape` | the rim of a planar mesh (circle to star), with a smoothness regularizer | primary, pixel, secondary |
| `multipose` | a cube, bumpy sphere or bunny from a sphere (large steps, one refinement) in 12 poses: shading, sun shadow, glossy mirror, environment map | all four |
| `isolated` | the same shapes from their sun shadows alone (`shadow`) or their reflection in a mirror alone (`mirror`) | secondary (and interior) |
| `texture` | a 256 x 256 base color texture from three views | interior (texels) |
| `neural_texture` | the same texture as a network (Fourier features and an MLP) trained through the renderer | interior |
| `env_map` | a 64 x 32 environment map (in log space, with a smoothness prior) from a metal and a diffuse sphere | interior (texels) |
| `moving_light` | the positions of a point light and a rect light from shading and shadows | interior, secondary |
| `optimize_scene` | parameters of a scene file (Mitsuba's Cornell-box tutorial: the wall colors), chosen by pattern | as needed |

`examples/render_meshes.py` renders the results of the shape examples next to their targets. `examples/timing.py`
times an optimization iteration (render, loss, backward, step) at 512 x 512, of the bunny's material or of its
vertices with all four PSDR terms and with each term alone.

Results with the default settings (surface distances: Chamfer distance relative to the target's bounding-box
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

`tests/test_psdr.py` checks falcor2's PSDR: its derivatives of every kind of parameter (StandardMaterial and OpenPBR
factors and textures, normal maps, vertices, light emission and translation, environment map texels, intensity and
rotation) and of every term against finite differences, forward against backward mode, antithetic sampling, and
the primal image against falcor2's reference path tracer. The first run compiles many kernel variants (about half
an hour).

## Layout

- `falcor2_inverse/`: `scene.py` (Scene), `params.py` (parameters and their gradients), `render.py` (Renderer),
  `loaders/` (Mitsuba, meshes), `optim.py` (large steps, AdamUniform), `contact.py` and `shaders/contact.slang`
  (self-contact), `geometry.py`, `transforms.py`, `losses.py`, `images.py`, `device.py`.
- `examples/`: the examples and their shared helpers (`common.py`).
- `assets/`: the Stanford bunny (from PSDR-Enzyme) and a Mitsuba Cornell box.
