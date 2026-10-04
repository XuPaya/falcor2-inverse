"""Scenes for inverse rendering: a falcor2 scene with named meshes, materials, lights and cameras, and their
differentiable parameters as PyTorch tensors."""

import tempfile
import weakref
from pathlib import Path

import numpy as np
import slangpy as spy
import torch

import falcor2 as f2

from falcor2_inverse.device import default_device, torch_device
from falcor2_inverse.geometry import tangents, vertex_normals
from falcor2_inverse.params import ParameterDict, _rgba, float_texture, scene_parameters
from falcor2_inverse.transforms import axis_angle_to_quaternion

MATERIAL_TYPES = {"standard": f2.StandardMaterial, "openpbr": "OpenPBRMaterial"}


def make_transform(translation=None, rotation=None, scale=None) -> f2.Transform:
    """falcor2 transform (scale, then rotate, then translate). rotation: axis-angle vector (radians),
    quaternion (x, y, z, w) or spy.quatf."""
    transform = f2.Transform()
    if translation is not None:
        transform.translation = spy.float3(*map(float, translation))
    if rotation is not None:
        if not isinstance(rotation, spy.quatf):
            r = np.asarray(rotation, np.float64).reshape(-1)
            rotation = spy.quatf(*map(float, axis_angle_to_quaternion(r) if len(r) == 3 else r))
        transform.rotation = rotation
    if scale is not None:
        transform.scale = spy.float3(*map(float, np.broadcast_to(np.asarray(scale, np.float64), (3,))))
    return transform


def look_rotation(direction, up=(0.0, 0.0, 1.0)) -> spy.quatf:
    """Rotation whose local -z axis points along direction (falcor2 cameras and lights look down -z)."""
    forward = np.asarray(direction, np.float64) / np.linalg.norm(direction)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, [1.0, 0.0, 0.0] if abs(forward[0]) < 0.9 else [0.0, 1.0, 0.0])
    up = np.cross(right / np.linalg.norm(right), forward)
    return spy.math.quat_from_look_at(spy.float3(*forward), spy.float3(*up))


def _f2_value(scene, value):
    """Property value for falcor2: 3-vectors as float3, image arrays (H, W, C) as float textures."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, (tuple, list, np.ndarray)):
        array = np.asarray(value, np.float32)
        if array.ndim == 3:
            return float_texture(scene.device, array)
        if array.shape == (3,):
            return spy.float3(*map(float, array))
        if array.size == 1:
            return float(array.reshape(-1)[0])
    return value


def _numpy(value, dtype) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype)


def set_mesh(geometry, positions, faces, texcoords=None, normals=None):
    """Replace a StaticMeshGeometry's data (arrays or tensors; vertex normals default to area-weighted ones).
    falcor2 may reorder the vertices: read them back with geometry.positions(0)."""
    positions = _numpy(positions, np.float32)
    faces = _numpy(faces, np.uint32).reshape(-1, 3)
    normals = vertex_normals(positions, faces) if normals is None else _numpy(normals, np.float64)
    geometry.set_mesh_data(
        positions=positions,
        sub_mesh_indices=[faces],
        normals=normals.astype(np.float32),
        tangents=tangents(normals).astype(np.float32),
        handedness=np.ones(len(positions), np.float32),
        texcoords=np.zeros((len(positions), 2), np.float32) if texcoords is None else _numpy(texcoords, np.float32),
        name="mesh",
    )


class Scene:
    """A falcor2 scene for inverse rendering.

    Objects are found by name: ``meshes`` (geometry instances), ``materials``, ``lights`` and ``cameras``. The
    differentiable parameters are named "<object>.<parameter>" (see ``parameters()``), e.g.
    "bunny.vertex_positions", "bunny.translation", "floor.base_color_factor", "floor.base_color_texture",
    "sun_lamp.radiance", "sky.rotation". Renderers write parameter values given to them into the scene.
    """

    def __init__(self, scene: f2.Scene = None, device=None, recompute_normals=True):
        self.device = device or default_device()
        self.f2 = scene if scene is not None else f2.Scene.create(self.device)
        # Recompute vertex normals and tangents when vertex positions are written.
        self.recompute_normals = recompute_normals
        # The structure version (objects, textures) and the states of the parameters and of the meshes' surfaces;
        # renderers rebuild their PSDR objects when the version changes and update them when the states change.
        self.version, self.state, self.geometry_state = 0, 0, 0
        self._params, self._written, self._temp = None, {}, None
        # Suggestions from scene files (e.g. Mitsuba's max_bounces and spp).
        self.metadata = {}

    @classmethod
    def load(cls, path, device=None, **options) -> "Scene":
        """Load a scene: USD (.usd, .usda, .usdc, .usdz), glTF (.gltf, .glb) and falcor2 Python scenes (.py) with
        falcor2's importers, Mitsuba 3 scenes (.xml), and meshes (.obj, .ply). See falcor2_inverse.loaders."""
        from falcor2_inverse.loaders import load_scene

        return load_scene(path, device=device, **options)

    # ------------------------------------------------------------------
    # Named objects

    @staticmethod
    def _named(pairs) -> dict:
        """Items by name: paths (USD prims) are shortened to their shortest unique suffix ("/scene/red" -> "red"),
        repeated names get numbers ("mesh", "mesh_1")."""
        pairs = list(pairs)
        parts = [[p for p in str(name).split("/") if p] or [str(name)] for name, _ in pairs]
        short = []
        for p in parts:
            for n in range(1, len(p) + 1):
                if n == len(p) or sum(q[-n:] == p[-n:] for q in parts) == 1:
                    break
            short.append("/".join(p[-n:]))
        names, named = {}, {}
        for name, (_, item) in zip(short, pairs):
            count = names.get(name, 0)
            names[name] = count + 1
            named[name if count == 0 else f"{name}_{count}"] = item
        return named

    def _components(self, kind):
        return [c for c in self.f2.components if isinstance(c, kind)]

    def __repr__(self):
        return (f"Scene(meshes={list(self.meshes)}, materials={list(self.materials)}, lights={list(self.lights)}, "
                f"cameras={list(self.cameras)})")

    @property
    def meshes(self) -> dict:
        """Geometry instances by entity name."""
        return self._named((c.entity.name or "mesh", c) for c in self._components(f2.GeometryInstance))

    @property
    def materials(self) -> dict:
        return self._named((m.name or "material", m) for m in self.f2.materials)

    @property
    def lights(self) -> dict:
        """Lights by entity name."""
        return self._named((c.entity.name or type(c).__name__, c) for c in self._components(f2.Light))

    @property
    def cameras(self) -> dict:
        """Cameras by entity name."""
        return self._named((c.entity.name or "camera", c) for c in self._components(f2.Camera))

    def instance_count(self, geometry) -> int:
        return sum(1 for c in self._components(f2.GeometryInstance) if c.geometry == geometry)

    def mesh_arrays(self, name):
        """Positions (V, 3, float32) and triangles (F, 3, int64) of a mesh (sub-mesh 0) in the geometry's stored
        vertex order, which vertex position parameters and their gradients use."""
        geometry = self.meshes[name].geometry
        return (np.asarray(geometry.positions(0), np.float32),
                np.asarray(geometry.indices(0), np.int64).reshape(-1, 3))

    # ------------------------------------------------------------------
    # Building

    def structure_changed(self):
        """Objects or textures were added, removed or replaced: renderers need new PSDR objects."""
        self.version += 1
        self._params = None
        self._written.clear()

    def update(self):
        self.f2.update()

    def add_material(self, name, kind="standard", **properties):
        """A material ("standard": StandardMaterial, "openpbr": OpenPBRMaterial). Colors may be 3-sequences;
        textures may be image arrays (H, W, 3 or 4) of linear values."""
        material = self.f2.create_material(MATERIAL_TYPES.get(kind, kind), f2.Properties(
            {k: _f2_value(self, v) for k, v in properties.items()}))
        material.name = name
        self.structure_changed()
        return material

    def _entity(self, name, translation=None, rotation=None, scale=None, parent=None):
        entity = self.f2.create_entity()
        entity.name = name
        entity.transform = make_transform(translation, rotation, scale)
        if parent is not None:
            entity.parent = parent
        return entity

    def add_mesh(self, name, positions, faces, material, texcoords=None, normals=None, translation=None,
                 rotation=None, scale=None):
        """A mesh instance; positions are object-space (falcor2 may reorder them, see mesh_arrays())."""
        geometry = self.f2.create_geometry(f2.StaticMeshGeometry)
        set_mesh(geometry, positions, faces, texcoords, normals)
        instance = self._entity(name, translation, rotation, scale).create_component(f2.GeometryInstance)
        instance.geometry = geometry
        instance.materials = [material]
        self.structure_changed()
        return instance

    def set_mesh(self, name, positions, faces, texcoords=None, normals=None):
        """Replace a mesh's vertices and triangles (a structural change, e.g. after remeshing)."""
        set_mesh(self.meshes[name].geometry, positions, faces, texcoords, normals)
        self.structure_changed()

    def _light(self, kind, name, position=None, direction=None, **properties):
        entity = self._entity(name, translation=position)
        if direction is not None:
            transform = entity.transform
            transform.rotation = look_rotation(direction)
            entity.transform = transform
        light = entity.create_component(kind)
        for key, value in properties.items():
            setattr(light, key, _f2_value(self, value))
        self.structure_changed()
        return light

    def add_point_light(self, name, position, intensity):
        return self._light(f2.PointLight, name, position, intensity=intensity)

    def add_rect_light(self, name, position, radiance, size, direction=(0.0, 0.0, -1.0)):
        """Rect light emitting along direction (default: down)."""
        width, height = np.broadcast_to(np.asarray(size, np.float64), (2,))
        return self._light(f2.RectLight, name, position, direction, radiance=radiance, width=float(width),
                           height=float(height))

    def add_disk_light(self, name, position, radiance, radius, direction=(0.0, 0.0, -1.0)):
        return self._light(f2.DiskLight, name, position, direction, radiance=radiance, radius=float(radius))

    def add_sphere_light(self, name, position, radiance, radius):
        return self._light(f2.SphereLight, name, position, radiance=radiance, radius=float(radius))

    def add_distant_light(self, name, direction, radiance, cutoff_angle=0.5):
        """Light from infinitely far away shining along direction, within a cone of cutoff_angle degrees (the sun:
        about 0.27)."""
        return self._light(f2.DistantLight, name, direction=direction, radiance=radiance,
                           cutoff_angle=float(cutoff_angle))

    def add_constant_light(self, name, radiance):
        return self._light(f2.ConstantLight, name, radiance=radiance)

    def add_env_map(self, name, env_map, intensity=(1.0, 1.0, 1.0), rotation=None):
        """Environment map from an image file (latitude-longitude) or linear texels (H, W, 3 or 4). falcor2's maps
        are centered on -z with +y up: rotation=(pi / 2, 0, 0) (axis-angle) turns their up to +z."""
        if not isinstance(env_map, (str, Path)):
            data = np.asarray(env_map.detach().cpu() if isinstance(env_map, torch.Tensor) else env_map, np.float32)
            if self._temp is None:
                self._temp = tempfile.TemporaryDirectory(prefix="falcor2_inverse_")
            env_map = Path(self._temp.name) / f"{name}_{self.version}.exr"
            spy.Bitmap(_rgba(data)).write(str(env_map))
        light = self._light(f2.EnvMapLight, name, env_map_path=str(env_map), intensity=intensity)
        if rotation is not None:
            light.entity.transform = make_transform(rotation=rotation)
        return light

    def add_camera(self, name, eye, target, up=(0.0, 0.0, 1.0), fov_y=45.0, resolution=(512, 512)):
        """Pinhole camera at eye looking at target (resolution: width, height; fov_y in degrees)."""
        entity = self._entity(name, translation=eye)
        transform = entity.transform
        transform.rotation = look_rotation(np.asarray(target, np.float64) - np.asarray(eye, np.float64), up)
        entity.transform = transform
        camera = entity.create_component(f2.Camera)
        camera.width, camera.height = int(resolution[0]), int(resolution[1])
        camera.fov_y = float(fov_y)
        self.structure_changed()
        return camera

    # ------------------------------------------------------------------
    # Parameters

    @property
    def params(self) -> dict:
        """Parameter descriptors by key (falcor2_inverse.params)."""
        if self._params is None:
            self.update()
            self._params = scene_parameters(self)
        return self._params

    def parameters(self, *patterns, device=None) -> ParameterDict:
        """Current values of the differentiable parameters (all, or the keys matching glob patterns) as float32
        tensors. Pass them, or any tensors computed from them, to a Renderer; tensors that require gradients
        receive them."""
        keys = ParameterDict.fromkeys(self.params).select(*patterns) if patterns else list(self.params)
        device = device or torch_device()
        return ParameterDict({k: torch.as_tensor(self.params[k].read(), device=device) for k in keys})

    def apply(self, values) -> tuple:
        """Write parameter values (tensors by key) into the scene, skipping a tensor that was the last one written
        for its key and has not been modified in place since, and update it. Returns (whether anything changed,
        whether geometry changed)."""
        changed = geometric = surfaces = False
        for key, value in values.items():
            param = self.params.get(key)
            if param is None:
                raise KeyError(f"Unknown scene parameter {key!r}; see Scene.parameters().keys()")
            # A weak reference: a new tensor cannot pass for a freed one that had the same id and memory.
            written = self._written.get(key)
            if written is not None and written[0]() is value and written[1:] == (value._version, self.version):
                continue
            param.write(value.detach().to(torch.float32).cpu().numpy())
            self._written[key] = (weakref.ref(value), value._version, self.version)
            changed, geometric = True, geometric or param.geometric
            surfaces = surfaces or param.moves_surfaces
        if changed:
            self.update()
            self.state += 1
            self.geometry_state += surfaces
        return changed, geometric

    def set_parameters(self, values):
        """Write parameter values (tensors or arrays by key) into the scene."""
        self.apply({k: torch.as_tensor(np.asarray(v.detach().cpu() if isinstance(v, torch.Tensor) else v,
                                                  np.float32)) for k, v in values.items()})
