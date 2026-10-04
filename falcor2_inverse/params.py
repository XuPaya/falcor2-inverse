"""Differentiable scene parameters.

Each parameter reads and writes its value in the falcor2 scene, and turns the output of PSDR.backward()
(object-space vertex gradients and parameter-slot gradients) into the gradient of its value. Values are float32
arrays: vertex positions (V, 3) in the geometry's stored vertex order, material factors (1 or 3 values), textures
(H, W, 4) linear RGBA texels, light emission (3), translations (3), rotations as axis-angle vectors (3, radians)
and scales (3).
"""

import fnmatch

import numpy as np
import slangpy as spy
import torch

import falcor2 as f2
from falcor2.psdr import MATERIAL_PARAMS, MATERIAL_TEXTURES

from falcor2_inverse.geometry import tangents, vertex_normals
from falcor2_inverse.transforms import (axis_angle_to_matrix, axis_angle_to_quaternion, quaternion_to_axis_angle,
                                        rotation_vector_of)

# Texture formats whose texels are written in place (dtype, channels); material textures in other formats are
# replaced by an RGBA32F copy on the first write.
_FLOAT_FORMATS = {spy.Format.rgba32_float: (np.float32, 4), spy.Format.rgba16_float: (np.float16, 4),
                  spy.Format.rgb32_float: (np.float32, 3)}


def _write_texels(texture, value):
    dtype, channels = _FLOAT_FORMATS[texture.format]
    texture.copy_from_numpy(np.ascontiguousarray(np.asarray(value)[..., :channels], dtype))
# Texture inputs that hold colors: 8-bit image files for them are sRGB-encoded.
_SRGB_TEXTURES = {"base_color", "emissive", "specular_color", "coat_color", "fuzz_color", "transmission_color",
                  "emission_color"}


def _array(value) -> np.ndarray:
    try:
        return np.array([float(value[i]) for i in range(len(value))], np.float32)
    except TypeError:
        return np.array([float(value)], np.float32)


def _matrix(entity) -> np.ndarray:
    return np.asarray(entity.world_from_object_matrix.to_numpy(), np.float64)


def _parent_matrix(entity) -> np.ndarray:
    parent = entity.parent
    return _matrix(parent) if parent is not None and parent.is_valid else np.eye(4)


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _rgba(data, srgb=False) -> np.ndarray:
    """Float RGBA (H, W, 4) of image data (H, W[, C]); integer data is normalized (and decoded from sRGB)."""
    data = np.asarray(data)
    if data.dtype.kind in "ui":
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
        if srgb:
            data[..., :3] = _srgb_to_linear(data[..., :3])
    data = data.astype(np.float32)
    if data.ndim == 2:
        data = data[..., None]
    if data.shape[-1] < 4:  # missing channels read as (r, 0, 0, 1)
        shape = data.shape[:2]
        data = np.concatenate([data, np.zeros(shape + (3 - min(3, data.shape[-1]),), np.float32),
                               np.ones(shape + (1,), np.float32)], -1)[..., :4]
    return np.ascontiguousarray(data)


def texels(texture) -> np.ndarray:
    """Linear float RGBA texels (H, W, 4) of a texture (mip level 0)."""
    name = str(texture.format).split(".")[-1]
    data = np.asarray(texture.to_numpy()).reshape(texture.height, texture.width, -1)
    if "bgra" in name:
        data = data[..., [2, 1, 0, 3]]
    return _rgba(data, srgb="srgb" in name)


def float_texture(device, data) -> spy.Texture:
    """RGBA32F texture of texels (H, W, 4) (or (H, W, 3), alpha 1)."""
    data = _rgba(np.asarray(data, np.float32))
    return device.create_texture(type=spy.TextureType.texture_2d, format=spy.Format.rgba32_float,
                                 width=data.shape[1], height=data.shape[0], data=data,
                                 usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination)


def _as_torch(array, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array, np.float32), device=like.device).reshape(like.shape)


class Param:
    """A scene parameter. ``geometric`` parameters move surfaces or lights, which needs PSDR's boundary terms;
    ``moves_surfaces`` ones change the meshes' edges."""

    geometric = moves_surfaces = False

    def __init__(self, scene, key):
        self.scene, self.key = scene, key

    def read(self) -> np.ndarray:
        raise NotImplementedError

    def write(self, value: np.ndarray):
        raise NotImplementedError

    def grad(self, psdr, grad_vertex, grad_params, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class VertexPositions(Param):
    """Object-space vertex positions (V, 3) of a mesh (sub-mesh 0), in the geometry's stored vertex order (falcor2
    may reorder vertices when a mesh is set: see Scene.mesh_arrays()). Writing them recomputes vertex normals and
    tangents unless the scene's ``recompute_normals`` is False."""

    geometric = moves_surfaces = True

    def __init__(self, scene, key, instance):
        super().__init__(scene, key)
        self.instance, self.geometry = instance, instance.geometry

    def read(self):
        return np.asarray(self.geometry.positions(0), np.float32)

    def write(self, value):
        value = np.ascontiguousarray(value, np.float32)
        self.geometry.set_positions(0, value)
        if self.scene.recompute_normals:
            normals = vertex_normals(value, np.asarray(self.geometry.indices(0), np.int64).reshape(-1, 3))
            self.geometry.set_normals(0, normals.astype(np.float32))
            self.geometry.set_tangents(0, tangents(normals).astype(np.float32))

    def grad(self, psdr, grad_vertex, grad_params, value):
        return _as_torch(grad_vertex[psdr.vertex_range(self.instance)], value)


class MaterialFactor(Param):
    """A StandardMaterial or OpenPBRMaterial factor (see falcor2.psdr.MATERIAL_PARAMS): 1 or 3 values. Some are
    stored as half floats."""

    def __init__(self, scene, key, material, name):
        super().__init__(scene, key)
        self.material, self.name = material, name
        self.size = len(self.read())

    def read(self):
        return _array(self.material[self.name])[:3]

    def write(self, value):
        self.material[self.name] = spy.float3(*map(float, value)) if self.size == 3 else float(value[0])

    def grad(self, psdr, grad_vertex, grad_params, value):
        slot = psdr.material_slot(self.material, self.name)
        return _as_torch(grad_params[slot:slot + self.size], value)


class MaterialTexture(Param):
    """Linear RGBA texels (H, W, 4) of a material texture (see falcor2.psdr.MATERIAL_TEXTURES). A texture that is
    not floating point, or only a file path, is replaced by an RGBA32F copy on the first write, which needs new
    renderers (the scene's structure changes)."""

    def __init__(self, scene, key, material, name):
        super().__init__(scene, key)
        self.material, self.name, self.property = material, name, f"{name}_texture"

    def read(self):
        texture = self.material[self.property]
        if texture is None:
            return _rgba(_bitmap(self.material[f"{self.property}_path"]), srgb=self.name in _SRGB_TEXTURES)
        return texels(texture)

    def write(self, value):
        texture = self.material[self.property]
        if texture is None or texture.format not in _FLOAT_FORMATS:
            self.material[self.property] = float_texture(self.scene.device, value)
            self.scene.structure_changed()
        else:
            _write_texels(texture, value)

    def grad(self, psdr, grad_vertex, grad_params, value):
        return _as_torch(grad_params[psdr.texture_slot(self.material, self.name)], value)


def _bitmap(path):
    return np.asarray(spy.Bitmap(str(path)))


class LightEmission(Param):
    """Emission of a light: PointLight / EnvMapLight intensity or the radiance of an area light (3)."""

    def __init__(self, scene, key, light):
        super().__init__(scene, key)
        self.light = light
        self.property = "intensity" if isinstance(light, (f2.PointLight, f2.EnvMapLight)) else "radiance"

    def read(self):
        return _array(getattr(self.light, self.property))

    def write(self, value):
        setattr(self.light, self.property, spy.float3(*map(float, value)))

    def grad(self, psdr, grad_vertex, grad_params, value):
        slot = psdr.light_slot(self.light, "intensity" if isinstance(self.light, f2.EnvMapLight) else "emission")
        return _as_torch(grad_params[slot:slot + 3], value)


class EntityTransform(Param):
    """Translation (3), rotation (axis-angle, 3) or scale (3) of an entity's local transform."""

    def __init__(self, scene, key, entity, component):
        super().__init__(scene, key)
        self.entity, self.component = entity, component

    def _components(self):
        transform = self.entity.transform
        q = transform.rotation
        return {"translation": _array(transform.translation), "scale": _array(transform.scale),
                "rotation": quaternion_to_axis_angle([q.x, q.y, q.z, q.w]).astype(np.float32)}

    def read(self):
        return self._components()[self.component]

    def write(self, value):
        transform = self.entity.transform
        if transform.composition_order != f2.Transform.CompositionOrder.srt:
            raise NotImplementedError(f"{self.key}: only scale-rotate-translate transforms are supported")
        if self.component == "translation":
            transform.translation = spy.float3(*map(float, value))
        elif self.component == "scale":
            transform.scale = spy.float3(*map(float, value))
        else:
            transform.rotation = spy.quatf(*map(float, axis_angle_to_quaternion(value)))
        self.entity.transform = transform

    def world_matrix(self, component: torch.Tensor) -> torch.Tensor:
        """The entity's world transform (4, 4, float64) with this component given as a tensor."""
        c = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in self._components().items()}
        c[self.component] = component
        top = torch.cat([axis_angle_to_matrix(c["rotation"]) * c["scale"], c["translation"][:, None]], 1)
        local = torch.cat([top, torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)])
        return torch.as_tensor(_parent_matrix(self.entity)) @ local


class MeshTransform(EntityTransform):
    """Translation, rotation or scale of a mesh instance's entity. Gradients come from the vertex gradients, so
    the geometry must not be shared with other instances."""

    geometric = moves_surfaces = True

    def __init__(self, scene, key, instance, component):
        super().__init__(scene, key, instance.entity, component)
        self.instance = instance

    def grad(self, psdr, grad_vertex, grad_params, value):
        if self.scene.instance_count(self.instance.geometry) > 1:
            raise NotImplementedError(f"{self.key}: the geometry is shared by several instances")
        # Object-space vertex gradients are M^T times world-space ones.
        g_object = np.asarray(grad_vertex[psdr.vertex_range(self.instance)], np.float64)
        g_world = torch.as_tensor(g_object @ np.linalg.inv(_matrix(self.entity)[:3, :3]))
        x_object = torch.as_tensor(np.asarray(self.instance.geometry.positions(0), np.float64))
        with torch.enable_grad():
            v = value.detach().to(torch.float64).cpu().requires_grad_()
            m = self.world_matrix(v)
            (grad,) = torch.autograd.grad((g_world * (x_object @ m[:3, :3].T + m[:3, 3])).sum(), v)
        return grad.to(value)


class LightTranslation(EntityTransform):
    """Translation of an analytic light's entity (3)."""

    geometric = True

    def __init__(self, scene, key, light):
        super().__init__(scene, key, light.entity, "translation")
        self.light = light

    def grad(self, psdr, grad_vertex, grad_params, value):
        slot = psdr.light_slot(self.light, "translation")
        g_world = np.asarray(grad_params[slot:slot + 3], np.float64)
        return _as_torch(_parent_matrix(self.entity)[:3, :3].T @ g_world, value)


class EnvMapRotation(EntityTransform):
    """Rotation (axis-angle, 3) of an environment map's entity."""

    def __init__(self, scene, key, light):
        super().__init__(scene, key, light.entity, "rotation")
        self.light = light

    def grad(self, psdr, grad_vertex, grad_params, value):
        slot = psdr.light_slot(self.light, "rotation")
        g_world = torch.as_tensor(np.asarray(grad_params[slot:slot + 3], np.float64))
        with torch.enable_grad():
            v = value.detach().to(torch.float64).cpu().requires_grad_()
            rotation = self.world_matrix(v)[:3, :3]
            (grad,) = torch.autograd.grad((g_world * rotation_vector_of(rotation, rotation.detach())).sum(), v)
        return grad.to(value)


class EnvMapTexture(Param):
    """Linear RGBA texels (H, W, 4) of an environment map, updated in place (its sampling distribution is not
    rebuilt, which keeps estimates unbiased)."""

    def __init__(self, scene, key, light):
        super().__init__(scene, key)
        self.light = light

    def _texture(self):
        handle = self.scene.f2.texture_manager.load_texture(self.light.env_map_path, generate_mips=False,
                                                           load_deferred=True)
        return handle.texture

    def read(self):
        return texels(self._texture())

    def write(self, value):
        texture = self._texture()
        if texture.format not in _FLOAT_FORMATS:
            raise NotImplementedError(f"{self.key}: environment map format {texture.format} is not floating point")
        _write_texels(texture, value)

    def grad(self, psdr, grad_vertex, grad_params, value):
        return _as_torch(grad_params[psdr.texture_slot(self.light)], value)


def scene_parameters(scene) -> dict:
    """The differentiable parameters of a scene by key (see Scene.parameters())."""
    params = {}

    def add(param):
        params[param.key] = param

    for name, instance in scene.meshes.items():
        add(VertexPositions(scene, f"{name}.vertex_positions", instance))
        for component in ("translation", "rotation", "scale"):
            add(MeshTransform(scene, f"{name}.{component}", instance, component))
    for name, material in scene.materials.items():
        kind = material.slang_type_name
        if kind not in MATERIAL_PARAMS:
            continue
        properties = set(material.properties.keys())
        textures = [t for t in MATERIAL_TEXTURES[kind] if f"{t}_texture" in properties
                    and (material[f"{t}_texture"] is not None or str(material[f"{t}_texture_path"] or ""))]
        for factor in MATERIAL_PARAMS[kind]:
            if factor in properties and (factor != "normal_texture_scale" or "normal" in textures):
                add(MaterialFactor(scene, f"{name}.{factor}", material, factor))
        for texture in textures:
            add(MaterialTexture(scene, f"{name}.{texture}_texture", material, texture))
    for name, light in scene.lights.items():
        if isinstance(light, (f2.ConstantLight, f2.DistantLight)):
            continue
        emission = "intensity" if isinstance(light, (f2.PointLight, f2.EnvMapLight)) else "radiance"
        add(LightEmission(scene, f"{name}.{emission}", light))
        if isinstance(light, f2.EnvMapLight):
            add(EnvMapRotation(scene, f"{name}.rotation", light))
            add(EnvMapTexture(scene, f"{name}.texture", light))
        else:
            add(LightTranslation(scene, f"{name}.translation", light))
    return params


class ParameterDict(dict):
    """Parameter values by key (torch tensors), with helpers to select the ones to optimize."""

    def select(self, *patterns):
        """Keys matching any of the glob patterns (e.g. "bunny.*", "*.base_color_factor")."""
        return [k for k in self if any(fnmatch.fnmatchcase(k, p) for p in patterns)]

    def requires_grad_(self, *patterns):
        """Make the values whose keys match any of the patterns differentiable leaf tensors; returns them."""
        for key in self.select(*patterns):
            if not self[key].requires_grad:
                self[key] = self[key].detach().clone().requires_grad_()
        return [self[key] for key in self.select(*patterns)]

    def trainable(self):
        """Values that require gradients (to give to a torch optimizer)."""
        return [v for v in self.values() if isinstance(v, torch.Tensor) and v.requires_grad]
