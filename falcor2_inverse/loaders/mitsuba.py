"""Mitsuba 3 scenes (XML) as falcor2 scenes.

Supported: <default>/$name substitution, <include>, <path>, <ref>; shapes obj, ply, serialized, rectangle, cube,
sphere, disk (with to_world transforms, baked into the vertices); BSDFs diffuse, plastic, roughplastic, conductor,
roughconductor, dielectric, roughdielectric, thindielectric, principled and twosided, normalmap (as
StandardMaterials); bitmap and checkerboard textures; area, point, spot, directional, envmap and constant emitters;
perspective and thinlens sensors (as pinhole cameras); rgb, spectrum (converted to RGB), float, integer, boolean,
string, point and vector properties. Other plugins are skipped with a warning. The scene's suggested path length
(path integrator max_depth - 1) and samples per pixel are in Scene.metadata.
"""

import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from falcor2_inverse.geometry import box, disk, quad, sphere
from falcor2_inverse.loaders.meshes import read_mesh

# Approximate F0 (linear RGB) of Mitsuba's named conductors.
CONDUCTORS = {"Au": (1.0, 0.71, 0.29), "Ag": (0.95, 0.93, 0.88), "Al": (0.91, 0.92, 0.92), "Cu": (0.95, 0.64, 0.54),
              "Cr": (0.55, 0.56, 0.55), "Fe": (0.56, 0.57, 0.58), "Ni": (0.66, 0.61, 0.53), "Pt": (0.67, 0.64, 0.59),
              "Ti": (0.54, 0.50, 0.45), "W": (0.51, 0.50, 0.47), "none": (1.0, 1.0, 1.0)}
IOR = {"vacuum": 1.0, "air": 1.00028, "water": 1.333, "acrylic glass": 1.49, "bk7": 1.5046, "polypropylene": 1.49,
       "diamond": 2.419, "fused quartz": 1.458, "sapphire": 1.77}


def _g(x, mu, s1, s2):
    return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)


def spectrum_to_rgb(wavelengths, values) -> np.ndarray:
    """Linear sRGB of a piecewise-linear spectrum (CIE 1931 fit of Wyman et al. 2013, equal-energy white)."""
    lam = np.linspace(380, 780, 401)
    s = np.interp(lam, wavelengths, values, left=values[0], right=values[-1])
    x = 1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7) - 0.065 * _g(lam, 501.1, 20.4, 26.2)
    y = 0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)
    xyz = np.array([(s * x).sum(), (s * y).sum(), (s * z).sum()]) / y.sum()
    to_rgb = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    white = to_rgb @ (np.array([x.sum(), y.sum(), z.sum()]) / y.sum())
    return np.maximum(to_rgb @ xyz / white, 0.0)


def _floats(text):
    return [float(x) for x in re.split(r"[,\s]+", text.strip()) if x]


def _rotation(axis, angle_degrees):
    axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    a = np.radians(angle_degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    m = np.eye(4)
    m[:3, :3] = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k
    return m


def _look_at(origin, target, up):
    d = np.asarray(target, np.float64) - origin
    d /= np.linalg.norm(d)
    left = np.cross(np.asarray(up, np.float64), d)
    left /= np.linalg.norm(left)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = left, np.cross(d, left), d, origin
    return m


class _Loader:
    def __init__(self, scene, path, defaults=None):
        self.scene, self.root = scene, Path(path).resolve().parent
        self.search = [self.root]
        self.defaults, self.objects = dict(defaults or {}), {}
        self.metadata = {}

    # ---- properties

    def value(self, element):
        text = element.get("value", "")
        for name, value in self.defaults.items():
            text = text.replace(f"${name}", value)
        return text

    def resolve(self, filename) -> Path:
        for base in self.search:
            if (base / filename).exists():
                return base / filename
        return self.root / filename

    def transform(self, element) -> np.ndarray:
        m = np.eye(4)
        for t in element:
            def vec(default):
                if t.get("value") is not None:
                    v = _floats(self.value(t))
                    return np.array(v * 3 if len(v) == 1 else v)
                return np.array([float(t.get(k, default)) for k in "xyz"])
            if t.tag == "translate":
                step = np.eye(4)
                step[:3, 3] = vec(0)
            elif t.tag == "scale":
                step = np.diag(np.append(vec(1), 1.0))
            elif t.tag == "rotate":
                step = _rotation(vec(0), float(t.get("angle", 0)))
            elif t.tag == "matrix":
                step = np.asarray(_floats(self.value(t)), np.float64).reshape(4, 4)
            elif t.tag == "lookat":
                step = _look_at(np.array(_floats(t.get("origin"))), _floats(t.get("target")),
                                _floats(t.get("up", "0, 1, 0")))
            else:
                warnings.warn(f"Mitsuba: unsupported transform {t.tag!r}")
                continue
            m = step @ m
        return m

    def properties(self, element) -> dict:
        props = {}
        for child in element:
            name = child.get("name")
            if child.tag in ("float", "integer"):
                props[name] = float(self.value(child))
            elif child.tag == "boolean":
                props[name] = self.value(child).lower() == "true"
            elif child.tag == "string":
                props[name] = self.value(child)
            elif child.tag == "rgb":
                v = _floats(self.value(child))
                props[name] = np.array(v * 3 if len(v) == 1 else v)
            elif child.tag == "spectrum":
                text = self.value(child)
                if ":" in text:
                    pairs = np.array([[float(x) for x in p.split(":")] for p in re.split(r",\s*", text.strip()) if p])
                    props[name] = spectrum_to_rgb(pairs[:, 0], pairs[:, 1])
                else:
                    props[name] = np.full(3, float(text))
            elif child.tag in ("point", "vector"):
                props[name] = (np.array(_floats(self.value(child))) if child.get("value")
                               else np.array([float(child.get(k, 0)) for k in "xyz"]))
            elif child.tag == "transform":
                props[name] = self.transform(child)
            elif child.tag == "texture":
                props[name or "texture"] = self.texture(child)
            elif child.tag == "ref":
                props[name or f"_ref_{len(props)}"] = self.objects.get(child.get("id"))
        return props

    def texture(self, element):
        props = self.properties(element)
        kind = element.get("type")
        if kind == "bitmap":
            import slangpy as spy

            data = np.asarray(spy.Bitmap(str(self.resolve(props["filename"]))))
            if data.dtype.kind in "ui":
                data = data.astype(np.float32) / np.iinfo(data.dtype).max
                if not props.get("raw", False):
                    data = np.where(data <= 0.04045, data / 12.92, ((data + 0.055) / 1.055) ** 2.4)
            return np.asarray(data, np.float32).reshape(data.shape[0], data.shape[1], -1)[..., :4]
        if kind == "checkerboard":
            n, tiles = 256, 2
            scale = props.get("to_uv", np.eye(4))
            u = (np.arange(n) + 0.5) / n
            uu, vv = np.meshgrid(u * scale[0, 0] * tiles / 2, u * scale[1, 1] * tiles / 2)
            on = ((np.floor(uu * 2) + np.floor(vv * 2)) % 2)[..., None] > 0
            c0, c1 = (np.broadcast_to(props.get(k, np.full(3, d)), (3,)) for k, d in (("color0", 0.4), ("color1", 0.2)))
            return np.where(on, c1, c0).astype(np.float32)
        warnings.warn(f"Mitsuba: unsupported texture {kind!r}, using gray")
        return np.full(3, 0.5)

    # ---- objects

    def name(self, element, default):
        """The element's id, or default (falcor2_inverse.Scene numbers repeated names)."""
        return element.get("id") or default

    def bsdf(self, element, two_sided=False):
        """A dict of StandardMaterial properties."""
        kind, props = element.get("type"), self.properties(element)
        nested = [c for c in element if c.tag == "bsdf"]
        if kind == "twosided":
            return self.bsdf(nested[0], True) if nested else {"double_sided": True}
        if kind == "normalmap":
            material = self.bsdf(nested[0], two_sided) if nested else {}
            normal = next((v for v in props.values() if isinstance(v, np.ndarray) and v.ndim == 3), None)
            if normal is not None:
                material["normal_texture"] = normal
            return material
        if kind in ("bumpmap", "mask", "blendbsdf"):
            warnings.warn(f"Mitsuba: {kind} is approximated by its (first) nested BSDF")
            return self.bsdf(nested[0], two_sided) if nested else {}

        def color(*names, default=0.5):
            for n in names:
                if n in props:
                    return props[n]
            return np.full(3, default)

        def roughness():
            alpha = props.get("alpha", props.get("alpha_u", 0.1))
            return float(np.sqrt(np.mean(alpha))) if not isinstance(alpha, np.ndarray) or alpha.ndim < 3 else 0.3

        def ior():
            def value(key, default):
                v = props.get(key, default)
                return IOR.get(v, 1.5) if isinstance(v, str) else float(v)
            return value("int_ior", 1.5046) / value("ext_ior", 1.00028)

        m = {"double_sided": two_sided}
        if kind == "diffuse":
            m.update(base_color=color("reflectance"), roughness_factor=1.0, ior=1.0)
        elif kind in ("plastic", "roughplastic"):
            m.update(base_color=color("diffuse_reflectance"), ior=ior(),
                     roughness_factor=roughness() if kind == "roughplastic" else 0.05)
        elif kind in ("conductor", "roughconductor"):
            f0 = props.get("specular_reflectance", 1.0)
            tint = np.asarray(CONDUCTORS.get(props.get("material", "Al"), CONDUCTORS["Al"]))
            m.update(base_color=tint * f0 if not isinstance(f0, np.ndarray) or f0.ndim < 3 else f0, metallic_factor=1.0,
                     roughness_factor=roughness() if kind == "roughconductor" else 0.0)
        elif kind in ("dielectric", "roughdielectric", "thindielectric"):
            m.update(base_color=np.ones(3), ior=ior(), specular_transmission_factor=1.0,
                     roughness_factor=roughness() if kind == "roughdielectric" else 0.0,
                     thin_walled=kind == "thindielectric")
        elif kind == "principled":
            m.update(base_color=color("base_color", default=0.8), metallic_factor=float(props.get("metallic", 0.0)),
                     roughness_factor=float(props.get("roughness", 0.5)),
                     specular_transmission_factor=float(props.get("spec_trans", 0.0)),
                     ior=float(props.get("eta", 1.5)))
        else:
            warnings.warn(f"Mitsuba: unsupported BSDF {kind!r}, using a gray diffuse material")
            m.update(base_color=np.full(3, 0.5), roughness_factor=1.0, ior=1.0)
        return m

    def material(self, properties, name):
        properties = dict(properties)
        base = properties.pop("base_color", np.full(3, 0.5))
        if isinstance(base, np.ndarray) and base.ndim == 3:
            properties["base_color_texture"], properties["base_color_factor"] = base, np.ones(3)
        else:
            properties["base_color_factor"] = base
        return self.scene.add_material(name, **properties)

    def shape(self, element):
        kind, props = element.get("type"), self.properties(element)
        name = self.name(element, Path(props["filename"]).stem if "filename" in props else kind)
        to_world = props.get("to_world", np.eye(4))
        if kind in ("obj", "ply", "serialized"):
            options = {"shape_index": int(props.get("shape_index", 0))} if kind == "serialized" else {}
            mesh = read_mesh(self.resolve(props["filename"]), **options)
            if props.get("face_normals", False):
                mesh.pop("normals", None)
        elif kind == "rectangle":
            positions, faces, uv = quad([0, 0, 0], [1, 0, 0], [0, 1, 0])
            mesh = {"positions": positions, "faces": faces, "texcoords": np.stack([uv[:, 0], 1 - uv[:, 1]], 1)}
        elif kind == "cube":
            positions, faces = box([0, 0, 0], 1.0)
            mesh = {"positions": positions, "faces": faces}
        elif kind == "sphere":
            positions, faces = sphere(props.get("center", np.zeros(3)), float(props.get("radius", 1.0)), 4)
            mesh = {"positions": positions, "faces": faces}
        elif kind == "disk":
            positions, faces = disk([0, 0, 0], 1.0, 64)
            mesh = {"positions": positions, "faces": faces}
        else:
            warnings.warn(f"Mitsuba: unsupported shape {kind!r} skipped")
            return
        positions = mesh["positions"] @ to_world[:3, :3].T + to_world[:3, 3]
        normals = mesh.get("normals")
        if normals is not None:
            normals = normals @ np.linalg.inv(to_world[:3, :3])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        faces = mesh["faces"]
        if props.get("flip_normals", False) ^ (np.linalg.det(to_world[:3, :3]) < 0):
            faces = faces[:, ::-1]
        bsdf = next((v for v in props.values() if isinstance(v, dict) and "double_sided" in v), None)
        for child in element:
            if child.tag == "bsdf":
                bsdf = self.bsdf(child)
        material = dict(bsdf or {"base_color": np.full(3, 0.5), "roughness_factor": 1.0, "ior": 1.0})
        emitter = next((c for c in element if c.tag == "emitter" and c.get("type") == "area"), None)
        emitter_props = self.properties(emitter) if emitter is not None else next(
            (v for v in props.values() if isinstance(v, tuple) and v[0] == "area"), (None, None))[1]
        if emitter_props is not None:
            material["emissive_factor"] = np.broadcast_to(emitter_props.get("radiance", np.ones(3)), (3,))
        key = (id(bsdf), "emissive_factor" in material and tuple(material["emissive_factor"]))
        if key not in self.materials:
            self.materials[key] = self.material(material, element.get("id", name) if emitter_props is not None
                                                else self.bsdf_names.get(id(bsdf), f"{name}_material"))
        self.scene.add_mesh(name, positions, faces, self.materials[key], texcoords=mesh.get("texcoords"),
                            normals=normals)

    def emitter(self, element):
        kind, props = element.get("type"), self.properties(element)
        name = self.name(element, kind)
        to_world = props.get("to_world", np.eye(4))
        scene = self.scene
        if kind == "point":
            position = props.get("position", to_world[:3, 3])
            scene.add_point_light(name, position, np.broadcast_to(props.get("intensity", np.ones(3)), (3,)))
        elif kind == "spot":
            light = scene.add_point_light(name, to_world[:3, 3], np.broadcast_to(props.get("intensity", np.ones(3)), (3,)))
            light.enable_shaping = True
            light.shaping_cone_angle = float(props.get("cutoff_angle", 20.0))
            transform = light.entity.transform
            from falcor2_inverse.scene import look_rotation
            transform.rotation = look_rotation(to_world[:3, 2])
            light.entity.transform = transform
        elif kind == "directional":
            direction = props.get("direction", to_world[:3, 2])
            irradiance = np.broadcast_to(props.get("irradiance", np.ones(3)), (3,))
            cutoff = 0.5  # degrees; radiance = irradiance / solid angle of the cone
            scene.add_distant_light(name, direction, irradiance / (2 * np.pi * (1 - np.cos(np.radians(cutoff)))),
                                    cutoff)
        elif kind == "envmap":
            # Mitsuba's maps are centered on -z like falcor2's, half a turn apart in longitude.
            rotation = to_world[:3, :3] @ _rotation([0, 1, 0], 180.0)[:3, :3]
            r = _axis_angle(rotation / np.cbrt(max(np.linalg.det(rotation), 1e-12)))
            scale = float(props.get("scale", 1.0))
            scene.add_env_map(name, str(self.resolve(props["filename"])), intensity=(scale,) * 3, rotation=r)
        elif kind == "constant":
            scene.add_constant_light(name, np.broadcast_to(props.get("radiance", np.ones(3)), (3,)))
        else:
            warnings.warn(f"Mitsuba: unsupported emitter {kind!r} skipped")

    def sensor(self, element):
        kind, props = element.get("type"), self.properties(element)
        if kind not in ("perspective", "thinlens"):
            warnings.warn(f"Mitsuba: unsupported sensor {kind!r} skipped")
            return
        film = next((c for c in element if c.tag == "film"), None)
        film_props = self.properties(film) if film is not None else {}
        width, height = int(film_props.get("width", 768)), int(film_props.get("height", 576))
        fov, axis = float(props.get("fov", 45.0)), props.get("fov_axis", "x")
        aspect = width / height
        if axis == "smaller":
            axis = "x" if width < height else "y"
        elif axis == "larger":
            axis = "x" if width > height else "y"
        half = np.tan(np.radians(fov) / 2)
        if axis == "x":
            half /= aspect
        elif axis == "diagonal":
            half /= np.sqrt(1 + aspect ** 2)
        m = props.get("to_world", np.eye(4))
        if np.linalg.det(m[:3, :3]) < 0:
            warnings.warn("Mitsuba: a mirrored camera transform renders mirrored")
        origin, forward, up = m[:3, 3], m[:3, 2], m[:3, 1]
        self.scene.add_camera(self.name(element, "camera"), origin, origin + forward, up,
                              fov_y=float(np.degrees(2 * np.arctan(half))), resolution=(width, height))
        sampler = next((c for c in element if c.tag == "sampler"), None)
        if sampler is not None:
            self.metadata.setdefault("spp", int(self.properties(sampler).get("sample_count", 4)))

    def load(self, path):
        self.materials, self.bsdf_names = {}, {}
        elements = list(self.elements(ET.parse(path).getroot(), Path(path).resolve().parent))
        # References may precede the objects' definitions: define the named objects first.
        for element in elements:
            if element.tag == "texture" and element.get("id"):
                self.objects[element.get("id")] = self.texture(element)
        for element in elements:
            if element.tag == "bsdf" and element.get("id"):
                bsdf = self.objects[element.get("id")] = self.bsdf(element)
                self.bsdf_names[id(bsdf)] = element.get("id")
            elif element.tag == "emitter" and element.get("type") == "area" and element.get("id"):
                self.objects[element.get("id")] = ("area", self.properties(element))
        for element in elements:
            if element.tag == "shape":
                self.shape(element)
            elif element.tag == "emitter" and element.get("type") != "area":
                self.emitter(element)
            elif element.tag == "sensor":
                self.sensor(element)
            elif element.tag == "integrator":
                depth = int(self.properties(element).get("max_depth", -1))
                if depth > 0:
                    self.metadata["max_bounces"] = max(depth - 1, 0)
        return self.metadata

    def elements(self, root, base):
        """The scene's elements with included files inlined, applying <default> and <path> on the way."""
        for element in root:
            if element.tag == "default":
                self.defaults.setdefault(element.get("name"), element.get("value"))
            elif element.tag == "path":
                self.search.insert(0, (base / self.value(element)).resolve())
            elif element.tag == "include":
                path = self.resolve(self.value(element) if element.get("value") else element.get("filename"))
                yield from self.elements(ET.parse(path).getroot(), path.parent)
            else:
                yield element


def _axis_angle(rotation):
    angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
    if angle < 1e-8:
        return np.zeros(3)
    if np.pi - angle < 1e-6:  # half turn: the axis is the dominant column of R + I
        axis = (rotation + np.eye(3))[:, np.argmax(np.diag(rotation))]
        return axis / np.linalg.norm(axis) * angle
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    return axis / (2 * np.sin(angle)) * angle


def load_mitsuba(scene, path, defaults=None) -> dict:
    """Add the objects of a Mitsuba XML scene to a Scene; returns its metadata (max_bounces, spp). defaults
    override the scene's <default> values."""
    return _Loader(scene, path, defaults).load(path)
