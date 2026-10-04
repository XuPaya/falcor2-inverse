"""Scene loaders by file type.

- USD (.usd, .usda, .usdc, .usdz), glTF (.gltf, .glb) and falcor2 Python scenes (.py): falcor2's importers.
- Mitsuba 3 scenes (.xml): falcor2_inverse.loaders.mitsuba.
- Meshes (.obj, .ply, Mitsuba .serialized): one mesh with a gray material, without lights or cameras.
"""

from pathlib import Path

import falcor2 as f2

from falcor2_inverse.device import default_device
from falcor2_inverse.loaders.meshes import read_mesh, read_obj, read_ply, read_serialized
from falcor2_inverse.loaders.mitsuba import load_mitsuba
from falcor2_inverse.scene import Scene

FALCOR2_FORMATS = (".usd", ".usda", ".usdc", ".usdz", ".gltf", ".glb", ".py")
MESH_FORMATS = (".obj", ".ply", ".serialized")


def load_scene(path, device=None, **options) -> Scene:
    """Load a scene file. Options: for Mitsuba scenes, values of its <default> parameters (e.g. spp=64)."""
    path = Path(path)
    suffix = path.suffix.lower()
    device = device or default_device()
    if suffix in FALCOR2_FORMATS:
        scene = Scene(f2.Scene.load(device, str(path)), device)
    elif suffix == ".xml":
        scene = Scene(device=device)
        scene.metadata = load_mitsuba(scene, path, {k: str(v) for k, v in options.items()})
    elif suffix in MESH_FORMATS:
        mesh = read_mesh(path)
        scene = Scene(device=device)
        material = scene.add_material(f"{path.stem}_material", base_color_factor=(0.7, 0.7, 0.7), roughness_factor=0.6)
        scene.add_mesh(path.stem, mesh["positions"], mesh["faces"], material, texcoords=mesh.get("texcoords"),
                       normals=mesh.get("normals"))
    else:
        raise ValueError(f"Unsupported scene format {suffix!r} ({path})")
    scene.update()
    return scene


__all__ = ["load_scene", "load_mitsuba", "read_mesh", "read_obj", "read_ply", "read_serialized"]
