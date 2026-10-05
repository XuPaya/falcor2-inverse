"""The falcor2 device (D3D12) and the PyTorch device that parameters and images live on."""

from pathlib import Path

import slangpy as spy
import torch

from falcor2.editor.utils import get_slang_include_paths

_DEVICE = None


def create_device(shader_cache: bool = True) -> spy.Device:
    """D3D12 device with falcor2's shader include paths and a persistent shader cache (.shader-cache/).

    PSDR kernels for scenes with OpenPBR materials take minutes to compile (more with shader debug info, which
    this device leaves off); the cache compiles each of them once.  One scene configuration (material types, light
    sampler, normal maps) takes ~14 MiB with all PSDR kernels but ~8 min to compile, and slangpy's default cache
    size (128 MiB, eviction from 70%) holds only ~6 of them, so experiments evicted and recompiled each other's
    kernels.  1 GiB holds ~50; the LMDB file grows only as entries are written.
    """
    root = Path(__file__).resolve().parents[1]
    return spy.Device(
        type=spy.DeviceType.d3d12,
        shader_cache_path=root / ".shader-cache" if shader_cache else None,
        shader_cache_size=1 << 30,
        compiler_options=spy.SlangCompilerOptions({
            "include_paths": get_slang_include_paths() + [Path(__file__).resolve().parent / "shaders"]}),
    )


def default_device() -> spy.Device:
    """A falcor2 device shared by the scenes that are not given one."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = create_device()
    return _DEVICE


def torch_device() -> torch.device:
    """Where rendered images and scene parameters are returned: the first CUDA device, or the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
