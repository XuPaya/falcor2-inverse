"""The falcor2 graphics device and the PyTorch device that parameters and images live on."""

import os
from pathlib import Path

import slangpy as spy
import torch

from falcor2.editor.utils import get_slang_include_paths

_DEVICE = None


def create_device(shader_cache: bool = True, *, device_type: spy.DeviceType = spy.DeviceType.automatic) -> spy.Device:
    """Device with falcor2's shader include paths and a persistent shader cache (.shader-cache/).

    SlangPy selects Vulkan on Linux and D3D12 on Windows by default. Pass device_type to select a backend explicitly.

    PSDR kernels for scenes with OpenPBR materials take minutes to compile (more with shader debug info, which
    this device leaves off). Cache usage varies with material families, light sampler, normal maps, and enabled
    derivative kernels. A 1 GiB cache reduces eviction and recompilation when experiments use different
    configurations; its LMDB file grows only as entries are written.
    """
    root = Path(__file__).resolve().parents[1]
    return spy.Device(
        type=device_type,
        shader_cache_path=root / ".shader-cache" if shader_cache else None,
        shader_cache_size=1 << 30,
        compiler_options=spy.SlangCompilerOptions({
            "include_paths": get_slang_include_paths() + [Path(__file__).resolve().parent / "shaders"],
            "floating_point_mode": floating_point_mode()}),
    )


def floating_point_mode() -> spy.SlangFloatingPointMode:
    """Floating-point mode of every kernel of the device: environment variable FALCOR2_FP_MODE = precise (default),
    default (Slang's default) or fast. Precise (no contraction into fused multiply-adds) makes PSDR's primal bitwise
    identical to the forward path tracer at the same seed: otherwise the same expressions are contracted differently in
    the ray-tracing pipeline and in PSDR's compute kernels (rounding-level differences that occasionally flip a sampling
    decision later in a path). It costs ~10 % of a forward frame."""
    mode = os.environ.get("FALCOR2_FP_MODE", "precise").strip().lower()
    if mode not in ("default", "precise", "fast"):
        raise ValueError(f"FALCOR2_FP_MODE must be default, precise or fast, not {mode!r}")
    return getattr(spy.SlangFloatingPointMode, mode)


def default_device() -> spy.Device:
    """A falcor2 device shared by the scenes that are not given one."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = create_device()
    return _DEVICE


def torch_device() -> torch.device:
    """Where rendered images and scene parameters are returned: the first CUDA device, or the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
