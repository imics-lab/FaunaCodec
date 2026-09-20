"""Replaceable upscaling backends.

Every backend takes a config dict and exposes
``upscale_sequence(frames, detections=None) -> list[np.ndarray]``, so the stage can be
swapped without touching the rest of the pipeline. Backends are imported lazily because
each pulls in heavy and mutually incompatible dependencies.
"""
from __future__ import annotations

from typing import Any

__all__ = ["UPSCALER_BACKENDS", "build_upscaler"]

# name -> (module, class). Order is the order the README documents them in.
UPSCALER_BACKENDS: dict[str, tuple[str, str]] = {
    "bicubic": ("bicubic_upscaler", "BicubicUpscaler"),
    "s3diff": ("s3diff_upscaler", "S3DiffUpscaler"),
    "osediff": ("osediff_upscaler", "OSEDiffUpscaler"),
    "pid": ("pid_upscaler", "PiDUpscaler"),
    "wan": ("wan_upscaler", "WanUpscaler"),
    "cogvideo": ("cogvideo_upscaler", "CogVideoUpscaler"),
}


def build_upscaler(name: str, cfg: dict[str, Any]) -> Any:
    """Instantiate the named backend. Raises KeyError with the valid names if unknown."""
    key = str(name).lower()
    if key not in UPSCALER_BACKENDS:
        raise KeyError(f"Unknown upscaler {name!r}. Available: {', '.join(UPSCALER_BACKENDS)}")
    module_name, class_name = UPSCALER_BACKENDS[key]
    module = __import__(f"{__name__}.{module_name}", fromlist=[class_name])
    return getattr(module, class_name)(cfg)
