"""Lanczos-windowed bicubic resampling: the fidelity reference for the upscaling stage.

It adds no detail, so full-reference metrics rate it highest among the upscalers while
no-reference metrics rate it lowest. That gap is the point of having it here.
"""
from __future__ import annotations

import cv2
import numpy as np


class BicubicUpscaler:
    """Config keys: out_w, out_h (both required)."""

    def __init__(self, cfg: dict) -> None:
        self.out_w = int(cfg["out_w"])
        self.out_h = int(cfg["out_h"])

    def upscale_sequence(
        self, frames: list[np.ndarray], detections: dict | None = None
    ) -> list[np.ndarray]:
        return [
            cv2.resize(f, (self.out_w, self.out_h), interpolation=cv2.INTER_LANCZOS4)
            for f in frames
        ]
