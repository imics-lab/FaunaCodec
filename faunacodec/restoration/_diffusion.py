"""GaussianDiffusion: cosine schedule, DDPM forward process, DDIM reverse sampler."""
from __future__ import annotations

import math

import torch


def _cosine_alpha_bar(t: torch.Tensor, T: int, s: float = 0.008) -> torch.Tensor:
    f  = torch.cos((t / T + s) / (1.0 + s) * math.pi / 2.0) ** 2
    f0 = math.cos(s / (1.0 + s) * math.pi / 2.0) ** 2
    return (f / f0).clamp(0.0, 1.0)


class GaussianDiffusion:
    """
    DDPM forward process + deterministic DDIM reverse.

    Args:
        T: Total diffusion timesteps (default 1000).
    """

    def __init__(self, T: int = 1000):
        self.T = T
        ts = torch.arange(T + 1, dtype=torch.float32)
        self.alpha_bar = _cosine_alpha_bar(ts, T)   # (T+1,) on CPU

    # ── Forward (noising) ─────────────────────────────────────────────────────

    # ── Reverse (DDIM, η=0) ───────────────────────────────────────────────────
