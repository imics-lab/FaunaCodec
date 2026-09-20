"""Per-frame super-resolution via S3Diff (degradation-guided one-step diffusion SR).

S3Diff conditions an SD-Turbo backbone on a degradation estimate produced by a small
ResNet, so a single diffusion step suffices. Upstream code and weights are fetched on
first use; set `repo_dir` and `weights_dir` to point at an existing checkout instead.
"""
from __future__ import annotations

import math
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from ..paths import REPO_ROOT

_REPO_URL = "https://github.com/ArcticHare105/S3Diff.git"
_WEIGHT_URLS = {
    "de_net.pth": "https://huggingface.co/zhangap/S3Diff/resolve/main/de_net.pth",
    "s3diff.pkl": "https://huggingface.co/zhangap/S3Diff/resolve/main/s3diff.pkl",
}
_PROMPT = "a clear and high quality image"


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[s3diff] downloading {dest.name} ...", flush=True)
    urllib.request.urlretrieve(url, dest)


class S3DiffUpscaler:
    """Upscale BGR frames with S3Diff.

    Config keys:
        repo_dir     (str) S3Diff checkout; cloned here on first use.
        weights_dir  (str) Directory holding de_net.pth and s3diff.pkl; downloaded on first use.
        out_w/out_h  (int) Output resolution. Required.
        device       (str) Torch device, default "cuda".
        half         (bool) fp16 inference, default True.
        color_fix    (str) "wavelet" or "" to disable, default "wavelet".
        seed         (int) RNG seed, default 42.
    """

    def __init__(self, cfg: dict) -> None:
        root = REPO_ROOT
        self.repo = Path(cfg.get("repo_dir", root / "S3Diff"))
        weights = Path(cfg.get("weights_dir", root / "weights"))

        self.out_w = int(cfg["out_w"])
        self.out_h = int(cfg["out_h"])
        self.device = torch.device(str(cfg.get("device", "cuda")))
        self.half = bool(cfg.get("half", True))
        self.color_fix = str(cfg.get("color_fix", "wavelet"))
        self.seed = int(cfg.get("seed", 42))

        if not self.repo.exists():
            print(f"[s3diff] cloning S3Diff into {self.repo} ...", flush=True)
            subprocess.run(["git", "clone", _REPO_URL, str(self.repo)], check=True)
        for name, url in _WEIGHT_URLS.items():
            if not (weights / name).exists():
                _fetch(url, weights / name)

        for path in (str(self.repo / "src"), str(self.repo)):
            if path not in sys.path:
                sys.path.insert(0, path)

        from de_net import DEResNet
        from huggingface_hub import snapshot_download
        from s3diff import S3Diff

        print("[s3diff] loading model ...", flush=True)
        self.net_sr = S3Diff(
            lora_rank_unet=32,
            lora_rank_vae=16,
            sd_path=snapshot_download(repo_id="stabilityai/sd-turbo"),
            pretrained_path=str(weights / "s3diff.pkl"),
            device=self.device,
        )
        self.net_sr.set_eval()
        if self.half:
            self.net_sr.half()

        self.net_de = DEResNet(num_in_ch=3, num_degradation=2)
        state = torch.load(str(weights / "de_net.pth"), map_location="cpu")
        self.net_de.load_state_dict(state.get("state_dict", state))
        self.net_de.to(self.device).eval()
        print("[s3diff] model loaded.", flush=True)

    def upscale_sequence(
        self, frames: list[np.ndarray], detections: dict | None = None
    ) -> list[np.ndarray]:
        """Upscale BGR frames to (out_h, out_w). `detections` is unused; S3Diff is per-frame."""
        out = []
        for i, frame in enumerate(frames):
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            out.append(self._upscale_frame(frame))
            print(f"[s3diff] frame {i + 1}/{len(frames)}", flush=True)
        return out

    def _upscale_frame(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lr = transforms.ToTensor()(rgb).unsqueeze(0).to(self.device)
        if self.half:
            lr = lr.half()

        up = F.interpolate(
            lr, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False
        ).contiguous()

        # S3Diff's UNet requires spatial dimensions that are multiples of 64.
        normalized = (up * 2.0 - 1.0).clamp(-1.0, 1.0)
        pad_h = math.ceil(self.out_h / 64) * 64 - self.out_h
        pad_w = math.ceil(self.out_w / 64) * 64 - self.out_w
        padded = F.pad(normalized, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            degradation = self.net_de(lr)
            sr = self.net_sr(padded, degradation, prompt=_PROMPT)

        sr = sr[:, :, : self.out_h, : self.out_w]
        out = (sr * 0.5 + 0.5).clamp(0, 1).cpu().float()

        if self.color_fix == "wavelet":
            from s3diff_utils import wavelet_color_fix

            fixed = wavelet_color_fix(
                transforms.ToPILImage()(out[0]),
                transforms.ToPILImage()(up[0].cpu().float()),
            )
            out = transforms.ToTensor()(fixed).unsqueeze(0)

        arr = (out[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
