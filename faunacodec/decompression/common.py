from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from ..config import coerce_bool
from ..paths import resolve_repo_path
from .interpolation_amt import AmtInterpolator


def _validate_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    dec = cfg.get("decompression", {}) or {}
    if not isinstance(dec, dict):
        raise ValueError("decompression config must be an object")
    out = dict(dec)
    out.setdefault("codec", "mp4v")
    out.setdefault("mask_source", "roi_detection")
    # Temporal ROI stabilization to reduce patch flicker/jitter.
    out.setdefault("roi_temporal_stabilize", True)
    out.setdefault("roi_temporal_alpha_still", 0.70)
    out.setdefault("roi_temporal_alpha_motion", 0.92)
    out.setdefault("roi_temporal_mask_dilate", 1)
    out.setdefault("roi_temporal_overlap_only", True)
    out.setdefault("roi_blend_edge_px", 2)
    interp = out.get("interpolate", {}) or {}
    if not isinstance(interp, dict):
        interp = {}
    interp.setdefault("enable", True)
    interp.setdefault("model", "amt-s")
    interp.setdefault("weights_path", None)
    interp.setdefault("repo_dir", "third_party/amt")
    interp.setdefault("device", "cuda")
    interp.setdefault("fp16", True)
    interp.setdefault("pad_to", 16)
    interp.setdefault("batch_size", 1)
    interp.setdefault("crop_margin", 8)
    interp.setdefault("max_crop_side", 768)
    interp_dev = str(interp.get("device", "cuda")).strip().lower()
    if interp_dev in {"cpu", "mps"}:
        raise ValueError("Strict GPU runtime forbids decompression.interpolate.device set to CPU/MPS.")
    if interp_dev not in {"", "auto", "cuda"} and not interp_dev.startswith("cuda:"):
        raise ValueError(
            "decompression.interpolate.device must be one of: auto, cuda, cuda:<index> for strict GPU runtime."
        )
    try:
        interp_batch_size = int(interp.get("batch_size", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("decompression.interpolate.batch_size must be an integer >= 1") from exc
    if interp_batch_size < 1:
        raise ValueError("decompression.interpolate.batch_size must be >= 1")
    interp["batch_size"] = int(interp_batch_size)
    try:
        interp_crop_margin = int(interp.get("crop_margin", 8))
    except (TypeError, ValueError) as exc:
        raise ValueError("decompression.interpolate.crop_margin must be an integer >= 0") from exc
    if interp_crop_margin < 0:
        raise ValueError("decompression.interpolate.crop_margin must be >= 0")
    interp["crop_margin"] = int(interp_crop_margin)
    try:
        interp_max_crop_side = int(interp.get("max_crop_side", 768))
    except (TypeError, ValueError) as exc:
        raise ValueError("decompression.interpolate.max_crop_side must be an integer >= 0") from exc
    if interp_max_crop_side < 0:
        raise ValueError("decompression.interpolate.max_crop_side must be >= 0")
    interp["max_crop_side"] = int(interp_max_crop_side)
    out["interpolate"] = interp
    try:
        roi_blend_edge_px = int(out.get("roi_blend_edge_px", 2))
    except (TypeError, ValueError) as exc:
        raise ValueError("decompression.roi_blend_edge_px must be an integer >= 0") from exc
    if roi_blend_edge_px < 0:
        raise ValueError("decompression.roi_blend_edge_px must be >= 0")
    out["roi_blend_edge_px"] = int(roi_blend_edge_px)

    dcvc = out.get("dcvc", {}) or {}
    if not isinstance(dcvc, dict):
        dcvc = {}
    if "use_cuda" in dcvc and coerce_bool(dcvc.get("use_cuda"), default=True) is False:
        raise ValueError("Strict GPU runtime forbids decompression.dcvc.use_cuda=false.")
    dcvc_dev = str(dcvc.get("device", "cuda")).strip().lower()
    if dcvc_dev in {"cpu", "mps"}:
        raise ValueError("Strict GPU runtime forbids decompression.dcvc.device set to CPU/MPS.")
    out["dcvc"] = dcvc
    return out


def _pick_stream_indices(frame_drop_json: dict[str, Any], meta: dict[str, Any], stream: str) -> list[int]:
    stream_meta = (meta.get("streams", {}) or {}).get(stream, {}) or {}
    m = stream_meta.get("frame_index_map", None)
    if isinstance(m, list) and m:
        return [int(x) for x in m]
    key = "roi_kept_frames" if stream == "roi" else "bg_kept_frames"
    arr = frame_drop_json.get(key, None)
    if isinstance(arr, list) and arr:
        return [int(x) for x in arr]
    return []


def _infer_total_frames(meta: dict[str, Any], frame_drop_json: dict[str, Any], roi_indices: Sequence[int], bg_indices: Sequence[int]) -> int:
    v = meta.get("video", {}) or {}
    n = int(v.get("frames_total", 0) or 0)
    if n > 0:
        return n
    stats = frame_drop_json.get("stats", {}) or {}
    n = int(stats.get("num_frames_read", 0) or 0)
    if n > 0:
        return n
    return max(max(roi_indices or [0]), max(bg_indices or [0])) + 1


def _resize_if_needed(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == int(width) and frame.shape[0] == int(height):
        return frame
    return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)


def _init_amt_interpolator(interp_cfg: dict[str, Any]) -> AmtInterpolator | None:
    if not bool(interp_cfg.get("enable", True)):
        return None
    variant = str(interp_cfg.get("model", "amt-s"))
    repo_dir = str(resolve_repo_path(interp_cfg.get("repo_dir", "third_party/amt")))
    weights_raw = interp_cfg.get("weights_path") or (
        "models/amt-l.pth" if variant == "amt-l" else "models/amt-s.pth"
    )
    weights_path = str(resolve_repo_path(weights_raw))
    device = str(interp_cfg.get("device", "auto"))
    fp16 = bool(interp_cfg.get("fp16", True))
    pad_to = int(interp_cfg.get("pad_to", 16))
    try:
        return AmtInterpolator(
            amt_repo_dir=repo_dir,
            variant=variant,
            weights_path=weights_path,
            device=device,
            fp16=fp16,
            pad_to=pad_to,
        )
    except Exception:
        if not fp16:
            raise
        return AmtInterpolator(
            amt_repo_dir=repo_dir,
            variant=variant,
            weights_path=weights_path,
            device=device,
            fp16=False,
            pad_to=pad_to,
        )
