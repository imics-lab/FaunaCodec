"""Dual-stream (ROI + background) encoders."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .dual_stream_dcvc import encode_roi_bg_dcvc
from .dual_stream_ffmpeg import encode_roi_bg_ffmpeg

__all__ = ["encode_dual_stream", "encode_roi_bg_dcvc", "encode_roi_bg_ffmpeg"]


def encode_dual_stream(
    *,
    source_video_path,
    roi_bbox_map: Mapping[Any, Any],
    frame_drop_result: dict[str, Any],
    compression_cfg: dict[str, Any],
    root_dir,
) -> dict[str, Any]:
    """Encode the ROI and background streams with the backend named by compression_cfg['codec'].

    Returns {"roi_bin_bytes": bytes, "bg_bin_bytes": bytes, "meta": dict}.
    """
    codec = str(compression_cfg.get("codec", "dcvc")).lower()
    encode = encode_roi_bg_dcvc if codec == "dcvc" else encode_roi_bg_ffmpeg
    return encode(
        source_video_path=source_video_path,
        roi_bbox_map=roi_bbox_map,
        frame_drop_result=frame_drop_result,
        compression_cfg=compression_cfg,
        root_dir=root_dir,
    )
