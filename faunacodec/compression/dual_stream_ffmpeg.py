"""
Encode frame sequences to bytes via ffmpeg using AV1, HEVC, or H.264.
Drop-in replacement for the DCVC encoder path; returns the same result dict shape.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from typing import Any

import numpy as np

from ..video_io import VideoInfo, probe_video

# Default encoder/preset/CRF per codec name.
# CRF scales differ: SVT-AV1 0-63, libx265 0-51, libx264 0-51 (lower = better).
_CODEC_DEFAULTS: dict[str, dict[str, Any]] = {
    "av1":  {"encoder": "libsvtav1", "preset": "5",      "roi_crf": 32, "bg_crf": 62},
    "hevc": {"encoder": "libx265",   "preset": "medium", "roi_crf": 22, "bg_crf": 32},
    "h264": {"encoder": "libx264",   "preset": "medium", "roi_crf": 19, "bg_crf": 30},
}


def _resolve_encoder(codec: str, ffmpeg_cfg: dict[str, Any]) -> tuple[str, str]:
    codec = codec.lower()
    d = _CODEC_DEFAULTS.get(codec, _CODEC_DEFAULTS["av1"])
    encoder = ffmpeg_cfg.get(f"{codec}_encoder", d["encoder"])
    preset  = (ffmpeg_cfg.get("preset", {}) or {}).get(codec) or ffmpeg_cfg.get(f"{codec}_preset", d["preset"])
    return str(encoder), str(preset)


def _crf_for(codec: str, stream: str, quality_cfg: dict[str, Any], ffmpeg_cfg: dict[str, Any]) -> int:
    codec = codec.lower()
    d = _CODEC_DEFAULTS.get(codec, _CODEC_DEFAULTS["av1"])
    key = "roi_crf" if stream == "roi" else "bg_crf"
    codec_quality = ((quality_cfg.get("ffmpeg", {}) or {}).get(codec, {}) or {})
    configured = [
        codec_quality.get(key),
        quality_cfg.get(key),
        (ffmpeg_cfg.get("crf", {}) or {}).get("roi" if stream == "roi" else "bg"),
    ]
    return int(next((value for value in configured if value is not None), d[key]))


def validate_ffmpeg_backend(compression_cfg: dict[str, Any]) -> None:
    """Fail before detection if the selected ffmpeg encoder is unavailable."""
    codec = str(compression_cfg.get("codec", "av1")).lower()
    encoder, _preset = _resolve_encoder(codec, compression_cfg.get("ffmpeg", {}) or {})
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for h264, hevc, and av1 compression")
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-h", f"encoder={encoder}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        detail = probe.stderr.strip()
        raise RuntimeError(
            f"ffmpeg encoder {encoder!r} is unavailable"
            + (f": {detail}" if detail else "")
        )


def encode_frames_ffmpeg_to_bytes(
    frames_iter: Iterable[tuple[int, np.ndarray]],
    *,
    info: VideoInfo,
    codec: str,
    crf: int,
    encoder: str,
    preset: str,
    extra_ffmpeg_args: list[str] = None,
) -> dict[str, Any]:
    """
    Pipe raw BGR frames into ffmpeg, return encoded bytes and metadata.

    Parameters
    ----------
    frames_iter : iterable of (source_frame_index, BGR uint8 array)
    info        : VideoInfo with width, height, fps
    codec       : logical codec name ("av1", "hevc", "h264")
    crf         : quality level (lower = better quality)
    encoder     : ffmpeg encoder name (e.g. "libsvtav1", "libx265")
    preset      : encoder preset string
    extra_ffmpeg_args : additional ffmpeg output args (inserted before output path)
    """
    with tempfile.TemporaryDirectory(prefix="ffenc_") as td:
        out_path = os.path.join(td, "stream.mp4")

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo",
            "-pixel_format", "bgr24",
            "-video_size", f"{info.width}x{info.height}",
            "-framerate", str(max(1, info.fps)),
            "-i", "pipe:0",
            "-c:v", encoder,
            "-crf", str(crf),
            "-preset", str(preset),
            "-pix_fmt", "yuv420p",
        ]
        if extra_ffmpeg_args:
            cmd.extend(extra_ffmpeg_args)
        cmd.append(out_path)

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        frame_indices: list[int] = []
        n = 0
        for src_idx, frame in frames_iter:
            frame_indices.append(int(src_idx))
            proc.stdin.write(frame.tobytes())
            n += 1

        proc.stdin.close()
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg ({encoder}) encoding failed (exit {proc.returncode}). "
                f"Check encoder availability: ffmpeg -encoders | grep {encoder}"
            )

        with open(out_path, "rb") as f:
            stream_bytes = f.read()

    return {
        "bitstream_bytes": stream_bytes,
        "meta": {
            "codec": codec,
            "encoder": encoder,
            "crf": crf,
            "preset": preset,
            "frames_encoded": n,
            "frame_index_map": frame_indices,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "compressed_bytes": len(stream_bytes),
        },
    }


def encode_roi_bg_ffmpeg(
    *,
    source_video_path,
    roi_bbox_map,
    frame_drop_result,
    compression_cfg: dict[str, Any],
    root_dir,
) -> dict[str, Any]:
    """
    Top-level ffmpeg compression entry point; mirrors encode_roi_bg_dcvc interface.
    Reuses the dual_stream_dcvc helpers for ROI masking and frame iteration.
    """
    from pathlib import Path as _Path

    from .dual_stream_dcvc import (
        _capture_rendered_kept_frames_single_pass,
        _iter_cached_frames,
        _iter_kept_bg_frames,
        _iter_kept_roi_frames,
        _pick_indices,
    )

    root = _Path(root_dir).expanduser().resolve()
    src_path = _Path(str(source_video_path)).expanduser()
    if not src_path.is_absolute():
        src_path = (root / src_path).resolve()

    codec       = str(compression_cfg.get("codec", "av1")).lower()
    quality_cfg = compression_cfg.get("quality", {}) or {}
    ffmpeg_cfg  = compression_cfg.get("ffmpeg", {}) or {}
    roi_cfg     = compression_cfg.get("roi", {}) or {}
    low_memory  = bool(compression_cfg.get("low_memory", False))

    encoder, preset = _resolve_encoder(codec, ffmpeg_cfg)
    roi_crf = _crf_for(codec, "roi", quality_cfg, ffmpeg_cfg)
    bg_crf  = _crf_for(codec, "bg",  quality_cfg, ffmpeg_cfg)
    min_conf = float(roi_cfg.get("min_conf", 0.0))

    info = probe_video(str(src_path))

    roi_indices = _pick_indices(frame_drop_result, "roi_kept_frames", "kept_frames")
    bg_indices  = _pick_indices(frame_drop_result, "bg_kept_frames",  "kept_frames")

    import tempfile as _tmp
    with _tmp.TemporaryDirectory(prefix="ffenc_dual_") as td:
        if low_memory:
            roi_enc = encode_frames_ffmpeg_to_bytes(
                _iter_kept_roi_frames(video_path=src_path, roi_kept_frames=roi_indices,
                                      roi_bbox_map=roi_bbox_map, roi_min_conf=min_conf),
                info=info, codec=codec, crf=roi_crf, encoder=encoder, preset=preset,
            )
            bg_enc = encode_frames_ffmpeg_to_bytes(
                _iter_kept_bg_frames(video_path=src_path, bg_kept_frames=bg_indices),
                info=info, codec=codec, crf=bg_crf, encoder=encoder, preset=preset,
            )
        else:
            cached = _capture_rendered_kept_frames_single_pass(
                video_path=src_path, info=info,
                roi_kept_frames=roi_indices, bg_kept_frames=bg_indices,
                roi_bbox_map=roi_bbox_map, roi_min_conf=min_conf,
                work_dir=_Path(td),
            )
            try:
                roi_enc = encode_frames_ffmpeg_to_bytes(
                    _iter_cached_frames(cached.get("roi_store"), cached.get("roi_indices", []),
                                        int(cached.get("roi_count", 0) or 0)),
                    info=info, codec=codec, crf=roi_crf, encoder=encoder, preset=preset,
                )
                bg_enc = encode_frames_ffmpeg_to_bytes(
                    _iter_cached_frames(cached.get("bg_store"), cached.get("bg_indices", []),
                                        int(cached.get("bg_count", 0) or 0)),
                    info=info, codec=codec, crf=bg_crf, encoder=encoder, preset=preset,
                )
            finally:
                from .dual_stream_dcvc import _close_memmap
                _close_memmap(cached.get("roi_store"))
                _close_memmap(cached.get("bg_store"))

    combined_meta = {
        "codec": codec,
        "video": {"path": str(src_path), "width": info.width,
                  "height": info.height, "fps": info.fps, "frames_total": info.frame_count},
        "streams": {"roi": roi_enc["meta"], "bg": bg_enc["meta"]},
    }

    return {
        "roi_bin_bytes": roi_enc["bitstream_bytes"],
        "bg_bin_bytes":  bg_enc["bitstream_bytes"],
        "meta": combined_meta,
    }
