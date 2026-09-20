"""The five pipeline stages as plain functions.

Each stage takes data in and returns data out; none of them reads a YAML file or writes
a video. The CLIs in ``faunacodec.cli`` and the demo notebook are thin layers over these.

    compress   video path  -> archive bytes        (edge device)
    decompress archive     -> DecodedVideo         (server)
    restore    frames      -> frames               (server, optional)
    upscale    frames      -> frames               (server, optional)
"""
from __future__ import annotations

import json
import platform
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .archive import read_payloads, write_archive
from .paths import REPO_ROOT
from .video_io import VideoInfo, probe_video

ProgressCallback = Callable[[str], None]

__all__ = [
    "DecodedVideo",
    "check_portability",
    "compress",
    "decompress",
    "encoder_provenance",
    "restore",
    "upscale",
]


@dataclass
class DecodedVideo:
    """What the server gets back from an archive."""

    frames: list[np.ndarray]
    fps: float
    width: int
    height: int
    #: frame index -> list of subject bounding boxes, as detected at encode time
    detections: dict[int, Any] = field(default_factory=dict)


def _noop(_message: str) -> None:
    pass


def _validate_detection_timeline(encoded: VideoInfo, detected: VideoInfo) -> None:
    if encoded.frame_count != detected.frame_count:
        raise ValueError(
            "The encoded and detection videos must have the same frame count: "
            f"{encoded.frame_count} != {detected.frame_count}"
        )
    if abs(encoded.fps - detected.fps) > 1e-3:
        raise ValueError(
            "The encoded and detection videos must have the same frame rate: "
            f"{encoded.fps:g} != {detected.fps:g}"
        )


def _scale_detections(
    detections: dict[str, Any],
    *,
    encoded_path: Path,
    encoded: VideoInfo,
    detected_path: Path,
    detected: VideoInfo,
) -> dict[str, Any]:
    scale_x = encoded.width / detected.width
    scale_y = encoded.height / detected.height
    frames: dict[int, list[Any]] = {}
    for frame_idx, boxes in (detections.get("frames", {}) or {}).items():
        scaled_boxes: list[Any] = []
        for box in boxes if isinstance(boxes, list) else []:
            if not isinstance(box, dict):
                scaled_boxes.append(box)
                continue
            scaled = dict(box)
            scaled["x1"] = max(0, min(encoded.width - 1, round(float(box["x1"]) * scale_x)))
            scaled["y1"] = max(0, min(encoded.height - 1, round(float(box["y1"]) * scale_y)))
            scaled["x2"] = max(0, min(encoded.width - 1, round(float(box["x2"]) * scale_x)))
            scaled["y2"] = max(0, min(encoded.height - 1, round(float(box["y2"]) * scale_y)))
            scaled_boxes.append(scaled)
        frames[int(frame_idx)] = scaled_boxes

    return {
        **detections,
        "video_path": str(encoded_path),
        "frame_count": encoded.frame_count,
        "fps": encoded.fps,
        "width": encoded.width,
        "height": encoded.height,
        "frames": frames,
        "detector_input": {
            "video_path": str(detected_path),
            "frame_count": detected.frame_count,
            "fps": detected.fps,
            "width": detected.width,
            "height": detected.height,
            "scale_x": scale_x,
            "scale_y": scale_y,
        },
    }


def compress(
    video_path: str | Path,
    config: dict[str, Any],
    *,
    detection_video_path: str | Path | None = None,
    root_dir: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> bytes:
    """Detect subjects, select frames, encode both streams, and pack an archive.

    ``config`` is the compression config: ``roi_detection``, ``frame_selection``, and
    ``compression`` sections (see configs/compression.yaml). ``detection_video_path``
    may name a higher-resolution video with the same timeline; its boxes are scaled to
    ``video_path`` before selection and encoding.
    """
    from .compression import encode_dual_stream
    from .config import validate_pipeline_config
    from .detection import run_roi_detection
    from .selection import apply_dual_timeline_policy, remove_redundant_frames

    say = progress or _noop
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    detection_path = Path(detection_video_path) if detection_video_path is not None else video_path
    if not detection_path.exists():
        raise FileNotFoundError(f"Detection video not found: {detection_path}")
    root = Path(root_dir) if root_dir is not None else REPO_ROOT

    # Fail on a malformed config before spending minutes on detection and encoding.
    validate_pipeline_config(config, str(video_path), root_dir=root)

    detection_cfg = config.get("roi_detection", {}) or {}
    selection_cfg = config.get("frame_selection", {}) or {}
    compression_cfg = config.get("compression", {}) or {}

    if str(compression_cfg.get("codec", "dcvc")).lower() != "dcvc":
        from .compression.dual_stream_ffmpeg import validate_ffmpeg_backend

        validate_ffmpeg_backend(compression_cfg)

    if detection_cfg.get("enable", True):
        say("detecting subjects")
        if detection_video_path is not None:
            encoded_info = probe_video(video_path)
            detected_info = probe_video(detection_path)
            _validate_detection_timeline(encoded_info, detected_info)
        detections = run_roi_detection(str(detection_path), detection_cfg)
        if detection_video_path is not None:
            detections = _scale_detections(
                detections,
                encoded_path=video_path,
                encoded=encoded_info,
                detected_path=detection_path,
                detected=detected_info,
            )
        boxes_by_frame = detections.get("frames", {}) or {}
        detected = sum(1 for boxes in boxes_by_frame.values() if boxes)
        say(f"detected subjects in {detected}/{len(boxes_by_frame)} frames")
        if detected == 0:
            # Without boxes the ROI stream is empty and the whole frame is coded at
            # background quality, leaving the subject unrecoverable. A tiny archive
            # looks like a great compression ratio, so say what actually happened.
            warnings.warn(
                "No subjects were detected, so the ROI stream will be empty and the "
                "whole frame will be coded at background quality. Check "
                "roi_detection.runtime (processing_scale, conf) against the input "
                "resolution.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        say("detection disabled; coding the whole frame as background")
        detections = {"video_path": str(video_path), "frames": {}}
        boxes_by_frame = {}

    say("selecting frames")
    selection = remove_redundant_frames(
        str(video_path), {int(k): v for k, v in boxes_by_frame.items()}, selection_cfg
    )
    selection = apply_dual_timeline_policy(selection, selection_cfg)

    say(f"encoding ({compression_cfg.get('codec', 'dcvc')})")
    encoded = encode_dual_stream(
        source_video_path=str(video_path),
        roi_bbox_map=boxes_by_frame,
        frame_drop_result=selection,
        compression_cfg=compression_cfg,
        root_dir=root,
    )

    meta = dict(encoded["meta"])
    meta["encoder"] = encoder_provenance(compression_cfg)

    return write_archive(
        roi_bin=encoded["roi_bin_bytes"],
        bg_bin=encoded["bg_bin_bytes"],
        meta=meta,
        roi_detections=detections,
        frame_drop=selection,
        runtime_config=config,
    )


def encoder_provenance(compression_cfg: dict[str, Any]) -> dict[str, Any]:
    """Record what encoded the archive, so the decoder can spot an unportable one."""
    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "codec": str(compression_cfg.get("codec", "dcvc")).lower(),
        "faunacodec_version": __version__,
    }


def check_portability(meta: dict[str, Any]) -> str | None:
    """Return a warning if this archive is one the local machine cannot decode faithfully.

    DCVC's floating-point entropy coder is not bit-exact across CPU architectures: a
    stream encoded on the ARM edge device decodes into noise on an x86 server, with no
    error raised. The deterministic integer runtime in third_party/dcvc_int16/
    exists for exactly this reason. Block codecs are unaffected.
    """
    encoder = meta.get("encoder") or {}
    encoded_on = str(encoder.get("machine", ""))
    if not encoded_on or encoded_on == platform.machine():
        return None
    if str(encoder.get("codec", "dcvc")).lower() != "dcvc":
        return None
    return (
        f"This archive was DCVC-encoded on {encoded_on} but is being decoded on "
        f"{platform.machine()}. DCVC's floating-point entropy coder is not bit-exact "
        "across architectures, so the output is likely to be corrupted. Use the "
        "deterministic integer runtime in third_party/dcvc_int16/ for "
        "cross-architecture transport."
    )


def decompress(
    archive: str | Path | bytes,
    config: dict[str, Any],
    *,
    interpolate: bool = True,
    strict_gpu: bool = False,
    max_frames: int = 0,
    progress: ProgressCallback | None = None,
) -> DecodedVideo:
    """Decode both streams, interpolate the dropped background frames, and composite.

    ``config`` is the decompression config (see configs/decompression.yaml), either the
    whole file or just its ``decompression`` section.
    """
    from .decompression.reconstruct import decompress_archive, decompress_archive_bytes

    say = progress or _noop

    payloads = read_payloads(archive)
    complaint = check_portability(json.loads(payloads["meta.json"]))
    if complaint:
        warnings.warn(complaint, RuntimeWarning, stacklevel=2)

    say("decoding streams")

    kwargs = {
        "no_interpolate": not interpolate,
        "strict_gpu": strict_gpu,
        "max_frames": max_frames,
    }
    if isinstance(archive, (bytes, bytearray)):
        frames, fps, width, height, roi_json = decompress_archive_bytes(
            bytes(archive), config, None, **kwargs
        )
    else:
        frames, fps, width, height, roi_json = decompress_archive(
            Path(archive), config, None, **kwargs
        )

    say(f"decoded {len(frames)} frames at {width}x{height}")
    return DecodedVideo(
        frames=frames,
        fps=fps,
        width=width,
        height=height,
        detections={int(k): v for k, v in (roi_json.get("frames", {}) or {}).items()},
    )


def restore(
    frames: list[np.ndarray],
    config: dict[str, Any],
    *,
    detections: dict[int, Any] | None = None,
    width: int | None = None,
    height: int | None = None,
    progress: ProgressCallback | None = None,
) -> list[np.ndarray]:
    """Remove compression artifacts with RestoreUNet (see configs/restoration.yaml)."""
    from .restoration import restore_frames

    say = progress or _noop
    if not frames:
        return []
    height = height or frames[0].shape[0]
    width = width or frames[0].shape[1]

    say("restoring frames")
    return restore_frames(frames, config, detections=detections or {}, width=width, height=height)


def upscale(
    frames: list[np.ndarray],
    backend: str,
    config: dict[str, Any],
    *,
    detections: dict[int, Any] | None = None,
    progress: ProgressCallback | None = None,
) -> list[np.ndarray]:
    """Upscale with one of the backends in ``faunacodec.upscaling.UPSCALER_BACKENDS``.

    ``config`` must carry ``out_w`` and ``out_h`` plus any backend-specific keys.
    """
    from .upscaling import build_upscaler

    say = progress or _noop
    if not frames:
        return []

    say(f"upscaling with {backend}")
    return build_upscaler(backend, config).upscale_sequence(frames, detections=detections)
