#!/usr/bin/env python3
"""Run any contiguous combination of pipeline stages end to end.

Stages run in the order listed and pass data in memory, so nothing is re-encoded
between them:

    compress     video    -> archive
    decompress   archive  -> frames
    restore      frames   -> frames
    upscale      frames   -> frames

Start from a raw video to run `compress`, or from an already-decompressed video to run
only the server-side stages.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ._common import (
    ROOT,
    load_config,
    load_stage_config,
    progress_reporter,
    setup_logging,
)

STAGES = ("compress", "decompress", "restore", "upscale")


def _parse_args(argv=None) -> argparse.Namespace:
    from ..upscaling import UPSCALER_BACKENDS

    parser = argparse.ArgumentParser(
        prog="faunacodec-pipeline",
        description=__doc__.split("\n\n")[0],
        epilog=(
            "examples:\n"
            "  faunacodec-pipeline data/bird1.mp4\n"
            "  faunacodec-pipeline data/bird1.mp4 --stages compress decompress\n"
            "  faunacodec-pipeline decompressed.mp4 --stages upscale --upscaler osediff\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", help="Raw video (for compress) or decompressed video")
    parser.add_argument(
        "-c", "--config", default="configs/pipeline.yaml", help="Pipeline config YAML"
    )
    parser.add_argument(
        "--detection-video",
        default=None,
        help="Optional higher-resolution video with the same timeline, used only for detection",
    )
    parser.add_argument(
        "-s", "--stages", nargs="+", default=list(STAGES), metavar="STAGE",
        help=f"Stages to run, in order. Choose from: {', '.join(STAGES)}",
    )
    parser.add_argument(
        "-u", "--upscaler", default=None, choices=sorted(UPSCALER_BACKENDS),
        help="Upscaling backend (default: the one named in the config)",
    )
    parser.add_argument("-o", "--output", default=None, help="Final video path")
    parser.add_argument(
        "--save-intermediate", action="store_true", help="Write the output of every stage"
    )
    parser.add_argument(
        "--lossless", action="store_true", help="Write the final video as FFV1 MKV, not MP4"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Log every step")
    return parser.parse_args(argv)


def _target_resolution(pipeline_cfg: dict[str, Any], frames: list[np.ndarray],
                       width: int, height: int, scale: float) -> tuple[int, int]:
    upscaling = pipeline_cfg.get("upscaling", {}) or {}
    if upscaling.get("out_w") and upscaling.get("out_h"):
        return int(upscaling["out_w"]), int(upscaling["out_h"])
    if not width or not height:
        if not frames:
            raise RuntimeError("Cannot infer the upscale target: no frames to measure")
        height, width = frames[0].shape[:2]
    return int(round(width * scale)), int(round(height * scale))


def main(argv=None) -> int:
    args = _parse_args(argv)
    setup_logging(args.verbose)
    say = progress_reporter("pipeline")

    from .. import stages as api
    from ..video_io import read_video, resize_video, write_video, write_video_lossless

    requested = [s.lower() for s in args.stages]
    unknown = [s for s in requested if s not in STAGES]
    if unknown:
        print(f"error: unknown stage(s) {unknown}; choose from {list(STAGES)}", file=sys.stderr)
        return 1

    source = Path(args.video)
    if not source.exists():
        print(f"error: video not found: {source}", file=sys.stderr)
        return 1
    detection_video = Path(args.detection_video) if args.detection_video else None
    if detection_video is not None and not detection_video.exists():
        print(f"error: detection video not found: {detection_video}", file=sys.stderr)
        return 1

    pipeline_cfg = load_config(args.config)
    out_dir = Path((pipeline_cfg.get("output", {}) or {}).get("out_dir", "outputs/pipeline"))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_intermediate = args.save_intermediate or bool(
        (pipeline_cfg.get("output", {}) or {}).get("save_intermediate", False)
    )

    archive: bytes | None = None
    frames: list[np.ndarray] = []
    fps, width, height = 30.0, 0, 0
    detections: dict[int, Any] = {}

    if "compress" not in requested:
        say(f"loading {source}")
        frames, fps = read_video(source)
        height, width = frames[0].shape[:2]

    started = time.perf_counter()
    scratch: Path | None = None

    try:
        for stage in requested:
            stage_started = time.perf_counter()

            if stage == "compress":
                compression_cfg = load_stage_config(pipeline_cfg, "compression")
                resolution = str(
                    (pipeline_cfg.get("input", {}) or {}).get("resolution", "") or ""
                )
                encode_from = source
                if resolution:
                    w, h = (int(v) for v in resolution.lower().split("x"))
                    scratch = out_dir / f".{source.stem}_{w}x{h}.mp4"
                    say(f"resizing to {w}x{h}")
                    encode_from = resize_video(source, scratch, w, h)
                    detection_video = detection_video or source
                archive = api.compress(
                    encode_from,
                    compression_cfg,
                    detection_video_path=detection_video,
                    root_dir=ROOT,
                    progress=say,
                )
                say(f"archive is {len(archive) / 1e6:.2f} MB")
                if save_intermediate:
                    path = out_dir / f"{source.stem}.zip"
                    path.write_bytes(archive)
                    say(f"wrote {path}")

            elif stage == "decompress":
                if archive is None:
                    print("error: decompress needs compress to run first", file=sys.stderr)
                    return 1
                decompression_cfg = load_stage_config(pipeline_cfg, "decompression")
                decoded = api.decompress(archive, decompression_cfg, progress=say)
                frames, fps = decoded.frames, decoded.fps
                width, height = decoded.width, decoded.height
                detections = decoded.detections
                if save_intermediate:
                    write_video(frames, out_dir / f"{source.stem}_decompressed.mp4", fps=fps)

            elif stage == "restore":
                restoration_cfg = load_stage_config(pipeline_cfg, "restoration")
                frames = api.restore(
                    frames, restoration_cfg,
                    detections=detections, width=width, height=height, progress=say,
                )
                if save_intermediate:
                    write_video(frames, out_dir / f"{source.stem}_restored.mp4", fps=fps)

            elif stage == "upscale":
                upscaling_cfg = load_stage_config(pipeline_cfg, "upscaling")
                backend = args.upscaler or str(upscaling_cfg.get("backend", "bicubic"))
                scale = float(upscaling_cfg.get("scale", 2))
                out_w, out_h = _target_resolution(pipeline_cfg, frames, width, height, scale)
                backend_cfg = dict(upscaling_cfg.get(backend, {}) or {})
                backend_cfg.setdefault("out_w", out_w)
                backend_cfg.setdefault("out_h", out_h)
                backend_cfg.setdefault("device", upscaling_cfg.get("device", "cuda"))
                frames = api.upscale(
                    frames, backend, backend_cfg, detections=detections, progress=say
                )

            say(f"{stage} took {time.perf_counter() - stage_started:.1f}s")
    finally:
        if scratch is not None and scratch.exists():
            scratch.unlink()

    if frames:
        suffix = ".mkv" if args.lossless else ".mp4"
        final = Path(args.output) if args.output else (
            out_dir / f"{source.stem}_{'_'.join(requested)}{suffix}"
        )
        writer = write_video_lossless if args.lossless else write_video
        writer(frames, final, fps=fps)
        say(f"complete in {time.perf_counter() - started:.1f}s -> {final}")
        print(final)
    else:
        say(f"complete in {time.perf_counter() - started:.1f}s (no video stage ran)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
