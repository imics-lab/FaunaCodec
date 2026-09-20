#!/usr/bin/env python3
"""Compress a video into a FaunaCodec archive. Runs on the edge device."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ._common import ROOT, add_common_arguments, load_config, progress_reporter, setup_logging


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="faunacodec-compress", description=__doc__.splitlines()[0]
    )
    parser.add_argument("video", help="Input video file")
    parser.add_argument(
        "-c", "--config", default="configs/compression.yaml", help="Compression config YAML"
    )
    parser.add_argument(
        "--resolution",
        default=None,
        metavar="WxH",
        help="Resize the input to this resolution before encoding (e.g. 640x360)",
    )
    parser.add_argument(
        "--detection-video",
        default=None,
        help="Optional higher-resolution video with the same timeline, used only for detection",
    )
    add_common_arguments(parser)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    setup_logging(args.verbose)
    say = progress_reporter("compress")

    from .. import stages
    from ..video_io import resize_video

    source = Path(args.video)
    if not source.exists():
        print(f"error: video not found: {source}", file=sys.stderr)
        return 1
    detection_video = Path(args.detection_video) if args.detection_video else None
    if detection_video is not None and not detection_video.exists():
        print(f"error: detection video not found: {detection_video}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    out_dir = Path((config.get("output", {}) or {}).get("out_dir", "outputs/compression"))
    output = Path(args.output) if args.output else out_dir / f"{source.stem}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    scratch = None
    video = source
    try:
        if args.resolution:
            width, height = (int(v) for v in args.resolution.lower().split("x"))
            scratch = output.parent / f".{source.stem}_{width}x{height}.mp4"
            say(f"resizing to {width}x{height}")
            video = resize_video(source, scratch, width, height)
            detection_video = detection_video or source

        archive = stages.compress(
            video,
            config,
            detection_video_path=detection_video,
            root_dir=ROOT,
            progress=say,
        )
    finally:
        if scratch is not None and scratch.exists():
            scratch.unlink()

    output.write_bytes(archive)
    source_mb = source.stat().st_size / 1e6
    archive_mb = len(archive) / 1e6
    say(
        f"done in {time.perf_counter() - started:.1f}s: "
        f"{source_mb:.2f} MB -> {archive_mb:.2f} MB ({source_mb / max(archive_mb, 1e-9):.1f}x)"
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
