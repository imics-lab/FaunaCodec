#!/usr/bin/env python3
"""Reconstruct a video from a FaunaCodec archive. Runs on the server."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ._common import add_common_arguments, load_config, progress_reporter, setup_logging


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="faunacodec-decompress", description=__doc__.splitlines()[0]
    )
    parser.add_argument("archive", help="Archive produced by faunacodec-compress")
    parser.add_argument(
        "-c", "--config", default="configs/decompression.yaml", help="Decompression config YAML"
    )
    parser.add_argument(
        "--no-interpolate",
        action="store_true",
        help="Hold dropped background frames instead of AMT-interpolating them",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Stop after N frames (0 = whole clip)"
    )
    add_common_arguments(parser)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    setup_logging(args.verbose)
    say = progress_reporter("decompress")

    from .. import stages
    from ..video_io import write_video

    archive = Path(args.archive)
    if not archive.exists():
        print(f"error: archive not found: {archive}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    out_dir = Path((config.get("output", {}) or {}).get("out_dir", "outputs/decompression"))
    output = Path(args.output) if args.output else out_dir / f"{archive.stem}_decompressed.mp4"

    started = time.perf_counter()
    decoded = stages.decompress(
        archive,
        config,
        interpolate=not args.no_interpolate,
        max_frames=args.max_frames,
        progress=say,
    )
    write_video(decoded.frames, output, fps=decoded.fps)

    say(f"done in {time.perf_counter() - started:.1f}s: {len(decoded.frames)} frames")
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
