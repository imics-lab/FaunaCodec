#!/usr/bin/env python3
"""Run the RestoreUNet artifact-removal stage on a decompressed video."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ._common import add_common_arguments, load_config, progress_reporter, setup_logging


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="faunacodec-restore", description=__doc__.splitlines()[0]
    )
    parser.add_argument("video", help="Decompressed video to restore")
    parser.add_argument(
        "-c", "--config", default="configs/restoration.yaml", help="Restoration config YAML"
    )
    parser.add_argument(
        "--ddim-steps", type=int, default=None, help="Override inference.ddim_steps"
    )
    add_common_arguments(parser)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    setup_logging(args.verbose)
    say = progress_reporter("restore")

    from .. import stages
    from ..video_io import read_video, write_video

    video = Path(args.video)
    if not video.exists():
        print(f"error: video not found: {video}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    if args.ddim_steps is not None:
        config.setdefault("inference", {})["ddim_steps"] = args.ddim_steps

    out_dir = Path((config.get("output", {}) or {}).get("out_dir", "outputs/restoration"))
    output = Path(args.output) if args.output else out_dir / f"{video.stem}_restored.mp4"

    started = time.perf_counter()
    frames, fps = read_video(video)
    restored = stages.restore(frames, config, progress=say)
    write_video(restored, output, fps=fps)

    say(f"done in {time.perf_counter() - started:.1f}s: {len(restored)} frames")
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
