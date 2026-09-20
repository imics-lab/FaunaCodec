#!/usr/bin/env python3
"""Run the upscaling stage on a video with any of the supported backends."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ._common import add_common_arguments, load_config, progress_reporter, setup_logging


def _parse_args(argv=None) -> argparse.Namespace:
    from ..upscaling import UPSCALER_BACKENDS

    parser = argparse.ArgumentParser(
        prog="faunacodec-upscale", description=__doc__.splitlines()[0]
    )
    parser.add_argument("video", help="Video to upscale")
    parser.add_argument(
        "-b", "--backend", default=None, choices=sorted(UPSCALER_BACKENDS),
        help="Upscaling backend (default: the one named in the config)",
    )
    parser.add_argument(
        "-c", "--config", default="configs/upscaling.yaml", help="Upscaling config YAML"
    )
    parser.add_argument("--scale", type=float, default=None, help="Override the scale factor")
    add_common_arguments(parser)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    setup_logging(args.verbose)
    say = progress_reporter("upscale")

    from .. import stages
    from ..video_io import probe_video, read_video, write_video

    video = Path(args.video)
    if not video.exists():
        print(f"error: video not found: {video}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    backend = args.backend or str(config.get("backend", "bicubic"))
    scale = args.scale if args.scale is not None else float(config.get("scale", 2))

    info = probe_video(video)
    backend_cfg = dict(config.get(backend, {}) or {})
    backend_cfg.setdefault("out_w", int(round(info.width * scale)))
    backend_cfg.setdefault("out_h", int(round(info.height * scale)))
    backend_cfg.setdefault("device", config.get("device", "cuda"))

    out_dir = Path((config.get("output", {}) or {}).get("out_dir", "outputs/upscaling"))
    output = Path(args.output) if args.output else out_dir / f"{video.stem}_{backend}.mp4"

    started = time.perf_counter()
    frames, fps = read_video(video)
    upscaled = stages.upscale(frames, backend, backend_cfg, progress=say)
    write_video(upscaled, output, fps=fps)

    say(
        f"done in {time.perf_counter() - started:.1f}s: "
        f"{info.width}x{info.height} -> {backend_cfg['out_w']}x{backend_cfg['out_h']}"
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
