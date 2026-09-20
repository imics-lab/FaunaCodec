#!/usr/bin/env python3
"""Run real H.264, HEVC, and AV1 round trips on the bundled wildlife clip.

This is intentionally separate from the regular test suite: it runs MegaDetector and
three full encodes/decodes, so it is meant for a CUDA test host such as the DGX Spark.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from faunacodec import stages  # noqa: E402
from faunacodec.archive import read_payloads  # noqa: E402
from faunacodec.cli._common import load_config  # noqa: E402
from faunacodec.masking import boxes_for_frame, build_boxes_mask  # noqa: E402
from faunacodec.video_io import probe_video, read_video, resize_video  # noqa: E402


def _psnr(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = np.mean((reference.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0**2 / max(mse, 1e-9)))


def run_codec(
    codec: str,
    source: Path,
    encoded_video: Path,
    reference: list[np.ndarray],
) -> dict[str, object]:
    config = copy.deepcopy(load_config("configs/compression.yaml"))
    config["compression"]["codec"] = codec
    expected = probe_video(encoded_video)
    detection_source = probe_video(source)

    started = time.perf_counter()
    archive = stages.compress(
        encoded_video,
        config,
        detection_video_path=source,
        root_dir=ROOT,
        progress=lambda message: print(f"[{codec}] {message}", flush=True),
    )
    compressed_seconds = time.perf_counter() - started

    started = time.perf_counter()
    decoded = stages.decompress(
        archive,
        load_config("configs/decompression.yaml"),
        progress=lambda message: print(f"[{codec}] {message}", flush=True),
    )
    decompressed_seconds = time.perf_counter() - started
    if len(decoded.frames) != expected.frame_count:
        raise RuntimeError(
            f"{codec}: decoded {len(decoded.frames)} frames, expected {expected.frame_count}"
        )

    payloads = read_payloads(archive)
    meta = json.loads(payloads["meta.json"])
    detections = json.loads(payloads["roi_detections.json"])
    selection = json.loads(payloads["frame_drop.json"])
    boxes_per_frame = [len(boxes) for boxes in detections["frames"].values()]
    detector_input = detections.get("detector_input", {})
    if (
        detector_input.get("width") != detection_source.width
        or detections.get("width") != expected.width
    ):
        raise RuntimeError(f"{codec}: detection coordinates were not scaled correctly")
    if max(boxes_per_frame, default=0) < 2:
        raise RuntimeError(f"{codec}: no frame retained both detected subjects")

    overall_scores: list[float] = []
    roi_scores: list[float] = []
    bg_scores: list[float] = []
    for frame_idx, (original, reconstructed) in enumerate(
        zip(reference, decoded.frames, strict=True)
    ):
        overall_scores.append(_psnr(original, reconstructed))
        mask = build_boxes_mask(
            width=expected.width,
            height=expected.height,
            boxes=boxes_for_frame(detections["frames"], frame_idx),
            roi_min_conf=0.0,
        ).astype(bool)
        if mask.any():
            roi_scores.append(_psnr(original[mask], reconstructed[mask]))
        if (~mask).any():
            bg_scores.append(_psnr(original[~mask], reconstructed[~mask]))

    return {
        "codec": codec,
        "archive_bytes": len(archive),
        "roi_stream_bytes": len(payloads["roi.bin"]),
        "bg_stream_bytes": len(payloads["bg.bin"]),
        "frames": len(decoded.frames),
        "roi_kept_frames": len(selection["roi_kept_frames"]),
        "bg_kept_frames": len(selection["bg_kept_frames"]),
        "max_subjects_in_frame": max(boxes_per_frame, default=0),
        "detection_resolution": [detector_input["width"], detector_input["height"]],
        "encoding_resolution": [detections["width"], detections["height"]],
        "roi_crf": meta["streams"]["roi"]["crf"],
        "bg_crf": meta["streams"]["bg"]["crf"],
        "mean_psnr_db": round(float(np.mean(overall_scores)), 3),
        "mean_roi_psnr_db": round(float(np.mean(roi_scores)), 3),
        "mean_bg_psnr_db": round(float(np.mean(bg_scores)), 3),
        "compression_seconds": round(compressed_seconds, 3),
        "decompression_seconds": round(decompressed_seconds, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "bird1.mp4")
    parser.add_argument("--output", type=Path, default=ROOT / ".dgx" / "codec-regression.json")
    parser.add_argument("--resolution", default="640x360", metavar="WxH")
    parser.add_argument(
        "--codecs", nargs="+", choices=("h264", "hevc", "av1"), default=("h264", "hevc", "av1")
    )
    args = parser.parse_args()

    width, height = (int(value) for value in args.resolution.lower().split("x"))
    with tempfile.TemporaryDirectory(prefix="faunacodec-regression-") as temp_dir:
        encoded_video = resize_video(
            args.source, Path(temp_dir) / f"input-{width}x{height}.mp4", width, height
        )
        reference, _fps = read_video(encoded_video)
        results = [
            run_codec(codec, args.source, encoded_video, reference) for codec in args.codecs
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
