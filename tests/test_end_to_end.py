"""Full compress/decompress round trip on the bundled clip.

These tests need the model weights (`python scripts/download_models.py --group compress
decompress`) and a CUDA device, so they skip automatically when either is missing. They
are the only tests that exercise the real codec rather than a mock.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from faunacodec.archive import read_payloads, write_archive
from faunacodec.cli._common import ROOT, load_config
from faunacodec.masking import boxes_for_frame
from faunacodec.video_io import read_video, resize_video

SAMPLE = ROOT / "data" / "bird1.mp4"
REQUIRED_WEIGHTS = [
    ROOT / "models" / "MDV6-yolov9-c.pt",
    ROOT / "models" / "cvpr2025_image.pth.tar",
    ROOT / "models" / "cvpr2025_video.pth.tar",
    ROOT / "models" / "amt-s.pth",
]


def _requirements_met() -> tuple[bool, str]:
    if not SAMPLE.exists():
        return False, f"{SAMPLE.name} is missing"
    missing = [p.name for p in REQUIRED_WEIGHTS if not p.exists()]
    if missing:
        return False, f"weights not downloaded: {', '.join(missing)}"
    try:
        import torch
    except ImportError:
        return False, "torch is not installed"
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    return True, ""


_ok, _why = _requirements_met()
pytestmark = pytest.mark.skipif(not _ok, reason=_why)

RESOLUTION = (640, 360)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return 10 * np.log10(255.0**2 / max(mse, 1e-9))


@pytest.fixture(scope="module")
def round_trip(tmp_path_factory):
    """Compress and decompress the sample clip once, and share it across the tests."""
    from faunacodec import stages

    work = tmp_path_factory.mktemp("round_trip")
    source = resize_video(SAMPLE, work / "source.mp4", *RESOLUTION)

    archive = stages.compress(
        source,
        load_config("configs/compression.yaml"),
        detection_video_path=SAMPLE,
        root_dir=ROOT,
    )
    (work / "clip.zip").write_bytes(archive)
    decoded = stages.decompress(archive, load_config("configs/decompression.yaml"))

    reference, _ = read_video(source)
    return {
        "work": work,
        "archive": archive,
        "decoded": decoded,
        "reference": reference,
        "payloads": read_payloads(work / "clip.zip"),
    }


def test_frame_count_survives_the_round_trip(round_trip):
    """Dropped frames are interpolated back, so the timeline length is preserved."""
    assert len(round_trip["decoded"].frames) == len(round_trip["reference"])


def test_both_streams_carry_data(round_trip):
    payloads = round_trip["payloads"]
    assert len(payloads["roi.bin"]) > 0, "the ROI stream is empty; detection likely failed"
    assert len(payloads["bg.bin"]) > 0


def test_the_background_stream_is_subsampled_more_than_the_subject(round_trip):
    """The dual timeline is the whole point: the background gets far fewer frames."""
    selection = json.loads(round_trip["payloads"]["frame_drop.json"])
    assert len(selection["bg_kept_frames"]) < len(selection["roi_kept_frames"])


def test_the_archive_is_much_smaller_than_the_source(round_trip):
    source_bytes = (round_trip["work"] / "source.mp4").stat().st_size
    assert len(round_trip["archive"]) < source_bytes / 10


def test_the_roi_stream_improves_the_subject(round_trip):
    """Re-decode with the boxes removed; the difference is what the ROI stream bought.

    Comparing the subject region against the background region instead would measure
    scene content, not coding quality, and the two are not comparable.
    """
    from faunacodec import stages

    payloads = round_trip["payloads"]
    detections = json.loads(payloads["roi_detections.json"])

    without_roi = write_archive(
        roi_bin=payloads["roi.bin"],
        bg_bin=payloads["bg.bin"],
        meta=json.loads(payloads["meta.json"]),
        roi_detections={**detections, "frames": {}},
        frame_drop=json.loads(payloads["frame_drop.json"]),
    )
    background_only = stages.decompress(
        without_roi, load_config("configs/decompression.yaml")
    ).frames

    reference = round_trip["reference"]
    composited = round_trip["decoded"].frames

    with_roi, plain = [], []
    for i in range(0, min(len(reference), len(composited), len(background_only)), 10):
        boxes = boxes_for_frame(detections["frames"], i)
        if not boxes:
            continue
        x1, y1, x2, y2 = (int(boxes[0][k]) for k in ("x1", "y1", "x2", "y2"))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        with_roi.append(_psnr(reference[i][y1:y2, x1:x2], composited[i][y1:y2, x1:x2]))
        plain.append(_psnr(reference[i][y1:y2, x1:x2], background_only[i][y1:y2, x1:x2]))

    assert with_roi, "no usable subject boxes were found"
    gain = float(np.mean(with_roi) - np.mean(plain))
    assert gain > 1.0, f"the ROI stream only gained {gain:.2f} dB inside the subject box"
