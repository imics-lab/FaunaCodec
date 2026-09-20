"""ROI mask construction, and the compress stage end to end with the encoder mocked."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from faunacodec import stages
from faunacodec.archive import REQUIRED_ENTRIES
from faunacodec.compression.dual_stream_ffmpeg import _crf_for, validate_ffmpeg_backend
from faunacodec.masking.masks import build_boxes_mask


class TestBuildBoxesMask:
    def test_empty_boxes_gives_zero_mask(self):
        mask = build_boxes_mask(width=320, height=180, boxes=[])
        assert mask.shape == (180, 320)
        assert mask.max() == 0

    def test_full_frame_box_gives_full_mask(self):
        boxes = [{"x1": 0, "y1": 0, "x2": 320, "y2": 180, "conf": 0.9}]
        mask = build_boxes_mask(width=320, height=180, boxes=boxes)
        assert mask.min() > 0

    def test_partial_box_covers_only_its_region(self):
        boxes = [{"x1": 10, "y1": 10, "x2": 50, "y2": 50, "conf": 0.9}]
        mask = build_boxes_mask(width=100, height=100, boxes=boxes)
        assert mask[30, 30] > 0
        assert mask[0, 0] == 0

    def test_multiple_boxes_union(self):
        boxes = [
            {"x1": 0, "y1": 0, "x2": 40, "y2": 40, "conf": 0.9},
            {"x1": 60, "y1": 60, "x2": 99, "y2": 99, "conf": 0.9},
        ]
        mask = build_boxes_mask(width=100, height=100, boxes=boxes)
        assert mask[20, 20] > 0
        assert mask[80, 80] > 0
        assert mask[50, 50] == 0

    def test_boxes_below_min_conf_are_dropped(self):
        boxes = [{"x1": 10, "y1": 10, "x2": 90, "y2": 90, "conf": 0.1}]
        mask = build_boxes_mask(width=100, height=100, boxes=boxes, roi_min_conf=0.5)
        assert mask.max() == 0


def _write_video(path: Path, *, size: tuple[int, int], frames: int = 5) -> Path:
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[height // 3 : 2 * height // 3, width // 8 + i : width // 3 + i] = 255
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """A 5-frame 320x180 clip with a bright square that moves, so motion is detectable."""
    return _write_video(tmp_path / "sample.mp4", size=(320, 180))


class TestFfmpegQuality:
    @pytest.mark.parametrize(
        ("codec", "expected"),
        [("h264", (19, 30)), ("hevc", (22, 32)), ("av1", (32, 62))],
    )
    def test_codec_specific_crf(self, codec: str, expected: tuple[int, int]):
        quality = {
            "ffmpeg": {
                "h264": {"roi_crf": 19, "bg_crf": 30},
                "hevc": {"roi_crf": 22, "bg_crf": 32},
                "av1": {"roi_crf": 32, "bg_crf": 62},
            }
        }

        assert (
            _crf_for(codec, "roi", quality, {}),
            _crf_for(codec, "bg", quality, {}),
        ) == expected
        assert (
            _crf_for(codec, "roi", {}, {}),
            _crf_for(codec, "bg", {}, {}),
        ) == expected

    def test_unavailable_selected_encoder_is_rejected(self):
        config = {
            "codec": "av1",
            "ffmpeg": {"av1_encoder": "not-a-real-encoder"},
        }
        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch(
                "subprocess.run",
                return_value=CompletedProcess([], returncode=1, stderr="unknown encoder"),
            ),
            pytest.raises(RuntimeError, match="not-a-real-encoder.*unavailable"),
        ):
            validate_ffmpeg_backend(config)


class TestCompressStage:
    """The compress stage should produce a well-formed archive regardless of backend."""

    def test_produces_archive_with_all_required_entries(self, sample_video: Path):
        config = {
            "roi_detection": {"enable": False},
            "frame_selection": {"dual_timeline": {"enable": False}},
            # av1 so the check does not need the DCVC checkpoints to be downloaded.
            "compression": {"codec": "av1", "quality": {}},
        }
        encoded = {
            "roi_bin_bytes": b"\x00" * 64,
            "bg_bin_bytes": b"\x01" * 64,
            "meta": {"video": {"width": 320, "height": 180, "fps": 30.0, "frames_total": 5}},
        }
        with patch("faunacodec.compression.encode_dual_stream", return_value=encoded):
            archive = stages.compress(sample_video, config)

        assert isinstance(archive, bytes)
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            names = set(zf.namelist())
        for entry in REQUIRED_ENTRIES:
            assert entry in names, f"{entry} missing from the archive"
        assert "archive_manifest.json" in names

    def test_missing_video_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            stages.compress(tmp_path / "nope.mp4", {})

    def test_high_resolution_detection_boxes_are_scaled_to_encoded_video(
        self, sample_video: Path, tmp_path: Path
    ):
        detection_video = _write_video(
            tmp_path / "detection.mp4", size=(640, 360)
        )
        weights = tmp_path / "detector.pt"
        weights.write_bytes(b"")
        config = {
            "roi_detection": {
                "enable": True,
                "paths": {"animal_model_path": str(weights)},
            },
            "frame_selection": {"dual_timeline": {"enable": False}},
            "compression": {"codec": "av1", "quality": {}},
        }
        detections = {
            "video_path": str(detection_video),
            "frame_count": 5,
            "fps": 30.0,
            "width": 640,
            "height": 360,
            "frames": {
                0: [
                    {"x1": 64, "y1": 36, "x2": 320, "y2": 180, "conf": 0.9},
                    {"x1": 400, "y1": 200, "x2": 639, "y2": 359, "conf": 0.8},
                ]
            },
        }
        selection = {
            "kept_frames": [0],
            "roi_kept_frames": [0],
            "bg_kept_frames": [0],
            "per_frame": {},
        }
        encoded = {
            "roi_bin_bytes": b"roi",
            "bg_bin_bytes": b"bg",
            "meta": {"video": {"width": 320, "height": 180, "fps": 30.0, "frames_total": 5}},
        }
        with (
            patch("faunacodec.detection.run_roi_detection", return_value=detections),
            patch("faunacodec.selection.remove_redundant_frames", return_value=selection),
            patch("faunacodec.selection.apply_dual_timeline_policy", return_value=selection),
            patch("faunacodec.compression.encode_dual_stream", return_value=encoded) as encode,
        ):
            archive = stages.compress(
                sample_video,
                config,
                detection_video_path=detection_video,
            )

        scaled = encode.call_args.kwargs["roi_bbox_map"][0]
        assert scaled[0] == {
            "x1": 32,
            "y1": 18,
            "x2": 160,
            "y2": 90,
            "conf": 0.9,
        }
        assert scaled[1]["x2"] == 319
        assert scaled[1]["y2"] == 179
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            stored = json.loads(zf.read("roi_detections.json"))
        assert stored["width"] == 320
        assert stored["height"] == 180
        assert stored["detector_input"]["width"] == 640
        assert stored["detector_input"]["scale_x"] == 0.5

    def test_detection_video_must_share_the_encoded_timeline(
        self, sample_video: Path, tmp_path: Path
    ):
        detection_video = _write_video(
            tmp_path / "short_detection.mp4", size=(640, 360), frames=4
        )
        weights = tmp_path / "detector.pt"
        weights.write_bytes(b"")
        config = {
            "roi_detection": {
                "enable": True,
                "paths": {"animal_model_path": str(weights)},
            },
            "compression": {"codec": "av1"},
        }
        with patch("faunacodec.detection.run_roi_detection") as detect:
            with pytest.raises(ValueError, match="same frame count"):
                stages.compress(
                    sample_video,
                    config,
                    detection_video_path=detection_video,
                )
        detect.assert_not_called()
