"""The pipeline CLI: config resolution and stage wiring, with the heavy stages mocked."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from faunacodec.cli import pipeline
from faunacodec.cli._common import ROOT, load_stage_config
from faunacodec.stages import DecodedVideo


def _write_video(path: Path, n: int = 5, size: tuple[int, int] = (64, 64)) -> Path:
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    for _ in range(n):
        writer.write(np.random.randint(0, 256, (height, width, 3), dtype=np.uint8))
    writer.release()
    return path


def _frames(n: int = 5, size: tuple[int, int] = (64, 64)) -> list[np.ndarray]:
    width, height = size
    return [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(n)]


class TestStageConfigResolution:
    """A stage section that names a file must be loaded from that file, not passed through.

    Passing the section through verbatim silently hands the encoder `{"config": "..."}`
    instead of its settings, which is how a pipeline run can appear to succeed while
    using nothing but defaults.
    """

    def test_referenced_file_is_loaded(self, tmp_path: Path):
        referenced = tmp_path / "compression.yaml"
        referenced.write_text("codec: av1\nquality:\n  ffmpeg:\n    av1:\n      roi_crf: 22\n")

        resolved = load_stage_config({"compression": {"config": str(referenced)}}, "compression")

        assert resolved["codec"] == "av1"
        assert resolved["quality"]["ffmpeg"]["av1"]["roi_crf"] == 22
        assert "config" not in resolved

    def test_inline_keys_override_the_referenced_file(self, tmp_path: Path):
        referenced = tmp_path / "compression.yaml"
        referenced.write_text("codec: av1\n")

        resolved = load_stage_config(
            {"compression": {"config": str(referenced), "codec": "hevc"}}, "compression"
        )

        assert resolved["codec"] == "hevc"

    def test_section_without_a_file_is_used_as_is(self):
        resolved = load_stage_config({"compression": {"codec": "dcvc"}}, "compression")
        assert resolved == {"codec": "dcvc"}

    def test_shipped_pipeline_config_resolves_every_stage(self):
        """The config the README tells people to run must wire up all four stages."""
        from faunacodec.cli._common import load_config

        pipeline_cfg = load_config(ROOT / "configs" / "pipeline.yaml")
        for section, expected_key in [
            ("compression", "compression"),
            ("decompression", "decompression"),
            ("restoration", "model"),
            ("upscaling", "backend"),
        ]:
            resolved = load_stage_config(pipeline_cfg, section)
            assert expected_key in resolved, f"{section} did not resolve to real settings"


class TestPipelineCli:
    @pytest.fixture
    def video(self, tmp_path: Path) -> Path:
        return _write_video(tmp_path / "clip.mp4")

    @pytest.fixture
    def config(self, tmp_path: Path) -> Path:
        path = tmp_path / "pipeline.yaml"
        path.write_text(
            "input:\n  resolution: ''\n"
            "compression: {}\n"
            "decompression: {}\n"
            f"output:\n  out_dir: {tmp_path / 'out'}\n"
        )
        return path

    def test_compress_then_decompress(self, video: Path, config: Path):
        decoded = DecodedVideo(frames=_frames(), fps=30.0, width=64, height=64)
        with patch("faunacodec.stages.compress", return_value=b"\x00" * 50) as mock_compress, \
             patch("faunacodec.stages.decompress", return_value=decoded) as mock_decompress:
            code = pipeline.main(
                [str(video), "--config", str(config), "--stages", "compress", "decompress"]
            )

        assert code == 0
        mock_compress.assert_called_once()
        mock_decompress.assert_called_once()

    def test_resized_encode_detects_on_the_original_video(self, video: Path, config: Path):
        config.write_text(
            "input:\n  resolution: 32x32\n"
            "compression: {}\n"
            f"output:\n  out_dir: {config.parent / 'out'}\n"
        )
        with patch("faunacodec.stages.compress", return_value=b"archive") as mock_compress:
            code = pipeline.main(
                [str(video), "--config", str(config), "--stages", "compress"]
            )

        assert code == 0
        assert mock_compress.call_args.kwargs["detection_video_path"] == video

    def test_upscale_only_reads_the_input_video(self, video: Path, config: Path):
        with patch("faunacodec.stages.upscale", side_effect=lambda f, *a, **k: f) as mock_upscale:
            code = pipeline.main(
                [str(video), "--config", str(config), "--stages", "upscale", "--upscaler", "bicubic"]
            )

        assert code == 0
        frames_in = mock_upscale.call_args.args[0]
        assert len(frames_in) == 5

    def test_decompress_without_compress_is_rejected(self, video: Path, config: Path):
        code = pipeline.main([str(video), "--config", str(config), "--stages", "decompress"])
        assert code == 1

    def test_unknown_stage_is_rejected(self, video: Path, config: Path):
        code = pipeline.main([str(video), "--config", str(config), "--stages", "enhance"])
        assert code == 1

    def test_missing_video_is_rejected(self, tmp_path: Path, config: Path):
        code = pipeline.main([str(tmp_path / "nope.mp4"), "--config", str(config)])
        assert code == 1
