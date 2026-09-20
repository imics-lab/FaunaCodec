"""The configs shipped in configs/ must load, validate, and name files that exist."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from faunacodec.cli._common import ROOT, load_config, load_stage_config
from faunacodec.config import validate_pipeline_config

CONFIG_DIR = ROOT / "configs"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text()) or {}


def test_every_shipped_config_is_valid_yaml():
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        assert isinstance(yaml.safe_load(path.read_text()), dict), path.name


class TestCompressionConfig:
    @pytest.fixture
    def cfg(self) -> dict:
        return _load("compression.yaml")

    def test_passes_the_schema_validator(self, cfg, tmp_path):
        """The validator also checks that the weights exist, so skip without them."""
        weights = [
            ROOT / cfg["roi_detection"]["paths"]["animal_model_path"],
            ROOT / cfg["compression"]["dcvc"]["model_i"],
            ROOT / cfg["compression"]["dcvc"]["model_p"],
        ]
        missing = [w.name for w in weights if not w.exists()]
        if missing:
            pytest.skip(f"weights not downloaded: {', '.join(missing)}")

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"")
        validate_pipeline_config(cfg, str(video), root_dir=ROOT)

    def test_dcvc_quantization_parameters_are_in_range(self, cfg):
        quality = cfg["compression"]["quality"]
        for key in ("roi_qp_i", "roi_qp_p", "bg_qp_i", "bg_qp_p"):
            assert 0 <= quality[key] <= 63, key

    def test_the_subject_is_encoded_at_higher_quality_than_the_background(self, cfg):
        quality = cfg["compression"]["quality"]
        # DCVC quantization parameters run the opposite way to CRF: higher is better.
        assert quality["roi_qp_i"] > quality["bg_qp_i"]
        for codec_quality in quality["ffmpeg"].values():
            assert codec_quality["roi_crf"] < codec_quality["bg_crf"]

    @pytest.mark.parametrize("codec", ["h264", "hevc", "av1"])
    def test_ffmpeg_codecs_do_not_require_dcvc_or_onnx_assets(self, cfg, tmp_path, codec):
        candidate = copy.deepcopy(cfg)
        candidate["roi_detection"]["enable"] = False
        candidate["roi_detection"]["paths"]["animal_model_path_onnx"] = "missing.onnx"
        candidate["compression"]["codec"] = codec
        candidate["compression"]["dcvc"]["model_i"] = "missing-image-model.pth"
        candidate["compression"]["dcvc"]["model_p"] = "missing-video-model.pth"
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"")

        validate_pipeline_config(candidate, str(video), root_dir=tmp_path)

    def test_named_model_files_are_in_the_manifest(self, cfg):
        import json

        manifest = json.loads((ROOT / "models" / "models.manifest.json").read_text())
        known = {entry["path"] for entry in manifest["models"]}
        referenced = [
            cfg["roi_detection"]["paths"]["animal_model_path"],
            cfg["roi_detection"]["paths"]["animal_model_path_onnx"],
            cfg["compression"]["dcvc"]["model_i"],
            cfg["compression"]["dcvc"]["model_p"],
        ]
        for path in referenced:
            assert path in known, f"{path} is not downloadable from the manifest"


class TestRestorationConfig:
    @pytest.fixture
    def cfg(self) -> dict:
        return _load("restoration.yaml")

    def test_temporal_window_is_odd(self, cfg):
        # The window is centered on the frame being restored, so it must be odd.
        assert cfg["model"]["temporal_window"] % 2 == 1

    def test_t_start_is_a_valid_timestep(self, cfg):
        assert 0 < cfg["inference"]["t_start"] < cfg["model"]["timesteps"]

    def test_ablation_configs_differ_only_in_temporal_window(self):
        base = _load("restoration.yaml")
        for name, window in [("restoration_T1.yaml", 1), ("restoration_T5.yaml", 5)]:
            variant = _load(name)
            assert variant["model"]["temporal_window"] == window
            assert variant["model"]["encoder_channels"] == base["model"]["encoder_channels"]


class TestUpscalingConfig:
    @pytest.fixture
    def cfg(self) -> dict:
        return _load("upscaling.yaml")

    def test_default_backend_is_available(self, cfg):
        from faunacodec.upscaling import UPSCALER_BACKENDS

        assert cfg["backend"] in UPSCALER_BACKENDS

    def test_every_backend_has_a_config_section(self, cfg):
        from faunacodec.upscaling import UPSCALER_BACKENDS

        for name in UPSCALER_BACKENDS:
            assert name in cfg, f"configs/upscaling.yaml has no section for {name}"


class TestPipelineConfig:
    @pytest.fixture
    def cfg(self) -> dict:
        return _load("pipeline.yaml")

    def test_referenced_stage_configs_exist(self, cfg):
        for section in ("compression", "decompression", "restoration", "upscaling"):
            referenced = Path(cfg[section]["config"])
            assert (ROOT / referenced).exists(), f"{section} points at a missing file"

    def test_every_stage_resolves_to_real_settings(self, cfg):
        for section in ("compression", "decompression", "restoration", "upscaling"):
            assert load_stage_config(cfg, section), section

    def test_input_resolution_is_parseable(self, cfg):
        resolution = cfg["input"]["resolution"]
        if resolution:
            width, height = (int(v) for v in resolution.lower().split("x"))
            assert width > 0 and height > 0


def test_configs_load_through_the_cli_helper():
    """The CLI defaults must resolve from any working directory."""
    assert load_config("configs/compression.yaml")["compression"]["codec"]


class TestDetectorBackendSelection:
    """prefer_onnx must degrade gracefully; prefer_onnx_strict must not."""

    @pytest.fixture
    def paths(self) -> dict:
        return {
            "animal_model_path": "models/MDV6-yolov9-c.pt",
            "animal_model_path_onnx": "models/MDV6-yolov9-c.onnx",
        }

    def test_pytorch_weights_by_default(self, paths):
        from faunacodec.detection.detector import _select_detector_weights

        assert _select_detector_weights(paths, {}).suffix == ".pt"

    def test_missing_onnx_path_falls_back(self, paths):
        from faunacodec.detection.detector import _select_detector_weights

        without_onnx = {"animal_model_path": paths["animal_model_path"]}
        assert _select_detector_weights(without_onnx, {"prefer_onnx": True}).suffix == ".pt"

    def test_strict_mode_reports_a_missing_onnx_path(self, paths):
        from faunacodec.detection.detector import _select_detector_weights

        without_onnx = {"animal_model_path": paths["animal_model_path"]}
        with pytest.raises(ValueError, match="animal_model_path_onnx"):
            _select_detector_weights(
                without_onnx, {"prefer_onnx": True, "prefer_onnx_strict": True}
            )

    def test_relative_paths_resolve_against_the_repository(self, paths):
        from faunacodec.detection.detector import _select_detector_weights

        resolved = _select_detector_weights(paths, {})
        assert resolved.is_absolute() or resolved.exists()


class TestDetectorBackendValidation:
    def _config(self, tmp_path: Path) -> tuple[dict, Path]:
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"")
        weights = tmp_path / "detector.pt"
        weights.write_bytes(b"")
        return {
            "roi_detection": {
                "enable": True,
                "paths": {
                    "animal_model_path": str(weights),
                    "animal_model_path_onnx": str(tmp_path / "missing.onnx"),
                },
                "runtime": {"prefer_onnx": True, "prefer_onnx_strict": False},
            },
            "compression": {"codec": "av1"},
        }, video

    def test_non_strict_onnx_can_fall_back_to_pytorch(self, tmp_path: Path):
        cfg, video = self._config(tmp_path)
        validate_pipeline_config(cfg, str(video), root_dir=tmp_path)

    def test_strict_onnx_requires_the_file(self, tmp_path: Path):
        cfg, video = self._config(tmp_path)
        cfg["roi_detection"]["runtime"]["prefer_onnx_strict"] = True
        with pytest.raises(ValueError, match="animal_model_path_onnx does not exist"):
            validate_pipeline_config(cfg, str(video), root_dir=tmp_path)
