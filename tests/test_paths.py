"""Config paths must resolve the same way from any working directory.

Resolving them against the process cwd instead of the repository is a recurring bug
class here: it makes the pipeline work from the repository root and fail from a
notebook, a subdirectory, or an installed console script, and the failure surfaces
deep inside a stage as a missing checkpoint.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from faunacodec.paths import REPO_ROOT, resolve_repo_path


@pytest.fixture
def elsewhere(tmp_path: Path):
    """Run the body from a directory that is not the repository root."""
    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(previous)


def test_relative_paths_resolve_against_the_repository(elsewhere):
    assert resolve_repo_path("models/amt-s.pth") == REPO_ROOT / "models" / "amt-s.pth"


def test_absolute_paths_are_left_alone(elsewhere, tmp_path: Path):
    target = tmp_path / "weights.pt"
    target.write_bytes(b"")
    assert resolve_repo_path(target) == target.resolve()


def test_an_existing_local_path_wins(elsewhere, tmp_path: Path):
    """A caller who puts the file next to them should get their copy."""
    local = tmp_path / "models"
    local.mkdir()
    (local / "amt-s.pth").write_bytes(b"")
    assert resolve_repo_path("models/amt-s.pth") == (local / "amt-s.pth").resolve()


def test_every_path_in_the_shipped_configs_resolves(elsewhere):
    """Catches a config that names a file no longer in the repository."""
    configured = []
    for name in ("compression.yaml", "decompression.yaml"):
        cfg = yaml.safe_load((REPO_ROOT / "configs" / name).read_text())
        detection = cfg.get("roi_detection", {}).get("paths", {})
        configured += list(detection.values())
        for section in (cfg.get("compression", {}), cfg.get("decompression", {})):
            dcvc = section.get("dcvc", {})
            configured += [dcvc[k] for k in ("repo_dir", "model_i", "model_p") if k in dcvc]
            interpolate = section.get("interpolate", {})
            configured += [interpolate[k] for k in ("repo_dir", "weights_path") if k in interpolate]

    assert configured, "no paths were found in the configs"
    for raw in configured:
        resolved = resolve_repo_path(raw)
        assert resolved.is_absolute()
        assert str(resolved).startswith(str(REPO_ROOT)), f"{raw} escaped the repository"


def test_vendored_runtime_directories_exist():
    for relative in (
        "third_party/dcvc",
        "third_party/dcvc_int16",
        "third_party/amt",
        "third_party/osediff",
    ):
        assert (REPO_ROOT / relative).is_dir(), relative


def test_legacy_dcvc_archive_path_falls_back_to_the_vendored_runtime():
    from faunacodec.decompression.roi_bg_decompress import _resolve_runtime_path

    expected = REPO_ROOT / "third_party" / "dcvc"
    assert _resolve_runtime_path("DCVC", kind="repo") == expected
    assert _resolve_runtime_path("third_party/dcvc", kind="repo") == expected


def test_detector_weights_resolve_from_anywhere(elsewhere):
    from faunacodec.detection.detector import _select_detector_weights

    cfg = yaml.safe_load((REPO_ROOT / "configs" / "compression.yaml").read_text())
    roi = cfg["roi_detection"]
    resolved = _select_detector_weights(roi["paths"], roi["runtime"])
    assert resolved == REPO_ROOT / roi["paths"]["animal_model_path"]


def test_cli_finds_its_default_configs_from_anywhere(elsewhere):
    from faunacodec.cli._common import load_config

    assert load_config("configs/compression.yaml")["compression"]["codec"]
