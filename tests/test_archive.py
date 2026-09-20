"""The archive format: what the encoder writes, the decoder must be able to read."""
from __future__ import annotations

import json
import zipfile

import pytest

from faunacodec.archive import (
    ARCHIVE_VERSION,
    MANIFEST_NAME,
    REQUIRED_ENTRIES,
    RUNTIME_CONFIG_NAME,
    frame_drop_for_transport,
    read_payloads,
    roi_detections_for_transport,
    write_archive,
)


def _archive(**overrides) -> bytes:
    payload = {
        "roi_bin": b"roi-bitstream",
        "bg_bin": b"bg-bitstream",
        "meta": {"video": {"width": 64, "height": 48, "fps": 30.0}},
        "roi_detections": {"frames": {"0": [{"x1": 1, "y1": 2, "x2": 3, "y2": 4, "conf": 0.9}]}},
        "frame_drop": {"roi_kept_frames": [0, 2], "bg_kept_frames": [0]},
        "runtime_config": {"compression": {"codec": "dcvc"}},
    }
    payload.update(overrides)
    return write_archive(**payload)


def test_reads_an_archive_that_was_never_written_to_disk():
    """faunacodec-pipeline holds the archive in memory, and must read it the same way."""
    payloads = read_payloads(_archive())
    assert set(payloads) == set(REQUIRED_ENTRIES)
    assert payloads["roi.bin"] == b"roi-bitstream"


def test_round_trip_preserves_every_entry(tmp_path):
    path = tmp_path / "clip.zip"
    path.write_bytes(_archive())

    payloads = read_payloads(path)
    assert set(payloads) == set(REQUIRED_ENTRIES)
    assert payloads["roi.bin"] == b"roi-bitstream"
    assert payloads["bg.bin"] == b"bg-bitstream"
    assert json.loads(payloads["meta.json"])["video"]["width"] == 64
    assert json.loads(payloads["frame_drop.json"])["roi_kept_frames"] == [0, 2]


def test_manifest_and_runtime_config_are_written(tmp_path):
    path = tmp_path / "clip.zip"
    path.write_bytes(_archive())

    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read(MANIFEST_NAME))
        assert RUNTIME_CONFIG_NAME in zf.namelist()
    assert set(manifest["entries"]) == set(REQUIRED_ENTRIES)
    assert manifest["version"] == ARCHIVE_VERSION


def test_transport_removes_per_frame_diagnostics_but_keeps_decoder_inputs():
    selection = {
        "roi_kept_frames": [0, 2],
        "per_frame": {
            "0": {
                "state": "STILL",
                "roi_motion_score": 0.125,
                "motion_urgency": 0.0,
                "roi_box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
            },
            "1": {
                "state": "MOTION",
                "motion_urgency": 0.75,
                "track_births": 1,
            },
        },
    }

    transport = frame_drop_for_transport(selection)

    assert "per_frame" not in transport
    assert transport["motion_urgency_by_frame"] == {"1": 0.75}
    assert transport["roi_box_by_frame"] == {
        "0": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}
    }
    assert "roi_motion_score" not in json.dumps(transport)
    assert "track_births" not in json.dumps(transport)


def test_archive_contains_transport_frame_drop_not_local_diagnostics():
    archive = _archive(
        frame_drop={
            "roi_kept_frames": [0],
            "per_frame": {
                "0": {"state": "MOTION", "motion_urgency": 1.0, "track_births": 2}
            },
        }
    )

    stored = json.loads(read_payloads(archive)["frame_drop.json"])
    assert "per_frame" not in stored
    assert stored["motion_urgency_by_frame"] == {"0": 1.0}
    assert "track_births" not in json.dumps(stored)


def test_transport_detections_remove_repeated_class_fields_only():
    detections = {
        "width": 64,
        "height": 48,
        "detector_input": {"width": 128, "height": 96, "scale_x": 0.5},
        "frames": {
            "0": [
                {
                    "x1": 1,
                    "y1": 2,
                    "x2": 30,
                    "y2": 40,
                    "conf": 0.9,
                    "cls": 0,
                    "label": "animal",
                    "track_id": 7,
                }
            ]
        },
    }

    transport = roi_detections_for_transport(detections)
    box = transport["frames"]["0"][0]

    assert transport["width"] == 64
    assert transport["detector_input"] == detections["detector_input"]
    assert box == {
        "x1": 1,
        "y1": 2,
        "x2": 30,
        "y2": 40,
        "conf": 0.9,
        "track_id": 7,
    }
    assert detections["frames"]["0"][0]["label"] == "animal"


def test_archive_stores_conservative_detection_payload():
    archive = _archive(
        roi_detections={
            "frames": {
                "0": [
                    {
                        "x1": 1,
                        "y1": 2,
                        "x2": 3,
                        "y2": 4,
                        "conf": 0.9,
                        "cls": 0,
                        "label": "animal",
                        "track_id": 5,
                    }
                ]
            }
        }
    )

    stored = json.loads(read_payloads(archive)["roi_detections.json"])
    box = stored["frames"]["0"][0]
    assert "cls" not in box
    assert "label" not in box
    assert box["track_id"] == 5
    assert box["conf"] == 0.9


def test_renamed_entries_are_resolved_through_the_manifest(tmp_path):
    """A reader must follow the manifest rather than assume canonical payload names."""
    path = tmp_path / "remapped.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            MANIFEST_NAME,
            json.dumps({"version": 2, "entries": {n: f"payload/{n}" for n in REQUIRED_ENTRIES}}),
        )
        for name in REQUIRED_ENTRIES:
            zf.writestr(f"payload/{name}", f"contents-of-{name}".encode())

    payloads = read_payloads(path)
    assert payloads["roi.bin"] == b"contents-of-roi.bin"


def test_missing_entry_is_reported_by_name(tmp_path):
    path = tmp_path / "broken.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("meta.json", "{}")

    with pytest.raises(FileNotFoundError, match="roi.bin"):
        read_payloads(path)


class TestPortabilityCheck:
    """A DCVC archive that crossed architectures decodes into noise; say so."""

    def test_mismatched_architecture_is_reported(self):
        import platform

        from faunacodec.stages import check_portability

        other = "aarch64" if platform.machine() != "aarch64" else "x86_64"
        complaint = check_portability({"encoder": {"machine": other, "codec": "dcvc"}})
        assert complaint is not None
        assert other in complaint
        assert "third_party/dcvc_int16" in complaint

    def test_same_architecture_is_silent(self):
        import platform

        from faunacodec.stages import check_portability

        meta = {"encoder": {"machine": platform.machine(), "codec": "dcvc"}}
        assert check_portability(meta) is None

    def test_block_codecs_cross_architectures_fine(self):
        import platform

        from faunacodec.stages import check_portability

        other = "aarch64" if platform.machine() != "aarch64" else "x86_64"
        assert check_portability({"encoder": {"machine": other, "codec": "av1"}}) is None

    def test_archives_without_provenance_are_not_flagged(self):
        from faunacodec.stages import check_portability

        assert check_portability({}) is None
