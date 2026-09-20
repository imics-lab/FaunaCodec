"""The FaunaCodec archive: the single interface between the edge device and the server.

An archive is a ZIP holding two encoded bitstreams plus everything the server needs to
reconstruct the timeline from them:

    archive_manifest.json   format version and entry-name map
    meta.json               source video geometry, codec settings, per-stream index maps
    roi_detections.json     per-frame subject bounding boxes
    frame_drop.json         which source frames each stream kept
    roi.bin                 ROI stream (subject at high quality, background masked out)
    bg.bin                  background stream (full frame, low quality, subsampled)
    compression_config.json the resolved config the encoder ran with, for provenance

Readers locate entries through the manifest, so the payload names may be remapped
without breaking older archives.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

ARCHIVE_VERSION = 3
MANIFEST_NAME = "archive_manifest.json"
RUNTIME_CONFIG_NAME = "compression_config.json"

#: Entries every archive must contain, in the order they are written.
REQUIRED_ENTRIES = (
    "meta.json",
    "roi_detections.json",
    "frame_drop.json",
    "roi.bin",
    "bg.bin",
)

__all__ = [
    "ARCHIVE_VERSION",
    "MANIFEST_NAME",
    "REQUIRED_ENTRIES",
    "RUNTIME_CONFIG_NAME",
    "frame_drop_for_transport",
    "read_payloads",
    "roi_detections_for_transport",
    "write_archive",
]


def frame_drop_for_transport(frame_drop: dict[str, Any]) -> dict[str, Any]:
    """Remove encoder diagnostics while retaining decoder inputs.

    Frame selection records a detailed trace for local analysis. The decoder only
    needs motion urgency and, for the optional ``frame_drop_roi_box`` mask source,
    the selected ROI box. Store those values in sparse maps instead of shipping the
    full diagnostic record for every frame.
    """
    transport = {key: value for key, value in frame_drop.items() if key != "per_frame"}
    per_frame = frame_drop.get("per_frame", {}) or {}
    if not isinstance(per_frame, dict):
        return transport

    urgency_by_frame: dict[str, float] = {}
    roi_box_by_frame: dict[str, dict[str, Any]] = {}
    for frame_idx, record in per_frame.items():
        if not isinstance(record, dict):
            continue
        try:
            urgency = float(record.get("motion_urgency", 0.0))
        except (TypeError, ValueError):
            urgency = 0.0
        if urgency != 0.0:
            urgency_by_frame[str(frame_idx)] = urgency

        roi_box = record.get("roi_box")
        if isinstance(roi_box, dict) and roi_box:
            roi_box_by_frame[str(frame_idx)] = roi_box

    if urgency_by_frame:
        transport["motion_urgency_by_frame"] = urgency_by_frame
    if roi_box_by_frame:
        transport["roi_box_by_frame"] = roi_box_by_frame
    return transport


def roi_detections_for_transport(roi_detections: dict[str, Any]) -> dict[str, Any]:
    """Remove repeated display-only class fields from archived bounding boxes."""
    transport = dict(roi_detections)
    frames = roi_detections.get("frames", {}) or {}
    if not isinstance(frames, dict):
        return transport

    transport_frames: dict[Any, Any] = {}
    for frame_idx, boxes in frames.items():
        if not isinstance(boxes, list):
            transport_frames[frame_idx] = boxes
            continue
        transport_frames[frame_idx] = [
            {key: value for key, value in box.items() if key not in {"cls", "label"}}
            if isinstance(box, dict)
            else box
            for box in boxes
        ]
    transport["frames"] = transport_frames
    return transport


def _json_bytes(value: Any, *, default: Any = None) -> str:
    """Serialize compact transport JSON; ZIP still provides a second compression layer."""
    if default is not None:
        return json.dumps(value, separators=(",", ":"), default=default)
    return json.dumps(value, separators=(",", ":"))


def write_archive(
    *,
    roi_bin: bytes,
    bg_bin: bytes,
    meta: dict[str, Any],
    roi_detections: dict[str, Any],
    frame_drop: dict[str, Any],
    runtime_config: dict[str, Any] | None = None,
) -> bytes:
    """Pack the encoder's output into archive bytes."""
    manifest = {
        "version": ARCHIVE_VERSION,
        "entries": {name: name for name in REQUIRED_ENTRIES},
        "extras": {"runtime_config": RUNTIME_CONFIG_NAME},
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, _json_bytes(manifest))
        zf.writestr("meta.json", _json_bytes(meta))
        zf.writestr(
            "roi_detections.json", _json_bytes(roi_detections_for_transport(roi_detections))
        )
        zf.writestr("frame_drop.json", _json_bytes(frame_drop_for_transport(frame_drop)))
        zf.writestr("roi.bin", bytes(roi_bin))
        zf.writestr("bg.bin", bytes(bg_bin))
        zf.writestr(
            RUNTIME_CONFIG_NAME, _json_bytes(runtime_config or {}, default=str)
        )
    return buf.getvalue()


def read_payloads(archive: str | Path | bytes) -> dict[str, bytes]:
    """Read the required entries out of an archive, resolving names via the manifest.

    Accepts a path or the archive bytes, so a caller that never wrote the archive to disk reads it the same way. Raises FileNotFoundError naming the entries that are absent.
    """
    source: Any = io.BytesIO(archive) if isinstance(archive, (bytes, bytearray)) else Path(archive)
    with zipfile.ZipFile(source, "r") as zf:
        names = set(zf.namelist())
        entry_map = {name: name for name in REQUIRED_ENTRIES if name in names}
        missing = [name for name in REQUIRED_ENTRIES if name not in entry_map]

        if missing:
            if MANIFEST_NAME not in names:
                raise FileNotFoundError(f"Missing archive entries: {missing}")
            try:
                manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
            except ValueError as exc:
                raise RuntimeError(f"Invalid archive manifest: {MANIFEST_NAME}") from exc
            entries = manifest.get("entries") if isinstance(manifest, dict) else None
            if not isinstance(entries, dict):
                raise RuntimeError(f"Invalid archive manifest: {MANIFEST_NAME}")

            entry_map = {}
            unresolved = []
            for canonical in REQUIRED_ENTRIES:
                actual = entries.get(canonical, canonical)
                if isinstance(actual, str) and actual.strip() and actual in names:
                    entry_map[canonical] = actual
                else:
                    unresolved.append(canonical)
            if unresolved:
                raise FileNotFoundError(f"Missing archive entries: {unresolved}")

        return {canonical: zf.read(entry_map[canonical]) for canonical in REQUIRED_ENTRIES}
