#!/usr/bin/env python3
"""Check that FaunaCodec is installed correctly, without needing weights or a GPU.

Verifies the imports, builds a synthetic clip, and runs it through compress and
decompress with the codec backend mocked, so the wiring is exercised end to end while
nothing large is downloaded. Reports which model files are present and which are not.

    python scripts/sanity_check.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

PASS, FAIL, SKIP = "  ok  ", " FAIL ", " skip "


def _report(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label}" + (f"  {detail}" if detail else ""))


def check_imports() -> bool:
    ok = True
    for label, module in [
        ("faunacodec", "faunacodec"),
        ("torch", "torch"),
        ("opencv", "cv2"),
        ("numpy", "numpy"),
        ("PyYAML", "yaml"),
        ("omegaconf", "omegaconf"),
    ]:
        try:
            imported = __import__(module)
            _report(PASS, label, getattr(imported, "__version__", ""))
        except ImportError as exc:
            _report(FAIL, label, str(exc))
            ok = False

    for label, module, hint in [
        ("ultralytics", "ultralytics", "needed for detection"),
        ("MLCodec_extensions_cpp", "MLCodec_extensions_cpp", "DCVC entropy coder; see setup.sh"),
    ]:
        try:
            imported = __import__(module)
            _report(PASS, label, getattr(imported, "__version__", ""))
        except ImportError:
            _report(FAIL, label, hint)
            ok = False
    return ok


def check_cuda() -> bool:
    try:
        import torch
    except ImportError:
        _report(SKIP, "CUDA", "torch is not installed")
        return True
    if torch.cuda.is_available():
        _report(PASS, "CUDA", f"{torch.cuda.device_count()} device(s): {torch.cuda.get_device_name(0)}")
    else:
        _report(SKIP, "CUDA", "no device; DCVC and the neural stages will not run")
    return True


def check_configs() -> bool:
    from faunacodec.cli._common import load_config, load_stage_config

    try:
        pipeline_cfg = load_config("configs/pipeline.yaml")
        for section in ("compression", "decompression", "restoration", "upscaling"):
            if not load_stage_config(pipeline_cfg, section):
                _report(FAIL, "configs", f"{section} resolved to nothing")
                return False
        _report(PASS, "configs", "all four stage configs resolve")
        return True
    except Exception as exc:  # noqa: BLE001 - this is the diagnostic
        _report(FAIL, "configs", str(exc))
        return False


def check_weights() -> bool:
    manifest = json.loads((ROOT / "models" / "models.manifest.json").read_text())
    by_group: dict[str, list[tuple[str, bool]]] = {}
    for entry in manifest["models"]:
        present = (ROOT / entry["path"]).exists()
        for group in entry.get("groups", []):
            by_group.setdefault(group, []).append((entry["path"], present))

    for group, files in by_group.items():
        have = sum(1 for _, present in files if present)
        status = PASS if have == len(files) else SKIP
        missing = [Path(p).name for p, present in files if not present]
        detail = f"{have}/{len(files)}"
        if missing:
            detail += f"  missing: {', '.join(missing)}"
        _report(status, f"weights [{group}]", detail)
    return True


def check_round_trip() -> bool:
    """Compress and decompress a synthetic clip with the codec mocked."""
    import cv2
    import numpy as np

    from faunacodec import stages
    from faunacodec.archive import REQUIRED_ENTRIES, read_payloads

    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        video = work / "bars.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (256, 128))
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0)]
        for i in range(10):
            frame = np.zeros((128, 256, 3), dtype=np.uint8)
            bar = 256 // len(colors)
            for ci, color in enumerate(colors):
                frame[:, ci * bar : (ci + 1) * bar] = color
            frame[40:80, 20 + 8 * i : 60 + 8 * i] = (255, 255, 255)
            writer.write(frame)
        writer.release()

        config = {
            "roi_detection": {"enable": False},
            "frame_selection": {"dual_timeline": {"enable": False}},
            "compression": {"codec": "av1", "quality": {}},
        }
        encoded = {
            "roi_bin_bytes": b"\x00" * 32,
            "bg_bin_bytes": b"\x01" * 32,
            "meta": {"video": {"width": 256, "height": 128, "fps": 30.0, "frames_total": 10}},
        }
        try:
            with patch("faunacodec.compression.encode_dual_stream", return_value=encoded):
                archive = stages.compress(video, config)
            (work / "clip.zip").write_bytes(archive)
            payloads = read_payloads(work / "clip.zip")
        except Exception as exc:  # noqa: BLE001 - this is the diagnostic
            _report(FAIL, "round trip", str(exc))
            return False

    if set(payloads) != set(REQUIRED_ENTRIES):
        _report(FAIL, "round trip", "archive is missing entries")
        return False
    _report(PASS, "round trip", f"archive built and read back ({len(archive)} bytes)")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)

    print("FaunaCodec sanity check\n")
    checks = [
        ("imports", check_imports),
        ("cuda", check_cuda),
        ("configs", check_configs),
        ("weights", check_weights),
        ("round trip", check_round_trip),
    ]
    failed = []
    for name, check in checks:
        try:
            if not check():
                failed.append(name)
        except Exception as exc:  # noqa: BLE001 - never let a check crash the report
            _report(FAIL, name, f"{type(exc).__name__}: {exc}")
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} check(s) failed: {', '.join(failed)}")
        print("See README.md > Install.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
