"""Argument parsing and config loading shared by the command-line entry points."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from ..paths import REPO_ROOT as ROOT

#: Repository root, so default config paths work when running from a checkout.


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving relative paths against the repository root."""
    candidate = Path(path)
    if not candidate.is_absolute() and not candidate.exists():
        candidate = ROOT / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(candidate) as handle:
        return yaml.safe_load(handle) or {}


def load_stage_config(pipeline_cfg: dict[str, Any], section: str) -> dict[str, Any]:
    """Resolve one stage's config out of a pipeline config.

    A stage section may either hold the settings inline, or name a separate file via a
    ``config:`` key. When it does both, the inline keys win.
    """
    stage = pipeline_cfg.get(section, {}) or {}
    referenced = stage.get("config", "")
    base = load_config(referenced) if referenced else {}
    overrides = {k: v for k, v in stage.items() if k != "config"}
    return {**base, **overrides}


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--output", default=None, help="Output path (auto-named if omitted)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log every step")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    os.environ["YOLO_VERBOSE"] = "true" if verbose else "false"
    if not verbose:
        for name in ("ultralytics", "onnxruntime"):
            logging.getLogger(name).setLevel(logging.ERROR)


def progress_reporter(prefix: str):
    """Return a callable that prints stage progress with a consistent prefix."""

    def report(message: str) -> None:
        print(f"[{prefix}] {message}", flush=True)

    return report
