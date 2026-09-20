"""Resolving the paths that appear in config files.

Configs name weights and vendored checkouts relative to the repository, as in
``models/amt-s.pth``. Resolving those against the working directory would make the
pipeline work only when launched from the repository root, which breaks notebooks,
subdirectories, and installed console scripts. Everything that reads a path out of a
config goes through here instead.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

__all__ = ["REPO_ROOT", "resolve_repo_path"]


def resolve_repo_path(raw: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a config path, treating a relative one as relative to the repository.

    An absolute path, or a relative one that already exists in the working directory,
    is kept as given, so a caller can still point at files outside the checkout.
    """
    path = Path(raw).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (Path(root or REPO_ROOT) / path).resolve()
