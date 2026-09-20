#!/usr/bin/env python3
"""Download and verify the model weights and sample data listed in the manifest.

Every file is checked against the SHA-256 in models/models.manifest.json, both before
downloading (so re-runs are cheap) and after (so a truncated transfer is caught).

    python scripts/download_models.py                      # everything
    python scripts/download_models.py --group compress     # just the edge-side models
    python scripts/download_models.py --group decompress restore
    python scripts/download_models.py --list               # show what is available

Set FAUNACODEC_ASSET_BASE to fetch from a mirror instead of the GitHub release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "models.manifest.json"
BASE_URL_ENV = "FAUNACODEC_ASSET_BASE"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def is_valid(path: Path, expected: str | None) -> bool:
    if not path.exists():
        return False
    return expected is None or sha256(path) == expected


def resolve_url(entry: dict[str, Any], manifest: dict[str, Any]) -> str | None:
    """Apply the FAUNACODEC_ASSET_BASE override to the manifest URL, if one is set."""
    url = entry.get("url")
    if not url:
        return None
    override = os.environ.get(BASE_URL_ENV)
    base = manifest.get("release_base", "")
    if override and base and url.startswith(base):
        return override.rstrip("/") + "/" + url[len(base):]
    return url


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    scratch = dest.with_suffix(dest.suffix + ".part")

    # Overwrite one line when attached to a terminal; otherwise stay quiet so logs
    # do not fill up with progress spam.
    interactive = sys.stdout.isatty()

    def report(blocks: int, block_size: int, total: int) -> None:
        if interactive and total > 0:
            done = min(blocks * block_size, total)
            print(f"\r    {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, str(scratch), reporthook=report)
        if interactive:
            print()
        scratch.replace(dest)
    finally:
        scratch.unlink(missing_ok=True)


def select(manifest: dict[str, Any], groups: list[str] | None) -> list[dict[str, Any]]:
    entries = manifest.get("models", [])
    if not groups:
        return entries
    wanted = set(groups)
    return [e for e in entries if wanted & set(e.get("groups", []))]


def main(argv=None) -> int:
    known_groups = sorted(json.loads(MANIFEST.read_text()).get("_groups", {}))
    parser = argparse.ArgumentParser(
        prog="download_models.py",
        description=__doc__.splitlines()[0],
        epilog=f"groups: {', '.join(known_groups)}",
    )
    parser.add_argument(
        "-g", "--group", nargs="+", choices=known_groups, default=None,
        help="Only fetch files in these groups (default: all)",
    )
    parser.add_argument("--manifest", default=str(MANIFEST), help="Manifest path")
    parser.add_argument("--force", action="store_true", help="Re-download files already present")
    parser.add_argument("--list", action="store_true", help="List files and exit")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text())

    if args.list:
        for name, description in manifest.get("_groups", {}).items():
            print(f"{name:<12} {description}")
            for entry in select(manifest, [name]):
                marker = "required" if entry.get("required") else "optional"
                print(f"             - {entry['path']}  ({marker})")
        return 0

    entries = select(manifest, args.group)
    print(f"{len(entries)} file(s) selected.\n")

    failures = 0
    for entry in entries:
        dest = ROOT / entry["path"]
        expected = entry.get("sha256")
        print(f"{entry['name']}\n  -> {dest.relative_to(ROOT)}")

        if is_valid(dest, expected) and not args.force:
            print("  already present and verified")
            continue

        url = resolve_url(entry, manifest)
        if url is None:
            level = "error" if entry.get("required", True) else "skip"
            print(f"  {level}: no download URL in the manifest")
            failures += entry.get("required", True)
            continue

        try:
            download(url, dest)
        except Exception as exc:  # noqa: BLE001 - report and continue to the next file
            print(f"  error: download failed: {exc}")
            failures += 1
            continue

        if is_valid(dest, expected):
            print("  downloaded and verified")
        else:
            print(f"  error: checksum mismatch, expected {expected}")
            failures += 1

    print(f"\n{failures} error(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
