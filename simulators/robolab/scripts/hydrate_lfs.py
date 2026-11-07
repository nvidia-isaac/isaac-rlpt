#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hydrate required public Git LFS artifacts without installing Git LFS."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote
from urllib.request import urlopen


DEFAULT_GR00T_LFS_PATH = Path(
    "scripts/deployment/dgpu/wheels/torchcodec-0.8.0-cp312-cp312-linux_aarch64.whl"
)
LFS_VERSION_LINE = "version https://git-lfs.github.com/spec/v1"


@dataclass(frozen=True)
class LfsPointer:
    """The object identity recorded in a Git LFS pointer file."""

    sha256: str
    size_bytes: int


def parse_lfs_pointer(content: str) -> LfsPointer:
    """Parse one canonical Git LFS pointer and reject every other file format."""
    lines = content.splitlines()
    if len(lines) != 3 or lines[0] != LFS_VERSION_LINE:
        raise ValueError("Expected a canonical three-line Git LFS pointer.")
    if not lines[1].startswith("oid sha256:"):
        raise ValueError("Git LFS pointer is missing a sha256 object ID.")
    sha256 = lines[1].removeprefix("oid sha256:")
    if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
        raise ValueError(f"Invalid Git LFS sha256 object ID: {sha256!r}")
    if not lines[2].startswith("size "):
        raise ValueError("Git LFS pointer is missing an object size.")
    try:
        size_bytes = int(lines[2].removeprefix("size "))
    except ValueError as error:
        raise ValueError(f"Invalid Git LFS object size: {lines[2]!r}") from error
    if size_bytes < 0:
        raise ValueError("Git LFS object size cannot be negative.")
    return LfsPointer(sha256=sha256, size_bytes=size_bytes)


def sha256_file(path: Path) -> str:
    """Compute a file digest without loading a model artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def media_url(repository: str, revision: str, relative_path: Path) -> str:
    """Return GitHub's public LFS media endpoint for one pinned artifact."""
    return (
        "https://media.githubusercontent.com/media/"
        f"{repository}/{revision}/{quote(relative_path.as_posix(), safe='/')}"
    )


def checked_destination(checkout_root: Path, relative_path: Path) -> Path:
    """Prevent a command-line path from escaping the requested checkout."""
    if relative_path.is_absolute():
        raise ValueError(f"LFS path must be relative: {relative_path}")
    resolved_root = checkout_root.resolve()
    destination = (resolved_root / relative_path).resolve()
    if not destination.is_relative_to(resolved_root):
        raise ValueError(f"LFS path escapes the checkout: {relative_path}")
    return destination


def hydrate_one(
    *, checkout_root: Path, repository: str, revision: str, relative_path: Path, timeout: float
) -> dict[str, object]:
    """Download, verify, and atomically replace one LFS pointer file."""
    destination = checked_destination(checkout_root, relative_path)
    if not destination.is_file():
        raise FileNotFoundError(destination)
    try:
        pointer = parse_lfs_pointer(destination.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return {
            "path": relative_path.as_posix(),
            "status": "already_materialized",
            "sha256": sha256_file(destination),
            "size_bytes": destination.stat().st_size,
        }

    url = media_url(repository, revision, relative_path)
    temporary_path: Path | None = None
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with urlopen(url, timeout=timeout) as response, tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                temporary_file.write(chunk)
                digest.update(chunk)
                size_bytes += len(chunk)
        if size_bytes != pointer.size_bytes:
            raise ValueError(
                f"Downloaded size mismatch for {relative_path}: "
                f"expected={pointer.size_bytes} actual={size_bytes}"
            )
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != pointer.sha256:
            raise ValueError(
                f"Downloaded sha256 mismatch for {relative_path}: "
                f"expected={pointer.sha256} actual={actual_sha256}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return {
        "path": relative_path.as_posix(),
        "status": "hydrated",
        "sha256": pointer.sha256,
        "size_bytes": pointer.size_bytes,
    }


def parse_args() -> argparse.Namespace:
    """Parse a pinned external checkout without assuming a host-specific root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout-root", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="Pinned Git revision used for the checkout.")
    parser.add_argument("--repository", default="NVIDIA/Isaac-GR00T")
    parser.add_argument(
        "--path",
        action="append",
        type=Path,
        help="Relative LFS artifact to hydrate. Defaults to the GR00T dGPU torchcodec wheel.",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    """Hydrate every requested artifact and emit machine-readable provenance."""
    args = parse_args()
    paths = args.path or [DEFAULT_GR00T_LFS_PATH]
    results = [
        hydrate_one(
            checkout_root=args.checkout_root,
            repository=args.repository,
            revision=args.revision,
            relative_path=path,
            timeout=args.timeout,
        )
        for path in paths
    ]
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
