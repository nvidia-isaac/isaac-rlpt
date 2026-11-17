# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remove every ``internal/`` directory of the repository and the internal-only repository paths for a release.

The scan covers the whole tree under ``--root`` (the repository root by default): nothing outside an
``internal/`` directory may name internal material. The internal-only paths
(``release.INTERNAL_ONLY_PATHS``: the internal CI and the release skill) are looked up in ``--root`` itself.
Release-excluded features (``release.RELEASE_FEATURES_FILE``) lose their paths and fenced lines; when that
changes the root ``pyproject.toml``, the root is locked again (needs ``uv`` and the package indexes) and the
workflows' lockfiles take its new metadata.

python tools/strip_internal.py --check   # list what would go, and verify nothing outside references it
python tools/strip_internal.py           # remove them (refuses while references remain)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from example_workflows._common import (
    release,  # the repository root was put on sys.path above
)


def _package_metadata(lock: str, name: str) -> str | None:
    """The ``[package.metadata]`` table of package ``name`` in a ``uv.lock`` text."""
    start = lock.find(f'[[package]]\nname = "{name}"\n')
    if start < 0:
        return None
    begin = lock.index("[package.metadata]", start)
    end = lock.find("\n[[package]]", begin)
    return lock[begin : None if end < 0 else end]


def relock(root: Path) -> None:
    """Lock the fenced root project again; the workflows' own lockfiles take its new metadata.

    A workflow lockfile records the root package's metadata verbatim, but locking it needs the workflow's
    external checkouts, so its other entries stay as they are.
    """
    print("locking . again", flush=True)
    subprocess.run(["uv", "lock", "--directory", str(root)], check=True)
    name = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["name"]
    metadata = _package_metadata((root / "uv.lock").read_text(encoding="utf-8"), name)
    for lock in release.files(root):
        if lock.name != "uv.lock" or lock == root / "uv.lock":
            continue
        text = lock.read_text(encoding="utf-8")
        old = _package_metadata(text, name)
        if old is not None and metadata is not None and old != metadata:
            lock.write_text(text.replace(old, metadata), encoding="utf-8")
            print(f"updated the {name} metadata in {lock.relative_to(root).as_posix()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="tree to strip (default: the repository root)",
    )
    parser.add_argument(
        "--check", action="store_true", help="report without removing anything"
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()

    directories = release.internal_dirs(root)
    references = release.internal_references(root)
    paths = release.internal_only_paths(root)
    features = release.release_features(root)
    excluded = {
        root / name: feature for feature in features.values() for name in feature.paths
    }
    fenced = {}
    if features and not references:
        for path in release.text_files(root):
            if release.is_internal(path, root, features.values()) or release.is_exempt(
                path
            ):
                continue
            text = path.read_text(encoding="utf-8")
            if (stripped := release.strip_fences(text, features)) != text:
                fenced[path] = stripped
    for path in directories:
        print(f"internal: {path.relative_to(root).as_posix()}/")
    for path in paths:
        print(f"internal-only path: {path.relative_to(root).as_posix()}")
    for path, feature in excluded.items():
        print(
            f"release-excluded feature {feature.name}: {path.relative_to(root).as_posix()}"
        )
    for path in fenced:
        print(f"fenced lines: {path.relative_to(root).as_posix()}")
    for finding in references:
        print(f"reference outside internal/: {finding.describe(root)}", file=sys.stderr)
    if references:
        print(
            f"{len(references)} reference(s) to internal material remain outside internal/; nothing removed",
            file=sys.stderr,
        )
        return 1
    if args.check:
        print(
            f"internal directories: {len(directories)}, internal-only paths: {len(paths)}, "
            f"release-excluded features: {len(features)} ({len(excluded)} paths, {len(fenced)} files with fenced lines); "
            "the tree is clean"
        )
        return 0
    for path, text in fenced.items():
        path.write_text(text, encoding="utf-8")
    for path in directories:
        if path.exists():
            shutil.rmtree(path)
            print(f"removed {path.relative_to(root).as_posix()}/")
    for path in [*paths, *excluded]:
        if not os.path.lexists(path):  # inside a directory removed above
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        print(f"removed {path.relative_to(root).as_posix()}")
    if root / "pyproject.toml" in fenced:
        relock(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
