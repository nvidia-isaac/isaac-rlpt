#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register the ``isaac_rlpt`` library module in an Isaac Lab checkout (3.0 or newer).

Isaac Lab dispatches ``./isaaclab.sh train --rl_library <name>`` through a table of backends: the
``LIBRARY_ENTRYPOINTS`` tables of ``scripts/reinforcement_learning/{train,play}.py`` (3.0.0-beta2), or
``_BACKEND_MODULES`` in ``source/isaaclab_rl/isaaclab_rl/entrypoints/dispatch.py`` (3.0.0 release branch).
This script links ``example_workflows/isaaclab_gr00t_posttraining/isaac_rlpt`` into the matching
directory and adds the table entries.
It is idempotent and reversible::

    WORKFLOW=example_workflows/isaaclab_gr00t_posttraining
    python $WORKFLOW/scripts/install.py --isaaclab /path/to/IsaacLab            # symlink + register
    python $WORKFLOW/scripts/install.py --isaaclab /path/to/IsaacLab --copy     # copy instead of symlink
    python $WORKFLOW/scripts/install.py --isaaclab /path/to/IsaacLab --check    # report only
    python $WORKFLOW/scripts/install.py --isaaclab /path/to/IsaacLab --uninstall
    python $WORKFLOW/scripts/install.py --rl_dir /writable/copy/of/scripts/reinforcement_learning
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

LIBRARY = "isaac_rlpt"
MODULE_SRC = Path(__file__).resolve().parent.parent / LIBRARY
DISPATCH = Path("source/isaaclab_rl/isaaclab_rl/entrypoints/dispatch.py")
BACKENDS = Path("source/isaaclab_rl/isaaclab_rl/entrypoints/backends")
ENTRY_LINES = {
    "train.py": f'    "{LIBRARY}": SCRIPT_DIR / "{LIBRARY}" / "train_{LIBRARY}.py",\n',
    "play.py": f'    "{LIBRARY}": SCRIPT_DIR / "{LIBRARY}" / "play_{LIBRARY}.py",\n',
}


def _register_dispatch(path: Path, check: bool) -> str:
    """Add the train and play backends to ``_BACKEND_MODULES`` of Isaac Lab's ``dispatch.py``."""
    text = path.read_text()
    if _registered(text):
        return "registered"
    for action in ("train", "play"):
        match = re.search(rf'^    "{action}": \{{\n', text, re.MULTILINE)
        if match is None:
            raise SystemExit(
                f"{path}: the {action} backend table not found; register {LIBRARY} by hand"
            )
        entry = f'        "{LIBRARY}": "isaaclab_rl.entrypoints.backends.{LIBRARY}.{action}_{LIBRARY}",\n'
        text = text[: match.end()] + entry + text[match.end() :]
    if not check:
        path.write_text(text)
    return "missing" if check else "added"


def _rl_dir(isaaclab: Path | None, rl_dir: Path | None) -> Path:
    if rl_dir is None:
        if isaaclab is None:
            raise SystemExit(
                "pass --isaaclab <checkout> or --rl_dir <scripts/reinforcement_learning>"
            )
        rl_dir = isaaclab / "scripts" / "reinforcement_learning"
    if not (rl_dir / "train.py").is_file() or not (rl_dir / "common.py").is_file():
        raise SystemExit(
            f"{rl_dir} is not Isaac Lab's scripts/reinforcement_learning directory "
            "(missing train.py or common.py)"
        )
    return rl_dir


def _registered(text: str) -> bool:
    return f'"{LIBRARY}":' in text


def _register(path: Path, line: str, check: bool) -> str:
    text = path.read_text()
    if _registered(text):
        return "registered"
    match = re.search(r"^LIBRARY_ENTRYPOINTS\s*=\s*\{\n", text, re.MULTILINE)
    if match is None:
        raise SystemExit(
            f"{path}: LIBRARY_ENTRYPOINTS table not found; register {LIBRARY} by hand"
        )
    if not check:
        path.write_text(text[: match.end()] + line + text[match.end() :])
    return "missing" if check else "added"


def _unregister(path: Path) -> str:
    text = path.read_text()
    if not _registered(text):
        return "absent"
    kept = "".join(
        line for line in text.splitlines(keepends=True) if not _registered(line)
    )
    path.write_text(kept)  # NOSONAR - the user names their own Isaac Lab checkout
    return "removed"


def _uninstall(rl_dir: Path) -> None:
    target = rl_dir / LIBRARY
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    for name in ENTRY_LINES:
        print(f"{name}: {_unregister(rl_dir / name)}")
    print(f"{LIBRARY}: removed from {rl_dir}")


def _link_module(target: Path, copy: bool, check: bool) -> str:
    """Place the module directory at ``target`` and return its state."""
    if target.exists() or target.is_symlink():
        return "linked" if target.is_symlink() else "copied"
    if check:
        return "missing"
    if copy:
        shutil.copytree(
            MODULE_SRC, target, ignore=shutil.ignore_patterns("__pycache__")
        )
        return "copied"
    target.symlink_to(MODULE_SRC, target_is_directory=True)
    return "linked"


def _install(rl_dir: Path, copy: bool, check: bool) -> None:
    target = rl_dir / LIBRARY
    print(f"{target}: {_link_module(target, copy, check)}")
    for name, line in ENTRY_LINES.items():
        print(f"{name}: {_register(rl_dir / name, line, check)}")


def _main_dispatch(args: argparse.Namespace) -> None:
    """The Isaac Lab 3.0.0 layout: the module becomes a package of the ``isaaclab_rl`` backends."""
    root = args.isaaclab.resolve()
    target, dispatch = root / BACKENDS / LIBRARY, root / DISPATCH
    if args.uninstall:
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        print(f"{DISPATCH.name}: {_unregister(dispatch)}")
        print(f"{LIBRARY}: removed from {root}")
        return
    print(f"{target}: {_link_module(target, args.copy, args.check)}")
    print(f"{DISPATCH.name}: {_register_dispatch(dispatch, args.check)}")
    if not args.check:
        print(
            f"\nReady: cd {root} && ./isaaclab.sh train --rl_library {LIBRARY} --help"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--isaaclab", type=Path, help="Isaac Lab checkout root.")
    where.add_argument(
        "--rl_dir",
        type=Path,
        help="A (possibly copied) scripts/reinforcement_learning directory to register into.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the module directory instead of symlinking it.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report the installation state without changing anything.",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove the module directory and the table entries.",
    )
    args = parser.parse_args(argv)

    if args.isaaclab and (args.isaaclab / DISPATCH).is_file():
        _main_dispatch(args)
        return
    rl_dir = _rl_dir(
        args.isaaclab.resolve() if args.isaaclab else None,
        args.rl_dir.resolve() if args.rl_dir else None,
    )
    if args.uninstall:
        _uninstall(rl_dir)
        return
    _install(rl_dir, copy=args.copy, check=args.check)
    if not args.check:
        if args.isaaclab:
            cmd = f"cd {args.isaaclab} && ./isaaclab.sh"
        else:
            cmd = f"python {rl_dir}/train.py"
        print(f"\nReady: {cmd} train --rl_library {LIBRARY} --help")


if __name__ == "__main__":
    main()
