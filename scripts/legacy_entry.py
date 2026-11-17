# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward a retired entry script to ``scripts/train.py``, translating its old overrides.

The pre-unification entries selected the algorithm through ``alg.online_alg_name`` or
``algorithm@online_algorithm``, exposed its hyperparameters as ``online_algorithm.*``,
placed the learner with ``ppo.num_learner_ranks`` / ``ppo.local_learner_device`` and
composed their own root config (``--config-name``). One release keeps them callable;
launchers should move to ``scripts/train.py algorithm=<name>``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

TRAIN_ENTRY = Path(__file__).resolve().with_name("train.py")

_RENAMED_PREFIXES = (
    ("online_algorithm.", "algorithm."),
    ("ppo.num_learner_ranks=", "placement.learner.ranks="),
    ("ppo.local_learner_device=", "learner.device="),
)
_ALGORITHM_OVERRIDE = "algorithm="
_ALGORITHM_SELECTORS = (
    "alg.online_alg_name=",
    "algorithm@online_algorithm=",
    _ALGORITHM_OVERRIDE,
)
_CONFIG_NAME_FLAGS = ("--config-name", "-cn")


def _respell(token: str) -> str:
    """The new spelling of one override."""
    for old, new in _RENAMED_PREFIXES:
        if token.startswith(old):
            token = new + token[len(old) :]
            break
    for selector in _ALGORITHM_SELECTORS:
        if token.startswith(selector):
            return _ALGORITHM_OVERRIDE + token[len(selector) :]
    return token


def translate_overrides(argv: Sequence[str], *, algorithm: str | None) -> list[str]:
    """Rewrite old override spellings for ``scripts/train.py``.

    ``algorithm`` binds an entry to one algorithm (the PPO entries): any
    algorithm selection in ``argv`` is dropped in its favor. ``--config-name``
    goes away because ``train`` is the one root config.
    """
    translated: list[str] = []
    tokens = iter(argv)
    for token in tokens:
        if token in _CONFIG_NAME_FLAGS:
            next(tokens, None)
            continue
        if token.startswith(tuple(f"{flag}=" for flag in _CONFIG_NAME_FLAGS)):
            continue
        token = _respell(token)
        if algorithm is not None and token.startswith(_ALGORITHM_OVERRIDE):
            continue
        translated.append(token)
    if algorithm is not None:
        translated.append(_ALGORITHM_OVERRIDE + algorithm)
    return translated


def forward(argv: Sequence[str], *, algorithm: str | None, entry: str) -> NoReturn:
    """Print the deprecation notice and replace this process with ``scripts/train.py``."""
    translated = translate_overrides(argv, algorithm=algorithm)
    print(
        f"[deprecated] {entry} forwards to scripts/train.py for one release; run: "
        f"python scripts/train.py {' '.join(translated)}",
        file=sys.stderr,
        flush=True,
    )
    os.execv(sys.executable, [sys.executable, str(TRAIN_ENTRY), *translated])
