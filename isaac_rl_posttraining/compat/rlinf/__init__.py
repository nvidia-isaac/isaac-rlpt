# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RLinf infrastructure backends behind this repository's interfaces.

`RLinf <https://github.com/RLinf/RLinf>`_ is an open-source RL
infrastructure project (Ray-based worker scheduling, channels, replay
buffers). This package adapts one of its subsystems to our seams:

* :class:`ParallelRolloutAdapter` (``rollout.py``) -- RLinf's scheduler
  (``Cluster`` / placed ``Worker`` groups / remote-call futures) driving N
  environment workers in parallel behind a collector-shaped surface
  (``run`` / ``collect`` / an ``EpisodeTracker``). (RLinf's ``Channel``
  would be the natural transition stream,
  but at the pinned version its worker-side path needs a CUDA torch build
  on GPU-bearing nodes -- see ``rollout.py``'s module docstring.)

Import policy: this package imports without ``rlinf`` installed; accessing
an adapter raises ``ImportError`` with the install hint. RLinf is **pinned
exactly** (:data:`RLINF_PINNED_VERSION`) because the adapters reach into
version-specific internals where RLinf's public API has no equivalent
(documented per call site). Install via ``pip install
"isaac-rl-posttraining[rlinf]"`` -- the same extra as the RLInf runtime,
preferably into a dedicated environment; RLinf's dependency tree is large (``ray[default]``,
``datasets``, ``accelerate``, ``wandb``, ...). Operational notes, benchmark
numbers, and the adopt/hold recommendation live in
``docs/infrastructure.md``.
"""

from __future__ import annotations

import importlib.util
from typing import Any

__all__ = [
    "NUMPY_MINIMUM_VERSION",
    "RAY_PINNED_VERSION",
    "RLINF_PINNED_COMMIT",
    "RLINF_PINNED_TAG",
    "RLINF_PINNED_VERSION",
    "ParallelRolloutAdapter",
    "is_available",
    "require_rlinf",
]

RLINF_PINNED_VERSION = "0.3.0"
"""The exact RLinf release these adapters are validated against."""

RLINF_PINNED_TAG = "v0.3"
"""Upstream git tag of :data:`RLINF_PINNED_VERSION`."""

RLINF_PINNED_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"
"""Upstream commit of :data:`RLINF_PINNED_TAG`."""

RAY_PINNED_VERSION = "2.58.0"
"""Ray client/server version used to validate the RLInf adapter."""

NUMPY_MINIMUM_VERSION = "1.26"
"""Oldest NumPy release accepted by the RLInf adapters."""

_INSTALL_HINT = (
    f"install the [rlinf] extra (pinned rlinf=={RLINF_PINNED_VERSION}, "
    f"ray=={RAY_PINNED_VERSION}): pip install "
    '"isaac-rl-posttraining[rlinf]"'
)


def is_available() -> bool:
    """Whether the ``rlinf`` package is importable in this environment."""
    try:
        return importlib.util.find_spec("rlinf") is not None
    except (ImportError, ValueError):
        return False


def require_rlinf() -> None:
    """Fail fast when the validated RLInf dependency set is unavailable.

    Exact RLInf and Ray matching is required because the adapter touches
    version-specific internals and Ray rejects mismatched clients and clusters.
    NumPy only has a floor: each container ships a PyTorch build matching its
    own NumPy major version, so the major version is not pinned here.

    Raises:
        ImportError: If ``rlinf`` is not installed.
    """
    if not is_available():
        raise ImportError(
            f"The RLinf adapters need the 'rlinf' package; {_INSTALL_HINT}"
        )
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import Version

    expected = {
        "rlinf": RLINF_PINNED_VERSION,
        "ray": RAY_PINNED_VERSION,
    }
    for package, wanted in expected.items():
        try:
            installed = version(package)
        except PackageNotFoundError as exc:
            raise ImportError(
                f"The RLinf adapters need {package}=={wanted}; {_INSTALL_HINT}"
            ) from exc
        if Version(installed) != Version(wanted):
            raise RuntimeError(
                f"RLInf runtime dependency mismatch: found {package}=={installed}, "
                f"but this adapter requires {package}=={wanted}. Ray clients and "
                "clusters must use the same version. Reinstall the [rlinf] extra."
            )

    try:
        numpy_version = version("numpy")
    except PackageNotFoundError as exc:
        raise ImportError(
            f"The RLinf adapters need NumPy>={NUMPY_MINIMUM_VERSION}; {_INSTALL_HINT}"
        ) from exc
    if Version(numpy_version) < Version(NUMPY_MINIMUM_VERSION):
        raise RuntimeError(
            f"RLInf runtime dependency mismatch: found numpy=={numpy_version}, but "
            f"this adapter requires numpy>={NUMPY_MINIMUM_VERSION}. Reinstall "
            "the [rlinf] extra."
        )


def __getattr__(name: str) -> Any:
    """Lazily resolve the adapter so importing this package never needs rlinf."""
    if name == "ParallelRolloutAdapter":
        require_rlinf()
        from isaac_rl_posttraining.compat.rlinf.rollout import ParallelRolloutAdapter

        return ParallelRolloutAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
