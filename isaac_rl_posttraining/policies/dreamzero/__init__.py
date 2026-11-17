# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DreamZero base models.

DreamZero is a world-action model (a WAN video-diffusion backbone that
jointly denoises future video and an action chunk). Two base models share this
package:

* ``dreamzero_local`` (:mod:`~isaac_rl_posttraining.policies.dreamzero.local_base_model`) —
  loads the frozen model **in-process** and supports DSRL steering by
  injecting the initial action-latent noise of the flow-matching denoise.
  Requires the ``groot`` package (the DreamZero model repo, installed with
  ``--no-deps``) plus the ``dreamzero-local`` extra.
* ``dreamzero_server`` (:mod:`~isaac_rl_posttraining.policies.dreamzero.
  server_base_model`) — a pure WebSocket protocol client driving a running
  DreamZero policy server; **native-only** (``supports_steering`` is
  ``False``, contract note 5) because the released serving protocol has no
  noise-injection channel. Requires the optional ``websockets`` and
  ``msgpack`` packages at request time (the ``dreamzero`` extra).

Importing this package (and :mod:`isaac_rl_posttraining` as a whole) works
without either dependency set; :func:`is_available` reports whether the
served client can talk to a server.
"""

from __future__ import annotations

import importlib.util

from isaac_rl_posttraining.policies.dreamzero.client import (
    DreamZeroClient,
    DreamZeroServerError,
    pack_payload,
    unpack_payload,
)
from isaac_rl_posttraining.policies.dreamzero.local_base_model import (
    DreamZeroBaseModelPolicy,
)
from isaac_rl_posttraining.policies.dreamzero.server_base_model import (
    DreamZeroServerBaseModelPolicy,
)

__all__ = [
    "DreamZeroBaseModelPolicy",
    "DreamZeroClient",
    "DreamZeroServerBaseModelPolicy",
    "DreamZeroServerError",
    "is_available",
    "pack_payload",
    "unpack_payload",
]


def is_available() -> bool:
    """Whether the optional DreamZero served-client dependencies are installed."""
    try:
        return (
            importlib.util.find_spec("websockets") is not None
            and importlib.util.find_spec("msgpack") is not None
        )
    except (ImportError, ValueError):
        return False
