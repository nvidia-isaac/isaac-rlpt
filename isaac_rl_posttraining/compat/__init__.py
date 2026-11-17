# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters for external RL-infrastructure libraries.

Everything under ``compat/`` wraps a third-party infrastructure library
behind one of this repository's own interfaces (replay buffers, rollout
collection, transports). The rules that keep these integrations honest:

* **Libraries, not frameworks.** The external project is imported *behind*
  our seams; it never hosts our training loop, owns our configs, or leaks
  into core code. Core packages (``algorithms/``, ``buffers/``,
  ``rollouts/``, ``envs/`` ...) never import anything from here or from the
  wrapped libraries -- the repo-hygiene tests enforce this.
* **Optional and pinned.** Each integration is an optional extra pinned to
  the exact upstream version its adapters were validated against
  (see ``pyproject.toml``); importing :mod:`isaac_rl_posttraining` (and the
  ``compat`` packages themselves) succeeds without any of them installed.
* **Contract-tested.** The same unit contracts that pin our native
  implementations run against the adapters (parameterized ``native`` |
  backend), so an adapter is only as good as the tests it passes.

Current integrations:

* :mod:`~isaac_rl_posttraining.compat.rlinf` -- RLinf replay buffer and
  Ray-scheduler-based parallel rollout adapters (extra ``[rlinf]``).
"""
