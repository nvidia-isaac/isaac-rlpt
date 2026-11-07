# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-session evaluation randomness: each seeded evaluation environment draws from its own stream."""

from __future__ import annotations

import random
from collections.abc import Hashable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

__all__ = ["SessionRandomness"]


class SessionRandomness:
    """Python, NumPy and torch RNG streams, one per seeded session.

    Inside :meth:`stream` a seeded session draws from its own generators, which
    continue where its last draw stopped; the process streams are restored
    afterwards. An unseeded session draws from the process streams. A
    session's streams can continue in another process (:meth:`export`,
    :meth:`load`): a served base model draws a session's decode noise where
    the actor's draws stopped.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._states: dict[Hashable, dict[str, Any]] = {}

    def seed(self, session_id: Hashable, seed: int | None) -> None:
        """Seed a session's streams for its next episode; ``None`` returns it to the process streams."""
        if seed is None:
            self._states.pop(session_id, None)
        else:
            self._states[session_id] = {"seed": int(seed)}

    def seeded(self, session_id: Hashable) -> bool:
        return session_id in self._states

    def export(self, session_id: Hashable) -> dict[str, Any] | None:
        """A copy of a seeded session's streams, to continue them in another process; ``None`` when unseeded."""
        state = self._states.get(session_id)
        return None if state is None else dict(state)

    def load(self, session_id: Hashable, state: dict[str, Any] | None) -> None:
        """Continue a session from streams :meth:`export` returned (``None``: the process streams)."""
        if state is None:
            self._states.pop(session_id, None)
        else:
            self._states[session_id] = dict(state)

    @contextmanager
    def stream(self, session_id: Hashable, *, enabled: bool) -> Iterator[None]:
        """Draw from the session's own streams while the block runs."""
        state = self._states.get(session_id)
        if not enabled or state is None:
            yield
            return

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        device = self.device
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [
                torch.cuda.current_device() if device.index is None else device.index
            ]
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                if "python" in state:
                    random.setstate(state["python"])
                    np.random.set_state(state["numpy"])
                    torch.set_rng_state(state["torch_cpu"])
                    if cuda_devices and "torch_cuda" in state:
                        torch.cuda.set_rng_state(state["torch_cuda"], device=device)
                    elif cuda_devices:
                        # Streams continued from a CPU process start this GPU's stream from the seed.
                        torch.cuda.manual_seed(int(state["seed"]))
                else:
                    seed = int(state["seed"])
                    random.seed(seed)
                    np.random.seed(seed % (2**32))
                    torch.manual_seed(seed)
                    if cuda_devices:
                        torch.cuda.manual_seed(seed)
                try:
                    yield
                finally:
                    state["python"] = random.getstate()
                    state["numpy"] = np.random.get_state()
                    state["torch_cpu"] = torch.get_rng_state()
                    if cuda_devices:
                        state["torch_cuda"] = torch.cuda.get_rng_state(device=device)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
