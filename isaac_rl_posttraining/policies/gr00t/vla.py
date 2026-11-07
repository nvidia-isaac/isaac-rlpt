# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delegated policy interface and registry for vision-language-action models.

A VLA policy owns its topology (backbone plus heads, ensembles, adapters) and
its loss computation; the RL harness drives it through two entry points only:
:meth:`VLAPolicy.act` for rollout/evaluation and :meth:`VLAPolicy.update` for
one optimization step. Adding a new model means subclassing :class:`VLAPolicy`
and registering it with :func:`register_vla` -- the harness itself never
changes.
"""

from __future__ import annotations

import abc
import os
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "VLAPolicy",
    "register_vla",
    "get_vla_policy_cls",
    "make_vla_policy",
]


class VLAPolicy(nn.Module, abc.ABC):
    """Abstract base for pretrained vision-language-action policies under RL.

    Subclasses own the full model (backbone, action head, any critic heads)
    and run their optimization internally: ``update(batch)`` performs
    forward, loss, backward, and optimizer step in one call, so the runner
    stays agnostic to model internals.

    Class attributes describe how the harness should feed the policy:

    Attributes:
        action_horizon: Number of environment steps one ``act`` call covers.
        obs_format: Observation layout the policy expects (``"dict"`` or a
            model-specific format name).
        needs_chunked_data: Whether ``update`` expects action-chunked batches.
        normalize_actions: Whether the rollout worker must normalize actions
            before execution. ``False`` means ``act`` already returns actions
            in the environment's raw action space.
    """

    action_horizon: int = 1
    obs_format: str = "dict"
    needs_chunked_data: bool = False
    normalize_actions: bool = False

    @abc.abstractmethod
    def act(
        self, obs: Any, deterministic: bool = False, actor_state: Any = None
    ) -> tuple[torch.Tensor, Any]:
        """Select an action for one observation.

        Args:
            obs: Observation in the policy's expected format.
            deterministic: Request deterministic action selection where the
                policy supports it.
            actor_state: Opaque recurrent state threaded between calls.

        Returns:
            Tuple ``(action, actor_state)`` with ``action`` in the
            environment action space.
        """
        raise NotImplementedError

    def set_phase(self, phase: str) -> None:
        """Record the current training phase (offline vs online)."""
        self._phase = phase

    def save(self, save_dir: str) -> None:
        """Save the policy state dict to ``save_dir``."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "vla_policy.pt"))

    def load(self, load_dir: str) -> None:
        """Load a state dict previously written by :meth:`save`."""
        state = torch.load(
            os.path.join(load_dir, "vla_policy.pt"),
            map_location="cpu",
            weights_only=True,
        )
        self.load_state_dict(state)


_REGISTRY: dict[str, type[VLAPolicy]] = {}


def register_vla(name: str):
    """Class decorator registering a :class:`VLAPolicy` subclass under ``name``.

    Args:
        name: Registry key used by :func:`make_vla_policy`.

    Returns:
        The decorator, which returns the class unchanged.
    """

    def _decorator(cls: type[VLAPolicy]) -> type[VLAPolicy]:
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_vla_policy_cls(name: str) -> type[VLAPolicy]:
    """Look up a registered :class:`VLAPolicy` subclass.

    Args:
        name: Registry key.

    Returns:
        The registered class.

    Raises:
        KeyError: If ``name`` was never registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown VLA policy {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def make_vla_policy(name: str, *args: Any, **kwargs: Any) -> VLAPolicy:
    """Instantiate the :class:`VLAPolicy` registered under ``name``.

    Args:
        name: Registry key.
        *args: Positional arguments forwarded to the class constructor.
        **kwargs: Keyword arguments forwarded to the class constructor.

    Returns:
        The constructed policy.
    """
    return get_vla_policy_cls(name)(*args, **kwargs)
