# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared network building blocks: MLP builder, target-net updates, ensembles."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence

import torch
from torch import nn

__all__ = ["build_mlp", "polyak_update", "TargetUpdater", "frozen_parameters", "CriticEnsemble"]

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
}


def build_mlp(
    input_size: int,
    hidden_dims: Sequence[int],
    activation: str = "relu",
    use_layer_norm: bool = False,
    dropout_rate: float = 0.0,
) -> nn.Sequential:
    """Build a feed-forward MLP as ``Linear [-> LayerNorm] -> act [-> Dropout]`` blocks.

    Args:
        input_size: Width of the input features.
        hidden_dims: Output width of each successive linear layer.
        activation: One of ``relu``, ``tanh``, ``elu``, ``leaky_relu``, ``gelu``.
        use_layer_norm: Insert a ``LayerNorm`` after every linear layer.
        dropout_rate: Dropout probability after each activation; 0 disables it.

    Returns:
        An ``nn.Sequential`` ending after the last block (no output head).

    Raises:
        ValueError: If ``activation`` is not a known activation name.
    """
    act_cls = _ACTIVATIONS.get(activation.lower())
    if act_cls is None:
        raise ValueError(f"Unknown activation: {activation!r} (choose from {sorted(_ACTIVATIONS)})")

    layers: list[nn.Module] = []
    prev_size = int(input_size)
    for hidden_size in hidden_dims:
        hidden_size = int(hidden_size)
        layers.append(nn.Linear(prev_size, hidden_size))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(act_cls())
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        prev_size = hidden_size
    return nn.Sequential(*layers)


def polyak_update(
    params: Iterable[torch.Tensor],
    target_params: Iterable[torch.Tensor],
    tau: float,
) -> None:
    """Update target parameters in place: ``target = tau * online + (1 - tau) * target``.

    Parameters that share storage between the online and target networks (an
    intentional pattern when heavy submodules are shared) already hold the
    online values and are left alone, so they are never scaled against themselves.

    Args:
        params: Online-network parameters.
        target_params: Target-network parameters, zipped 1:1 with ``params``.
        tau: Interpolation factor in ``[0, 1]``; ``1.0`` performs a hard copy.
    """
    with torch.no_grad():
        blend_targets: list[torch.Tensor] = []
        blend_sources: list[torch.Tensor] = []
        for p, tp in zip(params, target_params):
            if tp.data_ptr() == p.data_ptr():
                continue
            if tau >= 1.0:
                tp.copy_(p)
            else:
                blend_targets.append(tp)
                blend_sources.append(p)
        if blend_targets:
            # One batched multiply and add, the same arithmetic as per-tensor mul_ / add_.
            torch._foreach_mul_(blend_targets, 1.0 - tau)
            torch._foreach_add_(blend_targets, blend_sources, alpha=tau)


class TargetUpdater:
    """Soft-updates a target network from its online network, listing their parameters once.

    Walking a module tree for its parameters on every update costs more than the
    update itself for small networks, so the two parameter lists are kept until
    either module object changes. Parameters are matched by position, as in
    :func:`polyak_update`.
    """

    def __init__(self) -> None:
        self._modules: tuple[nn.Module, nn.Module] | None = None
        self._params: tuple[list[nn.Parameter], list[nn.Parameter]] = ([], [])

    def soft_update(self, online: nn.Module, target: nn.Module, tau: float) -> None:
        """``target = tau * online + (1 - tau) * target``, in place."""
        if self._modules is None or self._modules[0] is not online or self._modules[1] is not target:
            self._modules = (online, target)
            self._params = (list(online.parameters()), list(target.parameters()))
        polyak_update(self._params[0], self._params[1], tau)

    def __getstate__(self) -> dict[str, object]:
        # A pickled copy re-lists the parameters of the copied modules on first use.
        return {"_modules": None, "_params": ([], [])}


@contextmanager
def frozen_parameters(module: nn.Module) -> Iterator[None]:
    """Turn off ``requires_grad`` on ``module``'s trainable parameters for the duration of the block.

    Gradients still flow through the module to its inputs; only its own weight
    gradients are skipped.
    """
    trainable = [param for param in module.parameters() if param.requires_grad]
    for param in trainable:
        param.requires_grad_(False)
    try:
        yield
    finally:
        for param in trainable:
            param.requires_grad_(True)


class CriticEnsemble(nn.Module):
    """A list of Q-networks evaluated jointly, with deduplicated parameters.

    Members may intentionally share heavy submodules (e.g. a common feature
    trunk with per-member value heads). :meth:`parameters` yields each shared
    tensor exactly once so that a single optimizer step applies one gradient
    update per parameter — without deduplication, shared parameters would be
    updated once per member, silently multiplying their learning rate.
    """

    def __init__(self, critics: Sequence[nn.Module]) -> None:
        """Wrap ``critics`` into an ensemble module.

        Args:
            critics: The member Q-networks.
        """
        super().__init__()
        self.critics = nn.ModuleList(critics)
        self.num_critics = len(critics)

    def forward(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        critic_indices: Iterable[int] | None = None,
    ) -> list[torch.Tensor]:
        """Evaluate member critics on a batch.

        Args:
            obs: Batched observations (tensor or dict of tensors).
            action: Batched actions.
            critic_indices: Optional subset of member indices to evaluate
                (used for randomized-subset pessimism); ``None`` evaluates all.

        Returns:
            One Q-value tensor per evaluated member.
        """
        if critic_indices is not None:
            return [self.critics[int(i)](obs, action) for i in critic_indices]
        return [critic(obs, action) for critic in self.critics]

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Yield all member parameters with shared tensors emitted once."""
        seen_ids: set[int] = set()
        for critic in self.critics:
            for param in critic.parameters(recurse=recurse):
                if id(param) not in seen_ids:
                    seen_ids.add(id(param))
                    yield param
