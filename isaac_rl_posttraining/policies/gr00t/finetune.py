# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differential-finetuning specification for shared-backbone policies.

Describes which parts of a model are trainable, at which learning rate and
weight decay, and where low-rank (LoRA) adapters attach. Rules match
parameter names by glob; the first matching rule wins; parameters are
deduplicated by identity so a backbone shared by several heads is counted
once. LoRA configurations are only *described* here -- the model adapter
applies them (for example via ``peft``) before optimizers are built.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

__all__ = ["GroupSpec", "FinetuneSpec"]


@dataclass
class GroupSpec:
    """One parameter-group rule.

    Attributes:
        pattern: ``fnmatch`` glob over parameter names.
        trainable: Whether matched parameters receive gradients.
        lr: Learning rate for this group; ``None`` falls back to
            :attr:`FinetuneSpec.default_lr`.
        weight_decay: Weight decay for this group.
        lora: Optional LoRA description (``{"r", "alpha", "target_modules",
            "dropout"}``) consumed by the model adapter.
    """

    pattern: str
    trainable: bool = True
    lr: float | None = None
    weight_decay: float = 0.0
    lora: dict[str, Any] | None = None


@dataclass
class FinetuneSpec:
    """Ordered list of parameter-group rules; first match wins per parameter.

    Attributes:
        default_lr: Learning rate for unmatched parameters and for groups
            without an explicit ``lr``.
        groups: The ordered rules.
    """

    default_lr: float = 3e-4
    groups: list[GroupSpec] = field(default_factory=list)

    def lora_rules(self) -> list[GroupSpec]:
        """Return the rules that carry a LoRA configuration."""
        return [group for group in self.groups if group.lora is not None]

    def build_param_groups(self, model: nn.Module) -> list[dict]:
        """Route the model's parameters into torch optimizer param-groups.

        Frozen parameters get ``requires_grad_(False)`` and are excluded;
        parameters matched by no rule join a default group at
        :attr:`default_lr`. Parameters are deduplicated by identity.

        Args:
            model: Model whose named parameters are routed.

        Returns:
            Optimizer param-group dicts.
        """
        seen: set[int] = set()
        rule_params: list[list] = [[] for _ in self.groups]
        default_params: list = []

        for name, param in model.named_parameters():
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)

            matched = self._match_rule(name)
            if matched is None:
                param.requires_grad_(True)
                default_params.append(param)
                continue

            group = self.groups[matched]
            param.requires_grad_(group.trainable)
            if group.trainable:
                rule_params[matched].append(param)

        param_groups: list[dict] = [
            {
                "params": rule_params[i],
                "lr": group.lr if group.lr is not None else self.default_lr,
                "weight_decay": group.weight_decay,
            }
            for i, group in enumerate(self.groups)
            if group.trainable and rule_params[i]
        ]
        if default_params:
            param_groups.append({"params": default_params, "lr": self.default_lr})
        return param_groups

    def _match_rule(self, name: str) -> int | None:
        """Index of the first rule whose pattern matches ``name``."""
        for i, group in enumerate(self.groups):
            if fnmatch.fnmatch(name, group.pattern):
                return i
        return None
