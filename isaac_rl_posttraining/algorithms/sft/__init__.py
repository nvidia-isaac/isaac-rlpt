# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervised fine-tuning over a trainable base model.

:class:`SFT` trains any base model that implements ``actor_bc_loss``.
It does not build critics.
"""

from isaac_rl_posttraining.algorithms.sft.config import SFTConfig
from isaac_rl_posttraining.algorithms.sft.sft import SFT

__all__ = ["SFT", "SFTConfig"]
