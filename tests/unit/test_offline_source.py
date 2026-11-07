# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline demonstration sources: registration by base model."""

from __future__ import annotations

import pytest

from isaac_rl_posttraining.training.offline import open_offline_source


def test_open_offline_source_refuses_an_unknown_base_model():
    with pytest.raises(ValueError, match="no offline demonstration source"):
        open_offline_source("mock", "/tmp/demos", policy=object())


def test_open_offline_source_needs_a_dataset_dir():
    with pytest.raises(ValueError, match="base_model.dataset_dir"):
        open_offline_source("gr00t_n17", "", policy=object())
