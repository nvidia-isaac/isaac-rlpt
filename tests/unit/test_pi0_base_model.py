# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pi0 base model registers lazily and fails informatively without openpi.

pi0 is experimental (ported, not validated here); these tests pin the
registry surface so selecting it without its dependencies is a clear error
rather than a crash, and so the base package never imports it eagerly. pi0's
JAX stays on the process's own GPU.
"""

import os
import sys
import types

import pytest
import torch

from isaac_rl_posttraining.policies import available_base_models, get_base_model
from isaac_rl_posttraining.policies.pi0.policy import keep_jax_on


def test_pi0_is_registered_but_unavailable_without_openpi():
    """pi0 is a registered base model (an unknown name raises KeyError); without openpi it is reported
    unavailable and selecting it raises an informative ImportError."""
    try:
        import openpi  # noqa: F401

        pytest.skip("openpi installed; the unavailable-path is not exercisable")
    except ImportError:
        pass
    assert "pi0" not in available_base_models()
    with pytest.raises(ImportError, match="pi0"):
        get_base_model("pi0")


def test_pi0_keeps_jax_on_its_own_gpu(monkeypatch):
    """JAX would reserve 75% of every visible GPU; pi0 keeps it to its device and allocates on demand."""
    updates = {}
    fake_jax = types.SimpleNamespace(
        config=types.SimpleNamespace(
            update=lambda key, value: updates.update({key: value})
        )
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    keep_jax_on(torch.device("cuda:1"))
    assert updates == {"jax_cuda_visible_devices": "1"}
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    # A user's own setting wins.
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    keep_jax_on(torch.device("cuda:0"))
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
