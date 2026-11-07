# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``require_cuda_kernels`` names the GPU and torch build instead of a cryptic kernel error."""

from __future__ import annotations

from unittest import mock

import pytest

from isaac_rl_posttraining.utils.torch_utils import require_cuda_kernels


def _patch_cuda(capability, arches, available=True):
    return mock.patch.multiple(
        "torch.cuda",
        is_available=mock.Mock(return_value=available),
        get_device_capability=mock.Mock(return_value=capability),
        get_arch_list=mock.Mock(return_value=arches),
        get_device_name=mock.Mock(
            return_value="NVIDIA RTX PRO 6000 Blackwell Server Edition"
        ),
    )


def test_supported_gpu_passes():
    with _patch_cuda((8, 9), ["sm_80", "sm_89", "sm_90"]):
        require_cuda_kernels("cuda")


def test_unsupported_gpu_names_gpu_and_fix():
    with (
        _patch_cuda((12, 0), ["sm_50", "sm_90"]),
        pytest.raises(RuntimeError, match=r"sm_120.*torch.*CUDA 12\.8"),
    ):
        require_cuda_kernels("cuda")


def test_ada_runs_on_sm86_kernels():
    with _patch_cuda(
        (8, 9), ["sm_75", "sm_80", "sm_86", "sm_90", "sm_120", "compute_120"]
    ):
        require_cuda_kernels("cuda")


def test_ptx_fallback_passes():
    with _patch_cuda((12, 0), ["sm_90", "compute_90"]):
        require_cuda_kernels("cuda")


def test_cpu_and_missing_cuda_are_ignored():
    require_cuda_kernels("cpu")
    with _patch_cuda((12, 0), ["sm_90"], available=False):
        require_cuda_kernels("cuda")
