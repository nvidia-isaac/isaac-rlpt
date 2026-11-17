# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small torch helpers: nested data movement, scalar unwrapping, device precision."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

__all__ = [
    "move_to_device",
    "convert_to_tensor",
    "convert_to_numpy",
    "to_python_scalar",
    "apply_torch_perf_knobs",
    "resolve_amp_dtype",
    "require_cuda_kernels",
]

_perf_knobs_applied = False


def move_to_device(data: Any, device: torch.device | str) -> Any:
    """Recursively move tensors in a nested dict/list structure to ``device``.

    Args:
        data: A tensor, or an arbitrarily nested dict/list containing tensors.
            Non-tensor leaves are returned unchanged.
        device: Target device.

    Returns:
        The same structure with every tensor on ``device``.
    """
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def convert_to_tensor(data: Any) -> Any:
    """Recursively convert numpy arrays in a nested structure to tensors.

    Args:
        data: A numpy array, or an arbitrarily nested dict/list of arrays.
            Leaves that are neither arrays nor containers pass through.

    Returns:
        The same structure with numpy arrays replaced by ``torch.Tensor``.
    """
    if isinstance(data, dict):
        return {k: convert_to_tensor(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_to_tensor(v) for v in data]
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    return data


def convert_to_numpy(data: Any) -> Any:
    """Recursively convert tensors in a nested structure to numpy arrays.

    Args:
        data: A tensor, or an arbitrarily nested dict/list of tensors.

    Returns:
        The same structure with tensors replaced by detached CPU arrays.
    """
    if isinstance(data, dict):
        return {k: convert_to_numpy(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_to_numpy(v) for v in data]
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


def to_python_scalar(value: Any) -> Any:
    """Unwrap a 0-d tensor/array to a plain Python scalar; other values pass through."""
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return value.item()
    return value


def apply_torch_perf_knobs() -> None:
    """Enable cudnn autotuning and TF32 matmuls (idempotent, best-effort)."""
    global _perf_knobs_applied
    if _perf_knobs_applied:
        return
    _perf_knobs_applied = True
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def resolve_amp_dtype() -> torch.dtype | None:
    """Autocast dtype for training: BF16 on CUDA devices that support it, else none (FP32)."""
    # No FP16 fallback: FP16 needs a GradScaler on every optimizer, and FP32 is correct.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16
    return None


def require_cuda_kernels(device: torch.device | str | None = None) -> None:
    """Raise a clear error when ``device`` is a CUDA GPU this torch build has no kernels for."""
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability(device)
    arches = torch.cuda.get_arch_list()
    capability = major * 10 + minor
    # SASS runs on the same major with a lower-or-equal minor (sm_86 on sm_89); PTX at or below it is JIT-compiled.
    if any(
        (a.startswith("sm_") and int(a[3:]) // 10 == major and int(a[3:]) <= capability)
        or (a.startswith("compute_") and int(a[8:]) <= capability)
        for a in arches
    ):
        return
    raise RuntimeError(
        f"{torch.cuda.get_device_name(device)} is sm_{capability}, but torch {torch.__version__} "
        f"only ships kernels for {' '.join(arches)}. Install a torch build with CUDA 12.8 or newer "
        "(for example torch 2.9.0+cu128)."
    )
