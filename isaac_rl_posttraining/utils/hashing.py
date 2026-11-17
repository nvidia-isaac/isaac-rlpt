# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable content hashes for nested runtime records."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["digest_value", "digest_value_by_key"]


def digest_value(value: Any) -> str:
    """Hash one nested value without serialization ambiguity."""
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, Mapping):
            for key in sorted(item):
                digest.update(str(key).encode())
                update(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(str(len(item)).encode())
            for member in item:
                update(member)
            return
        array = np.ascontiguousarray(np.asarray(item))
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())

    update(value)
    return digest.hexdigest()


def digest_value_by_key(value: Any) -> dict[str, str]:
    """Hash each top-level mapping value, or return an empty mapping."""
    if not isinstance(value, Mapping):
        return {}
    return {str(key): digest_value(item) for key, item in sorted(value.items())}
