# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for public RoboLab external-artifact hydration helpers."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

HYDRATOR = Path(__file__).parents[1] / "scripts" / "hydrate_lfs.py"
MODULE = runpy.run_path(str(HYDRATOR))


def test_parse_lfs_pointer_accepts_the_canonical_format() -> None:
    pointer = MODULE["parse_lfs_pointer"](
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:3c5bf377f922d2126041b3c49846504083a015d44979ca594e368ccc8c4c0814\n"
        "size 512641\n"
    )

    assert (
        pointer.sha256
        == "3c5bf377f922d2126041b3c49846504083a015d44979ca594e368ccc8c4c0814"
    )
    assert pointer.size_bytes == 512641


def test_parse_lfs_pointer_rejects_noncanonical_or_invalid_values() -> None:
    with pytest.raises(ValueError, match="canonical"):
        MODULE["parse_lfs_pointer"]("not an LFS pointer\n")

    with pytest.raises(ValueError, match="Invalid Git LFS sha256"):
        MODULE["parse_lfs_pointer"](
            "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:not-a-digest\n"
            "size 1\n"
        )


def test_media_url_uses_the_pinned_revision_and_relative_path() -> None:
    url = MODULE["media_url"](
        "NVIDIA/Isaac-GR00T",
        "9c7e746b2cd37a810070a98ef41d290a07e806c2",
        Path("scripts/deployment/dgpu/wheels/torchcodec.whl"),
    )

    assert url == (
        "https://media.githubusercontent.com/media/NVIDIA/Isaac-GR00T/"
        "9c7e746b2cd37a810070a98ef41d290a07e806c2/"
        "scripts/deployment/dgpu/wheels/torchcodec.whl"
    )
