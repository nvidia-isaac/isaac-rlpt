# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The G1 whole-body-control dataset adapter and its layout invariants.

Covers the embodiment knowledge extracted out of the generic GR00T policy:

* :func:`assemble_wbc_arrays` reorders a teleoperation episode into the
  policy's contiguous 35-dim action (and passes the 43-dim state through),
  and
* the one action layout (:data:`G1_WBC_MODALITY`) the offline buffer and the
  Arena env server now share stays in agreement with the action layout a
  fine-tuned checkpoint's ``experiment_cfg`` implies. The buffer's
  ``WBC_MODALITY`` and the server's ``ACTION_SLICES`` were once separate
  copies that silently diverged; these tests pin them to one source.
"""

from __future__ import annotations

import numpy as np
import pytest

from isaac_rl_posttraining.envs.embodiments import get_dataset_adapter
from simulators.g1_arena.embodiment import (
    G1_DECOUPLED_STATE_GROUPS,
    G1_WBC_ACTION_DIM,
    G1_WBC_MODALITY,
    assemble_wbc_arrays,
    g1_wbc_adapter,
    is_wbc_dataset,
    wbc_base_height_slice,
    wbc_joint_action_slices,
    wbc_navigate_slice,
)

pd = pytest.importorskip("pandas")

# The action layout a checkpoint's experiment_cfg implies: the trained group
# order and per-group widths. Encoded as a fixture so a change to either the
# adapter (G1_WBC_MODALITY) or the sim env server has to move this too.
_CKPT_STATE_LAYOUT = [
    ("left_arm", 7),
    ("right_arm", 7),
    ("left_hand", 7),
    ("right_hand", 7),
    ("waist", 3),
]  # 31 consumed state dims
_CKPT_ACTION_LAYOUT = _CKPT_STATE_LAYOUT + [
    ("base_height_command", 1),
    ("navigate_command", 3),
]  # 35 action dims


def _contiguous(layout: list[tuple[str, int]]) -> tuple[dict, int]:
    """Lay groups end to end: ``[(name, width)] -> ({name: {start, end}}, total)``."""
    spans, offset = {}, 0
    for name, width in layout:
        spans[name] = {"start": offset, "end": offset + width}
        offset += width
    return spans, offset


def _wbc_episode(num_steps: int = 4) -> pd.DataFrame:
    """Synthesize one teleop episode with recognizable per-column values.

    Each joint-action column ``i`` holds the constant ``i`` so the assembled
    action can be checked against the exact source indices; the state column
    ``i`` holds ``1000 + i`` so the passthrough is distinguishable.
    """
    joint = np.tile(np.arange(43, dtype=np.float32), (num_steps, 1))  # col i == i
    state = np.tile(1000.0 + np.arange(43, dtype=np.float32), (num_steps, 1))
    base_height = np.full((num_steps, 1), 100.0, dtype=np.float32)
    navigate = np.tile(np.array([200.0, 201.0, 202.0], np.float32), (num_steps, 1))
    return pd.DataFrame(
        {
            "action": list(joint),
            "observation.state": list(state),
            "teleop.base_height_command": list(base_height),
            "teleop.navigate_command": list(navigate),
        }
    )


def test_is_wbc_dataset_detects_teleop_columns():
    assert is_wbc_dataset(_wbc_episode())
    plain = pd.DataFrame({"action": [[0.0]], "observation.state": [[0.0]]})
    assert not is_wbc_dataset(plain)


def test_assemble_wbc_arrays_shapes_and_group_order():
    """State passes through at 43-dim; action is the 35-dim reordered layout."""
    num_steps = 4
    state, action = assemble_wbc_arrays(_wbc_episode(num_steps))

    assert state.shape == (num_steps, 43)
    assert action.shape == (num_steps, 35)
    assert state.dtype == np.float32 and action.dtype == np.float32
    # The 43-dim state is passed straight through (columns 1000..1042).
    np.testing.assert_array_equal(
        state, np.tile(1000.0 + np.arange(43), (num_steps, 1))
    )

    # Each action group, sliced by G1_WBC_MODALITY, must equal the exact
    # source columns the assembler is expected to pull (verifies both the
    # group order and the joint-index mapping).
    expected_source = {
        "left_arm": list(range(15, 22)),
        "right_arm": list(range(29, 36)),
        "left_hand": list(range(22, 29)),
        "right_hand": list(range(36, 43)),
        "waist": list(range(12, 15)),
        "base_height_command": [100.0],
        "navigate_command": [200.0, 201.0, 202.0],
    }
    for group, span in G1_WBC_MODALITY["action"].items():
        got = action[:, span["start"] : span["end"]]
        exp = np.asarray(expected_source[group], dtype=np.float32)
        np.testing.assert_array_equal(got, np.broadcast_to(exp, got.shape))


def test_g1_wbc_adapter_bundle_and_resolver():
    """The adapter bundles the assembler, its modality, and fallback groups."""
    adapter = g1_wbc_adapter()
    assert adapter.assemble is assemble_wbc_arrays
    assert adapter.modality_json is G1_WBC_MODALITY
    assert list(adapter.fallback_state_groups) == G1_DECOUPLED_STATE_GROUPS
    # The env-aware resolver maps names to the adapter (or None for generic).
    assert get_dataset_adapter("g1_wbc").assemble is assemble_wbc_arrays
    assert get_dataset_adapter(None) is None
    assert get_dataset_adapter("none") is None
    with pytest.raises(KeyError):
        get_dataset_adapter("bogus")


def test_adapter_action_layout_matches_checkpoint_experiment_cfg():
    """Divergence guard: the adapter's action layout == the checkpoint's."""
    spans, total = _contiguous(_CKPT_ACTION_LAYOUT)
    assert total == 35 == G1_WBC_ACTION_DIM
    assert G1_WBC_MODALITY["action"] == spans, (
        "the dataset adapter's action layout drifted from the checkpoint's "
        "trained group order/widths"
    )


def test_adapter_state_consumed_width_matches_checkpoint_experiment_cfg():
    """The 43-dim state slices down to the checkpoint's 31 consumed dims."""
    _spans, total = _contiguous(_CKPT_STATE_LAYOUT)
    assert total == 31
    # The fallback groups the adapter registers are exactly the checkpoint's
    # consumed state groups, in order.
    assert G1_DECOUPLED_STATE_GROUPS == [name for name, _ in _CKPT_STATE_LAYOUT]
    # Each group's width in the 43-dim layout matches the checkpoint's width;
    # summed, the consumed groups total 31 (the legs are dropped at decode).
    consumed = 0
    for name, width in _CKPT_STATE_LAYOUT:
        span = G1_WBC_MODALITY["state"][name]
        assert span["end"] - span["start"] == width
        consumed += width
    assert consumed == 31


def test_sim_env_server_action_layout_matches_checkpoint():
    """The layout the Arena env server unpacks equals the checkpoint's.

    Reconstructs the 35-dim action from the slice helpers the env server now
    imports (rather than its own former literals) and pins it to the same
    fixture, so the two copies can never silently diverge again.
    """
    spans, total = _contiguous(_CKPT_ACTION_LAYOUT)
    rebuilt = {
        group: {"start": sl.start, "end": sl.stop}
        for group, sl in wbc_joint_action_slices(prefix="").items()
    }
    for group, sl in (
        ("base_height_command", wbc_base_height_slice()),
        ("navigate_command", wbc_navigate_slice()),
    ):
        rebuilt[group] = {"start": sl.start, "end": sl.stop}
    assert rebuilt == spans
    assert total == 35
