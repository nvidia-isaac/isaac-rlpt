# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# No-leg modality config for Unitree G1 WBC (NEW_EMBODIMENT).
# ACTION drops leg joints (waist+arms+hands = 31-DoF); the arena's low-level WBC policy
# handles the legs. STATE keeps full 43-DoF proprioception (incl. legs) as observation.
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

STATE_KEYS = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "left_hand",
    "right_arm",
    "right_hand",
]
ACTION_KEYS = ["waist", "left_arm", "left_hand", "right_arm", "right_hand"]  # NO legs

g1_noleg_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=list(STATE_KEYS)),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=list(ACTION_KEYS),
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in ACTION_KEYS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0], modality_keys=["annotation.human.task_description"]
    ),
}

register_modality_config(g1_noleg_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
