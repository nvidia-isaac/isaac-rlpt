# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Modality config for Unitree G1 WBC 43-DoF joint-space benchmark data (NEW_EMBODIMENT).
# Matches meta/modality.json of the arena_benchmark lerobot datasets:
#   state/action groups: left_leg(6) right_leg(6) waist(3) left_arm(7) left_hand(7)
#                        right_arm(7) right_hand(7)  -> 43-DoF absolute joint targets
#   video: ego_view ; language: annotation.human.task_description
# Action is ABSOLUTE joint targets (NON_EEF) -- these datasets store raw joint positions,
# NOT the 35-dim decoupled (base_height/navigate) action.
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

JOINT_KEYS = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "left_hand",
    "right_arm",
    "right_hand",
]

g1_43dof_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=list(JOINT_KEYS)),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=list(JOINT_KEYS),
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in JOINT_KEYS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(g1_43dof_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
