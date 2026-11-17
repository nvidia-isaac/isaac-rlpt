# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# 35-dim DECOUPLED WBC action config for Unitree G1 (NEW_EMBODIMENT).
# Action = upper body joints + high-level WBC commands; the low-level WBC controller
# consumes base_height/navigate to move the legs/locomote (legs NOT predicted directly).
#   left_arm7 + right_arm7 + left_hand7 + right_hand7 + waist3 + base_height1 + navigate3 = 35
# All fields present in these datasets' meta/modality.json (teleop.base_height_command,
# teleop.navigate_command). State = upper-body proprioception (no legs), matching the arena's
# canonical UnitreeG1SimWBCDataConfig.
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

STATE_KEYS = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]
ACTION_KEYS = [
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "waist",
    "base_height_command",
    "navigate_command",
]

g1_35dim_config = {
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
register_modality_config(g1_35dim_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
