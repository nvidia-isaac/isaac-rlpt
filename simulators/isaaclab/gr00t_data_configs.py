# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.5 data configs for IsaacLab checkpoints that ship none of their own.

A data config declares the checkpoint's modality keys and the transform
pipeline (crop/resize of the views, min-max normalization of state and
action) that :class:`~isaac_rl_posttraining.policies.gr00t.n15.Gr00tN15Policy`
binds to the checkpoint's ``experiment_cfg/metadata.json`` statistics. Isaac
Lab ships one for the trocar task (``assemble_trocar/config/gr00t_config.py``);
the Franka stack-cube checkpoint (``RLinf/RLinf-Gr00t-SFT-Stack-cube``) was
fine-tuned with RLinf's ``LiberoFrankaDataConfig``, reproduced here so the
integration needs no RLinf install. Requires the pinned ``gr00t`` package.
"""

from __future__ import annotations

from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import VideoColorJitter, VideoCrop, VideoResize, VideoToNumpy, VideoToTensor
from gr00t.experiment.data_config import BaseDataConfig
from gr00t.model.transforms import GR00TTransform

__all__ = ["IsaacLabFrankaDataConfig"]


class IsaacLabFrankaDataConfig(BaseDataConfig):
    """LIBERO-style single-arm layout of RLinf's Franka stack-cube SFT checkpoint.

    Two 256x256 views (table, wrist) cropped to 95% and resized to 224; a
    7-part state (end-effector position, axis-angle orientation, two finger
    joints); a 7-part action (relative pose delta and gripper), both min-max
    normalized with the checkpoint's statistics. Sixteen-step action window.
    The color jitter is a training-time augmentation the transform skips in
    evaluation mode.
    """

    video_keys = ["video.image", "video.wrist_image"]
    state_keys = ["state.x", "state.y", "state.z", "state.roll", "state.pitch", "state.yaw", "state.gripper"]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self) -> ModalityTransform:
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys, normalization_modes=dict.fromkeys(self.state_keys, "min_max")
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys, normalization_modes=dict.fromkeys(self.action_keys, "min_max")
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
