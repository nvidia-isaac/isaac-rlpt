# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LIBERO's entries into the runtime layer: one per base-policy convention."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar, TypeAlias

from isaac_rl_posttraining.data.spec import ActionSpec, ArraySpec, ObservationSpec
from isaac_rl_posttraining.envs.simulator_entry import (
    EnvironmentSpec,
    Phase,
    SimulatorEntry,
)
from isaac_rl_posttraining.runtimes.base import RolloutWorkerContext
from simulators.libero.env import DEFAULT_ACTION_DIM
from simulators.libero.policies.dreamzero import (
    DREAMZERO_LIBERO_DSRL_KEY_MAPPING,
    DREAMZERO_LIBERO_IMAGE_SIZE,
    LiberoDreamZeroAdapter,
    make_libero_dreamzero_env,
)
from simulators.libero.policies.gr00t import (
    GR00T_LIBERO_IMAGE_SIZE,
    GR00T_LIBERO_VIDEO_KEYS,
    LiberoGr00tAdapter,
    make_libero_gr00t_env,
)
from simulators.libero.policies.pi0 import (
    PI0_LIBERO_CAMERA_KEYS,
    PI0_LIBERO_DSRL_KEY_MAPPING,
    PI0_LIBERO_IMAGE_SIZE,
    PI0_LIBERO_STATE_KEY,
    LiberoPi0Adapter,
    make_libero_pi0_env,
)

__all__ = ["EEF_STATE_DIM", "EVALUATION_SEED_OFFSET", "Libero", "LiberoDreamZero", "LiberoPi0"]

#: Evaluation environments draw their initial scenes from a distinct stream so
#: they never replay the training sequence.
EVALUATION_SEED_OFFSET = 1000

LiberoTask: TypeAlias = LiberoGr00tAdapter | LiberoPi0Adapter | LiberoDreamZeroAdapter

#: Width of the end-effector state every convention emits (``EEF_STATE_KEYS``): position (3),
#: axis-angle (3), gripper joints (2).
EEF_STATE_DIM = 8


class _LiberoEntry(SimulatorEntry):
    """LIBERO stepped in-process: one task per worker, in the conventions of one base-policy family.

    ``env.env_name`` is ``<entry>/<suite>/<task>``. The default layout of one
    environment per worker applies. Evaluation workers are rebuilt per
    evaluation (fresh, independently seeded simulators) instead of sharing
    training state.
    """

    evaluation = "fresh_workers"
    #: MuJoCo renders the cameras through EGL on a GPU.
    simulates_on_gpu = True
    #: The steering contract of this entry's observations: state, language and image keys.
    key_mapping: ClassVar[dict[str, Any]]
    #: The camera entries of an observation in this convention, their square size, and the state entry.
    camera_keys: ClassVar[tuple[str, ...]]
    image_size: ClassVar[int]
    state_key: ClassVar[str] = "state"

    @abstractmethod
    def make_task(self, **kwargs: Any) -> LiberoTask:
        """Build one adapted LIBERO task in this entry's conventions."""

    def make_environment(self, spec: EnvironmentSpec, phase: Phase, context: RolloutWorkerContext) -> LiberoTask:
        parts = spec.env_name.split("/", 2)
        if len(parts) != 3 or parts[0] != self.name:
            raise ValueError(f"the {self.name} entry expects env.env_name={self.name}/<suite>/<task>")
        _, suite, task = parts
        seed = spec.seed
        if seed is not None:
            # Each environment draws its own seed; evaluation seeds come from their own stream.
            if phase == "eval":
                seed += EVALUATION_SEED_OFFSET
            seed += context.environment_indices[0]
        task_kwargs = {"task_id": int(task)} if task.isdigit() else {"task_name": task}
        return self.make_task(
            suite=suite,
            max_episode_steps=spec.max_episode_steps,
            use_init_states=spec.use_init_states,
            fixed_init_state_id=spec.fixed_init_state_id,
            seed=seed,
            **task_kwargs,
        )

    def steering_key_mapping(self, spec: EnvironmentSpec) -> dict[str, Any]:
        del spec
        return dict(self.key_mapping)

    def observation_spec(self, spec: EnvironmentSpec) -> ObservationSpec:
        """Every task of a suite emits the convention's cameras and the 8-dim end-effector state."""
        del spec
        arrays = {key: ArraySpec((self.image_size, self.image_size, 3), "uint8") for key in self.camera_keys}
        arrays[self.state_key] = ArraySpec((EEF_STATE_DIM,), "float32")
        return ObservationSpec(arrays)

    def action_spec(self, spec: EnvironmentSpec) -> ActionSpec:
        """A 7-dim operational-space command: end-effector delta (6) and gripper (1)."""
        del spec
        return ActionSpec(width=DEFAULT_ACTION_DIM)


class Libero(_LiberoEntry):
    """GR00T conventions: the cameras under ``video.*``, model-space actions."""

    name = "libero"
    base_models = ("gr00t", "gr00t_n15", "gr00t_n17")
    key_mapping: ClassVar[dict[str, Any]] = {
        "state": "state",
        "language": None,
        "image": [f"video.{key}" for key in GR00T_LIBERO_VIDEO_KEYS],
    }
    camera_keys = tuple(f"video.{key}" for key in GR00T_LIBERO_VIDEO_KEYS)
    image_size = GR00T_LIBERO_IMAGE_SIZE

    def make_task(self, **kwargs: Any) -> LiberoGr00tAdapter:
        return make_libero_gr00t_env(**kwargs)


class LiberoPi0(_LiberoEntry):
    """pi0 conventions: pad-resized cameras under ``observation/*``, end-effector state, env-space actions."""

    name = "libero_pi0"
    base_models = ("pi0",)
    key_mapping = PI0_LIBERO_DSRL_KEY_MAPPING
    camera_keys = PI0_LIBERO_CAMERA_KEYS
    image_size = PI0_LIBERO_IMAGE_SIZE
    state_key = PI0_LIBERO_STATE_KEY

    def make_task(self, **kwargs: Any) -> LiberoPi0Adapter:
        return make_libero_pi0_env(**kwargs)


class LiberoDreamZero(_LiberoEntry):
    """DreamZero conventions: rotated ``main_image``/``wrist_image`` cameras, end-effector state, env-space actions."""

    name = "libero_dreamzero"
    base_models = ("dreamzero_local", "dreamzero_server")
    key_mapping = DREAMZERO_LIBERO_DSRL_KEY_MAPPING
    camera_keys = ("main_image", "wrist_image")
    image_size = DREAMZERO_LIBERO_IMAGE_SIZE

    def make_task(self, **kwargs: Any) -> LiberoDreamZeroAdapter:
        return make_libero_dreamzero_env(**kwargs)
