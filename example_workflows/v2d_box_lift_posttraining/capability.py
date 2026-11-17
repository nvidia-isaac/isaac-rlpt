# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tissue-box task capability supplied to the reusable V2D server."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from example_workflows.v2d_box_lift_posttraining.config import (
    ResetPerturbation,
    TaskConfig,
    load_capability_launch,
    load_task,
)
from example_workflows.v2d_box_lift_posttraining.fixed_reset import (
    reset_to_fixed_trajectory_start,
)
from example_workflows.v2d_box_lift_posttraining.reset_events import (
    BatchedTextureReset,
    coalesce_texture_resets,
)
from example_workflows.v2d_box_lift_posttraining.success import (
    CombinedSuccessTracker,
    SuccessResult,
)
from isaac_rl_posttraining.envs.transport import PROTOCOL_VERSION, spec_payload
from simulators.v2d.contract import (
    IMAGE_KEYS,
    JOINT_DIM,
    JsonValue,
    Payload,
    batched_observation_from_record,
    object_height,
    v2d_action_spec,
    v2d_observation_spec,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg
    from torch import Tensor

    from example_workflows.v2d_box_lift_posttraining.collision_reset import (
        CollisionValidatedReset,
    )

TERMINAL_CAPTURE_KEY = "posttrain_terminal_object_state"


@dataclass(frozen=True)
class NativeBatchStep:
    record: dict[str, np.ndarray]
    terminal_record: dict[str, np.ndarray]
    completed: np.ndarray
    reasons: tuple[tuple[str, ...], ...]
    reset_diagnostics: dict[str, np.ndarray] = field(default_factory=dict)


class IsaacRuntime:
    """Small NumPy boundary around an already-created V2D Isaac environment."""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        *,
        render_ticks: int,
        termination_reasons: Callable[[ManagerBasedRLEnv, int], tuple[str, ...]],
        texture_resets: Sequence[BatchedTextureReset],
        collision_reset: CollisionValidatedReset | None = None,
    ) -> None:
        self.env = env
        self.render_ticks = render_ticks
        self.termination_reasons = termination_reasons
        self.texture_resets = tuple(texture_resets)
        self.collision_reset = collision_reset

    @staticmethod
    def _numpy(values: Mapping[str, Tensor]) -> dict[str, np.ndarray]:
        result = {}
        for key, value in values.items():
            result[key] = value.detach().cpu().numpy().copy()
        return result

    def _record(
        self, observation: Mapping[str, Mapping[str, Tensor]]
    ) -> dict[str, np.ndarray]:
        return self._numpy(observation["record"])

    def reset_envs(
        self, env_ids: Sequence[int] | None, *, seed: int | None
    ) -> dict[str, np.ndarray]:
        self._reset_states(env_ids, seed=seed)
        return self._observe_reset()

    def reset_cases(self, cases: Sequence[tuple[int, int]]) -> dict[str, np.ndarray]:
        """Reset independently seeded cases, then capture the settled batch once."""
        if not cases:
            raise ValueError("Require at least one reset case")
        last_case = len(cases) - 1
        try:
            for index, (env_index, seed) in enumerate(cases):
                for event in self.texture_resets:
                    event.apply_materials = index == last_case
                self._reset_states([env_index], seed=seed)
        finally:
            for event in self.texture_resets:
                event.apply_materials = True
        # Cameras render the whole scene, so per-case warmup scales quadratically.
        return self._observe_reset()

    def _reset_states(self, env_ids: Sequence[int] | None, *, seed: int | None) -> None:
        import torch

        # PhysX's indexed state writes require tensors on the simulation device.
        indices = None
        if env_ids is not None:
            indices = torch.as_tensor(
                env_ids, dtype=torch.int64, device=self.env.device
            )
        if self.collision_reset is not None and seed is not None:
            self.collision_reset.reseed(indices, seed)
        self.env.reset(seed=seed, env_ids=indices)

    def reset_diagnostics(self) -> dict[str, np.ndarray]:
        if self.collision_reset is None:
            return {}
        return self._numpy(self.collision_reset.diagnostics())

    def _observe_reset(self) -> dict[str, np.ndarray]:
        for _ in range(self.render_ticks):
            self.env.sim.render()
        observation = self.env.observation_manager.compute()
        command = self.env.command_manager.get_term("motion")
        for scale in (
            command.virtual_object_controller_scale_factor,
            command.virtual_object_controller_scale_factor_per_env,
        ):
            if scale.count_nonzero().item():
                raise RuntimeError("V2D virtual object assistance must be disabled")
        return self._record(observation)

    def step_batch(self, actions: np.ndarray) -> NativeBatchStep:
        import torch

        values = np.asarray(actions, dtype=np.float32)
        if values.shape != (self.env.num_envs, JOINT_DIM):
            raise ValueError(
                f"V2D actions must have shape ({self.env.num_envs}, {JOINT_DIM})"
            )
        # Native autoreset replaces these counters with the next episode's reset.
        reset_diagnostics = self.reset_diagnostics()
        with torch.no_grad():
            observation, _, terminated, truncated, extras = self.env.step(
                torch.as_tensor(values, dtype=torch.float32, device=self.env.device)
            )
        if TERMINAL_CAPTURE_KEY not in extras:
            raise RuntimeError("V2D terminal-state capture did not run")
        completed = (terminated | truncated).detach().cpu().numpy().astype(bool)
        reasons = tuple(
            self.termination_reasons(self.env, index) if completed[index] else ()
            for index in range(self.env.num_envs)
        )
        return NativeBatchStep(
            record=self._record(observation),
            terminal_record=self._numpy(extras[TERMINAL_CAPTURE_KEY]),
            completed=completed,
            reasons=reasons,
            reset_diagnostics=reset_diagnostics,
        )

    def close(self) -> None:
        self.env.close()


def _symmetric(value: float) -> tuple[float, float]:
    return (-value, value)


def _configure_trajectory_reset(
    event: EventTermCfg, task: TaskConfig, seed: int
) -> None:
    if task.reset_validation is not None and task.reset_validation.enabled:
        from example_workflows.v2d_box_lift_posttraining.collision_reset import (
            CollisionValidatedReset,
        )

        event.func = CollisionValidatedReset
        event.params.update(
            reset_validation=task.reset_validation,
            fixed_reset=task.fixed_reset,
            seed=seed,
        )
    elif task.fixed_reset:
        event.func = reset_to_fixed_trajectory_start
        event.params["seed"] = seed


def _apply_reset_perturbation(
    params: dict[str, object],
    expected_joint_groups: Mapping[str, tuple[str, ...]],
    perturbation: ResetPerturbation,
) -> None:
    groups = params["joint_position_noise_groups"]
    if not isinstance(groups, dict) or set(groups) != {"arm", "finger"}:
        raise ValueError(
            "V2D reset perturbations require exactly arm and finger groups"
        )
    configured = {}
    for name, group_value in groups.items():
        if not isinstance(group_value, dict):
            raise TypeError(f"reset joint group {name} must be a mapping")
        configured[name] = tuple(group_value["joint_names"])
    if configured != dict(expected_joint_groups):
        raise ValueError(
            "Environment reset joint groups do not match the embodiment contract"
        )
    groups["arm"]["range"] = _symmetric(perturbation.arm_joint_noise_rad)
    groups["finger"]["range"] = _symmetric(perturbation.finger_joint_noise_rad)
    params["object_xy_noise_range"] = _symmetric(perturbation.object_xy_noise_m)
    params["object_yaw_noise_range"] = _symmetric(perturbation.object_yaw_noise_rad)


def _fingerprint_initial(record: dict[str, np.ndarray], env_index: int) -> str:
    digest = hashlib.sha256()
    for key in ("arm_joint_pos", "finger_joint_pos", "object_position_e"):
        value = np.ascontiguousarray(record[key][env_index])
        digest.update(key.encode())
        digest.update(value.dtype.str.encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _terminal_info(
    env_index: int, outcome: SuccessResult, result: NativeBatchStep
) -> dict[str, JsonValue]:
    """Metrics, reset diagnostics and termination reasons of a completed episode."""
    info: dict[str, JsonValue] = {"task_metrics": outcome.task_metrics()}
    if result.reset_diagnostics:
        info["episode_diagnostics"] = {
            key: float(values[env_index])
            for key, values in result.reset_diagnostics.items()
        }
    termination_reasons = list(result.reasons[env_index])
    info["termination_reasons"] = termination_reasons
    if termination_reasons:
        info["terminal_reason"] = "+".join(termination_reasons)
    elif outcome.success:
        info["terminal_reason"] = "success"
    else:
        info["terminal_reason"] = "task_horizon"
    return info


def evaluation_case_seed(
    base_seed: int, round_index: int, environment_index: int
) -> int:
    """A stable, distinct reset stream for one evaluation case."""
    sequence = np.random.SeedSequence([base_seed, round_index, environment_index])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


class TissueBoxCapability:
    """Server-side full-episode controller for scalar and batched V2D."""

    def __init__(
        self,
        runtime: IsaacRuntime,
        tracker: CombinedSuccessTracker,
        initial_record: dict[str, np.ndarray],
        *,
        task: TaskConfig,
        prompt: str,
        default_seed: int,
        receipt: dict[str, JsonValue],
    ) -> None:
        self.runtime = runtime
        self.tracker = tracker
        self.record = initial_record
        self.task = task
        self.prompt = prompt
        self.default_seed = default_seed
        self.receipt = receipt
        self.num_envs = len(next(iter(initial_record.values())))
        self.steps: NDArray[np.int64] = np.zeros(self.num_envs, dtype=np.int64)
        self.active: NDArray[np.bool_] = np.zeros(self.num_envs, dtype=np.bool_)
        self.initial_fingerprints = ["" for _ in range(self.num_envs)]
        self.case_seeds: list[int | None] = [None] * self.num_envs
        self.round_index = -1

    def spec(self) -> dict[str, object]:
        observation = batched_observation_from_record(self.record)
        return {
            "protocol": PROTOCOL_VERSION,
            "device": str(self.runtime.env.device),
            "task_id": self.task.task_id,
            "episode_horizon": self.task.episode_horizon,
            "action_dim": JOINT_DIM,
            "num_envs": self.num_envs,
            "prompt": self.prompt,
            "image_shapes": {
                key: list(observation[key].shape[1:]) for key in IMAGE_KEYS
            },
            "specs": spec_payload(v2d_observation_spec(), v2d_action_spec()),
            "evaluator": self.task.success_definition.evaluator,
            "success_definition_sha256": self.task.success_definition.sha256,
            "receipt": self.receipt,
        }

    def reset(self, seed: int | None, env_ids: list[int] | None) -> Payload:
        selected = self._select_envs(seed, env_ids)
        self._reset_selected(selected, seed)
        for env_index in selected:
            self.tracker.complete(int(env_index), next_initial_record=self.record)
            self.initial_fingerprints[env_index] = _fingerprint_initial(
                self.record, int(env_index)
            )
        self.steps[selected] = 0
        self.active[selected] = True
        observation = batched_observation_from_record(self.record)
        if self.num_envs == 1:
            # The scalar client expects unbatched observations, as step() returns.
            observation = {key: value[0] for key, value in observation.items()}
        return {
            **observation,
            "info": {
                "round_index": self.round_index,
                "case_seed": list(self.case_seeds),
                "initial_state_fingerprint": list(self.initial_fingerprints),
            },
        }

    def _select_envs(
        self, seed: int | None, env_ids: list[int] | None
    ) -> NDArray[np.int64]:
        """The environments to reset; a full reset also advances the evaluation round."""
        if env_ids is None:
            selected: NDArray[np.int64] = np.arange(self.num_envs, dtype=np.int64)
            if seed is None:
                self.round_index += 1
            else:
                # The rollout worker advances an explicit seed for each round.
                # Treat that seed as the complete case identity so repeating an
                # evaluation replays the same cases on a persistent server.
                self.round_index = 0
        else:
            selected = np.asarray(env_ids, dtype=np.int64)
            if (
                selected.ndim != 1
                or selected.size == 0
                or len(np.unique(selected)) != len(selected)
                or (selected < 0).any()
                or (selected >= self.num_envs).any()
            ):
                raise ValueError(
                    "env_ids must contain unique valid environment indices"
                )
        return selected

    def _reset_selected(self, selected: NDArray[np.int64], seed: int | None) -> None:
        if seed is None:
            # Inactive slots already hold startup or simulator autoreset states.
            # Only interrupted episodes need another physical reset.
            reset_ids = selected[self.active[selected]]
            if reset_ids.size:
                self.record = self.runtime.reset_envs(reset_ids.tolist(), seed=None)
            for env_index in selected:
                self.case_seeds[env_index] = None
        else:
            cases = []
            for env_index in selected:
                case_seed = evaluation_case_seed(seed, self.round_index, int(env_index))
                cases.append((int(env_index), case_seed))
                self.case_seeds[env_index] = case_seed
            self.record = self.runtime.reset_cases(cases)

    def _step_batch(self, actions: np.ndarray, *, need_obs: bool) -> Payload:
        if not self.active.any():
            raise RuntimeError("RESET required before STEP")
        active_before = self.active.copy()
        self.tracker.sample_pre_action(self.record, active_before)
        result = self.runtime.step_batch(actions)
        self.tracker.update_post_action(result.terminal_record)
        self.record = result.record
        counts: NDArray[np.int64] = active_before.astype(np.int64)
        self.steps += counts
        completed = np.asarray(result.completed, dtype=np.bool_) & active_before
        successes: NDArray[np.bool_] = np.zeros(self.num_envs, dtype=np.bool_)
        infos: list[dict[str, JsonValue]] = []
        for env_index in range(self.num_envs):
            info: dict[str, JsonValue] = {
                "episode_steps": int(self.steps[env_index]),
                "initial_state_fingerprint": self.initial_fingerprints[env_index],
                "case_seed": self.case_seeds[env_index],
                "round_index": self.round_index,
            }
            if completed[env_index]:
                outcome = self.tracker.complete(
                    env_index, next_initial_record=self.record
                )
                successes[env_index] = outcome.success
                self.active[env_index] = False
                info.update(_terminal_info(env_index, outcome, result))
            elif (
                active_before[env_index]
                and self.steps[env_index] >= self.task.episode_horizon
            ):
                raise RuntimeError(
                    f"Simulator failed to terminate environment {env_index} at step "
                    f"{self.task.episode_horizon}"
                )
            infos.append(info)
        observation = batched_observation_from_record(self.record)
        if not need_obs:
            for key in IMAGE_KEYS:
                del observation[key]
        return {
            **observation,
            "reward": successes.astype(np.float32),
            "done": completed & successes,
            "truncated": completed & ~successes,
            "success": successes,
            "num_steps": counts,
            "info": infos,
        }

    def step(self, action: object, *, need_obs: bool) -> Payload:
        values = np.asarray(action, dtype=np.float32)
        if self.num_envs > 1:
            if values.shape != (self.num_envs, JOINT_DIM):
                raise ValueError(
                    f"Batched actions must have shape ({self.num_envs}, {JOINT_DIM})"
                )
            return self._step_batch(values, need_obs=need_obs)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != JOINT_DIM or len(values) == 0:
            raise ValueError("Scalar V2D actions must be [58] or [chunk,58]")
        executed = 0
        response = None
        for row in values:
            response = self._step_batch(row[None, :], need_obs=True)
            executed += 1
            if bool(response["done"][0]) or bool(response["truncated"][0]):
                break
        assert response is not None
        scalar: Payload = {}
        for key, value in response.items():
            if isinstance(value, np.ndarray) or key == "info":
                scalar[key] = value[0]
            else:
                scalar[key] = value
        scalar["num_steps"] = executed
        return scalar

    def close(self) -> None:
        self.runtime.close()


def _configure_visual_events(events: object, task: TaskConfig) -> None:
    from robotic_grounding.rendering.dr.controller import (
        disable_visual_event_terms,
        set_visual_event_mode,
    )

    if task.visual_mode == "off":
        disable_visual_event_terms(events)
    elif task.visual_mode == "training_matched":
        visual_event_mode = "startup" if task.fixed_reset else "reset"
        set_visual_event_mode(events, visual_event_mode, exclude_suffixes=("_startup",))
    else:
        raise ValueError(f"Unsupported visual mode {task.visual_mode!r}")


def _validate_env(
    env: ManagerBasedRLEnv, task: TaskConfig, terminations: Sequence[str]
) -> CollisionValidatedReset | None:
    """Close and reject an environment that differs from the task; return its collision reset."""
    collision_reset = None
    if task.reset_validation is not None and task.reset_validation.enabled:
        from example_workflows.v2d_box_lift_posttraining.collision_reset import (
            CollisionValidatedReset,
        )

        collision_reset = env.event_manager.get_term_cfg(
            "reset_to_trajectory_frame"
        ).func
        if not isinstance(collision_reset, CollisionValidatedReset):
            env.close()
            raise TypeError("Collision reset event was not installed")
    if env.max_episode_length != task.episode_horizon:
        env.close()
        raise ValueError("Resolved task horizon does not match configuration")
    if set(env.termination_manager.active_terms) != set(terminations):
        env.close()
        raise ValueError("Resolved termination terms differ from released contract")
    return collision_reset


def _build_runtime(
    launch_config: Path, device: str, seed: int, num_envs: int
) -> tuple[IsaacRuntime, TaskConfig, str, int, dict[str, JsonValue]]:
    config = load_capability_launch(launch_config)
    revision = subprocess.check_output(
        ["git", "-C", str(config.v2d_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != config.expected_v2d_revision:
        raise ValueError(
            f"Expected V2D revision {config.expected_v2d_revision}, got {revision}"
        )
    sys.path.insert(0, str(config.v2d_root / "robotic_grounding"))
    sys.path.insert(
        0, str(config.v2d_root / "robotic_grounding/source/robotic_grounding")
    )
    import gymnasium as gym
    import robotic_grounding.tasks  # noqa: F401
    from groot_finetune.contracts import load_embodiment_contract
    from groot_finetune.source_rollout import termination_reasons_for_env
    from groot_finetune.task_profile import load_task_profile
    from isaaclab.envs import mdp
    from isaaclab.managers import TerminationTermCfg
    from isaaclab_tasks.utils import parse_env_cfg
    from robotic_grounding.tasks.scene_utils import SceneConfig, apply_scene_config

    task = load_task(config.task_config)
    contract = load_embodiment_contract(task.embodiment)
    profile = load_task_profile(
        config.v2d_root / "robotic_grounding" / task.task_profile
    )
    if task.gym_task != contract.inference_task or task.control_hz != contract.fps:
        raise ValueError("Task and embodiment runtime contracts differ")
    cfg = parse_env_cfg(task.gym_task, device=device, num_envs=num_envs)
    _apply_reset_perturbation(
        cfg.events.reset_to_trajectory_frame.params,
        contract.reset_joint_groups,
        task.reset_perturbation,
    )
    cfg.motion_file = str(config.motion_file)
    scene = SceneConfig.from_motion_file(str(config.motion_file))
    if scene.robot_name != "vega_sharpa" or scene.sequence_id is None:
        raise ValueError("Require a partitioned Vega Sharpa motion scene")
    supports = [item for item in scene.fixed_objects if item.name == "support_surface"]
    if len(supports) != 1:
        raise ValueError("Require exactly one discovered support surface")
    target_index = profile.target_object.resolve(
        tuple(item.name for item in scene.scene_objects)
    )
    apply_scene_config(cfg, scene, use_primitive_urdfs=task.use_primitive_urdfs)
    control_dt = float(cfg.decimation) * float(cfg.sim.dt)
    if not np.isclose(control_dt, 1 / task.control_hz):
        raise ValueError("Simulator control frequency differs from the task")
    cfg.episode_length_s = task.episode_horizon * control_dt

    def capture_timeout(env):
        command = env.command_manager.get_term("motion")
        positions = command.object_position_e
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise RuntimeError("Tracking command exposes invalid object positions")
        env.extras[TERMINAL_CAPTURE_KEY] = {
            "object_position_e": positions.reshape(positions.shape[0], -1).clone()
        }
        return mdp.time_out(env)

    cfg.terminations.timeout = TerminationTermCfg(func=capture_timeout, time_out=True)
    command = cfg.commands.motion
    command.always_reset_to_first_frame = True
    command.reset_to_first_frame_prob = 1.0
    command.reset_finger_openness = task.reset_finger_openness
    command.reset_freeze_steps = 0
    command.initial_virtual_object_control_curriculum_scale = 0.0
    command.voc_reset_scale = 0.0
    command.voc_decay_steps = 0
    command.debug_vis = False
    cfg.curriculum = None
    _configure_visual_events(cfg.events, task)
    cfg.num_rerenders_on_reset = max(1, cfg.num_rerenders_on_reset)
    _configure_trajectory_reset(cfg.events.reset_to_trajectory_frame, task, seed)
    cfg.seed = seed
    env = gym.make(task.gym_task, cfg=cfg).unwrapped
    collision_reset = _validate_env(env, task, contract.evaluation_terminations)
    receipt: dict[str, JsonValue] = {
        "v2d_revision": revision,
        "task_sha256": task.sha256,
        "gym_task": task.gym_task,
        "num_envs": num_envs,
        "motion_file": str(config.motion_file.resolve()),
        "sequence_id": scene.sequence_id,
        "target_object": scene.scene_objects[target_index].name,
        "episode_horizon": task.episode_horizon,
        "execution_length": task.execution_length,
        "reset_perturbation": asdict(task.reset_perturbation),
        "fixed_reset": task.fixed_reset,
        "fixed_reset_seed": seed if task.fixed_reset else None,
        "reset_validation": None
        if task.reset_validation is None
        else task.reset_validation.as_dict(),
        "evaluator": task.success_definition.evaluator,
        "success_definition_sha256": task.success_definition.sha256,
        "metric_sample_timing": "success_pre_action_and_final_height_diagnostic_post_action_pre_autoreset",
    }

    def reasons(environment: ManagerBasedRLEnv, index: int) -> tuple[str, ...]:
        return tuple(
            termination_reasons_for_env(environment.termination_manager, index)
        )

    return (
        IsaacRuntime(
            env,
            render_ticks=task.visual_warmup_steps,
            termination_reasons=reasons,
            texture_resets=coalesce_texture_resets(env),
            collision_reset=collision_reset,
        ),
        task,
        profile.instruction,
        target_index,
        receipt,
    )


def create_capability(
    launch_config: Path, device: str, seed: int, num_envs: int
) -> TissueBoxCapability:
    """Capability factory loaded by :mod:`simulators.v2d.server`."""
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    runtime, task, prompt, target_index, receipt = _build_runtime(
        launch_config, device, seed, num_envs
    )
    try:
        initial = runtime.reset_envs(None, seed=seed)
    except Exception:
        runtime.close()
        raise
    receipt["initial_reset_diagnostics"] = {
        key: values.tolist() for key, values in runtime.reset_diagnostics().items()
    }
    from groot_finetune.closed_loop.evaluation import LiftHoldTracker
    from groot_finetune.task_profile import load_task_profile

    config = load_capability_launch(launch_config)
    profile = load_task_profile(
        config.v2d_root / "robotic_grounding" / task.task_profile
    )
    lift_tracker = LiftHoldTracker(
        object_height(initial, target_index), config=profile.evaluator
    )
    tracker = CombinedSuccessTracker(
        initial,
        object_index=target_index,
        definition=task.success_definition,
        lift_tracker=lift_tracker,
    )
    return TissueBoxCapability(
        runtime,
        tracker,
        initial,
        task=task,
        prompt=prompt,
        default_seed=seed,
        receipt=receipt,
    )
