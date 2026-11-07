# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the provider-neutral reference-edit policy used by rollout runtimes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from isaac_rl_posttraining.algorithms.sac_edit import (
    ReferenceEditActor,
    ReferenceEditActorConfig,
)
from isaac_rl_posttraining.algorithms.sac_edit.input_normalizer import (
    RunningMeanStdNormalizer,
)
from isaac_rl_posttraining.envs.reference_action import (
    environment_actions_from_learner,
    reference_observation_entries,
)
from isaac_rl_posttraining.runtimes import (
    GymRolloutWorker,
    RolloutRequest,
    RolloutWorkerContext,
    SparseOutcomeReward,
)
from isaac_rl_posttraining.runtimes.edit_policy import EditActionChunkPolicy
from isaac_rl_posttraining.runtimes.policy_snapshot import policy_state_from_tensors

HORIZON, ACTION_DIM, ENV_ACTION_DIM, STATE_DIM, FEAT_DIM = 3, 2, 4, 4, 2
#: The learner controls two of the four environment dimensions.
ACTION_INDICES = (1, 3)
DEFAULTS = np.asarray([0.5, 0.0, -0.5, 0.0], dtype=np.float32)
EDIT_SCALE = 0.02
CPU = torch.device("cpu")


class _FrozenReference:
    """Frozen-policy double: a state-dependent reference chunk and pooled features, counting decodes."""

    def __init__(self) -> None:
        self.policy = SimpleNamespace(model=SimpleNamespace(feat_dim=FEAT_DIM))
        self.calls = 0
        self.modes: list[str] = []

    def infer_reference(
        self, observation: dict[str, Any], *, image_transform_mode: str = "eval"
    ):
        self.calls += 1
        self.modes.append(image_transform_mode)
        step = float(np.asarray(observation["state"])[0, 0])
        return torch.full((1, 2 * FEAT_DIM), step), _raw_reference(step)


def _raw_reference(step: float) -> np.ndarray:
    """The frozen chunk ``[1, HORIZON, ENV_ACTION_DIM]``: row ``h`` holds ``0.1 (h + 1) + 0.01 step``."""
    reference = np.zeros((1, HORIZON, ENV_ACTION_DIM), dtype=np.float32)
    for row in range(HORIZON):
        reference[0, row, :] = 0.1 * (row + 1) + 0.01 * step
    return reference


def _expected_reference(step: float) -> np.ndarray:
    """The controlled dimensions of the frozen chunk, ``[HORIZON, ACTION_DIM]``."""
    return _raw_reference(step)[0][:, list(ACTION_INDICES)]


def _observation(step: float) -> dict[str, np.ndarray]:
    return {
        "state": np.asarray([step, 0.5, -0.5, 1.0], dtype=np.float32),
        "ego_view_image": np.zeros((4, 4, 3), dtype=np.uint8),
    }


def _actor(*, include_state: bool = False) -> ReferenceEditActor:
    spaces = {
        "vlm_features": gym.spaces.Box(-np.inf, np.inf, (2 * FEAT_DIM,), np.float32),
        "reference_action": gym.spaces.Box(
            -1.0, 1.0, (HORIZON, ACTION_DIM), np.float32
        ),
    }
    if include_state:
        spaces["state"] = gym.spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32)
    return ReferenceEditActor(
        ReferenceEditActorConfig(
            observation_space=gym.spaces.Dict(spaces),
            action_space=gym.spaces.Box(-1.0, 1.0, (HORIZON, ACTION_DIM), np.float32),
            remove_obs_keys=[] if include_state else ["state"],
            simnorm_dim=4,
            simnorm_groups=2,
            reference_action_key="reference_action",
            edit_scale=EDIT_SCALE,
            hidden_dims=(8, 8),
            use_tanh_output=True,
            deterministic=False,
            log_std_init=0.0,
            log_std_min=-20.0,
            log_std_max=2.0,
        )
    )


def _policy(
    base_model: _FrozenReference, actor: ReferenceEditActor, **overrides: Any
) -> EditActionChunkPolicy:
    return EditActionChunkPolicy(
        actor=actor,
        base_model=base_model,
        device=CPU,
        action_indices=ACTION_INDICES,
        environment_action_defaults=DEFAULTS,
        action_horizon=HORIZON,
        reference_action_transform="identity",
        relative_action_scale=0.025,
        reference_image_transform_mode="train",
        state_key="state",
        image_keys=("ego_view_image",),
        **overrides,
    )


class _ScriptedEnv(gym.Env):
    """Succeeds after ``terminal_after`` environment steps; accepts one command or a whole chunk."""

    def __init__(self, terminal_after: int = 5) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32),
                "ego_view_image": gym.spaces.Box(0, 255, (4, 4, 3), np.uint8),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ENV_ACTION_DIM,), np.float32)
        self.steering_remove_obs_keys = ["ego_view_image"]
        self.terminal_after = terminal_after
        self.commands: list[np.ndarray] = []
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._t = 0
        return _observation(0.0), {}

    def step(self, action: np.ndarray):
        executed = 0
        terminated = False
        for command in np.asarray(action, dtype=np.float32).reshape(-1, ENV_ACTION_DIM):
            self.commands.append(command.copy())
            self._t += 1
            executed += 1
            terminated = self._t >= self.terminal_after
            if terminated:
                break
        info = {"success": terminated, "is_success": terminated, "num_steps": executed}
        return _observation(float(self._t)), 0.0, terminated, False, info

    def close(self) -> None:
        # Test double: holds no resources to release.
        pass


def test_warmup_executes_the_reference_and_records_it_as_native() -> None:
    base_model = _FrozenReference()
    policy = _policy(base_model, _actor())

    chunk = policy.action_chunk(_observation(2.0), mode="warmup")

    reference = _expected_reference(2.0)
    assert chunk.source == "native"
    assert chunk.execution_length == HORIZON
    np.testing.assert_allclose(chunk.learner_action[0], reference, atol=1e-6)
    environment_actions = chunk.environment_actions[0]
    assert environment_actions.shape == (HORIZON, ENV_ACTION_DIM)
    np.testing.assert_allclose(environment_actions[:, list(ACTION_INDICES)], reference)
    np.testing.assert_allclose(
        environment_actions[:, [0, 2]], np.tile(DEFAULTS[[0, 2]], (HORIZON, 1))
    )
    assert set(chunk.learner_observation) == {
        "state",
        "vlm_features",
        "reference_action",
        "action_mask",
    }
    np.testing.assert_array_equal(
        chunk.learner_observation["action_mask"], np.ones((1, HORIZON, 1))
    )
    assert chunk.learner_observation["vlm_features"].shape == (1, 2 * FEAT_DIM)
    np.testing.assert_allclose(
        chunk.learner_observation["reference_action"][0], reference
    )
    assert base_model.calls == 1 and base_model.modes == ["train"]


def test_train_decisions_edit_the_reference_within_the_actor_bound() -> None:
    torch.manual_seed(0)
    actor = _actor()
    chunk = _policy(_FrozenReference(), actor).action_chunk(
        _observation(1.0), mode="train"
    )

    assert chunk.source == "policy"
    reference = chunk.learner_observation["reference_action"][0]
    assert np.all(np.abs(chunk.learner_action[0] - reference) <= EDIT_SCALE + 1e-6)
    np.testing.assert_allclose(
        chunk.environment_actions[0][:, list(ACTION_INDICES)], chunk.learner_action[0]
    )


def test_evaluation_samples_the_actor_unless_deterministic() -> None:
    torch.manual_seed(0)
    actor = _actor()
    sampled = _policy(_FrozenReference(), actor)
    deterministic = _policy(_FrozenReference(), actor, deterministic_evaluation=True)
    observation = _observation(1.0)

    first = deterministic.action_chunk(observation, mode="eval")
    second = deterministic.action_chunk(observation, mode="eval")
    drawn = sampled.action_chunk(observation, mode="eval")

    assert first.source == second.source == drawn.source == "policy"
    np.testing.assert_array_equal(first.learner_action, second.learner_action)
    # A freshly initialized edit actor has a zero mean edit, so its mode is the reference.
    np.testing.assert_allclose(
        first.learner_action[0], _expected_reference(1.0), atol=1e-6
    )
    assert not np.array_equal(drawn.learner_action, first.learner_action)


def test_prepared_successor_is_the_next_decisions_decode_until_reset() -> None:
    base_model = _FrozenReference()
    policy = _policy(base_model, _actor())

    policy.action_chunk(_observation(0.0), mode="warmup")
    prepared = policy.prepare_observation(_observation(1.0))
    reused = policy.action_chunk(_observation(1.0), mode="train")

    assert base_model.calls == 2
    assert set(prepared) == {"state", "vlm_features", "reference_action", "action_mask"}
    np.testing.assert_array_equal(
        prepared["reference_action"], reused.learner_observation["reference_action"][0]
    )

    policy.prepare_observation(_observation(2.0))
    policy.reset_episode()
    fresh = policy.action_chunk(_observation(3.0), mode="warmup")

    assert base_model.calls == 4
    np.testing.assert_allclose(
        fresh.learner_action[0], _expected_reference(3.0), atol=1e-6
    )


def test_warmup_collection_decodes_once_per_observation() -> None:
    """A warm-up episode executes the frozen reference whole, decoding once per observation."""
    actor = _actor()
    base_model = _FrozenReference()
    worker = GymRolloutWorker(
        RolloutWorkerContext(
            worker_index=0,
            num_rollout_workers=1,
            environment_indices=(0,),
            device="cpu",
        ),
        _ScriptedEnv(),
        _policy(base_model, actor),
        reward_transform=SparseOutcomeReward(
            success_reward=0.0, failure_reward=-1.0, step_reward=0.0
        ),
    )
    result = worker.collect(RolloutRequest(episodes_per_environment=1, mode="warmup"))

    assert base_model.calls == 3  # the reset plus one per decision
    assert len(result.decisions) == 2
    for decision in result.decisions:
        transition = decision.transition
        assert set(transition.observation.arrays()) == {
            "state",
            "vlm_features",
            "reference_action",
            "action_mask",
        }
        assert decision.action_chunk.source == "native"
        # The stored action is the executed chunk: the frozen reference itself during warm-up.
        np.testing.assert_allclose(
            transition.learner_action.value,
            _expected_reference(float(transition.observation.arrays()["state"][0])),
            atol=1e-6,
        )
    assert [d.transition.steps for d in result.decisions] == [HORIZON, 2]
    last = result.decisions[-1].transition
    assert last.terminated[-1]
    final_reward = last.decision_reward("sparse_outcome", aggregation="last")
    assert final_reward == pytest.approx(0.0)


def test_native_policy_evaluates_the_frozen_reference_only() -> None:
    base_model = _FrozenReference()
    policy = _policy(base_model, _actor(), native=True)

    chunk = policy.action_chunk(_observation(1.0), mode="eval")

    assert chunk.source == "native"
    np.testing.assert_allclose(
        chunk.learner_action[0], _expected_reference(1.0), atol=1e-6
    )
    for mode in ("train", "warmup"):
        with pytest.raises(ValueError, match="native"):
            policy.action_chunk(_observation(1.0), mode=mode)


def test_policy_state_carries_the_input_normalizer_buffers() -> None:
    learner_actor = _actor(include_state=True)
    learner_actor.input_normalizer = RunningMeanStdNormalizer(STATE_DIM)
    learner_actor.input_normalizer_key = "state"
    learner_actor.input_normalizer.update(
        torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 2.0, 1.0, 0.0]])
    )
    rollout_actor = _actor(include_state=True)
    rollout_actor.input_normalizer = RunningMeanStdNormalizer(STATE_DIM)
    rollout_actor.input_normalizer_key = "state"
    policy = _policy(_FrozenReference(), rollout_actor)

    assert (
        policy.set_policy_state(
            policy_state_from_tensors(learner_actor.state_dict(), version=1)
        )
        == 1
    )

    torch.testing.assert_close(
        rollout_actor.input_normalizer.running_mean,
        learner_actor.input_normalizer.running_mean,
    )
    torch.testing.assert_close(
        rollout_actor.input_normalizer.count, learner_actor.input_normalizer.count
    )
    with pytest.raises(ValueError, match="must increase"):
        policy.set_policy_state(
            policy_state_from_tensors(learner_actor.state_dict(), version=1)
        )


def test_environment_actions_keep_defaults_outside_the_controlled_dimensions() -> None:
    learner_action = np.asarray([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)

    environment_actions = environment_actions_from_learner(
        learner_action,
        action_indices=ACTION_INDICES,
        environment_action_defaults=DEFAULTS,
    )

    np.testing.assert_allclose(
        environment_actions, [[[0.5, 0.1, -0.5, 0.2], [0.5, 0.3, -0.5, 0.4]]]
    )
    with pytest.raises(ValueError, match="learner action width"):
        environment_actions_from_learner(
            np.zeros((1, 2, 3), np.float32),
            action_indices=ACTION_INDICES,
            environment_action_defaults=DEFAULTS,
        )


def test_reference_entries_make_targets_relative_to_the_previous_one() -> None:
    reference = np.asarray(
        [[[1.0, 0.0, 2.0, 0.0], [1.0, 0.0, 2.5, 0.0]]], dtype=np.float32
    )
    state = np.asarray([[0.0, 0.0, 1.5, 0.0]], dtype=np.float32)

    entries = reference_observation_entries(
        torch.zeros(1, 2 * FEAT_DIM),
        reference,
        state,
        action_indices=(0, 2),
        action_horizon=2,
        reference_action_transform="absolute_to_relative",
        relative_action_scale=0.5,
    )

    # Row 0 moves from the state (1.0 - 0.0)/0.5 -> clipped to 1, (2.0 - 1.5)/0.5 = 1;
    # row 1 moves from row 0: 0 and (2.5 - 2.0)/0.5 = 1.
    np.testing.assert_allclose(entries["reference_action"], [[[1.0, 1.0], [0.0, 1.0]]])
    assert entries["vlm_features"].shape == (1, 2 * FEAT_DIM)
    np.testing.assert_array_equal(entries["action_mask"], np.ones((1, 2, 1)))
    with pytest.raises(ValueError, match="does not cover action_indices"):
        reference_observation_entries(
            torch.zeros(1, 2),
            reference,
            state,
            action_indices=(0, 4),
            action_horizon=2,
            reference_action_transform="identity",
            relative_action_scale=0.5,
        )


class _BatchedScriptedEnv:
    """Two lockstep copies of the scripted task; the first succeeds after four steps."""

    num_envs = 2

    def __init__(self) -> None:
        self.t = 0

    def _observations(self) -> dict[str, np.ndarray]:
        return {
            "state": np.stack([_observation(float(self.t))["state"]] * 2),
            "ego_view_image": np.zeros((2, 4, 4, 3), dtype=np.uint8),
        }

    def reset(self, *, seed: int | None = None):
        del seed
        self.t = 0
        return self._observations(), {}

    def step(self, actions: np.ndarray):
        assert actions.shape == (2, ENV_ACTION_DIM)
        self.t += 1
        terminated = np.asarray([self.t >= 4, False])
        infos = [{"success": bool(done)} for done in terminated]
        return self._observations(), np.zeros(2), terminated, np.zeros(2, bool), infos

    def close(self) -> None:
        # Test double: holds no resources to release.
        pass


def test_sac_edit_runs_on_a_batched_worker_with_the_default_bootstrap_values() -> None:
    """The edit actor has no value head; a batched worker only encodes its next observations."""
    from entry_doubles import ScriptedSimulator, scripted_environment

    from isaac_rl_posttraining.runtimes import BatchedWorkerSpec

    torch.manual_seed(0)
    base_model = _FrozenReference()
    spec = BatchedWorkerSpec(
        entry=ScriptedSimulator(lambda context: _BatchedScriptedEnv()),
        environment=scripted_environment(2),
        phase="train",
        policy=lambda context: _policy(base_model, _actor()),
        chunks_per_episode=3,
    )
    assert spec.bootstrap_values == "evaluate_next_observation"
    worker = spec(
        RolloutWorkerContext(
            worker_index=0,
            num_rollout_workers=1,
            environment_indices=(0, 1),
            device="cpu",
        )
    )
    result = worker.collect(RolloutRequest(episodes_per_environment=1, mode="train"))
    # Environment 0 succeeds during its second chunk; environment 1 runs all three.
    assert [decision.environment_index for decision in result.decisions] == [
        0,
        1,
        0,
        1,
        1,
    ]
    for decision in result.decisions:
        transition = decision.transition
        assert transition.policy_outputs.next_value is None
        assert "reference_action" in transition.next_observation.arrays()
    assert result.timings["next_observation_inference_seconds"] > 0.0
    worker.close()
