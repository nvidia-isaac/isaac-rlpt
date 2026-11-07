# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunk-level transition assembly in the steering rollout worker.

DSRL trains on CHUNK transitions, not raw env steps: the SAC actor emits one
steering latent, the frozen policy decodes an action chunk, the worker
executes the first ``action_exec_len`` env steps and stores ONE transition
(obs at decode time, the latent as the action, the chunk-terminal reward,
next obs after the chunk). These tests pin that assembly on the
``GymRolloutWorker`` driving the steering policy:

* an N-step episode with exec length K yields ceil(N/K) transitions, whose
  stored action is the latent (not a decoded env action),
* within-chunk reward aggregation keeps the LAST executed step's reward (the
  sparse terminal-outcome design: intermediate rewards of a chunk are not
  summed; the chunk-terminal step carries the signal),
* the terminal success flag propagates into the stored transition and the
  scored episode,
* termination and truncation propagate distinctly, including a truncation
  that cuts a chunk short mid-execution.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import (
    ARENA_STEERING_KEY_MAPPING,
    MockEnv,
    MockPolicy,
    StubBasePolicy,
    UnbatchedMockEnv,
)

from isaac_rl_posttraining.runtimes import (
    GymRolloutWorker,
    RolloutRequest,
    RolloutWorkerContext,
    SteeringActionChunkPolicy,
)
from isaac_rl_posttraining.training import episode_outcomes

ACTION_DIM = 35
EXEC_LEN = 4


class _LatentSpaceStub:
    """Space-only stub for the latent action space plumbing (no VLM features)."""

    use_vlm_features = False

    def sample_action(self):
        return np.zeros((1, ACTION_DIM), dtype=np.float32)


def _collect_one_episode(env, latent_constant=0.5):
    """Run one training episode of ``env`` through the worker; returns (result, base model, actor)."""
    base_model = StubBasePolicy(horizon=40, action_dim=ACTION_DIM)
    actor = MockPolicy(action_shape=(1, ACTION_DIM), constant=latent_constant)
    policy = SteeringActionChunkPolicy(
        actor=actor,
        base_model=base_model,
        latent_sampler=_LatentSpaceStub(),
        device=torch.device("cpu"),
        steering_key_mapping=dict(ARENA_STEERING_KEY_MAPPING),
        action_exec_len=EXEC_LEN,
    )
    worker = GymRolloutWorker(
        RolloutWorkerContext(
            worker_index=0,
            num_rollout_workers=1,
            environment_indices=(0,),
            device="cpu",
        ),
        UnbatchedMockEnv(env),
        policy,
    )
    result = worker.collect(RolloutRequest(episodes_per_environment=1, mode="train"))
    return result, base_model, actor


def _transitions(result):
    return [decision.transition for decision in result.decisions]


def test_chunk_transition_assembly_from_env_steps():
    """An 8-step episode with exec_len=4 stores exactly 2 latent-action transitions."""
    env = MockEnv(
        episode_outcomes=("fail",),
        horizon=2 * EXEC_LEN,
        dino_dim=8,
        action_dim=ACTION_DIM,
        hard_step_limit=100,
        step_rewards=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    result, base_model, _ = _collect_one_episode(env, latent_constant=0.5)
    transitions = _transitions(result)

    assert len(transitions) == 2, (
        f"8 env steps / exec_len 4 must yield 2 chunk transitions, got {len(transitions)}"
    )
    assert env.total_steps == 2 * EXEC_LEN
    assert result.env_steps == 2 * EXEC_LEN, (
        "the result must count raw env steps, not chunks"
    )
    assert len(base_model.noises) == 2, "one decode per chunk"

    for transition in transitions:
        # The stored learner action is the STEERING LATENT; the executed chunk travels next to it.
        assert transition.learner_action.kind == "latent"
        action = np.asarray(transition.learner_action.value)
        np.testing.assert_allclose(
            action.reshape(-1), 0.5, err_msg="stored action must be the latent"
        )
        assert action.reshape(-1).size == ACTION_DIM
        assert transition.env_action.commands.shape == (EXEC_LEN, ACTION_DIM)
        # The stored observation is the SAC-facing dict (state + dino embedding).
        components = set(transition.observation.arrays())
        assert "state" in components, (
            f"chunk transition obs must include the state, got {components}"
        )
        assert transition.episode_id == transitions[0].episode_id
        assert transition.steps == EXEC_LEN

    # Env actions came from the frozen policy's decode (position-coded rows 1..exec_len).
    first_round = [a.reshape(-1)[0] for a in env.actions_received[:EXEC_LEN]]
    np.testing.assert_allclose(
        first_round, np.arange(1, EXEC_LEN + 1, dtype=np.float32)
    )


def test_within_chunk_reward_aggregation():
    """The chunk transition keeps the LAST executed step's reward (sparse-terminal design)."""
    env = MockEnv(
        episode_outcomes=("fail",),
        horizon=2 * EXEC_LEN,
        dino_dim=8,
        action_dim=ACTION_DIM,
        hard_step_limit=100,
        step_rewards=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    result, _, _ = _collect_one_episode(env)

    rewards = [
        transition.decision_reward("env", aggregation="last")
        for transition in _transitions(result)
    ]
    assert rewards == [4.0, 8.0], (
        "each chunk transition must carry its final step's reward "
        f"(not a within-chunk sum); got {rewards}"
    )
    assert [
        tuple(transition.reward("env").tolist()) for transition in _transitions(result)
    ] == [
        (1.0, 2.0, 3.0, 4.0),
        (5.0, 6.0, 7.0, 8.0),
    ], "the per-step rewards of the chunk are what the transition stores"


def test_success_flag_propagation():
    """A terminal success reaches the stored transition and the scored episode."""
    env = MockEnv(
        episode_outcomes=("success",),
        horizon=2 * EXEC_LEN,
        dino_dim=8,
        action_dim=ACTION_DIM,
        hard_step_limit=100,
    )
    result, _, _ = _collect_one_episode(env)
    transitions = _transitions(result)

    assert len(transitions) == 2
    non_terminal, terminal = transitions
    assert not non_terminal.succeeded, (
        "non-terminal chunk transitions must not be flagged successful"
    )
    assert terminal.succeeded, (
        "the terminal transition must carry the success flag for the buffer"
    )
    assert bool(terminal.terminated[-1]) is True and terminal.outcome == "success"
    (outcome,) = episode_outcomes([result])
    assert outcome.success is True and outcome.decisions == 2


def test_termination_vs_truncation_propagation():
    """done and truncated propagate distinctly; truncation may cut a chunk short."""
    # Case 1: natural termination at the chunk boundary.
    env_done = MockEnv(
        episode_outcomes=("fail",),
        horizon=2 * EXEC_LEN,
        dino_dim=8,
        action_dim=ACTION_DIM,
        hard_step_limit=100,
    )
    result_done, _, _ = _collect_one_episode(env_done)
    last_done = _transitions(result_done)[-1]
    assert bool(last_done.terminated[-1]) is True
    assert bool(last_done.truncated[-1]) is False
    (outcome,) = episode_outcomes([result_done])
    assert outcome.success is False

    # Case 2: the episode cap truncates mid-chunk (step 6 of an exec_len-4 schedule).
    env_trunc = MockEnv(
        episode_outcomes=("never",),
        horizon=10**9,
        max_episode_steps=EXEC_LEN + 2,
        dino_dim=8,
        action_dim=ACTION_DIM,
        hard_step_limit=100,
    )
    result_trunc, _, _ = _collect_one_episode(env_trunc)
    transitions = _transitions(result_trunc)

    assert env_trunc.total_steps == EXEC_LEN + 2, (
        "a mid-chunk truncation must stop the chunk immediately "
        f"(expected {EXEC_LEN + 2} env steps, got {env_trunc.total_steps})"
    )
    assert len(transitions) == 2
    last_trunc = transitions[-1]
    assert bool(last_trunc.truncated[-1]) is True
    assert bool(last_trunc.terminated[-1]) is False, (
        "truncation must not masquerade as termination"
    )
    assert last_trunc.steps == 2, "the cut chunk records the steps it actually executed"
    assert (
        last_trunc.env_action.commands.shape[0] == 2
        and last_trunc.env_action.execution_length == EXEC_LEN
    )
    assert result_trunc.env_steps == EXEC_LEN + 2


def test_records_reject_nan_and_inf_in_actions_rewards_and_policy_outputs() -> None:
    import numpy as np
    import pytest
    from conftest import make_transition

    from isaac_rl_posttraining.data import EnvAction, LearnerAction, PolicyOutputs

    with pytest.raises(ValueError, match="LearnerAction.value must be finite"):
        LearnerAction("latent", np.asarray([np.nan, 1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="EnvAction.commands must be finite"):
        EnvAction(
            commands=np.asarray([[np.inf, 0.0]], dtype=np.float32), execution_length=1
        )
    with pytest.raises(ValueError, match="PolicyOutputs.value must be finite"):
        PolicyOutputs(value=np.asarray(np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match=r"rewards\['env'\] must be finite"):
        make_transition(rewards=np.asarray([np.nan, 0.0], dtype=np.float32))
    # Integer and boolean arrays cannot hold NaN and pass untouched.
    EnvAction(commands=np.asarray([[1, 2]], dtype=np.int32), execution_length=1)
