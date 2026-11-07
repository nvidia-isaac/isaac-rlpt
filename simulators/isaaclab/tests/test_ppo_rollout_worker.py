# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The GR00T PPO rollout worker's episode bookkeeping, with the simulator and policy faked.

The worker is the generic batched worker over the Isaac Lab entry's scenes and
the flow-SDE policy, composed as the PPO launch composes it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from isaac_rl_posttraining.envs.simulator_entry import EnvironmentSpec
from isaac_rl_posttraining.policies.gr00t.flow_sde import (
    FlowSDEConfig,
    FlowSDESample,
    Gr00tFlowSDEActorCritic,
    pack_model_inputs,
    unpack_model_inputs,
)
from isaac_rl_posttraining.policies.gr00t.flow_sde_policy import FlowSDEPolicySpec
from isaac_rl_posttraining.policies.trainable import TrainablePolicySpec
from isaac_rl_posttraining.runtimes import (
    BatchedEpisodeRolloutWorker,
    BatchedWorkerSpec,
    CollectedDecision,
    RolloutRequest,
    RolloutWorkerContext,
    SparseOutcomeReward,
    policy_state_from_tensors,
)
from simulators.isaaclab import runtime as isaaclab_runtime

SCENES = 2
WINDOW = 4
CHUNKS = 3
CONTEXT = RolloutWorkerContext(
    worker_index=1, num_rollout_workers=2, environment_indices=(2, 3), device="cpu"
)


class _FakeEnv:
    """Two scenes: scene 0 succeeds during chunk 1, scene 1 never terminates.

    Like the simulator child, ``success`` in the infos is the termination flag.
    """

    # Test double, shared on purpose.
    instances: list[_FakeEnv] = []  # noqa: RUF012
    # The flow family never sizes a learner from the scenes.
    single_observation_space = None
    single_action_space = None

    def __init__(self, spec) -> None:
        self.spec = spec
        self.num_envs = SCENES
        self.seeds: list[int | None] = []
        self.steps = 0
        _FakeEnv.instances.append(self)

    def _obs(self):
        return {
            "video.cam": np.zeros((SCENES, 4, 4, 3), dtype=np.uint8),
            "state.q": np.full((SCENES, 2), float(self.steps), dtype=np.float32),
        }

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.steps = 0
        return self._obs(), {}

    def step(self, actions):
        assert actions.shape == (SCENES, 2)
        self.steps += 1
        terminated = np.zeros(SCENES, dtype=bool)
        # Scene 0 terminates on the last step of its second chunk.
        terminated[0] = self.steps == 2 * WINDOW
        infos = [
            {"success": bool(flag), "is_success": bool(flag), "num_steps": 1}
            for flag in terminated
        ]
        return (
            self._obs(),
            np.zeros(SCENES),
            terminated,
            np.zeros(SCENES, dtype=bool),
            infos,
        )

    def restart(self):
        pass  # the fake has no child process to restart

    def close(self):
        pass  # nothing to release


class _FakePolicy(nn.Module):
    frozen_submodules = ("model.backbone",)
    cached_feature_names = ("backbone_features",)
    requires_cached_features = False
    pack_inputs = staticmethod(pack_model_inputs)
    unpack_inputs = staticmethod(unpack_model_inputs)

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.backbone = nn.Linear(2, 2)
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(False)
        self.head = nn.Linear(2, 2)
        self.video_keys = ["video.cam"]
        self.state_keys = ["state.q"]
        self.action_window = WINDOW
        self.denoising_steps = 4
        self.calls = 0

    def build_inputs(self, frames, states, instruction):
        assert instruction == "stack the cubes"
        batch = frames["video.cam"].shape[0]
        return {
            "state": torch.as_tensor(states["state.q"]),
            "eagle_pixel_values": torch.zeros(batch, 3, 2, 2),
        }

    def learner_inputs(self, inputs):
        return {
            key: value for key, value in inputs.items() if not key.startswith("eagle_")
        }

    def act(self, inputs, *, stochastic_step):
        batch = inputs["state"].shape[0]
        self.calls += 1
        value = torch.full((batch,), float(self.calls))
        return FlowSDESample(
            actions=torch.zeros(batch, WINDOW, 2),
            chains=torch.zeros(batch, self.denoising_steps + 1, WINDOW, 2),
            log_prob=torch.full((batch,), -1.0),
            value=value,
            stochastic_step=stochastic_step,
            cached_features={
                "backbone_features": torch.zeros(batch, 3, 2).to(torch.bfloat16)
            },
        )

    def decode(self, normalized, *, inputs, states):
        return np.zeros((normalized.shape[0], WINDOW, 2), dtype=np.float32)


def _worker_spec() -> BatchedWorkerSpec:
    """The launch's composition: a worker's Isaac Lab scenes, the flow-SDE policy, RLinf's sparse reward."""
    return BatchedWorkerSpec(
        entry=isaaclab_runtime.IsaacLab(),
        environment=EnvironmentSpec(
            env_name="isaaclab/Isaac-Fake-v0",
            num_envs=2 * SCENES,
            max_episode_steps=CHUNKS * WINDOW,
            seed=3,
            action_dim=2,
            image_resolution=4,
        ),
        phase="train",
        scenes=True,
        policy=FlowSDEPolicySpec(
            spec=TrainablePolicySpec(
                base_model="gr00t_n15",
                checkpoint="/ckpt",
                flow_sde=FlowSDEConfig(),
                options={"data_config": "m:C", "embodiment_tag": "tag"},
            ),
            action_window=WINDOW,
            instruction="stack the cubes",
            base_seed=3,
            cache_features=True,
        ),
        chunks_per_episode=CHUNKS,
        reward_transform=SparseOutcomeReward(
            success_reward=1.0, failure_reward=0.0, step_reward=0.0
        ),
        successor_values="next_decision",
        episode_attempts=3,
    )


@pytest.fixture
def fake_simulation(monkeypatch):
    _FakeEnv.instances.clear()
    monkeypatch.setattr(isaaclab_runtime, "SubprocessIsaacLabEnv", _FakeEnv)
    monkeypatch.setattr(
        Gr00tFlowSDEActorCritic,
        "load",
        classmethod(lambda cls, checkpoint, **kwargs: _FakePolicy()),
    )


@pytest.fixture
def worker(fake_simulation) -> BatchedEpisodeRolloutWorker:
    return _worker_spec()(CONTEXT)


def _ordered(decisions) -> list[CollectedDecision]:
    return sorted(
        decisions,
        key=lambda item: (
            item.environment_index,
            item.transition.episode_id,
            item.transition.metadata.step_in_episode,
        ),
    )


def test_worker_emits_complete_episodes_with_seeded_resets_and_successor_values(worker):
    result = worker.collect(RolloutRequest(episodes_per_environment=2, reset_seed=50))
    env = _FakeEnv.instances[0]
    # The environment adds the worker's seed stride; the decisions record the request's seed.
    assert env.seeds == [50 + 1_000_003 + 0, 50 + 1_000_003 + 1]
    # Scene 0 stops after its terminal chunk; scene 1 runs to the time limit.
    per_env = {index: [] for index in (2, 3)}
    for decision in _ordered(result.decisions):
        per_env[decision.environment_index].append(decision)
    assert [len(per_env[2]), len(per_env[3])] == [2 * 2, 3 * 2]
    assert result.episodes_completed == 4
    assert result.env_steps == len(result.decisions) * WINDOW
    assert result.worker_index == 1
    assert {
        "environment_reset_seconds",
        "environment_step_seconds",
        "action_inference_seconds",
    } <= set(result.timings)
    first_episode_scene0 = [
        d
        for d in per_env[2]
        if d.transition.episode_id == per_env[2][0].transition.episode_id
    ]
    last = first_episode_scene0[-1].transition
    last_reward = last.decision_reward("sparse_outcome", aggregation="last")
    assert last.terminated[-1] and not last.truncated[-1]
    assert last_reward == pytest.approx(1.0)
    assert last.succeeded
    assert float(last.policy_outputs.next_value) == pytest.approx(0.0)
    first = first_episode_scene0[0].transition
    first_reward = first.decision_reward("sparse_outcome", aggregation="last")
    assert not first.terminated[-1] and first_reward == pytest.approx(0.0)
    # The successor value is the next decision's value.
    assert float(first.policy_outputs.next_value) == float(last.policy_outputs.value)
    scene1_episode = [
        d
        for d in per_env[3]
        if d.transition.episode_id == per_env[3][0].transition.episode_id
    ]
    assert [bool(d.transition.truncated[-1]) for d in scene1_episode] == [
        False,
        False,
        True,
    ]
    assert all(
        d.transition.metadata.reset_seed_scope == "worker_batch"
        for d in result.decisions
    )
    assert all(d.transition.metadata.reset_seed == 50 for d in first_episode_scene0)
    assert all(
        d.transition.policy_outputs.features is not None for d in result.decisions
    )
    components = first.observation.arrays()
    assert "eagle_pixel_values_bf16" not in components and "state" in components
    assert first.learner_action.value.shape == (5, WINDOW, 2) and first.steps == WINDOW
    assert all(d.action_chunk.policy_version == 0 for d in result.decisions)


def test_worker_rejects_decision_budgets_and_unseeded_resets_pass_none(worker):
    with pytest.raises(ValueError, match="successor_values='policy'"):
        worker.collect(RolloutRequest(decisions_per_environment=4))
    worker.collect(RolloutRequest(episodes_per_environment=1, mode="eval"))
    assert _FakeEnv.instances[0].seeds == [None]


def test_worker_applies_published_weights_without_frozen_keys(worker):
    policy = worker.policy.actor_critic
    new_head = {name: value + 1.0 for name, value in policy.head.state_dict().items()}
    state = policy_state_from_tensors(
        {f"head.{name}": value.to(torch.bfloat16) for name, value in new_head.items()},
        version=1,
    )
    assert worker.set_policy_state(state) == 1
    assert torch.allclose(
        policy.head.weight.detach(),
        new_head["weight"].to(torch.bfloat16).float(),
        atol=1e-2,
    )
    with pytest.raises(ValueError, match="must advance"):
        worker.set_policy_state(state)
    with_frozen = policy_state_from_tensors(policy.state_dict(), version=2)
    with pytest.raises(ValueError, match="unexpected"):
        worker.set_policy_state(with_frozen)
