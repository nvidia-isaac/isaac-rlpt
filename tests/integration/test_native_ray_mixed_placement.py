# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-part placement on several processes: a base model served by Ray actors, the actor in each worker.

Seeded evaluation draws travel with each decode request, so a base model
served by another process decodes exactly the numbers an all-local policy in
one process does. Training and evaluation groups share the policy-server
actors. A trainable policy served whole (PPO) takes the learner's weights
by Ray object reference. The ``native_ray`` marker skips the module without
the ``ray`` extra.
"""

from __future__ import annotations

import copy
import functools

import numpy as np
import pytest

ray = pytest.importorskip("ray")
torch = pytest.importorskip("torch")

from base_model_doubles import (
    DrawingBaseModel,
    ReferenceBaseModel,
    batched_spec,
    collect,
    edit_spec,
    gym_spec,
    steering_spec,
)
from entry_doubles import ScriptedSimulator, scripted_environment

from isaac_rl_posttraining.algorithms.ppo import PPO, GaussianActorCritic, PPOConfig
from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
from isaac_rl_posttraining.runtimes import (
    GymWorkerSpec,
    LearnerBatch,
    NativeRuntime,
    NativeRuntimeConfig,
    RolloutRequest,
)
from isaac_rl_posttraining.runtimes.placement import (
    Envs,
    Learner,
    Placement,
    Resources,
    Served,
    plan_placement,
)
from isaac_rl_posttraining.runtimes.policy import PolicyFeatures, RolloutPolicy
from isaac_rl_posttraining.runtimes.policy_snapshot import apply_policy_state

pytestmark = [pytest.mark.native_ray, pytest.mark.timeout(900)]

STEERING_PARTS = ("base", "actor")


@pytest.fixture(scope="module", autouse=True)
def local_ray():
    """One local Ray instance for the module; each runtime joins it instead of starting its own."""
    ray.init(num_cpus=16, include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()


def _runtime(placement: Placement | None = None, *, parts=STEERING_PARTS, **settings):
    config = NativeRuntimeConfig(device="cpu", **settings)
    if placement is None:
        return NativeRuntime(config)
    plan = plan_placement(
        placement,
        Resources.uniform(1, 0),
        policy_parts=parts,
        served_supported=True,
        distributed_learner=True,
        served_parts=True,
    )
    return NativeRuntime(config, plan=plan)


SEEDED_EVALUATION = (
    RolloutRequest(episodes_per_environment=2, mode="eval", reset_seed=11),
    RolloutRequest(episodes_per_environment=1, mode="eval", reset_seed=40),
)


@pytest.mark.parametrize("family", ["dsrl", "rfs", "edit"])
@pytest.mark.parametrize("worker", ["gym", "batched"])
@pytest.mark.parametrize("deterministic", [False, True], ids=["sampled", "mean"])
def test_a_base_model_served_by_ray_actors_decides_like_one_local_policy(
    family, worker, deterministic
):
    """Seeded evaluation (and mean-mode SAC-edit) matches an all-local run in one process."""
    if family == "edit" and not deterministic:
        pytest.skip(
            "SAC-edit draws from each process's own generator; mean mode is comparable"
        )
    num_envs, workers = (4, 2) if worker == "batched" else (3, 3)
    make = batched_spec if worker == "batched" else gym_spec

    def spec():
        if family == "edit":
            return make(edit_spec(ReferenceBaseModel(), deterministic=True), num_envs)
        return make(
            steering_spec(DrawingBaseModel(), family, deterministic=deterministic),
            num_envs,
        )

    single = collect(
        _runtime(policy_mode="local"),
        spec(),
        workers=workers,
        num_envs=num_envs,
        requests=SEEDED_EVALUATION,
    )
    runtime = _runtime(
        Placement(envs=Envs(workers=workers), parts={"base": Served(replicas=2)})
    )
    several = collect(
        runtime, spec(), workers=workers, num_envs=num_envs, requests=SEEDED_EVALUATION
    )
    assert several == single


def test_training_and_evaluation_groups_share_the_policy_server_actors():
    """Both a served base model and a whole served policy: one set of server actors per served role."""
    base_model = DrawingBaseModel()
    for placement in (
        Placement(envs=Envs(workers=2), parts={"base": Served(replicas=2)}),
        Placement(envs=Envs(workers=2), policy=Served(replicas=2)),
    ):
        runtime = _runtime(placement)
        try:
            train = runtime.open_rollout_group(
                gym_spec(steering_spec(base_model, "dsrl"), 2),
                num_rollout_workers=2,
                num_envs=2,
                name="train",
            )
            evaluation = runtime.open_rollout_group(
                gym_spec(steering_spec(base_model, "dsrl", deterministic=True), 2),
                num_rollout_workers=2,
                num_envs=2,
                name="eval",
            )
            (servers,) = runtime._servers.values()
            assert len(servers._actors) == 2
            base_alone = placement.parts != {}
            # The base model is one policy for both groups; a whole policy is one per group.
            assert len(servers._policies) == (1 if base_alone else 2)
            train.collect([RolloutRequest(decisions_per_environment=2)] * 2)
            results = evaluation.collect(
                [RolloutRequest(episodes_per_environment=1, mode="eval", reset_seed=2)]
                * 2
            )
            assert all(result.decisions for result in results)
            evaluation.close()
            assert len(servers._policies) == 1
        finally:
            runtime.close()


class _ActorCriticEnv:
    def __init__(self, offset: float) -> None:
        self.offset, self.t = offset, 0

    def _observation(self):
        return {"state": np.asarray([self.offset, self.t, 1.0], np.float32)}

    def reset(self, *, seed=None):
        self.t = 0
        return self._observation(), {}

    def step(self, action):
        self.t += 1
        return self._observation(), 0.0, self.t >= 3, False, {"success": False}

    def close(self):
        pass


class _ActorCriticPolicy(RolloutPolicy):
    """Acts with the mean action of a trainable actor-critic (the whole policy is the trained model)."""

    features = PolicyFeatures(encodes_observations=False)

    def __init__(self, model: GaussianActorCritic, context=None) -> None:
        del context  # a policy spec is called with the server's context
        self.model = copy.deepcopy(model)
        self.version = 0

    def action_chunks(self, observations, *, mode, session_ids):
        states = torch.as_tensor(np.stack([o["state"] for o in observations]))
        with torch.no_grad():
            actions, _, _ = self.model.act(states, deterministic=True)
        return tuple(
            ActionChunk.from_arrays(
                learner_action=action[None].numpy(),
                environment_actions=action[None, None].numpy(),
                execution_length=1,
                learner_action_kind="latent",
                policy_version=self.version,
            )
            for action in actions
        )

    def set_policy_state(self, state):
        self.version = apply_policy_state(
            self.model, state, current_version=self.version
        )
        return self.version


def _ppo_batches(ranks: int, seed: int) -> list[LearnerBatch]:
    generator = np.random.default_rng(seed)
    samples = 8
    tensors = {
        "observations": generator.normal(size=(samples, 3)).astype(np.float32),
        "actions": generator.uniform(-0.9, 0.9, size=(samples, 2)).astype(np.float32),
        "old_log_probs": generator.normal(-2.0, 0.1, size=samples).astype(np.float32),
        "advantages": generator.normal(size=samples).astype(np.float32),
        "returns": generator.normal(size=samples).astype(np.float32),
        "old_values": generator.normal(size=samples).astype(np.float32),
    }
    return list(LearnerBatch(tensors=tensors, samples=samples).split(ranks))


def test_a_served_trainable_policy_takes_the_learners_weights_by_reference():
    """PPO: the learner ranks publish once to the object store, the policy servers fetch, workers take the version."""
    torch.manual_seed(4)
    model = GaussianActorCritic(3, 2, hidden_sizes=(16,))
    algorithm = PPO(
        PPOConfig(epochs=2, minibatch_size=4, normalize_advantages=False, seed=8)
    )
    runtime = _runtime(
        Placement(
            envs=Envs(workers=2), policy=Served(replicas=2), learner=Learner(ranks=2)
        ),
        parts=("policy",),
    )
    try:
        learner = runtime.open_learner_group(
            algorithm.learner(functools.partial(copy.deepcopy, model)),
            num_learner_ranks=2,
            name="ppo",
        )
        group = runtime.open_rollout_group(
            GymWorkerSpec(
                entry=ScriptedSimulator(
                    lambda context: _ActorCriticEnv(
                        float(context.environment_indices[0])
                    )
                ),
                environment=scripted_environment(2),
                phase="train",
                policy=functools.partial(_ActorCriticPolicy, model),
                supports_batched_policy=True,
            ),
            num_rollout_workers=2,
            num_envs=2,
            name="ppo_rollouts",
        )
        for step in range(2):
            learner.train_step(_ppo_batches(2, step))
        weights = learner.publish(1)
        assert weights.handle is not None and weights.inline is None
        assert group.set_weights(weights) == (1, 1)
        results = group.collect([RolloutRequest(decisions_per_environment=2)] * 2)
        trained = _ActorCriticPolicy(model)
        trained.set_policy_state(weights.resolve())
        for result in results:
            for decision in result.decisions:
                assert decision.action_chunk.policy_version == 1
                (expected,) = trained.action_chunks(
                    [decision.transition.observation.arrays()],
                    mode=["train"],
                    session_ids=[0],
                )
                np.testing.assert_array_equal(
                    decision.action_chunk.environment_actions,
                    expected.environment_actions,
                )
        # The update changed the policy the servers act with.
        (untrained,) = _ActorCriticPolicy(model).action_chunks(
            [results[0].decisions[0].transition.observation.arrays()],
            mode=["train"],
            session_ids=[0],
        )
        assert not np.array_equal(
            untrained.environment_actions,
            results[0].decisions[0].action_chunk.environment_actions,
        )
    finally:
        runtime.close()
