# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A served base model with the actor on each rollout worker decides exactly as an all-local policy.

``runtime.native.base_mode=served`` (``placement.parts.base.mode=served``)
keeps the actor step on every worker and sends decodes and reference
requests to one in-process base-model server. The base model below draws from
torch's generator in every call, so the tests see any change to the order
or the streams of the draws: seeded evaluation streams, unseeded training
draws, RFS residuals, SAC-edit references, Gym and batched workers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from base_model_doubles import (
    WIDTH,
    DrawingBaseModel,
    batched_spec,
    collect,
    gym_spec,
    policy_spec,
    steering_spec,
)

from isaac_rl_posttraining.runtimes import (
    FrozenBaseModelSpec,
    NativeRuntime,
    NativeRuntimeConfig,
    RolloutRequest,
)
from isaac_rl_posttraining.runtimes.base import PolicyContext
from isaac_rl_posttraining.runtimes.base_part import (
    LocalBasePart,
    ServedBasePart,
    ServedBasePartSpec,
    with_served_base_part,
)
from isaac_rl_posttraining.runtimes.policy_snapshot import policy_state_from_tensors
from isaac_rl_posttraining.runtimes.serving import (
    BasePartInputs,
    Batcher,
    DecodeOptions,
    InProcessPolicyService,
    PolicyRequest,
    PolicyResponse,
)
from isaac_rl_posttraining.runtimes.session_randomness import SessionRandomness

LOCAL = NativeRuntimeConfig(device="cpu", policy_mode="local")
# Workers collect one after another: their actors draw from this process's generators in a fixed order.
BASE_SERVED = NativeRuntimeConfig(
    device="cpu", base_mode="served", concurrent_collection=False
)
ROUNDS = (
    RolloutRequest(episodes_per_environment=1, mode="warmup"),
    RolloutRequest(episodes_per_environment=1, mode="train"),
    RolloutRequest(episodes_per_environment=2, mode="eval", reset_seed=11),
    RolloutRequest(episodes_per_environment=1, mode="eval"),
)


@pytest.mark.parametrize("family", ["dsrl", "rfs", "rfs_reference", "edit"])
@pytest.mark.parametrize("worker", ["gym", "batched"])
@pytest.mark.parametrize("deterministic", [False, True], ids=["sampled", "mean"])
def test_a_served_base_model_decides_like_a_local_one(family, worker, deterministic):
    """Warm-up, training and (seeded) evaluation rounds match draw for draw."""
    num_envs, workers = (4, 2) if worker == "batched" else (3, 3)
    make = batched_spec if worker == "batched" else gym_spec
    runs = []
    for config in (LOCAL, BASE_SERVED):
        base_model = DrawingBaseModel()
        spec = make(
            policy_spec(family, base_model, deterministic=deterministic), num_envs
        )
        runs.append(
            collect(
                NativeRuntime(config),
                spec,
                workers=workers,
                num_envs=num_envs,
                requests=ROUNDS,
            )
        )
    assert runs[0] == runs[1]
    assert {row[2] for row in runs[0]} >= {"policy"}


@pytest.mark.parametrize("worker", ["gym", "batched"])
def test_a_served_base_model_evaluates_the_native_policy_like_a_local_one(worker):
    num_envs, workers = (4, 2) if worker == "batched" else (3, 3)
    make = batched_spec if worker == "batched" else gym_spec
    request = RolloutRequest(episodes_per_environment=2, mode="eval", reset_seed=3)
    runs = [
        collect(
            NativeRuntime(config),
            make(steering_spec(DrawingBaseModel(), "dsrl", native=True), num_envs),
            workers=workers,
            num_envs=num_envs,
            requests=(request,),
        )
        for config in (LOCAL, BASE_SERVED)
    ]
    assert runs[0] == runs[1]


def test_training_and_evaluation_groups_share_one_base_model_server():
    base_model = DrawingBaseModel()
    runtime = NativeRuntime(
        NativeRuntimeConfig(
            device="cpu", base_mode="served", collect_policy_server_metrics=True
        )
    )
    try:
        train = runtime.open_rollout_group(
            gym_spec(steering_spec(base_model, "dsrl"), 2),
            num_rollout_workers=2,
            num_envs=2,
            name="train",
        )
        evaluation = runtime.open_rollout_group(
            batched_spec(steering_spec(base_model, "dsrl", deterministic=True), 2),
            num_rollout_workers=1,
            num_envs=2,
            name="eval",
        )
        assert len(runtime._base_services) == 1 and len(runtime._services) == 1
        train.collect([RolloutRequest(decisions_per_environment=2)] * 2)
        evaluation.collect([RolloutRequest(episodes_per_environment=1, mode="eval")])
        # Both groups' decodes reached the one server; three clients kept their sessions apart.
        (stats,) = train.policy_server_stats()
        assert evaluation.policy_server_stats() == (stats,)
        assert stats["operation_batch_sizes"]["decode"]
        assert len({session[0] for session in base_model.started}) == 3
    finally:
        runtime.close()


def test_actor_weights_reach_the_workers_and_the_frozen_base_model_takes_none():
    base_model = DrawingBaseModel()
    runtime = NativeRuntime(BASE_SERVED)
    try:
        group = runtime.open_rollout_group(
            gym_spec(steering_spec(base_model, "dsrl", deterministic=True), 2),
            num_rollout_workers=2,
            num_envs=2,
            name="weights",
        )
        state = policy_state_from_tensors(
            {"bias": torch.full((WIDTH,), 0.25)}, version=1
        )
        assert group.set_policy_state(state) == (1, 1)
        request = [RolloutRequest(decisions_per_environment=1, mode="eval")] * 2
        (first, _) = group.collect(request)
        latent = first.decisions[0].transition.learner_action.value
        expected = (
            np.tanh(first.decisions[0].transition.observation.arrays()["state"][0])
            + 0.25
        )
        np.testing.assert_allclose(
            latent.reshape(-1), np.full(WIDTH, expected, np.float32), rtol=1e-6
        )
        (service,) = runtime._base_services.values()
        assert service.queue_for(0).server.version == 0
    finally:
        runtime.close()


def test_seeded_streams_continue_across_the_served_decode():
    """A seeded session's actor draw, served decode and next actor draw follow one stream."""
    observations = [{"state": np.asarray([1.0, 2.0], np.float32)}] * 2
    decoded = []
    for served in (False, True):
        base_model = DrawingBaseModel()
        if served:
            service = InProcessPolicyService(
                FrozenBaseModelSpec(prebuilt=base_model),
                device="cpu",
                batcher=Batcher(),
            )
            base = ServedBasePartSpec(service.handle())(
                PolicyContext(
                    policy_index=0,
                    num_policies=1,
                    device="cpu",
                    placement="rollout_worker",
                )
            )
        else:
            base = LocalBasePart(base_model, torch.device("cpu"))
        randomness = SessionRandomness(torch.device("cpu"))
        randomness.seed(0, 7)
        randomness.seed(1, 8)
        rows = []
        for _ in range(2):
            with randomness.stream(0, enabled=True):
                rows.append(torch.rand(1).item())
            rows.append(
                base.decode(
                    observations,
                    [np.ones((1, WIDTH), np.float32)] * 2,
                    session_ids=(0, 1),
                    modes=("eval", "eval"),
                    randomness=randomness,
                    options=DecodeOptions(),
                )
                .round(6)
                .tolist()
            )
        decoded.append(rows)
        if served:
            assert isinstance(base, ServedBasePart)
            service.close()
    assert decoded[0] == decoded[1]


def test_a_base_model_that_keeps_one_episode_cannot_be_served():
    class OneEpisode(DrawingBaseModel):
        def start_episode(self):
            pass

    with pytest.raises(TypeError, match="cannot share it"):
        InProcessPolicyService(
            FrozenBaseModelSpec(prebuilt=OneEpisode()), device="cpu", batcher=Batcher()
        )


def test_a_base_model_that_declares_it_cannot_be_served_is_refused_at_load():
    class Declared(DrawingBaseModel):
        servable = False

    with pytest.raises(TypeError, match="servable=False"):
        InProcessPolicyService(
            FrozenBaseModelSpec(prebuilt=Declared()), device="cpu", batcher=Batcher()
        )


def test_only_steering_and_edit_policies_have_a_base_model_to_serve():
    spec = gym_spec(lambda context: None, 1)
    with pytest.raises(TypeError, match="has none"):
        with_served_base_part(spec, lambda base_model: None)


def test_base_model_requests_and_responses_validate_their_payload():
    with pytest.raises(ValueError, match="needs its base-model inputs"):
        PolicyRequest("decode", 0, {}, "eval")
    with pytest.raises(ValueError, match="needs a decision mode"):
        PolicyRequest("decode", 0, {}, None, base=BasePartInputs())
    with pytest.raises(ValueError, match="image_transform_mode"):
        BasePartInputs(image_transform_mode="test")
    with pytest.raises(ValueError, match="needs its result"):
        PolicyResponse(0, "reference", 0)
    assert PolicyRequest("features", 0, {}).base is None


def test_workers_collecting_on_threads_share_the_base_model_server():
    runtime = NativeRuntime(
        NativeRuntimeConfig(
            device="cpu",
            base_mode="served",
            concurrent_collection=True,
            policy_max_wait_ms=20.0,
            collect_policy_server_metrics=True,
        )
    )
    try:
        group = runtime.open_rollout_group(
            gym_spec(steering_spec(DrawingBaseModel(), "dsrl"), 3),
            num_rollout_workers=3,
            num_envs=3,
            name="threads",
        )
        results = group.collect([RolloutRequest(decisions_per_environment=2)] * 3)
        assert [len(result.decisions) for result in results] == [2, 2, 2]
        (stats,) = group.policy_server_stats()
        assert sum(stats["operation_batch_sizes"]["decode"]) == 6
    finally:
        runtime.close()
