# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral learner contracts and RLInf FSDP orchestration tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from isaac_rl_posttraining.runtimes import (
    LearnerBatch,
    LearnerStepResult,
    NativeRuntime,
    PolicyState,
    RLInfRuntime,
    RLInfRuntimeConfig,
    open_distributed_learner_group,
)
from isaac_rl_posttraining.runtimes.rlinf.learner import (
    RLInfLearnerGroup,
    validate_learner_checkpoint_manifest,
)


class _Completed:
    def __init__(self, value):
        self.value = value

    def wait(self):
        return self.value


class _CollectiveGroup:
    def __init__(self, num_learner_ranks: int = 2) -> None:
        self.num_learner_ranks = num_learner_ranks
        self.calls = []
        self.worker_info_list = []
        self.fail_step = False
        self.rank_batches = []

    def execute_on(self, rank):
        provider = self

        class _Rank:
            def train_step(self, batch, step):
                provider.calls.append(("train_step", rank, step))
                provider.rank_batches.append((rank, batch))
                if provider.fail_step:
                    provider.fail_step = False
                    raise RuntimeError("injected collective failure")
                return _Completed(
                    [
                        LearnerStepResult(
                            learner_rank=rank,
                            step=step,
                            samples=batch.samples,
                            metrics={"loss": rank + step / 10},
                        )
                    ]
                )

        return _Rank()

    def policy_state(self, version):
        self.calls.append(("policy_state", version))
        return _Completed(
            [
                PolicyState(version=version, tensors={"weight": np.ones(2)}),
                *([None] * (self.num_learner_ranks - 1)),
            ]
        )

    def save_checkpoint(self, directory):
        self.calls.append(("save_checkpoint", directory))
        return _Completed(list(range(self.num_learner_ranks)))

    def load_checkpoint(self, directory):
        self.calls.append(("load_checkpoint", directory))
        return _Completed([(rank, 1) for rank in range(self.num_learner_ranks)])

    def close(self):
        return _Completed([None] * self.num_learner_ranks)


def _batch(value: float) -> LearnerBatch:
    return LearnerBatch(
        tensors={
            "obs": {"state": np.full((2, 3), value, dtype=np.float32)},
            "action": np.full((2, 1), value, dtype=np.float32),
        },
        samples=2,
    )


def test_learner_batch_copies_transport_arrays_and_validates_names() -> None:
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    batch = LearnerBatch(tensors={"obs": {"state": source}}, samples=2)
    source.fill(-1)
    nested = batch.tensors["obs"]
    assert isinstance(nested, dict)
    assert np.array_equal(nested["state"], np.arange(6).reshape(2, 3))

    with pytest.raises(ValueError, match="nested learner batch names"):
        LearnerBatch(tensors={"obs": {"": source}}, samples=2)
    with pytest.raises(TypeError, match="object dtype"):
        LearnerBatch(tensors={"bad": np.array([object()])}, samples=1)


def test_learner_collectives_fail_closed_after_provider_error() -> None:
    provider = _CollectiveGroup()
    group = RLInfLearnerGroup(provider, num_learner_ranks=2)
    batches = (_batch(1), _batch(2))

    first = group.train_step(batches)
    provider.fail_step = True
    with pytest.raises(RuntimeError, match="injected collective failure"):
        group.train_step(batches)
    with pytest.raises(RuntimeError, match="unusable after a failed collective"):
        group.train_step(batches)

    assert [result.step for result in first] == [1, 1]
    assert provider.calls == [
        ("train_step", 0, 1),
        ("train_step", 1, 1),
        ("train_step", 0, 2),
    ]


def test_learner_collectives_have_one_monotonic_order() -> None:
    provider = _CollectiveGroup()
    group = RLInfLearnerGroup(provider, num_learner_ranks=2)
    batches = (_batch(1), _batch(2))

    group.train_step(batches)
    state = group.policy_state(version=3)
    group.save_checkpoint("/shared/checkpoint")
    group.load_checkpoint("/shared/checkpoint")

    assert state.version == 3
    assert provider.calls == [
        ("train_step", 0, 1),
        ("train_step", 1, 1),
        ("policy_state", 3),
        ("save_checkpoint", "/shared/checkpoint"),
        ("load_checkpoint", "/shared/checkpoint"),
    ]
    # Each rank received only its own batch.
    assert provider.rank_batches == [(0, batches[0]), (1, batches[1])]


def test_distributed_learner_extension_fails_before_native_launch() -> None:
    runtime = NativeRuntime()
    with pytest.raises(ValueError, match="runs one learner rank in this process"):
        open_distributed_learner_group(
            runtime,
            lambda context, modules: None,
            num_learner_ranks=2,
            name="unsupported",
        )


def test_rlinf_distributed_update_capability() -> None:
    runtime = RLInfRuntime(RLInfRuntimeConfig(learner_mixed_precision="bf16"))
    assert runtime.features.distributed_learner


def test_rlinf_learner_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="learner_mixed_precision"):
        RLInfRuntimeConfig(learner_mixed_precision="fp16")
    with pytest.raises(ValueError, match="learner_first_accelerator"):
        RLInfRuntimeConfig(learner_first_accelerator=-1)
    with pytest.raises(ValueError, match="learner_seed"):
        RLInfRuntimeConfig(learner_seed=True)
    with pytest.raises(TypeError, match="deterministic_evaluation"):
        RLInfRuntimeConfig(deterministic_evaluation=1)  # type: ignore[arg-type]


def test_learner_checkpoint_manifest_fixes_world_size_and_topology() -> None:
    manifest = {
        "schema_version": 1,
        "checkpoint_id": "checkpoint-123",
        "world_size": 2,
        "module_names": ["actor", "critic"],
        "optimizer_names": ["actor", "critic"],
        "learner_step": 17,
    }
    assert validate_learner_checkpoint_manifest(
        manifest,
        world_size=2,
        module_names=("critic", "actor"),
        optimizer_names=("critic", "actor"),
    ) == ("checkpoint-123", 17)

    with pytest.raises(ValueError, match="world size mismatch"):
        validate_learner_checkpoint_manifest(
            manifest,
            world_size=4,
            module_names=("actor", "critic"),
            optimizer_names=("actor", "critic"),
        )


class _FakeWorker:
    @classmethod
    def create_group(cls, *args):
        del cls, args
        return _FakeGroupBuilder()


class _FakeGroupBuilder:
    def launch(self, **kwargs):
        placement = kwargs["placement_strategy"]
        return _FakeLaunchedGroup(placement.end - placement.start + 1)


class _FakeLaunchedGroup(_CollectiveGroup):
    def initialize(self):
        return _Completed(list(range(self.num_learner_ranks)))


class _FakePackedPlacement:
    def __init__(self, start, end) -> None:
        self.start = start
        self.end = end


class _FakeFlexiblePlacement:
    def __init__(self, ranks) -> None:
        self.ranks = ranks


#: (device, placement) of every rollout group the fake scheduler launched.
_LAUNCHED: list = []


class _FakeRolloutWorker:
    @classmethod
    def create_group(cls, spec, device, assignments):
        del spec
        return _FakeRolloutBuilder(device, len(assignments))


class _FakeRolloutBuilder:
    def __init__(self, device, workers) -> None:
        self.device, self.workers = device, workers

    def launch(self, **kwargs):
        _LAUNCHED.append((self.device, kwargs["placement_strategy"]))
        return SimpleNamespace(
            ping=lambda: _Completed(list(range(self.workers))), worker_info_list=[]
        )


@pytest.mark.parametrize(
    ("accelerators", "ranks"),
    [(5, [[1], [2], [3], [4]]), (2, [[1], [1], [1], [1]]), (3, [[1], [1], [2], [2]])],
)
def test_rlinf_local_workers_take_a_card_each_and_share_when_there_are_fewer(
    monkeypatch, accelerators, ranks
) -> None:
    """Workers holding their own policy get cards from first_accelerator on, shared when too few."""
    monkeypatch.setattr(
        "isaac_rl_posttraining.runtimes.rlinf.runtime.make_rollout_worker",
        lambda worker_base: _FakeRolloutWorker,
    )
    scheduler = SimpleNamespace(
        Worker=object,
        FlexiblePlacementStrategy=_FakeFlexiblePlacement,
        Channel=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(qsize=lambda: 0)
        ),
    )
    runtime = RLInfRuntime(
        RLInfRuntimeConfig(
            device="gpu", first_accelerator=1, serialize_specs_by_value=False
        )
    )
    runtime._cluster_owner.scheduler = scheduler
    runtime._cluster_owner.cluster = SimpleNamespace(num_accelerators=accelerators)
    _LAUNCHED.clear()
    runtime.open_rollout_group(
        lambda context: None, num_rollout_workers=4, num_envs=4, name="rollout"
    )
    ((device, placement),) = _LAUNCHED
    assert device == "gpu" and placement.ranks == ranks


def test_rlinf_rollout_group_waits_for_its_result_channel(monkeypatch) -> None:
    """A group closed right after it opens (an evaluation group replaced before use) must not kill the process."""
    monkeypatch.setattr(
        "isaac_rl_posttraining.runtimes.rlinf.runtime.make_rollout_worker",
        lambda worker_base: _FakeRolloutWorker,
    )
    waited: list[str] = []
    channel = SimpleNamespace(qsize=lambda: waited.append("qsize") or 0)
    scheduler = SimpleNamespace(
        Worker=object,
        FlexiblePlacementStrategy=_FakeFlexiblePlacement,
        NodePlacementStrategy=lambda nodes: nodes,
        Channel=SimpleNamespace(create=lambda **kwargs: channel),
    )
    runtime = RLInfRuntime(
        RLInfRuntimeConfig(device="cpu", serialize_specs_by_value=False)
    )
    runtime._cluster_owner.scheduler = scheduler
    runtime._cluster_owner.cluster = SimpleNamespace(num_accelerators=0)
    runtime.open_rollout_group(
        lambda context: None, num_rollout_workers=1, num_envs=1, name="rollout"
    )
    assert waited == ["qsize"]
