# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Placement planner: default layouts, explicit overrides, multi-node, errors and the dry-run lines."""

from __future__ import annotations

import dataclasses
import json
import pickle
import types

import pytest

from isaac_rl_posttraining.runtimes.placement import (
    Envs,
    Learner,
    Local,
    Placement,
    Resources,
    Served,
    plan_placement,
    plan_policy_service,
)

ONE_NODE = Resources.uniform(1, 8)
PARTS = ("base", "actor")


def _plan(
    placement=None,
    resources=ONE_NODE,
    *,
    served=True,
    distributed=True,
    parts=PARTS,
    served_parts=True,
):
    return plan_placement(
        placement or Placement(),
        resources,
        policy_parts=parts,
        served_supported=served,
        distributed_learner=distributed,
        served_parts=served_parts,
    )


def test_default_served_layout_on_one_node():
    plan = _plan(Placement(policy=Served()))
    assert plan.envs.by_node() == {"node0": (0, 1, 2)}
    assert plan.envs.processes == 3
    (service,) = plan.services
    assert service.parts == PARTS
    assert service.by_node() == {"node0": (3, 4)}
    assert service.processes == 2 and service.served.replicas == 2
    assert plan.learner.by_node() == {"node0": (5, 6, 7)}
    assert plan.learner.processes == 3
    assert not plan.shared


def test_default_local_layout_on_one_node():
    plan = _plan()
    assert plan.envs.num_gpus == 5 and plan.learner.num_gpus == 3
    (policy,) = plan.policy
    assert (
        policy.served is None
        and policy.gpus == ()
        and policy.processes == plan.envs.processes
    )


@pytest.mark.parametrize(
    ("placement", "gpus", "expected"),
    [
        (Placement(policy=Served()), 3, (1, 1, 1)),
        (Placement(policy=Served(), learner=Learner(gpus=("0",))), 3, (1, 1, 1)),
        (Placement(), 2, (1, None, 1)),
    ],
)
def test_rounding_never_leaves_a_role_without_a_gpu_it_could_have(
    placement, gpus, expected
):
    plan = _plan(placement, Resources(nodes=("node0",), gpus_per_node=(gpus,)))
    served = plan.services[0].num_gpus if plan.services else None
    assert (plan.envs.num_gpus, served, plan.learner.num_gpus) == expected
    assert not plan.shared


def test_single_rank_learner_takes_one_gpu_and_envs_the_rest():
    plan = _plan(Placement(policy=Served()), distributed=False)
    assert (plan.envs.num_gpus, plan.services[0].num_gpus, plan.learner.num_gpus) == (
        5,
        2,
        1,
    )


def test_every_role_gets_a_gpu_when_there_is_one_per_role():
    # Native Ray pins the single learner rank to GPU 0; three GPUs then leave two for envs and policy.
    plan = _plan(
        Placement(policy=Served(), learner=Learner(ranks=1, gpus=("0",))),
        Resources.uniform(1, 3),
        distributed=False,
    )
    assert (plan.envs.num_gpus, plan.services[0].num_gpus, plan.learner.num_gpus) == (
        1,
        1,
        1,
    )
    assert not plan.shared


@pytest.mark.parametrize(("gpus", "shared_gpu"), [(2, ("node0:1",)), (1, ("node0:0",))])
def test_roles_left_without_a_gpu_of_their_own_share_one(gpus, shared_gpu):
    # Native Ray's learner rank holds GPU 0; the rollout workers and the policy server share the rest.
    plan = _plan(
        Placement(policy=Served(), learner=Learner(ranks=1, gpus=("0",))),
        Resources.uniform(1, gpus),
        distributed=False,
    )
    assert plan.shared
    assert plan.learner.devices() == ("node0:0",)
    assert plan.envs.devices() == plan.services[0].devices() == shared_gpu
    assert plan.services[0].processes == 1
    # Explicit requests that take every GPU leave the rollout workers sharing one of them.
    taken = _plan(Placement(policy=Served(gpus=4), learner=Learner(gpus=4)))
    assert taken.shared and taken.envs.devices() == ("node0:0",)
    assert taken.services[0].num_gpus == taken.learner.num_gpus == 4


def test_no_role_takes_more_gpus_than_its_processes_use():
    # One rollout worker uses one GPU; two learner ranks take two; one GPU stays free.
    plan = _plan(
        Placement(envs=Envs(workers=1), learner=Learner(ranks=2)),
        Resources.uniform(1, 4),
    )
    assert plan.envs.devices() == ("node0:0",) and plan.learner.num_gpus == 2
    assert plan.lines()[0].endswith("; 1 gpu(s) unused")
    # Two workers use at most two replicas; the GPUs a capped role cannot use go to the other.
    served = _plan(
        Placement(policy=Served(), envs=Envs(workers=2), learner=Learner(ranks=1)),
        distributed=False,
    )
    assert (served.envs.num_gpus, served.services[0].processes) == (2, 2)
    assert served.lines()[0].endswith("; 3 gpu(s) unused")


def test_evaluation_workers_with_a_local_policy_use_the_gpus_training_leaves_free():
    """Each evaluation worker holds a policy copy next to the training workers', so a free GPU goes to it."""
    two_workers = Placement(envs=Envs(workers=2), learner=Learner(ranks=1))
    four_gpus = Resources.uniform(1, 4)
    plan = plan_placement(
        two_workers,
        four_gpus,
        policy_parts=PARTS,
        served_supported=True,
        distributed_learner=False,
        evaluation_workers=1,
    )
    assert (plan.envs.num_gpus, plan.envs.processes, plan.learner.num_gpus) == (3, 2, 1)
    # A served policy keeps the evaluation workers thin: they share the training workers' GPUs.
    served = plan_placement(
        dataclasses.replace(two_workers, policy=Served(replicas=1)),
        four_gpus,
        policy_parts=PARTS,
        served_supported=True,
        distributed_learner=False,
        evaluation_workers=1,
    )
    assert served.envs.num_gpus == 2


def test_a_second_rollout_group_starts_on_the_least_used_gpus():
    """The evaluation group's workers land on GPUs the training workers do not hold."""
    from isaac_rl_posttraining.runtimes.native_ray.cluster import RayCluster

    cluster = object.__new__(RayCluster)
    cluster._reservations = {
        "envs": types.SimpleNamespace(
            slots=[("node0", 1), ("node0", 2), ("node0", 3)], bundles=[]
        )
    }
    cluster._slot_load = {}
    training = cluster.claim_slots("envs", 2)
    assert training == [0, 1]
    evaluation = cluster.claim_slots("envs", 1)
    assert evaluation == [2]
    cluster.release_slots("envs", evaluation)
    assert cluster.claim_slots("envs", 1) == [2]
    # More workers than GPUs wrap around as before.
    assert cluster.claim_slots("envs", 3) == [0, 1, 2]


def test_explicit_values_win_and_defaults_fill_the_rest():
    plan = _plan(Placement(policy=Served(gpus=2), learner=Learner(gpus=4)))
    assert (plan.envs.num_gpus, plan.services[0].num_gpus, plan.learner.num_gpus) == (
        2,
        2,
        4,
    )
    plan = _plan(Placement(envs=Envs(workers=16), learner=Learner(ranks=2)))
    assert (
        plan.envs.processes == 16
        and plan.envs.num_gpus == 6
        and plan.learner.num_gpus == 2
    )


def test_explicit_devices_are_honoured_and_kept_from_defaults():
    plan = _plan(Placement(envs=Envs(gpus=["7"]), policy=Served(gpus=["0:0", "0:1"])))
    assert plan.envs.devices() == ("node0:7",)
    assert plan.services[0].devices() == ("node0:0", "node0:1")
    assert "node0:7" not in plan.learner.devices()


def test_mixed_parts_get_their_own_roles():
    plan = _plan(Placement(parts={"actor": Local(), "base": Served(gpus=2)}))
    assert plan.part("actor").served is None
    assert plan.part("base").num_gpus == 2
    with pytest.raises(KeyError):
        plan.part("critic")


def test_equal_served_parts_share_one_server():
    plan = _plan(Placement(parts={"actor": Served(), "base": Served()}))
    assert len(plan.services) == 1 and plan.services[0].parts == PARTS


@pytest.mark.parametrize(
    ("parts", "match"),
    [
        ({"actor": Served(), "base": Local()}, "only the base model is served alone"),
        (
            {"actor": Served(max_batch=4), "base": Served()},
            "different policy servers",
        ),
    ],
)
def test_only_the_base_model_is_served_alone(parts, match):
    with pytest.raises(ValueError, match=match):
        _plan(Placement(parts=parts))


def test_a_runtime_without_served_parts_serves_whole_policies_only():
    with pytest.raises(ValueError, match="whole policies only"):
        _plan(Placement(parts={"base": Served()}), served_parts=False)
    whole = _plan(Placement(policy=Served()), served_parts=False)
    assert whole.services[0].parts == PARTS and not whole.base_served_alone
    assert _plan(Placement(parts={"base": Served()})).base_served_alone


def test_multi_node_assignment_keeps_replicas_on_one_node():
    plan = _plan(Placement(policy=Served(gpus_per_replica=2)), Resources.uniform(2, 8))
    assert (plan.envs.num_gpus, plan.services[0].num_gpus, plan.learner.num_gpus) == (
        6,
        4,
        6,
    )
    devices = plan.services[0].devices()
    replicas = [devices[i : i + 2] for i in range(0, len(devices), 2)]
    assert all(
        len({device.split(":")[0] for device in replica}) == 1 for replica in replicas
    )
    assert set(plan.envs.by_node()) | set(plan.learner.by_node()) == {"node0", "node1"}


def test_nodes_restrict_the_plan():
    plan = _plan(Placement(nodes=["node1"]), Resources.uniform(2, 8))
    assert set(plan.envs.by_node()) == {"node1"} and set(plan.learner.by_node()) == {
        "node1"
    }
    with pytest.raises(ValueError, match="not free"):
        _plan(Placement(nodes=["1"], envs=Envs(gpus=["0:0"])), Resources.uniform(2, 8))


def test_small_or_cpu_resources_share_the_first_gpu():
    plan = _plan(Placement(policy=Served()), Resources.uniform(1, 1))
    assert plan.shared
    assert (
        plan.envs.devices()
        == plan.learner.devices()
        == plan.services[0].devices()
        == ("node0:0",)
    )
    one_rank = _plan(
        Placement(learner=Learner(ranks=1)),
        Resources.uniform(1, 1),
        served=False,
        distributed=False,
    )
    assert one_rank.shared
    assert one_rank.envs.devices() == one_rank.learner.devices() == ("node0:0",)
    cpu = _plan(
        Placement(policy=Served(replicas=2), learner=Learner(ranks=1)),
        Resources.uniform(1, 0),
    )
    assert cpu.envs.gpus == () and cpu.services[0].processes == 2


def test_one_gpu_with_a_single_learner_rank_shares_it():
    # The one-GPU steering and PPO recipes resolve learner.ranks=1 before planning.
    plan = _plan(Placement(learner=Learner(ranks=1)), Resources.uniform(1, 1))
    assert plan.shared
    assert plan.envs.devices() == plan.learner.devices() == ("node0:0",)
    assert plan.learner.processes == 1
    with pytest.raises(ValueError, match="over-subscribes"):
        _plan(Placement(learner=Learner(ranks=2)), Resources.uniform(1, 1))


@pytest.mark.parametrize("policy", [Local(), Served()])
def test_one_gpu_shares_it_with_the_configured_single_learner_rank(policy):
    # The run config always states one learner rank (placement.learner.ranks=1); that names no GPU.
    plan = _plan(
        Placement(policy=policy, learner=Learner(ranks=1)),
        Resources.uniform(1, 1),
        distributed=False,
    )
    assert plan.shared
    assert plan.envs.devices() == plan.learner.devices() == ("node0:0",)


def test_one_gpu_still_refuses_several_learner_ranks():
    with pytest.raises(ValueError, match="requested 2"):
        _plan(Placement(learner=Learner(ranks=2)), Resources.uniform(1, 1))


@pytest.mark.parametrize(
    ("placement", "match"),
    [
        (Placement(learner=Learner(gpus=9)), "over-subscribes"),
        (
            Placement(envs=Envs(gpus=["3"]), learner=Learner(gpus=["0:3"])),
            r"envs \(placement.envs.gpus\) and learner \(placement.learner.gpus\) both ask for GPU\(s\) node0:3",
        ),
        (Placement(envs=Envs(gpus=["0:8"])), "does not exist"),
        (Placement(envs=Envs(gpus=["node7:0"])), "unknown node"),
        (Placement(learner=Learner(ranks=2, gpus=3)), "disagree"),
        (Placement(policy=Served(replicas=3, gpus=4)), "disagree"),
        (
            Placement(parts={"critic": Served()}),
            r"placement.parts.critic names a policy part .* placement.policy.mode=served",
        ),
    ],
)
def test_bad_placements_fail_with_a_clear_error(placement, match):
    with pytest.raises(ValueError, match=match):
        _plan(placement)


def test_unsupported_placements_fail_before_planning():
    with pytest.raises(ValueError, match="this runtime has no policy server"):
        _plan(Placement(policy=Served()), served=False)
    with pytest.raises(ValueError, match="distributed"):
        _plan(Placement(learner=Learner(ranks=2)), distributed=False)


def test_user_types_validate_their_fields():
    with pytest.raises(ValueError):
        Envs(workers=0)
    with pytest.raises(ValueError):
        Served(routing="random")
    with pytest.raises(TypeError):
        Served(gpus="0")
    with pytest.raises(ValueError, match="twice"):
        Learner(gpus=["1", "1"])
    with pytest.raises(TypeError):
        Placement(policy="served")


def test_plan_lines_json_and_pickle():
    plan = _plan(Placement(parts={"actor": Local(), "base": Served()}))
    assert plan.lines() == [
        "plan.placement.resources=node0 (8 gpus)",
        "plan.placement.envs=3 worker(s) on node0 gpus [0, 1, 2]",
        (
            "plan.placement.policy.base=served by 2 replica(s) x 1 gpu on node0 gpus [3, 4], "
            "engine=builtin, routing=modulo, max_batch=unbounded, max_wait_ms=5.0"
        ),
        "plan.placement.policy.actor=local (a copy in every rollout worker)",
        "plan.placement.learner=3 rank(s) on node0 gpus [5, 6, 7]",
    ]
    assert pickle.loads(pickle.dumps(plan)) == plan
    payload = json.loads(plan.to_json())
    assert payload["learner"]["processes"] == 3
    default_part = _plan(parts=("policy",)).lines()
    assert (
        "plan.placement.policy=local (a copy in every rollout worker)" in default_part
    )


def test_an_unknown_part_names_the_parts_of_the_algorithm_and_how_to_serve_the_whole_policy():
    with pytest.raises(
        ValueError, match=r"is one part that it trains as a whole"
    ) as whole:
        _plan(Placement(parts={"base": Served()}), parts=("policy",))
    with pytest.raises(
        ValueError, match=r"has the parts \['base', 'actor'\]"
    ) as steering:
        _plan(Placement(parts={"critic": Served()}), parts=("base", "actor"))
    for error in (whole, steering):
        assert "placement.policy.mode=served" in str(error.value)


def test_gpus_may_be_listed_as_integer_indices():
    assert Served(replicas=2, gpus=[0, 1]).gpus == ("0", "1")
    assert Learner(gpus=(3, "node0:4")).gpus == ("3", "node0:4")
    with pytest.raises(ValueError, match="GPU indices"):
        Served(gpus=[-1])


def test_a_policy_service_plans_only_its_replicas():
    role = plan_policy_service(ONE_NODE, Served(replicas=2, gpus=[1, 3]))
    assert role.by_node() == {"node0": (1, 3)}
    assert role.processes == role.served.replicas == 2
    # Without a count, one replica per listed GPU; without GPUs, the first free ones.
    assert plan_policy_service(ONE_NODE, Served(gpus=[2, 5])).processes == 2
    assert plan_policy_service(ONE_NODE, Served(replicas=3)).devices() == (
        "node0:0",
        "node0:1",
        "node0:2",
    )
    with pytest.raises(ValueError, match="does not exist"):
        plan_policy_service(Resources.uniform(1, 1), Served(gpus=[0, 1]))
    with pytest.raises(ValueError, match="disagree"):
        plan_policy_service(ONE_NODE, Served(replicas=3, gpus=[0, 1]))


def test_a_policy_service_without_gpus_runs_on_the_cpu_and_says_so():
    role = plan_policy_service(
        Resources.uniform(1, 0), Served(replicas=2, max_batch=32)
    )
    assert role.gpus == () and role.processes == 2
    assert role.describe() == (
        "served by 2 replica(s) on the cpu, engine=builtin, routing=modulo, "
        "max_batch=32, max_wait_ms=5.0"
    )


def test_served_roles_take_whole_replicas_and_envs_take_the_remainder():
    plan = _plan(Placement(policy=Served(gpus_per_replica=4)))
    assert (plan.envs.num_gpus, plan.services[0].num_gpus, plan.learner.num_gpus) == (
        1,
        4,
        3,
    )
    plan = _plan(Placement(policy=Served(gpus_per_replica=2)), Resources.uniform(1, 4))
    assert (plan.envs.num_gpus, plan.services[0].num_gpus, plan.learner.num_gpus) == (
        1,
        2,
        1,
    )


def test_shared_gpus_still_give_a_replica_all_of_its_gpus():
    plan = _plan(Placement(policy=Served(gpus_per_replica=2)), Resources.uniform(1, 2))
    assert plan.shared and plan.services[0].num_gpus == 2
