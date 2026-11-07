# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The simulator entry a robot or simulator subclasses: defaults, declaration checks, registration."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from isaac_rl_posttraining.data.spec import ArraySpec, ObservationSpec
from isaac_rl_posttraining.envs import providers
from isaac_rl_posttraining.envs.simulator_entry import (
    CONVENTIONAL_STEERING_KEYS,
    EnvironmentSpec,
    SimulatorEntry,
    WorkerLayout,
)
from isaac_rl_posttraining.rollouts.steering_step import SteeringStep
from isaac_rl_posttraining.runners.build import train_plan
from isaac_rl_posttraining.runners.simulator_entries import resolve_simulator_entry
from isaac_rl_posttraining.runtimes import (
    BatchedWorkerSpec,
    GymWorkerSpec,
    RolloutWorkerContext,
)

SPEC = EnvironmentSpec(env_name="toy/reach", num_envs=2, max_episode_steps=50, seed=3)
CONTEXT = RolloutWorkerContext(
    worker_index=1, num_rollout_workers=2, environment_indices=(1,), device="cpu"
)


class ToyRobot(SimulatorEntry):
    """The Tier-1 entry: a name and how to build the environment, nothing else."""

    name = "toy"

    def make_environment(self, spec, phase, context):
        return ("env", spec.env_name, phase, context.environment_indices)


@pytest.fixture
def registry():
    """Leave the registry as it was, whatever a test registers."""
    saved = dict(providers._SIMULATOR_ENTRIES)
    yield providers
    providers._SIMULATOR_ENTRIES.clear()
    providers._SIMULATOR_ENTRIES.update(saved)


def test_a_minimal_entry_gets_the_single_robot_defaults() -> None:
    entry = ToyRobot()

    assert entry.name == "toy"
    assert entry.evaluation == "shared"
    assert entry.server_url_variables is None
    assert entry.base_models is None
    assert entry.decode_batch_size is None
    assert not entry.supports_chunked_step
    assert entry.worker_layout(4) == WorkerLayout(workers=4, batched=False)
    assert entry.steering_key_mapping(SPEC) == dict(CONVENTIONAL_STEERING_KEYS)
    assert entry.observation_spec(SPEC) is None and entry.action_spec(SPEC) is None
    with pytest.raises(NotImplementedError, match="flow policy family"):
        entry.make_scenes(SPEC, CONTEXT)
    with pytest.raises(NotImplementedError, match="instruction"):
        entry.instruction(SPEC)


def test_a_declared_observation_derives_the_steering_keys() -> None:
    class Declared(ToyRobot):
        def observation_spec(self, spec):
            return ObservationSpec(
                {
                    "wrist_cam": ArraySpec((None, None, 3), "uint8"),
                    "proprio": ArraySpec((12,), "float32"),
                }
            )

    assert Declared().steering_key_mapping(SPEC) == {
        "state": "proprio",
        "language": None,
        "image": ["wrist_cam"],
    }

    class Ambiguous(ToyRobot):
        def observation_spec(self, spec):
            return ObservationSpec(
                {
                    "cam": ArraySpec((None, None, 3), "uint8"),
                    "a": ArraySpec((3,), "float32"),
                    "b": ArraySpec((3,), "float32"),
                }
            )

    with pytest.raises(
        ValueError,
        match="toy declares an observation whose state entry is ambiguous.*steering_key_mapping",
    ):
        Ambiguous().steering_key_mapping(SPEC)

    class Blind(ToyRobot):
        def observation_spec(self, spec):
            return ObservationSpec({"proprio": ArraySpec((12,), "float32")})

    with pytest.raises(
        ValueError,
        match="toy declares an observation without a camera entry.*steering_key_mapping",
    ):
        Blind().steering_key_mapping(SPEC)


def test_an_entry_without_make_environment_fails_when_instantiated() -> None:
    class Incomplete(SimulatorEntry):
        name = "incomplete"

    with pytest.raises(TypeError, match="make_environment"):
        Incomplete()


def test_an_entry_without_a_name_fails_when_instantiated() -> None:
    class Nameless(SimulatorEntry):
        def make_environment(self, spec, phase, context):
            return object()

    with pytest.raises(TypeError, match="class attribute `name`"):
        Nameless()


def test_the_renamed_backends_attribute_fails_at_class_definition() -> None:
    with pytest.raises(TypeError, match="backends was renamed to base_models"):

        class Old(SimulatorEntry):
            name = "old"
            backends = ("gr00t",)

            def make_environment(self, spec, phase, context):
                return object()


def test_malformed_declarations_fail_at_class_definition() -> None:
    with pytest.raises(ValueError, match="evaluation must be one of"):

        class BadEvaluation(SimulatorEntry):  # NOSONAR - defining it is the check
            name = "bad"
            evaluation = "sometimes"

            def make_environment(self, spec, phase, context):
                return object()

    with pytest.raises(ValueError, match="decode_batch_size"):

        class BadDecode(SimulatorEntry):  # NOSONAR - defining it is the check
            name = "bad"
            decode_batch_size = 0

            def make_environment(self, spec, phase, context):
                return object()

    with pytest.raises(ValueError, match="server_url_variables"):

        class BadServer(SimulatorEntry):  # NOSONAR - defining it is the check
            name = "bad"
            server_url_variables = ("ONLY_ONE",)

            def make_environment(self, spec, phase, context):
                return object()

    with pytest.raises(ValueError, match="workers"):
        WorkerLayout(workers=0)


def test_the_default_layout_runs_on_one_simulator() -> None:
    """Only an entry that spreads a phase over several servers overrides worker_layout for num_servers > 1."""
    entry = ToyRobot()
    assert entry.worker_layout(4, num_servers=1) == WorkerLayout(
        workers=4, batched=False
    )
    with pytest.raises(ValueError, match="one simulator; 2 server addresses"):
        entry.worker_layout(4, num_servers=2)


def test_environment_spec_lists_its_server_addresses() -> None:
    from isaac_rl_posttraining.envs.simulator_entry import server_addresses

    assert server_addresses(None) == ()
    assert server_addresses("tcp://a:1,tcp://a:2") == ("tcp://a:1", "tcp://a:2")
    assert SPEC.server_addresses() == () and SPEC.num_servers == 1
    one = EnvironmentSpec(
        env_name="toy/reach",
        num_envs=2,
        max_episode_steps=50,
        seed=3,
        server_url="tcp://a:1",
    )
    assert one.server_addresses() == ("tcp://a:1",) and one.num_servers == 1
    two = EnvironmentSpec(
        env_name="toy/reach",
        num_envs=4,
        max_episode_steps=50,
        seed=3,
        server_url="tcp://a:1, tcp://a:2",
    )
    assert two.server_addresses() == ("tcp://a:1", "tcp://a:2") and two.num_servers == 2
    with pytest.raises(ValueError, match="comma-separated list"):
        EnvironmentSpec(
            env_name="toy/reach",
            num_envs=4,
            max_episode_steps=50,
            seed=3,
            server_url="tcp://a:1,",
        )


def test_registry_accepts_a_class_an_instance_or_a_module_path(registry) -> None:
    registry.register_simulator_entry("toy", ToyRobot)
    first = registry.get_simulator_entry("toy/reach")
    assert isinstance(first, ToyRobot)
    assert registry.get_simulator_entry("toy/other") is first, (
        "one instance per entry class"
    )

    instance = ToyRobot()
    registry.register_simulator_entry("toy", instance, override=True)
    assert registry.get_simulator_entry("toy/reach") is instance

    registry.register_simulator_entry("toy", f"{__name__}:ToyRobot", override=True)
    assert registry.get_simulator_entry("toy/reach") is first
    with pytest.raises(ValueError, match="already registered"):
        registry.register_simulator_entry("toy", ToyRobot)
    with pytest.raises(ValueError, match="module:Class"):
        registry.register_simulator_entry("toy2", "not_a_path")
    with pytest.raises(TypeError, match="subclass SimulatorEntry"):
        registry.register_simulator_entry("toy3", object())


def test_registry_lists_the_segments_it_serves(registry) -> None:
    registry.register_simulator_entry("toy", ToyRobot)
    segments = registry.available_simulator_entries()
    assert "toy" in segments and "arena" in segments
    assert segments == sorted(segments)


def test_registry_names_the_missing_method_of_an_incomplete_entry(registry) -> None:
    class Incomplete(SimulatorEntry):
        name = "incomplete"

    registry.register_simulator_entry("incomplete", Incomplete)
    with pytest.raises(
        TypeError, match="not a complete simulator entry.*make_environment"
    ):
        registry.get_simulator_entry("incomplete/task")


def test_config_attaches_an_out_of_tree_entry_by_module_path(registry) -> None:
    cfg = OmegaConf.create(
        {
            "env": {"env_name": "toy/reach"},
            "simulator": {"entry": f"{__name__}:ToyRobot"},
        }
    )

    entry = resolve_simulator_entry(cfg)

    assert isinstance(entry, ToyRobot)
    assert registry.find_simulator_entry("toy/reach") is entry
    with pytest.raises(ValueError, match="simulator.entry=<module>:<Class>"):
        resolve_simulator_entry(
            OmegaConf.create({"env": {"env_name": "nowhere/task"}, "simulator": {}})
        )


def test_a_simulator_group_that_disagrees_with_the_env_name_is_refused() -> None:
    def cfg(env_name: str, provider: str):
        return OmegaConf.create(
            {"env": {"env_name": env_name}, "simulator": {"provider": provider}}
        )

    with pytest.raises(
        ValueError, match="runs the g1_arena simulator, but simulator= selects a libero"
    ):
        resolve_simulator_entry(cfg("arena/drill_lift", "libero"))
    assert (
        resolve_simulator_entry(cfg("libero_pi0/libero_90/task", "libero")).name
        == "libero_pi0"
    )
    assert (
        resolve_simulator_entry(cfg("arena/drill_lift", "g1_arena")).name == "g1_arena"
    )


def test_dry_run_plan_names_the_config_selected_entry(
    registry, tmp_path, monkeypatch
) -> None:
    from hydra import compose, initialize_config_dir

    from isaac_rl_posttraining.configs import register_configs
    from isaac_rl_posttraining.train import CONFIG_DIR

    register_configs()
    monkeypatch.chdir(tmp_path)

    def toy_plan(*overrides: str):
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            cfg = compose(
                config_name="train",
                overrides=[
                    "algorithm=dsrl_sac",
                    "base_model=mock",
                    "env.env_name=toy/reach",
                    # The root config's simulator group (g1_arena) has no cap for this task.
                    "env.max_episode_steps=50",
                    f"+simulator.entry={__name__}:ToyRobot",
                    *overrides,
                ],
            )
        return train_plan(cfg)

    plan = toy_plan()

    # The plan names the entry that runs, not the root config's simulator group.
    assert plan.simulator == "toy"
    assert plan.entry.endswith("ToyRobot(name='toy')")
    assert plan.image_keys == "image from toy.steering_key_mapping"
    assert plan.layout == "1 rollout worker(s) x 1 environment(s)"
    assert plan.evaluation_mode == "shared"
    # Re-scoring a checkpoint has no training workers to share: it evaluates on workers of its own.
    rescore = toy_plan(
        "training.num_rollouts=0",
        "eval.eval_on_first_step=true",
        "training.load_dir=/ckpt/final",
    )
    assert rescore.evaluation_mode == "dedicated"
    assert "on workers of their own" in rescore.evaluation
    assert (
        toy_plan("env.image_keys=[wrist_cam]").image_keys
        == "wrist_cam from env.image_keys"
    )


def _arena_plan_config(*overrides: str):
    from hydra import compose, initialize_config_dir

    from isaac_rl_posttraining.configs import register_configs
    from isaac_rl_posttraining.train import CONFIG_DIR

    register_configs()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(
            config_name="train",
            overrides=["algorithm=dsrl_sac", "base_model=mock", *overrides],
        )


def test_plan_lists_a_served_simulators_addresses_and_the_launch_requires_them(
    monkeypatch, tmp_path
) -> None:
    from isaac_rl_posttraining.runners.build import build_run

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ARENA_SERVER_URL", raising=False)
    monkeypatch.delenv("ARENA_EVAL_SERVER_URL", raising=False)
    # The dry run names the missing addresses as launch inputs and runs every other check.
    assert train_plan(_arena_plan_config()).launch_inputs == (
        "ARENA_SERVER_URL (train) needs to be exported at launch; "
        "ARENA_EVAL_SERVER_URL (eval) needs to be exported at launch"
    )
    with pytest.raises(
        ValueError, match="g1_arena needs ARENA_SERVER_URL for its train environments"
    ):
        build_run(_arena_plan_config())
    monkeypatch.setenv("ARENA_SERVER_URL", "tcp://127.0.0.1:7300")
    with pytest.raises(
        ValueError,
        match="g1_arena needs ARENA_EVAL_SERVER_URL for its eval environments",
    ):
        build_run(_arena_plan_config())
    # A run that only evaluates needs only the evaluation address; one that neither trains nor evaluates needs none.
    monkeypatch.delenv("ARENA_SERVER_URL")
    assert train_plan(_arena_plan_config("training.num_rollouts=0")).launch_inputs == (
        "ARENA_EVAL_SERVER_URL (eval) needs to be exported at launch"
    )
    assert (
        train_plan(
            _arena_plan_config(
                "training.num_rollouts=0",
                "eval.eval_freq=0",
                "eval.eval_on_first_step=false",
            )
        ).launch_inputs
        == "none (no phase opens environments)"
    )
    monkeypatch.setenv("ARENA_SERVER_URL", "tcp://127.0.0.1:7300")
    monkeypatch.setenv("ARENA_EVAL_SERVER_URL", "tcp://127.0.0.1:7301")
    assert train_plan(_arena_plan_config()).evaluation_mode == "dedicated"


def test_worker_specs_build_the_entrys_environment_on_the_worker() -> None:
    entry = ToyRobot()
    gym_spec = GymWorkerSpec(
        entry=entry, environment=SPEC, phase="eval", policy=lambda context: object()
    )
    assert gym_spec.make_environment(CONTEXT) == ("env", "toy/reach", "eval", (1,))

    class Scenes(ToyRobot):
        def make_scenes(self, spec, context):
            return ("scenes", context.environment_indices)

    batched = BatchedWorkerSpec(
        entry=Scenes(),
        environment=SPEC,
        phase="train",
        policy=lambda context: object(),
        chunks_per_episode=2,
    )
    assert batched.make_environment(CONTEXT) == ("env", "toy/reach", "train", (1,))
    scenes = BatchedWorkerSpec(
        entry=Scenes(),
        environment=SPEC,
        phase="train",
        scenes=True,
        policy=lambda context: object(),
        chunks_per_episode=2,
    )
    assert scenes.make_environment(CONTEXT) == ("scenes", (1,))
    with pytest.raises(TypeError, match="entry must be a SimulatorEntry"):
        GymWorkerSpec(
            entry=object(),
            environment=SPEC,
            phase="train",
            policy=lambda context: object(),
        )
    with pytest.raises(ValueError, match="phase"):
        GymWorkerSpec(
            entry=entry, environment=SPEC, phase="test", policy=lambda context: object()
        )


def test_steering_step_names_the_key_mapping_when_an_observation_lacks_a_mapped_entry() -> (
    None
):
    import torch

    step = SteeringStep(
        base_model=object(),
        device=torch.device("cpu"),
        steering_key_mapping=dict(CONVENTIONAL_STEERING_KEYS),
    )

    with pytest.raises(KeyError, match="no state entry 'state'.*steering_key_mapping"):
        step.process_obs({"proprio": np.zeros(3, dtype=np.float32)}, squeeze=True)
    with pytest.raises(KeyError, match="no image entry 'image'.*steering_key_mapping"):
        step.process_obs({"state": np.zeros((1, 3), dtype=np.float32)})


def test_base_model_support_check_exempts_the_mock_base_model() -> None:
    from isaac_rl_posttraining.runners.simulator_entries import (
        require_base_model_support,
    )

    class Gr00tOnly(ToyRobot):
        base_models = ("gr00t",)

    entry = Gr00tOnly()
    require_base_model_support(
        OmegaConf.create({"base_model": {"name": "gr00t"}}), entry
    )
    require_base_model_support(
        OmegaConf.create({"base_model": {"name": "mock"}}), entry
    )
    with pytest.raises(
        ValueError,
        match="builds environments for base model\\(s\\) \\['gr00t'\\]; got base_model='pi0'",
    ):
        require_base_model_support(
            OmegaConf.create({"base_model": {"name": "pi0"}}), entry
        )


def test_offline_family_does_not_open_training_environments() -> None:
    from isaac_rl_posttraining.runners.simulator_entries import required_phases

    train = OmegaConf.create(
        {
            "algorithm": {"family": "steering"},
            "training": {"num_rollouts": 1000},
            "eval": {"eval_freq": 0, "eval_on_first_step": False},
            "ppo": {"eval_only": False},
        }
    )
    assert required_phases(train) == ("train",)
    offline = OmegaConf.create(
        {
            "algorithm": {"family": "offline"},
            "training": {"num_rollouts": 1000},
            "eval": {"eval_freq": 5000, "eval_on_first_step": True},
            "ppo": {"eval_only": False},
        }
    )
    assert required_phases(offline) == ("eval",)
    eval_only = OmegaConf.create(
        {
            "algorithm": {"family": "offline"},
            "training": {"num_rollouts": 0},
            "eval": {"eval_freq": 0, "eval_on_first_step": True},
            "ppo": {"eval_only": False},
        }
    )
    assert required_phases(eval_only) == ("eval",)
