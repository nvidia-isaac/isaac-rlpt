# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resource lifecycle and pre-flight checks of the off-policy run."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from isaac_rl_posttraining.envs.simulator_entry import SimulatorEntry
from isaac_rl_posttraining.runners import off_policy_evaluation, off_policy_launch
from isaac_rl_posttraining.runtimes.policy_snapshot import policy_state_from_tensors
from isaac_rl_posttraining.training.checkpoints import policy_state_fingerprint

CONFIG_DIR = Path(__file__).resolve().parents[2] / "isaac_rl_posttraining" / "configs"


class _Resource:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _Logger:
    def __init__(self) -> None:
        self.finished = 0
        self.records = []

    def finish(self) -> None:
        self.finished += 1

    def log(self, values, *, step, prefix) -> None:
        self.records.append((prefix, step, values))


class _TestSimulator(SimulatorEntry):
    name = "test"
    evaluation = "fresh_workers"

    def make_environment(self, spec, phase, context):
        del spec, phase, context
        return object()

    def steering_key_mapping(self, env_name):
        del env_name
        return {"state": "state", "language": None, "image": []}


ENTRY = _TestSimulator()


def test_run_closes_setup_resources_when_training_fails(monkeypatch) -> None:
    """The local environment, every opened runtime and the logger close even when the loop raises."""
    env = _Resource()
    logger = _Logger()
    runtime = _Resource()
    components = SimpleNamespace(env=env, wandb_logger=logger)

    monkeypatch.setattr(off_policy_launch, "require_simulator_entry", lambda cfg: ENTRY)
    monkeypatch.setattr(
        off_policy_launch, "setup_dsrl_training", lambda cfg: components
    )

    def fail_after_opening_the_runtime(cfg, built, entry, runtimes):
        del cfg, built, entry
        runtimes.append(runtime)
        raise RuntimeError("attachment failed")

    monkeypatch.setattr(off_policy_launch, "_train", fail_after_opening_the_runtime)

    with pytest.raises(RuntimeError, match="attachment failed"):
        off_policy_launch.run(OmegaConf.create({"training": {"num_rollouts": 1}}))

    assert env.closed == 1
    assert runtime.closed == 1
    assert logger.finished == 1


def test_run_requires_a_simulator_entry() -> None:
    with pytest.raises(ValueError, match="simulator entry"):
        off_policy_launch.run(
            OmegaConf.create(
                {
                    "training": {"num_rollouts": 1},
                    "runtime": {"name": "native"},
                    "env": {"env_name": "CartPole-v1"},
                    "algorithm": {"name": "dsrl_sac"},
                }
            )
        )


def _compose(*overrides: str):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="train", overrides=["base_model=mock", *overrides])


@pytest.fixture
def dry_run(monkeypatch, tmp_path):
    """The entry point's dry run, with any simulator or model work turned into a test failure."""
    monkeypatch.setenv("ISAAC_RL_POSTTRAINING_DRYRUN", "1")
    monkeypatch.chdir(tmp_path)

    def never_setup(*args, **kwargs):
        raise AssertionError("the dry run must fail before any simulator work")

    monkeypatch.setattr(off_policy_launch, "setup_dsrl_training", never_setup)
    from scripts.train import run_from_hydra_cfg

    return run_from_hydra_cfg


@pytest.mark.parametrize(
    "overrides",
    [
        ("training.num_envs=2", "training.rollout_count_by=step"),
        ("training.num_envs=2", "training.rollout_count_by=episode"),
        ("eval.eval_num_envs=2", "eval.eval_num_episodes=2"),
        ("eval.eval_num_envs=2", "eval.eval_freq=0", "eval.eval_on_first_step=false"),
    ],
    ids=["count_by_step", "count_by_episode", "batched_eval", "batched_eval_disabled"],
)
def test_dry_run_accepts_batched_collection_and_evaluation(dry_run, overrides) -> None:
    dry_run(_compose(*overrides))


def test_dry_run_rejects_evaluation_episodes_that_do_not_fill_the_environments(
    dry_run,
) -> None:
    """The evaluator scores whole rounds of eval_num_envs; the plan refuses a count it cannot score."""
    with pytest.raises(
        SystemExit,
        match=r"eval\.eval_num_episodes \(6\) must be an exact multiple of eval\.eval_num_envs \(4\)",
    ):
        dry_run(_compose("eval.eval_num_envs=4", "eval.eval_num_episodes=6"))
    with pytest.raises(
        SystemExit,
        match=r"eval\.eval_num_episodes \(2\) must be at least eval\.eval_num_envs \(4\)",
    ):
        dry_run(_compose("eval.eval_num_envs=4", "eval.eval_num_episodes=2"))
    # Without evaluation the counts are irrelevant.
    dry_run(
        _compose(
            "eval.eval_num_envs=4",
            "eval.eval_num_episodes=6",
            "eval.eval_freq=0",
            "eval.eval_on_first_step=false",
        )
    )


def test_evaluation_only_opens_only_the_evaluation_path(monkeypatch, tmp_path) -> None:
    env = _Resource()
    logger = _Logger()
    phases = []
    delegated = []
    eval_spec = SimpleNamespace(reward_channel="env")

    monkeypatch.setattr(
        off_policy_launch,
        "rollout_worker_spec_for",
        lambda cfg, components, entry, *, phase, num_envs, configured_workers: (
            phases.append((phase, num_envs, configured_workers)) or eval_spec
        ),
    )
    monkeypatch.setattr(
        off_policy_launch,
        "run_evaluation_only",
        lambda *args, **kwargs: delegated.append((args, kwargs)),
    )
    components = SimpleNamespace(
        env=env,
        actor=SimpleNamespace(state_dict=dict),
        algorithm=SimpleNamespace(step_counter=0),
        wandb_logger=logger,
        save_dir=str(tmp_path / "checkpoints"),
        buffer=object(),
        device="cpu",
        algorithm_name="dsrl_sac",
    )
    cfg = OmegaConf.create(
        {
            "training": {"num_rollouts": 0, "num_envs": 128},
            "eval": {
                "eval_freq": 0,
                "eval_on_first_step": True,
                "eval_num_envs": 2,
            },
            "buffer": {"reward_aggregation": "sum"},
            "runtime": {},
            "dsrl": {"action_exec_len": 4},
        }
    )
    runtimes = []

    off_policy_launch._train(cfg, components, ENTRY, runtimes)

    assert phases == [("eval", 2, None)]
    assert env.closed == 1
    assert components.env is None
    ((args, kwargs),) = delegated
    assert args[0:3] == (cfg, ENTRY, eval_spec)
    assert kwargs["eval_num_envs"] == 2
    assert kwargs["runtimes"] is runtimes
    assert kwargs["save_dir"] == str(tmp_path / "checkpoints")
    assert kwargs["output_dir"] == str(tmp_path)


def test_evaluation_only_runner_logs_and_closes(monkeypatch, tmp_path) -> None:
    runtime = _Resource()
    evaluator = _Resource()
    evaluator.evaluate = lambda state: {
        "success_rate": 0.5,
        "num_eval_episodes": 2.0,
    }
    logger = _Logger()
    eval_spec = SimpleNamespace(reward_channel="env")
    open_arguments = []
    monkeypatch.setattr(off_policy_evaluation, "open_runtime", lambda config: runtime)

    def open_evaluator(cfg, entry, opened_runtime, train_group, spec, **kwargs):
        open_arguments.append((opened_runtime, train_group, spec, kwargs))
        return evaluator

    monkeypatch.setattr(off_policy_evaluation, "open_evaluator", open_evaluator)
    cfg = OmegaConf.create({"buffer": {"reward_aggregation": "sum"}})
    actor = SimpleNamespace(state_dict=dict)
    runtimes = []

    off_policy_evaluation.run_evaluation_only(
        cfg,
        ENTRY,
        eval_spec,
        eval_num_envs=2,
        runtimes=runtimes,
        actor=actor,
        optimizer_updates=0,
        experiment_logger=logger,
        output_dir=str(tmp_path),
        save_dir=str(tmp_path / "checkpoints"),
    )

    assert open_arguments[0][0:3] == (runtime, None, eval_spec)
    assert evaluator.closed == 1
    assert runtimes == [runtime]
    # The score is on disk where a training run's is, with the digest of the scored weights.
    summary = json.loads(
        (tmp_path / "checkpoints" / "evaluation_summary.json").read_text()
    )
    assert summary["latest"]["step"] == 0
    assert summary["latest"]["success_rate"] == pytest.approx(0.5)
    assert summary["history"] == [summary["latest"]]
    (fingerprint,) = (
        (tmp_path / "checkpoints" / "fingerprints.jsonl").read_text().splitlines()
    )
    record = json.loads(fingerprint)
    assert (record["event"], record["step"], record["success_rate"]) == ("eval", 0, 0.5)
    assert record["actor_sha256"] == policy_state_fingerprint(
        policy_state_from_tensors({}, version=1)
    )
    assert logger.records == [
        (
            "eval",
            0,
            {
                "success_rate": 0.5,
                "num_eval_episodes": 2.0,
                "simulator_steps": 0.0,
                "optimizer_updates": 0.0,
            },
        )
    ]
