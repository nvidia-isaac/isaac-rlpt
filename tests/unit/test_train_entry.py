# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""scripts/train.py: family dispatch on one root config, the launch plan, and the retired entries."""

from __future__ import annotations

import functools
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from isaac_rl_posttraining.compat import rlinf
from isaac_rl_posttraining.configs import register_configs
from isaac_rl_posttraining.runners import (
    dsrl_ppo_launch,
    flow_sde_ppo_launch,
    off_policy_launch,
    offline_launch,
)
from isaac_rl_posttraining.runners.build import build_run, train_plan

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.legacy_entry import translate_overrides

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "isaac_rl_posttraining" / "configs"
RETIRED_ENTRIES = {
    "scripts/ppo/train_dsrl_ppo.py": [
        "--config-name",
        "dsrl",
        "base_model=mock",
        "ppo.local_learner_device=cpu",
    ],
    "scripts/ppo/train_isaaclab_gr00t_ppo.py": [
        "base_model.ckpt=/ckpt/x",
        "ppo.num_learner_ranks=1",
        "runtime=native",
        "runtime.num_rollout_workers=1",
        "training.num_envs=16",
    ],
}


def _compose(*overrides: str):
    register_configs()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="train", overrides=list(overrides))


@pytest.mark.parametrize(
    "overrides,run",
    [
        (("algorithm=dsrl_sac", "base_model=mock"), off_policy_launch.run),
        (("algorithm=rfs", "base_model=mock"), off_policy_launch.run),
        (
            ("algorithm=sac_edit", "training.chunk_size=null", "base_model=mock"),
            off_policy_launch.run,
        ),
        (("algorithm=dsrl_ppo", "base_model=mock"), dsrl_ppo_launch.run),
        (("algorithm=ppo", "base_model.ckpt=/ckpt/x"), flow_sde_ppo_launch.run),
        (("algorithm=iql", "base_model=gr00t_n17"), offline_launch.run),
        (("algorithm=sft", "base_model=gr00t_n17"), offline_launch.run),
    ],
)
def test_build_run_dispatches_on_the_algorithm_family(overrides, run, monkeypatch):
    # ppo runs on runtime=rlinf, which a launch requires installed.
    monkeypatch.setattr(rlinf, "is_available", lambda: True)
    built = build_run(_compose(*overrides))
    assert isinstance(built, functools.partial) and built.func is run


def test_off_policy_plan_comes_from_the_simulator_entry():
    plan = train_plan(_compose("algorithm=rfs", "base_model=mock"))
    assert (plan.simulator, plan.layout, plan.evaluation_mode) == (
        "g1_arena",
        "1 rollout worker(s) x 1 environment(s)",
        "dedicated",
    )
    assert plan.entry == "simulators.g1_arena.runtime.G1Arena(name='g1_arena')"
    assert (plan.algorithm, plan.family, plan.policy_family) == (
        "rfs",
        "steering",
        "steering",
    )
    assert plan.schedule.startswith(
        "40000 env steps, learning_starts=1200, eval_freq=5000 (50 episodes)"
    )
    assert plan.checkpoint_dir.endswith("/dsrl_<timestamp>/checkpoints")


def test_edit_family_plans_like_the_steering_family():
    plan = train_plan(
        _compose("algorithm=sac_edit", "training.chunk_size=null", "base_model=mock")
    )
    assert (plan.family, plan.policy_family, plan.evaluation_mode) == (
        "edit",
        "edit",
        "dedicated",
    )
    assert plan.layout == "1 rollout worker(s) x 1 environment(s)"
    assert (
        "plan.placement.policy.base+actor=served by one in-process policy server"
        in "\n".join(plan.lines())
    )


def test_plan_without_evaluation_says_so():
    plan = train_plan(
        _compose(
            "algorithm=dsrl_sac",
            "base_model=mock",
            "eval.eval_freq=0",
            "eval.eval_on_first_step=false",
        )
    )
    assert plan.evaluation_mode == "disabled"


@pytest.mark.parametrize(
    "overrides,match",
    [
        (
            ("placement.learner.ranks=2", "runtime.native.processes=single"),
            "placement.learner.ranks must be 1",
        ),
        (("learner.device=gpu",), "learner.device"),
    ],
)
def test_off_policy_families_train_in_the_controller_process(overrides, match):
    cfg = _compose("algorithm=dsrl_sac", "base_model=mock", *overrides)
    with pytest.raises(ValueError, match=match):
        train_plan(cfg)
    with pytest.raises(ValueError, match=match):
        build_run(cfg)


def test_learner_section_is_validated():
    with pytest.raises(ValueError, match="placement.learner.ranks"):
        train_plan(
            _compose(
                "algorithm=ppo",
                "base_model.ckpt=/ckpt/x",
                "placement.learner.ranks=-1",
            )
        )
    with pytest.raises(ValueError, match="learner.device"):
        train_plan(
            _compose("algorithm=ppo", "base_model.ckpt=/ckpt/x", "learner.device=tpu")
        )


def test_legacy_override_translation():
    assert translate_overrides(
        [
            "--config-name=dsrl",
            "alg.online_alg_name=rfs",
            "online_algorithm.batch_size=128",
            "base_model=mock",
        ],
        algorithm=None,
    ) == ["algorithm=rfs", "algorithm.batch_size=128", "base_model=mock"]
    assert translate_overrides(
        ["-cn", "dsrl", "algorithm@online_algorithm=sac_edit"], algorithm=None
    ) == ["algorithm=sac_edit"]
    assert translate_overrides(
        [
            "ppo.num_learner_ranks=4",
            "ppo.local_learner_device=cpu",
            "algorithm@online_algorithm=rfs",
        ],
        algorithm="dsrl_ppo",
    ) == ["placement.learner.ranks=4", "learner.device=cpu", "algorithm=dsrl_ppo"]


@pytest.mark.parametrize("entry", sorted(RETIRED_ENTRIES))
def test_retired_entries_forward_to_train(entry, tmp_path):
    env = {
        k: v for k, v in os.environ.items() if k != "ISAAC_RL_POSTTRAINING_OUTPUT_DIR"
    }
    env.update({"ISAAC_RL_POSTTRAINING_DRYRUN": "1", "PYTHONPATH": str(REPO_ROOT)})
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / entry), *RETIRED_ENTRIES[entry]],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert f"[deprecated] {entry} forwards to scripts/train.py" in result.stderr
    expected = {
        "train_dsrl_ppo.py": "dsrl_ppo",
        "train_isaaclab_gr00t_ppo.py": "ppo",
    }[Path(entry).name]
    assert f"plan.algorithm={expected}" in result.stdout
    assert (
        "online_algorithm" not in result.stdout
        and "num_learner_ranks" not in result.stdout
    )


def test_dsrl_ppo_ignores_demonstration_buffers(monkeypatch):
    """PPO trains on fresh episodes: a configured demonstration buffer is dropped before setup."""
    seen = {}

    class _Stop(Exception):
        pass

    def _setup(cfg):
        seen["demonstrations"] = cfg.buffer.demonstrations
        seen["fraction"] = cfg.buffer.demonstration_fraction
        raise _Stop

    monkeypatch.setattr(dsrl_ppo_launch, "setup_dsrl_training", _setup)
    cfg = _compose(
        "algorithm=dsrl_ppo",
        "base_model=mock",
        "buffer.demonstrations=/runs/x/replay_buffers/buffer_latest",
        "buffer.demonstration_fraction=0.25",
    )
    with pytest.raises(_Stop):
        dsrl_ppo_launch.run(cfg)

    assert seen == {"demonstrations": None, "fraction": None}


@pytest.mark.parametrize(
    "override,match",
    [
        ("eval.eval_freq=100", r"remove eval\.eval_freq \(or set eval\.eval_freq=0\)"),
        (
            "eval.eval_on_first_step=true",
            None,
        ),  # the default: switched off without complaint
        ("eval.eval_record_video=false", None),
    ],
)
def test_dsrl_ppo_rejects_the_eval_keys_it_would_overwrite(override, match):
    cfg = _compose("algorithm=dsrl_ppo", "base_model=mock", override)
    if match is None:
        dsrl_ppo_launch.reject_off_policy_eval_keys(cfg)
        return
    with pytest.raises(ValueError, match=match):
        dsrl_ppo_launch.reject_off_policy_eval_keys(cfg)
    with pytest.raises(ValueError, match=match):
        train_plan(cfg)


def test_dsrl_ppo_reports_other_eval_keys_as_ignored(monkeypatch):
    warnings = []
    monkeypatch.setattr(dsrl_ppo_launch.logger, "warning", warnings.append)
    dsrl_ppo_launch.reject_off_policy_eval_keys(
        _compose("algorithm=dsrl_ppo", "base_model=mock", "eval.eval_num_episodes=2")
    )
    assert warnings == [
        (
            "algorithm=dsrl_ppo ignores eval.eval_num_episodes (off-policy evaluator settings); "
            "its evaluation follows ppo.eval_*"
        )
    ]


def test_the_plan_rejects_step_scheduled_settings_that_would_be_inert():
    with pytest.raises(ValueError, match="respect_step_schedules=true"):
        train_plan(
            _compose(
                "algorithm=dsrl_sac",
                "base_model=mock",
                "algorithm.actor_learning_starts=10",
            )
        )


def test_on_policy_schedules_bound_the_policy_lag_before_anything_loads():
    from omegaconf import OmegaConf

    from isaac_rl_posttraining.configs.normalize import normalize_train_config
    from isaac_rl_posttraining.runners.run_schedule import (
        check_run_config,
        schedule_spec,
    )
    from isaac_rl_posttraining.training.schedule import ScheduleSpec

    cfg = OmegaConf.create({"schedule": {"kind": "async"}})
    assert schedule_spec(cfg, on_policy=True) == ScheduleSpec("async", max_policy_lag=1)
    check_run_config(cfg, on_policy=True)
    cfg.schedule.max_policy_lag = 3
    with pytest.raises(ValueError, match="max_policy_lag=1"):
        check_run_config(cfg, on_policy=True)
    overlapped = _compose(
        "algorithm=dsrl_sac", "base_model=mock", "schedule=overlapped"
    )
    assert schedule_spec(normalize_train_config(overlapped)).kind == "overlapped"


def test_off_policy_load_dir_is_logged_as_a_warm_start(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import torch
    from omegaconf import OmegaConf

    messages = []
    monkeypatch.setattr(
        off_policy_launch, "load_checkpoint", lambda algorithm, load_dir: 0
    )
    monkeypatch.setattr(off_policy_launch.logger, "info", messages.append)
    algorithm = SimpleNamespace(actor=torch.nn.Linear(1, 1))
    cfg = OmegaConf.create({"training": {"load_dir": "/runs/a/checkpoints/final"}})
    off_policy_launch._warm_start(cfg, algorithm, str(tmp_path))
    assert messages[0] == (
        "Warm start from /runs/a/checkpoints/final: learner state loaded; env steps and replay start empty"
    )
    assert not any("Resumed" in message for message in messages)


@pytest.mark.parametrize(
    "overrides,message",
    [
        (
            ["placement.envs.workers=2", "training.num_envs=2"],
            "G1 Factory runs one batched worker per simulator server",
        ),
    ],
)
def test_a_failed_plan_prints_its_problems_without_a_traceback(
    overrides, message, tmp_path
):
    env = {
        **os.environ,
        "ISAAC_RL_POSTTRAINING_DRYRUN": "1",
        "PYTHONPATH": str(REPO_ROOT),
        "ARENA_SERVER_URL": "tcp://127.0.0.1:5555",
        "ARENA_EVAL_SERVER_URL": "tcp://127.0.0.1:5556",
    }
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/train.py"),
            "base_model=mock",
            *overrides,
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=600,
        check=False,
    )
    assert result.returncode == 1, result.stderr[-2000:]
    assert message in result.stderr
    assert "Traceback" not in result.stderr, result.stderr[-2000:]


def test_an_old_config_key_is_refused_in_one_line_naming_its_replacement(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/train.py"),
            "backend=mock",
            "backend.base_ckpt=/ckpt",
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ISAAC_RL_POSTTRAINING_DRYRUN": "1",
            "PYTHONPATH": str(REPO_ROOT),
        },
        cwd=tmp_path,
        timeout=600,
        check=False,
    )
    assert result.returncode == 1, result.stderr[-2000:]
    assert result.stderr.strip() == (
        "config keys this version does not accept: "
        "backend: renamed to base_model; backend.base_ckpt: renamed to base_model.ckpt"
    )


def test_the_dry_run_prints_only_the_new_spelling_of_a_renamed_key(monkeypatch, capsys):
    from scripts.train import run_from_hydra_cfg

    monkeypatch.setenv("ISAAC_RL_POSTTRAINING_DRYRUN", "1")
    monkeypatch.setenv("ARENA_SERVER_URL", "tcp://127.0.0.1:5555")
    monkeypatch.setenv("ARENA_EVAL_SERVER_URL", "tcp://127.0.0.1:5556")
    run_from_hydra_cfg(
        _compose(
            "base_model=mock",
            "eval.eval_deterministic=true",
        )
    )
    printed = capsys.readouterr().out
    assert "action_selection: mean" in printed
    assert "eval_deterministic" not in printed


def test_sigterm_unwinds_like_ctrl_c_and_the_handler_is_restored():
    import os
    import signal

    from isaac_rl_posttraining.runners.off_policy_launch import sigterm_as_interrupt

    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt, match="SIGTERM"), sigterm_as_interrupt():
        os.kill(os.getpid(), signal.SIGTERM)
        signal.sigtimedwait([], 1)  # let the handler run
    assert signal.getsignal(signal.SIGTERM) is before


def test_an_interrupted_ppo_run_saves_an_interrupted_checkpoint(monkeypatch):
    from types import SimpleNamespace

    from isaac_rl_posttraining.runners import dsrl_ppo_launch

    saved = []

    class Trainer:
        def run(self):
            raise KeyboardInterrupt

        def save_checkpoint(self, tag):
            saved.append(tag)

    session = SimpleNamespace(
        trainer=Trainer(), run_dir="run", close=lambda error: None
    )
    monkeypatch.setattr(dsrl_ppo_launch, "open_session", lambda cfg: session)
    monkeypatch.setattr(
        dsrl_ppo_launch, "ppo_settings", lambda cfg: SimpleNamespace(eval_only=False)
    )

    with pytest.raises(KeyboardInterrupt):
        dsrl_ppo_launch.run(None)
    assert saved == ["interrupted"]


@pytest.mark.parametrize("algorithm", ["iql", "sft"])
@pytest.mark.parametrize("base_model", ["gr00t_n17", "pi0"])
def test_offline_plan_resolves_without_a_checkpoint_or_server(algorithm, base_model):
    plan = train_plan(_compose(f"algorithm={algorithm}", f"base_model={base_model}"))
    assert (plan.algorithm, plan.family, plan.loop, plan.policy_family) == (
        algorithm,
        "offline",
        "offline",
        "flow",
    )
    assert plan.layout == "offline (no collection workers)"
    assert f"{algorithm} updates on {base_model}" in plan.schedule


@pytest.mark.parametrize("algorithm", ["iql", "sft"])
@pytest.mark.parametrize(
    "base_model,message",
    [
        ("mock", "base_model=mock cannot be trained"),
        ("gr00t_n15", "no demonstration reader for base_model=gr00t_n15"),
    ],
)
def test_offline_refuses_a_base_model_it_cannot_train(algorithm, base_model, message):
    cfg = _compose(f"algorithm={algorithm}", f"base_model={base_model}")
    with pytest.raises(ValueError, match=message):
        train_plan(cfg)
    with pytest.raises(ValueError, match=message):
        build_run(cfg)


def test_offline_refuses_a_runtime_that_moves_evaluation_out_of_process():
    with pytest.raises(ValueError, match="select runtime=native"):
        train_plan(_compose("algorithm=sft", "base_model=gr00t_n17", "runtime=rlinf"))


@pytest.mark.parametrize(
    "overrides,key,given",
    [
        (("algorithm=dsrl_sac", "base_model=mock"), "training.load_dir", "runs/final"),
        (
            ("algorithm=ppo", "base_model.ckpt=/ckpt/x"),
            "ppo.resume_from",
            "runs/iteration_3",
        ),
    ],
)
def test_relative_checkpoint_directories_resolve_against_the_callers_directory(
    tmp_path, monkeypatch, overrides, key, given
):
    from omegaconf import OmegaConf

    from scripts.train import run_from_hydra_cfg

    cfg = _compose(*overrides, f"{key}={given}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ISAAC_RL_POSTTRAINING_DRYRUN", "1")
    run_from_hydra_cfg(cfg)
    expected = str(tmp_path / given)
    assert OmegaConf.select(cfg, key) == expected
