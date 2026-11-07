# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launcher dry-run contracts for example_workflows/g1_arena_posttraining/scripts/{train,eval}.sh.

With ``ISAAC_RL_POSTTRAINING_DRYRUN=1`` each launcher prints its fully resolved plan (task,
max-steps, sim devices, hydra overrides) and exits 0 without booting a
simulator or learner. These tests parse that plan and pin:

* the 9-task max-episode-steps map (a wrong flat cap distorts success rates),
* ``GR00T_MULTITASK_DIRS`` is force-unset (single-task normalization stats
  must not silently inherit a multi-task override from the caller's shell),
* native image resolution (``--image-resolution 0``): pre-resizing the ego
  camera is out-of-distribution for the policy and cripples manipulation,
* eval base mode maps to an eval-only run of the native policy; policy mode
  loads the given checkpoint directory,
* ``TRAIN_SIM_DEVICE`` / ``EVAL_SIM_DEVICE`` plumb through to the respective
  simulator processes (per-sim CPU/GPU physics selection).
"""

import re
import sys
from pathlib import Path

import pytest

from example_workflows.conftest import (
    EVAL_LAUNCHER,
    TASK_MAX_STEPS,
    TRAIN_LAUNCHER,
    dryrun_text,
    run_launcher_dryrun,
)

# Launch with the LM prefix stripped: the launchers accept both forms and
# must resolve to the canonical LM task name.
_SHORT = {task: task[2:] for task in TASK_MAX_STEPS}


def _assert_dryrun_ok(result):
    assert result.returncode == 0, (
        f"dry-run must exit 0; rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _assert_max_steps(text, expected):
    pattern = re.compile(
        rf"(env\.max_episode_steps\s*=\s*{expected}\b|max[_ ]?steps\D{{0,3}}{expected}\b)",
        re.IGNORECASE,
    )
    assert pattern.search(text), (
        f"resolved plan does not show max episode steps = {expected}:\n{text}"
    )


def _device_mentions(text, role):
    """Devices mentioned on plan lines that concern the given sim role (train/eval)."""
    devices = set()
    env_var = f"{role.upper()}_SIM_DEVICE"
    for line in text.splitlines():
        low = line.lower()
        m = re.search(rf"{env_var}\s*[:=]\s*([a-z0-9:]+)", line, re.IGNORECASE)
        if m:
            devices.add(m.group(1).lower())
            continue
        if role in low and ("device" in low or "sim" in low):
            for dev in ("cpu", "gpu", "cuda"):
                if re.search(rf"\b{dev}\b", low):
                    devices.add(dev)
    return devices


def test_train_launcher_uses_the_explicit_simulator_interpreter() -> None:
    text = Path(TRAIN_LAUNCHER).read_text(encoding="utf-8")

    assert "source .venv/bin/activate" not in text
    assert 'exec setsid "$SIM_PY"' in text


@pytest.mark.parametrize(
    "task,expected_steps",
    # One drill task and one non-drill task: every task goes through the same yaml lookup, and the
    # config's map is pinned against TASK_MAX_STEPS in tests/unit/test_simulators.py.
    [(task, TASK_MAX_STEPS[task]) for task in ("LMDrillLift", "LMPushShelfForward")],
)
def test_max_steps_resolve_from_the_simulator_config(
    launcher_sandbox, task, expected_steps
):
    """A benchmark task resolves to its per-task episode cap in the train plan."""
    result = run_launcher_dryrun(TRAIN_LAUNCHER, [_SHORT[task]], launcher_sandbox)
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert task in text, f"plan must echo the canonical task name {task}:\n{text}"
    _assert_max_steps(text, expected_steps)


def test_multitask_env_vars_gone_from_plan(launcher_sandbox):
    """The retired multitask env vars must not appear in (or leak into) the plan.

    Multitask stats selection moved from GR00T_MULTITASK_DIRS/GR00T_TASK_DIR
    environment variables to explicit config; the launchers must neither
    mention those vars nor let caller-set values reach the resolved plan.
    """
    sentinel = "/nonexistent/BOGUS_MULTITASK_DIR_SENTINEL"
    result = run_launcher_dryrun(
        TRAIN_LAUNCHER,
        ["DrillLift"],
        launcher_sandbox,
        extra_env={"GR00T_MULTITASK_DIRS": sentinel, "GR00T_TASK_DIR": sentinel},
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert sentinel not in text, (
        "a caller-set multitask env var leaked into the plan "
        "(single-task stats would silently become multi-task):\n" + text
    )
    for retired in ("GR00T_MULTITASK_DIRS", "GR00T_TASK_DIR"):
        assert retired not in text, (
            f"the launcher still references the retired env var {retired} "
            f"(tuning must flow through config, not the environment):\n{text}"
        )


def test_default_plan_pins_native_resolution_and_boots_nothing(launcher_sandbox):
    """The plan pins native image resolution (0 = no pre-resize) for sim and learner, and a dry run
    never waits on or leaves behind a simulator process."""
    result = run_launcher_dryrun(
        TRAIN_LAUNCHER, ["DrillLift"], launcher_sandbox, timeout=30
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    pattern = re.compile(
        r"(--image-resolution[= ]0\b|ARENA_IMAGE_RESOLUTION\s*[:=]\s*0\b|image[-_]resolution\s*[:=]\s*0\b)",
        re.IGNORECASE,
    )
    assert pattern.search(text), (
        f"plan must resolve image resolution 0 (native):\n{text}"
    )
    assert "listening on" not in text, "dry-run must not boot an env server"


def test_base_mode_maps_to_eval_only_flags(launcher_sandbox):
    """dsrl/eval.sh TASK base N: eval-only (num_rollouts=0), native base policy, N episodes."""
    result = run_launcher_dryrun(
        EVAL_LAUNCHER, ["DrillLift", "base", 50], launcher_sandbox
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert re.search(r"num_rollouts\s*=\s*0\b", text), (
        f"base mode must map to an eval-only run (num_rollouts=0):\n{text}"
    )
    assert re.search(r"eval_base\s*=\s*[Tt]rue", text), (
        f"base mode must pass eval.eval_base=true to the learner (config field, not env; "
        f"a mode label alone does not flip the policy to native):\n{text}"
    )
    assert re.search(r"eval_num_episodes\s*=\s*50\b", text), (
        f"the episode count must plumb through to the eval config:\n{text}"
    )
    assert not re.search(r"DSRL_EVAL_BASE", text), (
        f"base mode must use the eval.eval_base config field, not the retired env var:\n{text}"
    )


def test_policy_mode_maps_to_load_dir(launcher_sandbox):
    """dsrl/eval.sh TASK policy N CKPT loads the SAC checkpoint via training.load_dir."""
    ckpt = launcher_sandbox["tmp"] / "ckpt_sentinel_dir"
    ckpt.mkdir()
    result = run_launcher_dryrun(
        EVAL_LAUNCHER, ["DrillLift", "policy", 10, str(ckpt)], launcher_sandbox
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert re.search(r"load_dir\s*=\s*\S*ckpt_sentinel_dir", text), (
        f"policy mode must resolve training.load_dir to the given checkpoint:\n{text}"
    )
    assert re.search(r"num_rollouts\s*=\s*0\b", text), (
        f"policy-mode eval is still an eval-only run:\n{text}"
    )


def test_policy_mode_without_ckpt_fails(launcher_sandbox):
    """policy mode without a checkpoint dir is a usage error, even in dry-run."""
    result = run_launcher_dryrun(
        EVAL_LAUNCHER, ["DrillLift", "policy", 10], launcher_sandbox
    )
    assert result.returncode != 0, "policy mode without CKPT_DIR must fail loudly"


def _ppo_checkpoint(sandbox, manifest='{"iteration": 1}', run_config=True):
    """A finetune_ppo.sh run's ``final`` checkpoint; older runs name the algorithm only in the run's config."""
    run = sandbox["tmp"] / "ppo_LMDrillPnP_run"
    ckpt = run / "checkpoints" / "final"
    ckpt.mkdir(parents=True)
    for name in ("learner.pt", "policy_state.pt"):
        (ckpt / name).write_bytes(b"")
    (ckpt / "manifest.json").write_text(manifest, encoding="utf-8")
    if run_config:
        (run / "ppo_config.yaml").write_text(
            "algorithm:\n  name: ppo\n", encoding="utf-8"
        )
    Path(sandbox["ckpt"], "config.json").write_text("{}\n", encoding="utf-8")
    return ckpt


@pytest.mark.parametrize(
    "manifest,algorithm",
    [
        ('{"iteration": 1, "algorithm": "ppo"}', "ppo"),
        ('{"env_steps": 5, "algorithm": "dsrl_sac"}', "dsrl_sac"),
    ],
    ids=["ppo", "dsrl_sac"],
)
def test_policy_mode_routes_by_the_manifest_algorithm(
    launcher_sandbox, manifest, algorithm
):
    """The manifest's algorithm alone (no run config next to it) picks the evaluator."""
    ckpt = _ppo_checkpoint(launcher_sandbox, manifest=manifest, run_config=False)
    result = run_launcher_dryrun(
        EVAL_LAUNCHER, ["DrillPnP", "policy", 2, str(ckpt)], launcher_sandbox
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert ("launch.override=ppo.eval_only=true\n" in text) == (algorithm == "ppo")
    assert (f"launch.override=training.load_dir={ckpt}\n" in text) == (
        algorithm != "ppo"
    )


def test_policy_mode_scores_a_full_ppo_checkpoint_through_finetune_ppo(
    launcher_sandbox,
):
    """A full-PPO checkpoint goes to finetune_ppo.sh's evaluation-only mode, not DSRL's load_dir."""
    ckpt = _ppo_checkpoint(launcher_sandbox)
    result = run_launcher_dryrun(
        EVAL_LAUNCHER,
        ["DrillPnP", "policy", 2, str(ckpt), "logging.wandb_offline=true"],
        launcher_sandbox,
        extra_env={"SIM_VIZ": "kit", "DISPLAY": ":0"},
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    for override in (
        "algorithm=ppo",
        "ppo.eval_only=true",
        "placement.learner.ranks=0",
        "ppo.eval_episodes_per_environment=1",
        f"ppo.resume_from={ckpt}",
        "training.num_envs=2",
        "logging.wandb_offline=true",
    ):
        assert f"launch.override={override}\n" in text, f"missing {override}:\n{text}"
    assert "training.load_dir" not in text
    assert "launch.viz_sim=train" in text


def test_policy_mode_refuses_a_full_ppo_checkpoint_under_an_offline_algo(
    launcher_sandbox,
):
    ckpt = _ppo_checkpoint(launcher_sandbox)
    result = run_launcher_dryrun(
        EVAL_LAUNCHER,
        ["DrillPnP", "policy", 2, str(ckpt)],
        launcher_sandbox,
        extra_env={"ALGO": "sft"},
    )
    assert result.returncode == 2
    assert "is a full-PPO checkpoint; unset ALGO" in result.stderr


def test_train_sim_device_plumbs_to_env(launcher_sandbox):
    """TRAIN_SIM_DEVICE=cpu shows up as the training simulator's device in the plan."""
    result = run_launcher_dryrun(
        TRAIN_LAUNCHER,
        ["DrillLift"],
        launcher_sandbox,
        extra_env={"TRAIN_SIM_DEVICE": "cpu", "EVAL_SIM_DEVICE": "gpu"},
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    train_devices = _device_mentions(text, "train")
    assert "cpu" in train_devices, (
        f"TRAIN_SIM_DEVICE=cpu must plumb through to the train sim; saw {train_devices or 'nothing'}:\n{text}"
    )


def test_eval_sim_device_plumbs_to_env(launcher_sandbox):
    """EVAL_SIM_DEVICE plumbs through to the evaluation simulator independently."""
    result = run_launcher_dryrun(
        TRAIN_LAUNCHER,
        ["DrillLift"],
        launcher_sandbox,
        extra_env={"TRAIN_SIM_DEVICE": "gpu", "EVAL_SIM_DEVICE": "cpu"},
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    eval_devices = _device_mentions(text, "eval")
    assert "cpu" in eval_devices, (
        f"EVAL_SIM_DEVICE=cpu must plumb through to the eval sim; saw {eval_devices or 'nothing'}:\n{text}"
    )
    train_devices = _device_mentions(text, "train")
    assert "cpu" not in train_devices or "gpu" in train_devices, (
        f"the two sims' devices must be independent; train saw {train_devices}:\n{text}"
    )


def test_ppo_eval_reset_state_directory_is_simulator_only(launcher_sandbox):
    reset_dir = launcher_sandbox["tmp"] / "evaluation-states"
    result = run_launcher_dryrun(
        TRAIN_LAUNCHER,
        ["DrillLift"],
        launcher_sandbox,
        extra_env={
            "TRAINER": "ppo",
            "EVAL_RESET_STATE_DIR": str(reset_dir),
        },
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    eval_line = next(
        line for line in text.splitlines() if line.startswith("launch.cmd.eval_sim=")
    )
    train_line = next(
        line for line in text.splitlines() if line.startswith("launch.cmd.train_sim=")
    )
    assert f"--reset-state-dir {reset_dir}" in eval_line
    assert "--reset-state-dir" not in train_line


def test_ppo_reset_observation_capture_is_eval_simulator_only(launcher_sandbox):
    capture_dir = launcher_sandbox["tmp"] / "evaluation-observations"
    result = run_launcher_dryrun(
        TRAIN_LAUNCHER,
        ["DrillLift"],
        launcher_sandbox,
        extra_env={
            "TRAINER": "ppo",
            "EVAL_RESET_OBSERVATION_DIR": str(capture_dir),
        },
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    eval_line = next(
        line for line in text.splitlines() if line.startswith("launch.cmd.eval_sim=")
    )
    train_line = next(
        line for line in text.splitlines() if line.startswith("launch.cmd.train_sim=")
    )
    assert f"--reset-observation-dir {capture_dir}" in eval_line
    assert "--reset-observation-dir" not in train_line
    assert f"launch.paths.eval_reset_observation_dir={capture_dir}" in text


def test_renderer_antialiasing_applies_to_both_ppo_simulators(launcher_sandbox):
    result = run_launcher_dryrun(
        TRAIN_LAUNCHER,
        ["DrillLift"],
        launcher_sandbox,
        extra_env={
            "TRAINER": "ppo",
            "SIM_ANTIALIASING_MODE": "fxaa",
        },
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    eval_line = next(
        line for line in text.splitlines() if line.startswith("launch.cmd.eval_sim=")
    )
    train_line = next(
        line for line in text.splitlines() if line.startswith("launch.cmd.train_sim=")
    )
    assert "--renderer-antialiasing fxaa" in eval_line
    assert "--renderer-antialiasing fxaa" in train_line
    assert "sim.antialiasing_mode=fxaa" in text


def test_strict_reproducibility_applies_to_both_ppo_simulators(launcher_sandbox):
    result = run_launcher_dryrun(
        TRAIN_LAUNCHER,
        ["DrillLift"],
        launcher_sandbox,
        extra_env={
            "TRAINER": "ppo",
            "SIM_STRICT_REPRODUCIBILITY": "1",
        },
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    eval_line = next(
        line for line in text.splitlines() if line.startswith("launch.cmd.eval_sim=")
    )
    train_line = next(
        line for line in text.splitlines() if line.startswith("launch.cmd.train_sim=")
    )
    assert "--strict-reproducibility" in eval_line
    assert "--strict-reproducibility" in train_line
    assert "sim.strict_reproducibility=1" in text


def test_ppo_eval_only_does_not_start_training_simulator(launcher_sandbox):
    reset_dir = launcher_sandbox["tmp"] / "evaluation-states"
    result = run_launcher_dryrun(
        TRAIN_LAUNCHER,
        ["DrillLift", "ppo.eval_only=true", "ppo.resume_from=/tmp/checkpoint"],
        launcher_sandbox,
        extra_env={
            "TRAINER": "ppo",
            "EVAL_RESET_STATE_DIR": str(reset_dir),
        },
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert "launch.ppo_eval_only=1" in text
    assert "launch.cmd.train_sim=<disabled for PPO evaluation-only>" in text
    eval_line = next(
        line for line in text.splitlines() if line.startswith("launch.cmd.eval_sim=")
    )
    assert f"--reset-state-dir {reset_dir}" in eval_line
    assert "launch.override=ppo.eval_only=true" in text
    assert "launch.env.learner.ARENA_SERVER_URL=tcp://127.0.0.1:7301" in text
    assert "launch.env.learner.ARENA_EVAL_SERVER_URL=tcp://127.0.0.1:7301" in text


@pytest.mark.parametrize(
    "script,env,overrides,viz_sim",
    [
        ("finetune_ppo.sh", {}, ["ppo.eval_only=true"], "train"),
        ("finetune_ppo.sh", {}, [], "eval"),
        ("train.sh", {"TRAINER": "ppo"}, ["ppo.eval_only=true"], "eval"),
        ("train.sh", {}, [], "eval"),
    ],
    ids=["ppo-eval-only", "ppo-train", "dsrl_ppo-eval-only", "dsrl_sac"],
)
def test_viz_opens_on_the_simulator_that_steps_the_rollouts(
    launcher_sandbox, script, env, overrides, viz_sim
):
    launcher = str(Path(TRAIN_LAUNCHER).with_name(script))
    Path(launcher_sandbox["ckpt"], "config.json").write_text("{}\n", encoding="utf-8")
    result = run_launcher_dryrun(
        launcher, ["DrillLift", *overrides], launcher_sandbox, extra_env=env
    )
    _assert_dryrun_ok(result)
    assert f"launch.viz_sim={viz_sim}" in dryrun_text(result)


@pytest.mark.parametrize(
    "launcher,args",
    [(TRAIN_LAUNCHER, ["DrillLift"]), (EVAL_LAUNCHER, ["DrillLift", "base", 10])],
    ids=["train", "eval"],
)
def test_config_parser_uses_the_learner_python(launcher_sandbox, launcher, args):
    """Launcher setup must not fall back to an unrelated system Python."""
    wrapper = launcher_sandbox["tmp"] / "learner-python"
    marker = launcher_sandbox["tmp"] / "learner-python-invoked"
    wrapper.write_text(f'#!/bin/sh\ntouch \'{marker}\'\nexec "$REAL_PYTHON" "$@"\n')
    wrapper.chmod(0o755)

    result = run_launcher_dryrun(
        launcher,
        args,
        launcher_sandbox,
        extra_env={"LEARNER_PYTHON": str(wrapper), "REAL_PYTHON": sys.executable},
    )

    _assert_dryrun_ok(result)
    assert marker.is_file()


@pytest.mark.parametrize(
    "launcher,args",
    [(TRAIN_LAUNCHER, ["DrillLift"]), (EVAL_LAUNCHER, ["DrillLift", "base", 10])],
    ids=["train", "eval"],
)
def test_base_model_defaults_to_gr00t_in_plan(launcher_sandbox, launcher, args):
    """Without BASE_MODEL set, both launchers plan the gr00t base model group."""
    result = run_launcher_dryrun(launcher, args, launcher_sandbox)
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert re.search(r"^launch\.override=base_model=gr00t\b", text, re.MULTILINE), (
        f"the launcher must pass the base model group override to the learner:\n{text}"
    )


def test_num_envs_defaults_to_single_env(launcher_sandbox):
    """Without NUM_ENVS the plan keeps the validated single-env layout."""
    result = run_launcher_dryrun(TRAIN_LAUNCHER, ["DrillLift"], launcher_sandbox)
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert re.search(r"launch\.num_envs\s*=\s*1\b", text), (
        f"plan must state num_envs:\n{text}"
    )
    assert re.search(r"launch\.cmd\.train_sim=.*--num-envs 1\b", text), (
        f"the train sim command must pin --num-envs 1:\n{text}"
    )
    assert "training.num_envs" not in text, (
        f"single-env runs must not override the learner's environment count:\n{text}"
    )


def test_num_envs_boots_one_batched_train_sim(launcher_sandbox):
    """NUM_ENVS>1 batches the ONE train sim and tells the learner how many environments it hosts.

    The eval simulator must stay single-env: the evaluator scores episodes
    sequentially on its own dedicated server.
    """
    result = run_launcher_dryrun(
        TRAIN_LAUNCHER,
        ["DrillLift"],
        launcher_sandbox,
        extra_env={"NUM_ENVS": "8", "ENV_SPACING": "6", "RENDER_ONCE_PRIMS": "none"},
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert re.search(r"launch\.num_envs\s*=\s*8\b", text), (
        f"plan must echo NUM_ENVS:\n{text}"
    )
    assert re.search(
        r"^launch\.override=training\.rollout_count_by=episode$", text, re.MULTILINE
    ), (
        f"a batched simulator collects whole episodes; the runtime rejects step-counted rollouts:\n{text}"
    )
    train_cmd = next(
        line for line in text.splitlines() if "launch.cmd.train_sim=" in line
    )
    assert (
        "--num-envs 8" in train_cmd
        and "--env-spacing 6" in train_cmd
        and "--render-once-prims none" in train_cmd
    ), (
        f"the batched train sim must receive --num-envs/--env-spacing/--render-once-prims:\n{train_cmd}"
    )
    eval_cmd = next(
        line for line in text.splitlines() if "launch.cmd.eval_sim=" in line
    )
    assert "--num-envs" not in eval_cmd, (
        f"the eval sim must stay single-env:\n{eval_cmd}"
    )
    assert re.search(r"^launch\.override=training\.num_envs=8\b", text, re.MULTILINE), (
        f"the learner must receive training.num_envs=8:\n{text}"
    )


def test_ppo_plan_isolates_batched_training_and_evaluation(launcher_sandbox):
    """PPO uses equal-sized, separate simulator processes for train and eval."""
    result = run_launcher_dryrun(
        TRAIN_LAUNCHER,
        ["DrillLift"],
        launcher_sandbox,
        extra_env={"TRAINER": "ppo", "NUM_ENVS": "8"},
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert "plan.algorithm=dsrl_ppo" in text
    train_cmd = next(
        line for line in text.splitlines() if "launch.cmd.train_sim=" in line
    )
    eval_cmd = next(
        line for line in text.splitlines() if "launch.cmd.eval_sim=" in line
    )
    assert "--num-envs 8" in train_cmd
    assert "--num-envs 8" in eval_cmd
    assert "ARENA_EVAL_SERVER_URL=tcp://127.0.0.1:" in text
    # The rest of the recipe is dsrl_ppo.yaml's; the library's dry run prints it resolved.
    assert "launch.override=placement.learner.ranks=1" in text
    assert "official_policy_init: true" in text and "use_tanh_output: false" in text


def _commands(text, role):
    """(port, scenes) of every launch.cmd.<role>_sim line; a server without --num-envs hosts one scene."""
    commands = []
    for line in text.splitlines():
        if line.startswith(f"launch.cmd.{role}_sim="):
            scenes = re.search(r"--num-envs (\d+)", line)
            commands.append(
                (
                    re.search(r"--port (\d+)", line).group(1),
                    scenes.group(1) if scenes else "1",
                )
            )
    return commands


@pytest.mark.parametrize(
    "trainer,eval_sims",
    [("dsrl", [("7302", "1")]), ("ppo", [("7302", "4"), ("7303", "4")])],
)
def test_num_servers_boots_one_train_sim_per_rollout_worker(
    launcher_sandbox, trainer, eval_sims
):
    """NUM_SERVERS=2 splits NUM_ENVS over two training servers on consecutive ports (one batched worker
    each); DSRL keeps one single-scene evaluation server after them, DSRL-PPO evaluates on two shaped
    like the training ones, since its trainer swaps rollout groups of one topology."""
    result = run_launcher_dryrun(
        TRAIN_LAUNCHER,
        ["DrillLift"],
        launcher_sandbox,
        extra_env={"TRAINER": trainer, "NUM_ENVS": "8", "NUM_SERVERS": "2"},
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert _commands(text, "train") == [("7300", "4"), ("7301", "4")], text
    assert _commands(text, "eval") == eval_sims, text
    urls = ",".join(f"tcp://127.0.0.1:{port}" for port, _ in eval_sims)
    assert (
        "launch.env.learner.ARENA_SERVER_URL=tcp://127.0.0.1:7300,tcp://127.0.0.1:7301"
        in text
    )
    assert f"launch.env.learner.ARENA_EVAL_SERVER_URL={urls}" in text
    assert (
        "plan.layout=2 rollout worker(s) x 4 environment(s), batched on 2 simulator servers"
        in text
    )
    assert re.search(r"^launch\.override=training\.num_envs=8$", text, re.MULTILINE), (
        text
    )


@pytest.mark.parametrize(
    "launcher,args",
    [(TRAIN_LAUNCHER, ["DrillLift"]), (EVAL_LAUNCHER, ["DrillLift", "base", 10])],
    ids=["train", "eval"],
)
def test_rlinf_on_one_gpu_shares_it_between_its_groups(
    launcher_sandbox, launcher, args
):
    """RLInf on the one GPU the launcher gives the learner plans every group onto it instead of refusing."""
    result = run_launcher_dryrun(
        launcher,
        [*args, "placement.gpus_per_node=1"],
        launcher_sandbox,
        extra_env={"RUNTIME": "rlinf"},
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert "launch.override=runtime.rlinf.first_accelerator=0" in text
    assert "plan.placement.learner=1 rank(s) on node0 gpus [0]" in text
    two = dryrun_text(
        run_launcher_dryrun(
            TRAIN_LAUNCHER,
            ["DrillLift"],
            launcher_sandbox,
            extra_env={"RUNTIME": "rlinf", "LEARNER_GPUS": "0,1"},
        )
    )
    assert "first_accelerator=0" not in two


def test_finetune_runs_one_rollout_worker_per_training_simulator(launcher_sandbox):
    """Full PPO fine-tuning keeps NUM_SERVERS' layout, not the Isaac Lab recipe's four workers."""
    finetune = str(Path(TRAIN_LAUNCHER).with_name("finetune_ppo.sh"))
    # The dry run reads the checkpoint's config.json to pick the GR00T generation.
    Path(launcher_sandbox["ckpt"], "config.json").write_text("{}\n", encoding="utf-8")
    for servers, layout in (
        ("1", "1 rollout worker(s) x 8 scene(s)"),
        ("2", "2 rollout worker(s) x 4 scene(s)"),
    ):
        result = run_launcher_dryrun(
            finetune,
            ["PushButton"],
            launcher_sandbox,
            extra_env={"NUM_SERVERS": servers},
        )
        _assert_dryrun_ok(result)
        assert f"plan.layout={layout}" in dryrun_text(result), dryrun_text(result)


@pytest.mark.parametrize(
    "env,message",
    [
        ({"NUM_SERVERS": "0"}, "NUM_SERVERS must be a positive integer"),
        (
            {"NUM_ENVS": "8", "NUM_SERVERS": "3"},
            "must split NUM_ENVS=8 environments equally",
        ),
        ({"NUM_ENVS": "2", "NUM_SERVERS": "2"}, "at least two per server"),
    ],
)
def test_num_servers_rejects_a_split_it_cannot_serve(launcher_sandbox, env, message):
    result = run_launcher_dryrun(
        TRAIN_LAUNCHER, ["DrillLift"], launcher_sandbox, extra_env=env
    )
    assert result.returncode != 0
    assert message in dryrun_text(result)


@pytest.mark.parametrize(
    "launcher,args",
    [(TRAIN_LAUNCHER, ["DrillLift"]), (EVAL_LAUNCHER, ["DrillLift", "base", 10])],
    ids=["train", "eval"],
)
def test_base_model_env_var_selects_base_model(launcher_sandbox, launcher, args):
    """BASE_MODEL=<name> flips the plan and the learner's Hydra group override."""
    result = run_launcher_dryrun(
        launcher, args, launcher_sandbox, extra_env={"BASE_MODEL": "dreamzero_local"}
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert re.search(
        r"^launch\.override=base_model=dreamzero_local\b", text, re.MULTILINE
    ), (
        f"BASE_MODEL=dreamzero_local must become the Hydra base model group override:\n{text}"
    )
    assert not re.search(r"^launch\.override=base_model=gr00t\b", text, re.MULTILINE), (
        f"the default base model override must not also be emitted:\n{text}"
    )


# --- station launcher (docker-level GPU pin, two-host split) ---------------------------------

STATION_LAUNCHER = str(Path(TRAIN_LAUNCHER).parent / "station" / "station_train.sh")
_REPO_ROOT = Path(TRAIN_LAUNCHER).resolve().parents[3]


def _station_env(launcher_sandbox):
    """A fake G1 Factory tree whose posttrain/ is this repo, so the launcher finds its scripts."""
    g1f = Path(launcher_sandbox["tmp"]) / "g1f"
    g1f.mkdir()
    (g1f / "posttrain").symlink_to(_REPO_ROOT)
    return {
        "G1F": str(g1f),
        "HF_CACHE": str(Path(launcher_sandbox["tmp"]) / "hf_cache"),
    }


def _container_cmd(text):
    return next(
        line for line in text.splitlines() if line.startswith("launch.container_cmd=")
    )


def test_station_train_pins_the_card_at_the_docker_level(launcher_sandbox):
    """GPUS=<uuid> becomes a docker device pin, and --privileged is dropped so the pin holds.

    --privileged mounts every host device node, which would expose all cards to
    Isaac Sim regardless of the pin.
    """
    result = run_launcher_dryrun(
        STATION_LAUNCHER,
        ["DrillLift", "64", "training.max_updates_per_burst=20"],
        launcher_sandbox,
        extra_env={
            **_station_env(launcher_sandbox),
            "GPUS": "GPU-abc",
            "EVAL_SIM": "0",
            "NUM_ROLLOUTS": "25600",
        },
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert "launch.docker.gpus=device=GPU-abc" in text, text
    assert "launch.docker.privileged=none" in text, text
    cmd = _container_cmd(text)
    for token in (
        "N=64",
        "EVAL_SIM=0",
        "NUM_ROLLOUTS=25600",
        "training.max_updates_per_burst=20",
    ):
        assert token in cmd, f"{token} must reach the container command:\n{cmd}"


def test_station_train_defaults_expose_every_card_on_the_native_image(launcher_sandbox):
    result = run_launcher_dryrun(
        STATION_LAUNCHER,
        ["DrillLift", "8"],
        launcher_sandbox,
        extra_env=_station_env(launcher_sandbox),
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert "launch.docker.gpus=all" in text, text
    assert "launch.docker.privileged=--privileged" in text, text
    assert (
        "launch.runtime=native" in text and "launch.image=g1_factory_learner:" in text
    ), text
    assert "RUNTIME=native" in _container_cmd(text)


def test_station_train_sims_only_needs_no_checkpoint(launcher_sandbox):
    """The simulator host of a two-host split has no learner, hence no checkpoint or dataset."""
    result = run_launcher_dryrun(
        STATION_LAUNCHER,
        ["DrillLift", "256"],
        launcher_sandbox,
        extra_env={
            **_station_env(launcher_sandbox),
            "SIMS_ONLY": "1",
            "GPUS": "0",
            "POSTTRAIN_BASE_CKPT": "",
            "POSTTRAIN_DATA_ROOT": "",
        },
    )
    _assert_dryrun_ok(result)
    cmd = _container_cmd(dryrun_text(result))
    assert "SIMS_ONLY=1" in cmd and "N=256" in cmd, cmd


def test_station_train_passes_remote_simulator_urls(launcher_sandbox):
    result = run_launcher_dryrun(
        STATION_LAUNCHER,
        ["DrillLift", "64"],
        launcher_sandbox,
        extra_env={
            **_station_env(launcher_sandbox),
            "TRAIN_SIM_URL": "tcp://127.0.0.1:7300",
            "EVAL_SIM": "0",
            "GPUS": "1",
        },
    )
    _assert_dryrun_ok(result)
    cmd = _container_cmd(dryrun_text(result))
    assert "TRAIN_SIM_URL=tcp://127.0.0.1:7300" in cmd and "EVAL_SIM=0" in cmd, cmd


def test_image_shortest_edge_reaches_both_sims(launcher_sandbox):
    """IMAGE_SHORTEST_EDGE shrinks frames on the server for the train and the eval sim alike."""
    result = run_launcher_dryrun(
        TRAIN_LAUNCHER,
        ["DrillLift"],
        launcher_sandbox,
        extra_env={"IMAGE_SHORTEST_EDGE": "256"},
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    for tag in ("launch.cmd.train_sim=", "launch.cmd.eval_sim="):
        cmd = next(line for line in text.splitlines() if tag in line)
        assert "--image-shortest-edge 256" in cmd, (
            f"{tag} must carry --image-shortest-edge:\n{cmd}"
        )
    plain = dryrun_text(
        run_launcher_dryrun(TRAIN_LAUNCHER, ["DrillLift"], launcher_sandbox)
    )
    assert "--image-shortest-edge" not in plain, (
        "native frames stay the default for train.sh"
    )


# --- run_container_train.sh: the in-container plan (runtime, cards, learner overrides) ---

RUN_CONTAINER_LAUNCHER = str(
    Path(TRAIN_LAUNCHER).parent / "station" / "run_container_train.sh"
)


def _container_env(launcher_sandbox, **overrides):
    """run_container_train.sh's world: WORKDIR/posttrain is this repo, PY this interpreter."""
    workdir = Path(launcher_sandbox["tmp"]) / "workdir"
    workdir.mkdir(exist_ok=True)
    link = workdir / "posttrain"
    if not link.exists():
        link.symlink_to(_REPO_ROOT)
    env = {
        "WORKDIR": str(workdir),
        "PY": sys.executable,
        "TASK": "DrillLift",
        "N": "8",
        "EVAL_SIM": "0",
        "WANDB_NAME": "probe",
    }
    env.update(overrides)
    return env


def _overrides(text):
    """The learner's overrides, space-joined, from the dry run's launch.override= lines."""
    prefix = "launch.override="
    return " ".join(
        line[len(prefix) :] for line in text.splitlines() if line.startswith(prefix)
    )


def _plan_value(text, key):
    line = next(line for line in text.splitlines() if line.startswith(f"launch.{key}="))
    return line.split("=", 1)[1]


def test_run_container_defaults_to_the_native_runtime(launcher_sandbox):
    result = run_launcher_dryrun(
        RUN_CONTAINER_LAUNCHER,
        [],
        launcher_sandbox,
        extra_env=_container_env(launcher_sandbox),
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert _plan_value(text, "runtime") == "native"
    assert _plan_value(text, "rollout_gpu") == _plan_value(text, "learner_gpu") == "0"
    overrides = _overrides(text)
    assert "runtime=native" in overrides and "first_accelerator" not in overrides, (
        overrides
    )
    assert (
        "training.rollout_count_by=episode" in overrides
        and "eval.eval_freq=0" in overrides
    ), overrides


def test_run_container_rlinf_places_the_rollout_actor_on_the_learner_card_by_default(
    launcher_sandbox,
):
    """One card: the actor shares the learner's card, and the evaluation group may share it too."""
    result = run_launcher_dryrun(
        RUN_CONTAINER_LAUNCHER,
        ["training.max_updates_per_burst=20"],
        launcher_sandbox,
        extra_env=_container_env(launcher_sandbox, RUNTIME="rlinf"),
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert _plan_value(text, "runtime") == "rlinf"
    assert _plan_value(text, "rollout_gpu") == "0"
    overrides = _overrides(text)
    for token in (
        "runtime=rlinf",
        "runtime.rlinf.first_accelerator=0",
        "runtime.rlinf.exclusive_accelerators=false",
        "training.max_updates_per_burst=20",
    ):
        assert token in overrides, f"{token} must reach the learner:\n{overrides}"
    ray_tmpdir = _plan_value(text, "ray_tmpdir")
    assert ray_tmpdir.startswith("/tmp/ray_"), text  # NOSONAR - launcher-pinned path


def test_run_container_rlinf_split_puts_the_actor_on_the_other_card(launcher_sandbox):
    result = run_launcher_dryrun(
        RUN_CONTAINER_LAUNCHER,
        [],
        launcher_sandbox,
        extra_env=_container_env(
            launcher_sandbox, RUNTIME="rlinf", LEARNER_GPU="0", ROLLOUT_GPU="1"
        ),
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert (
        _plan_value(text, "learner_gpu") == "0"
        and _plan_value(text, "rollout_gpu") == "1"
    )
    assert "runtime.rlinf.first_accelerator=1" in _overrides(text), text


def test_run_container_rejects_a_rollout_card_without_the_rlinf_runtime(
    launcher_sandbox,
):
    result = run_launcher_dryrun(
        RUN_CONTAINER_LAUNCHER,
        [],
        launcher_sandbox,
        extra_env=_container_env(launcher_sandbox, ROLLOUT_GPU="1"),
    )
    assert result.returncode != 0
    assert "ROLLOUT_GPU applies to RUNTIME=rlinf" in dryrun_text(result)


def test_run_container_boots_one_server_per_rollout_worker_on_consecutive_ports(
    launcher_sandbox,
):
    """NUM_SERVERS=2: two training servers with half the environments each, the eval port after
    them, the learner's address list in server order, and the RLinf workers sharing the learner's card."""
    result = run_launcher_dryrun(
        RUN_CONTAINER_LAUNCHER,
        [],
        launcher_sandbox,
        extra_env=_container_env(
            launcher_sandbox, RUNTIME="rlinf", NUM_SERVERS="2", N="8"
        ),
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert _plan_value(text, "num_servers") == "2"
    assert (
        _plan_value(text, "train_server_urls")
        == "tcp://127.0.0.1:7300,tcp://127.0.0.1:7301"
    )
    assert _plan_value(text, "eval_port") == "7302"
    overrides = _overrides(text)
    assert "runtime.rlinf.rollout_workers_share_accelerator=true" in overrides, (
        overrides
    )
    one = dryrun_text(
        run_launcher_dryrun(
            RUN_CONTAINER_LAUNCHER,
            [],
            launcher_sandbox,
            extra_env=_container_env(launcher_sandbox, RUNTIME="rlinf"),
        )
    )
    assert (
        _plan_value(one, "num_servers") == "1"
        and _plan_value(one, "eval_port") == "7301"
    )
    assert "rollout_workers_share_accelerator" not in _overrides(one)


def test_a_simulator_only_container_serves_several_servers_on_any_runtime(
    launcher_sandbox,
):
    """The split's simulator container runs no learner: it boots the servers whatever runtime the
    learner's container uses (the sweep keeps it on the native image)."""
    result = run_launcher_dryrun(
        RUN_CONTAINER_LAUNCHER,
        [],
        launcher_sandbox,
        extra_env=_container_env(
            launcher_sandbox, SIMS_ONLY="1", NUM_SERVERS="2", N="256"
        ),
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert (
        _plan_value(text, "runtime") == "native"
        and _plan_value(text, "num_servers") == "2"
    )
    assert (
        _plan_value(text, "train_server_urls")
        == "tcp://127.0.0.1:7300,tcp://127.0.0.1:7301"
    )


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"NUM_SERVERS": "2"}, "needs RUNTIME=rlinf"),
        ({"NUM_SERVERS": "3", "RUNTIME": "rlinf"}, "split N=8 environments equally"),
        ({"NUM_SERVERS": "2", "RUNTIME": "rlinf", "N": "2"}, "at least two per server"),
        (
            {"NUM_SERVERS": "2", "RUNTIME": "rlinf", "ROLLOUT_GPU": "1"},
            "share the learner's card",
        ),
        ({"NUM_SERVERS": "0"}, "positive integer"),
    ],
)
def test_run_container_rejects_a_server_count_it_cannot_serve(
    launcher_sandbox, overrides, message
):
    result = run_launcher_dryrun(
        RUN_CONTAINER_LAUNCHER,
        [],
        launcher_sandbox,
        extra_env=_container_env(launcher_sandbox, **overrides),
    )
    assert result.returncode != 0
    assert message in dryrun_text(result), dryrun_text(result)


def test_station_train_forwards_the_server_count(launcher_sandbox):
    result = run_launcher_dryrun(
        STATION_LAUNCHER,
        ["DrillLift", "64"],
        launcher_sandbox,
        extra_env={
            **_station_env(launcher_sandbox),
            "RUNTIME": "rlinf",
            "NUM_SERVERS": "2",
            "GPUS": "1",
        },
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert "launch.num_servers=2" in text
    assert " NUM_SERVERS=2 " in _container_cmd(text), _container_cmd(text)


def test_run_container_boots_the_eval_server_with_the_evaluation_width(
    launcher_sandbox,
):
    """EVAL_NUM_ENVS: the eval server hosts that many environments and the learner scores that many
    episodes in parallel; the default keeps one."""
    result = run_launcher_dryrun(
        RUN_CONTAINER_LAUNCHER,
        [],
        launcher_sandbox,
        extra_env=_container_env(launcher_sandbox, EVAL_SIM="1", EVAL_NUM_ENVS="25"),
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert _plan_value(text, "eval_num_envs") == "25"
    assert "eval.eval_num_envs=25" in _overrides(text), text
    one = dryrun_text(
        run_launcher_dryrun(
            RUN_CONTAINER_LAUNCHER,
            [],
            launcher_sandbox,
            extra_env=_container_env(launcher_sandbox, EVAL_SIM="1"),
        )
    )
    assert _plan_value(one, "eval_num_envs") == "1"
    assert "eval.eval_num_envs=1" in _overrides(one), one
    off = dryrun_text(
        run_launcher_dryrun(
            RUN_CONTAINER_LAUNCHER,
            [],
            launcher_sandbox,
            extra_env=_container_env(
                launcher_sandbox, EVAL_SIM="0", EVAL_NUM_ENVS="25"
            ),
        )
    )
    assert "eval.eval_num_envs" not in _overrides(off), off


def test_run_container_rejects_a_bad_evaluation_width(launcher_sandbox):
    result = run_launcher_dryrun(
        RUN_CONTAINER_LAUNCHER,
        [],
        launcher_sandbox,
        extra_env=_container_env(launcher_sandbox, EVAL_NUM_ENVS="0"),
    )
    assert result.returncode != 0
    assert "EVAL_NUM_ENVS must be a positive integer" in dryrun_text(result)


def test_station_train_forwards_the_evaluation_width(launcher_sandbox):
    result = run_launcher_dryrun(
        STATION_LAUNCHER,
        ["DrillLift", "64"],
        launcher_sandbox,
        extra_env={
            **_station_env(launcher_sandbox),
            "EVAL_NUM_ENVS": "25",
            "GPUS": "1",
        },
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert "launch.eval_num_envs=25" in text
    assert " EVAL_NUM_ENVS=25 " in _container_cmd(text), _container_cmd(text)


def test_run_container_rejects_an_unknown_runtime(launcher_sandbox):
    result = run_launcher_dryrun(
        RUN_CONTAINER_LAUNCHER,
        [],
        launcher_sandbox,
        extra_env=_container_env(launcher_sandbox, RUNTIME="bogus"),
    )
    assert result.returncode != 0
    assert "RUNTIME must be native or rlinf" in dryrun_text(result)


def test_station_train_selects_the_rlinf_image_and_forwards_the_runtime(
    launcher_sandbox,
):
    result = run_launcher_dryrun(
        STATION_LAUNCHER,
        ["DrillLift", "64"],
        launcher_sandbox,
        extra_env={
            **_station_env(launcher_sandbox),
            "RUNTIME": "rlinf",
            "ROLLOUT_GPU": "1",
            "GPUS": "0,1",
            "EVAL_SIM": "0",
        },
    )
    _assert_dryrun_ok(result)
    text = dryrun_text(result)
    assert "launch.runtime=rlinf" in text, text
    assert "launch.image=g1_factory_learner_rlinf:" in text, text
    cmd = _container_cmd(text)
    assert "RUNTIME=rlinf" in cmd and "ROLLOUT_GPU=1" in cmd, cmd
