# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every launcher and recipe script resolves its plan from any working directory, without hardware.

With ``ISAAC_RL_POSTTRAINING_DRYRUN=1`` a launcher prints its own ``launch.*`` lines (task, ports,
simulator commands), every override it passes as ``launch.override=`` lines, then the library's
dry run of those overrides (``plan.*``), and exits 0 before starting a simulator or learner.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parent.parent
REPO_ROOT = WORKFLOW.parent.parent
SCRIPTS = WORKFLOW / "scripts"
SIMULATOR_CONFIG = (
    REPO_ROOT / "simulators" / "g1_arena" / "configs" / "simulator" / "g1_arena.yaml"
)

#: Launcher knobs a developer's shell may carry; each test sets what it needs explicitly.
_CLEARED = (
    "ACTION_HORIZON",
    "ALGO",
    "BASE_MODEL",
    "BATCH_SIZE",
    "DENOISING_STEPS",
    "EMBODIMENT_TAG",
    "EVAL_MAXSTEPS",
    "EVAL_SEED",
    "EVAL_SIM_DEVICE",
    "GPU",
    "IMAGE_SHORTEST_EDGE",
    "ISAACLAB_ARENA_PATH",
    "LEARNER_PYTHON",
    "NUM_ENVS",
    "NUM_ROLLOUTS",
    "POSTTRAIN_BASE_CKPT",
    "POSTTRAIN_DATA_ROOT",
    "RUNTIME",
    "SAVE_INTERVAL",
    "SIMULATOR",
    "TRAINER",
    "TRAIN_MAXSTEPS",
    "TRAIN_SIM_DEVICE",
    "VARIANT",
    "WANDB_NAME",
)

#: The run-length, evaluation and retention settings of the September-2026 protocol.
_PROTOCOL = {
    "training.num_rollouts=40000",
    "eval.eval_num_episodes=50",
    "eval.eval_freq=5000",
    "training.save_interval=5000",
    "training.checkpoint_retention=best_last_final",
    "eval.action_selection=sample",
    "base_model.dataset_dir=null",
}


def episode_caps() -> dict[str, int]:
    return yaml.safe_load(SIMULATOR_CONFIG.read_text(encoding="utf-8"))[
        "max_episode_steps"
    ]


def run_dry(
    script: str, *args: str, cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    run_env = {key: value for key, value in os.environ.items() if key not in _CLEARED}
    run_env.update(
        {
            "ISAAC_RL_POSTTRAINING_DRYRUN": "1",
            "LEARNER_PYTHON": sys.executable,
            "BASE_MODEL": "mock",
        }
    )
    if env:
        run_env.update(env)
    return subprocess.run(
        ["bash", str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        env=run_env,
        cwd=cwd,
        timeout=120,
        check=False,
    )


def plan(script: str, *args: str, cwd: Path, env: dict[str, str] | None = None) -> str:
    """The dry-run plan of a launcher; a non-zero exit fails the test with the full output."""
    completed = run_dry(script, *args, cwd=cwd, env=env)
    assert completed.returncode == 0, (
        f"{script} {' '.join(args)} exited {completed.returncode}\n"
        f"{completed.stdout}\n{completed.stderr}"
    )
    return completed.stdout


def hydra_overrides(text: str) -> set[str]:
    prefix = "launch.override="
    return {
        line[len(prefix) :] for line in text.splitlines() if line.startswith(prefix)
    }


def command_line(text: str, prefix: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(prefix))


@pytest.mark.parametrize("task,cap", sorted(episode_caps().items()))
def test_train_plan_resolves_every_task_from_another_directory(
    tmp_path: Path, task: str, cap: int
) -> None:
    text = plan("train.sh", task.removeprefix("LM"), cwd=tmp_path)
    assert f"launch.task={task}" in text
    assert f"env.max_episode_steps={cap}" in hydra_overrides(text)
    assert "launch.env.learner.ARENA_IMAGE_RESOLUTION=0" in text
    assert "plan.algorithm=dsrl_sac" in text and "base_model=mock" in hydra_overrides(
        text
    )
    assert "plan.placement.learner=" in text


def test_eval_base_mode_scores_the_frozen_policy_from_the_checkpoint_alone(
    tmp_path: Path,
) -> None:
    overrides = hydra_overrides(
        plan("eval.sh", "DrillLift", "base", "50", cwd=tmp_path)
    )
    assert {
        "eval.eval_base=true",
        "training.num_rollouts=0",
        "eval.eval_on_first_step=true",
        "eval.eval_num_episodes=50",
        "base_model.dataset_dir=null",
        "base_model.task_instruction='Walk to the table and pick up the power drill'",
    } <= overrides


def test_eval_policy_mode_loads_the_given_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "16088"
    checkpoint.mkdir(parents=True)
    overrides = hydra_overrides(
        plan("eval.sh", "DrillLift", "policy", "100", str(checkpoint), cwd=tmp_path)
    )
    assert f"training.load_dir={checkpoint}" in overrides
    assert "eval.eval_base=true" not in overrides
    assert "eval.eval_num_episodes=100" in overrides


def test_eval_policy_mode_without_a_checkpoint_is_a_usage_error(
    tmp_path: Path,
) -> None:
    assert run_dry("eval.sh", "DrillLift", "policy", "10", cwd=tmp_path).returncode != 0


def test_ppo_launcher_selects_dsrl_ppo_on_equal_batched_simulators(
    tmp_path: Path,
) -> None:
    text = plan("train_ppo.sh", "DrillLift", cwd=tmp_path, env={"NUM_ENVS": "8"})
    assert "plan.algorithm=dsrl_ppo" in text
    assert {"algorithm=dsrl_ppo", "training.num_envs=8"} <= hydra_overrides(text)
    assert "--num-envs 8" in command_line(text, "launch.cmd.train_sim=")
    assert "--num-envs 8" in command_line(text, "launch.cmd.eval_sim=")


@pytest.mark.parametrize(
    "mode,algorithm,expected",
    [
        ("rfs", "rfs", {"training.chunk_size=40"}),
        (
            "rfs_pc",
            "rfs",
            {
                "rfs.proposal_conditioned_residual=true",
                "rfs.reference_action_conditioning=true",
            },
        ),
        ("rfs_nb2", "rfs", {"dsrl.noise_action_bound=2.0"}),
        ("sac_edit", "sac_edit", set()),
    ],
)
def test_zoo_recipes_select_their_algorithm_under_the_shared_protocol(
    tmp_path: Path, mode: str, algorithm: str, expected: set[str]
) -> None:
    text = plan("zoo_recipes.sh", mode, "DrillPnP", "1", cwd=tmp_path)
    assert f"plan.algorithm={algorithm}" in text
    overrides = hydra_overrides(text)
    assert _PROTOCOL | expected | {"training.seed=1"} <= overrides
    assert (
        "base_model.task_instruction='Move the power drill from one table to another'"
        in overrides
    )


def test_zoo_recipes_reject_an_unknown_mode(tmp_path: Path) -> None:
    assert run_dry("zoo_recipes.sh", "dsrl", "DrillPnP", cwd=tmp_path).returncode != 0


def test_repro_script_reproduces_the_results_protocol(tmp_path: Path) -> None:
    train = hydra_overrides(
        plan("repro_sep2026.sh", "train", "DrillPnP", "1", cwd=tmp_path)
    )
    assert _PROTOCOL | {"training.seed=1", "training.chunk_size=40"} <= train
    assert "dsrl.noise_action_bound=2.0" not in train
    assert not any(override.startswith("eval.reset_seed=") for override in train)

    variant = hydra_overrides(
        plan(
            "repro_sep2026.sh",
            "train",
            "DrillLift",
            cwd=tmp_path,
            env={"VARIANT": "b2"},
        )
    )
    assert {"dsrl.noise_action_bound=2.0", "training.seed=0"} <= variant

    # The PushButton row of the README table: the wider bound plus the +1-on-success reward.
    success = hydra_overrides(
        plan(
            "repro_sep2026.sh",
            "train",
            "PushButton",
            cwd=tmp_path,
            env={"VARIANT": "b2_succ"},
        )
    )
    assert {
        "dsrl.noise_action_bound=2.0",
        "env.success_reward=1.0",
        "env.failure_reward=0.0",
    } <= success

    base = hydra_overrides(plan("repro_sep2026.sh", "base", "DrillPnP", cwd=tmp_path))
    assert {
        "eval.eval_base=true",
        "eval.eval_num_episodes=100",
        "training.num_rollouts=0",
    } <= base
    # The published protocol scores unseeded episodes; EVAL_SEED opts into seeded scenes.
    assert not any(override.startswith("eval.reset_seed=") for override in base)

    checkpoint = tmp_path / "checkpoints" / "16088"
    checkpoint.mkdir(parents=True)
    rescore = hydra_overrides(
        plan("repro_sep2026.sh", "rescore", "DrillPnP", str(checkpoint), cwd=tmp_path)
    )
    assert {
        f"training.load_dir={checkpoint}",
        "eval.eval_num_episodes=100",
        "training.num_rollouts=0",
    } <= rescore
    assert "eval.eval_base=true" not in rescore
    assert not any(override.startswith("eval.reset_seed=") for override in rescore)

    seeded_base = hydra_overrides(
        plan(
            "repro_sep2026.sh",
            "base",
            "DrillPnP",
            cwd=tmp_path,
            env={"EVAL_SEED": "1000"},
        )
    )
    assert "eval.reset_seed=1000" in seeded_base
    seeded_rescore = hydra_overrides(
        plan(
            "repro_sep2026.sh",
            "rescore",
            "DrillPnP",
            str(checkpoint),
            cwd=tmp_path,
            env={"EVAL_SEED": "1000"},
        )
    )
    assert "eval.reset_seed=1000" in seeded_rescore


def _prompt(task: str) -> subprocess.CompletedProcess[str]:
    prompts = SCRIPTS / "task_prompts.sh"
    return subprocess.run(
        ["bash", "-c", f'source "{prompts}" && task_prompt {task}'],
        capture_output=True,
        text=True,
        check=False,
    )


def test_every_task_has_a_language_prompt() -> None:
    for task in episode_caps():
        completed = _prompt(task)
        assert completed.returncode == 0 and completed.stdout.strip(), task
    assert _prompt("LMNoSuchTask").returncode != 0


@pytest.mark.parametrize("algorithm", ["dsrl_sac", "rfs", "sac_edit", "dsrl_ppo"])
def test_each_recipe_composes_a_launch_plan_with_the_mock_policy(
    monkeypatch: pytest.MonkeyPatch, algorithm: str
) -> None:
    """What the learner does first: compose the config and validate the plan, before anything loads."""
    from hydra import compose, initialize_config_dir

    from isaac_rl_posttraining.configs import register_configs
    from isaac_rl_posttraining.runners.build import train_plan

    monkeypatch.setenv("ARENA_SERVER_URL", "tcp://127.0.0.1:7300")
    monkeypatch.setenv("ARENA_EVAL_SERVER_URL", "tcp://127.0.0.1:7301")
    register_configs()
    config_dir = REPO_ROOT / "isaac_rl_posttraining" / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                f"algorithm={algorithm}",
                "base_model=mock",
                "simulator=g1_arena",
            ],
        )
    resolved = train_plan(cfg)
    assert resolved.simulator == "g1_arena"
    assert resolved.algorithm == algorithm


def test_image_shortest_edge_reaches_both_simulators(tmp_path: Path) -> None:
    """IMAGE_SHORTEST_EDGE shrinks frames on the server for the train and the eval simulator alike."""
    text = plan(
        "train.sh", "DrillLift", cwd=tmp_path, env={"IMAGE_SHORTEST_EDGE": "256"}
    )
    for prefix in ("launch.cmd.train_sim=", "launch.cmd.eval_sim="):
        assert "--image-shortest-edge 256" in command_line(text, prefix), command_line(
            text, prefix
        )
    plain = plan("train.sh", "DrillLift", cwd=tmp_path)
    assert "--image-shortest-edge" not in plain, "native frames stay the default"


def test_several_environments_count_collection_by_episode(tmp_path: Path) -> None:
    """One batched simulator hosts every environment and returns whole episodes; the plan must compose for it."""
    overrides = hydra_overrides(
        plan("train.sh", "PushButton", cwd=tmp_path, env={"NUM_ENVS": "8"})
    )
    assert {"training.num_envs=8", "training.rollout_count_by=episode"} <= overrides
    single = hydra_overrides(plan("train.sh", "PushButton", cwd=tmp_path))
    assert "training.rollout_count_by=episode" not in single


@pytest.mark.parametrize("script", ["train.sh", "eval.sh"])
def test_the_renamed_backend_knob_is_refused(script: str, tmp_path: Path) -> None:
    args = ("DrillLift",) if script == "train.sh" else ("DrillLift", "base", "2")
    completed = run_dry(script, *args, cwd=tmp_path, env={"BACKEND": "mock"})
    assert completed.returncode == 2
    assert (
        "BACKEND was renamed to BASE_MODEL; set BASE_MODEL=mock instead"
        in completed.stderr
    )


def test_an_unrelated_backend_variable_is_not_refused(tmp_path: Path) -> None:
    plan("train.sh", "DrillLift", cwd=tmp_path, env={"BACKEND": "nccl"})


def launch_ports(text: str) -> set[int]:
    lines = [line for line in text.splitlines() if line.startswith("launch.ports.")]
    return {int(port) for line in lines for port in line.split("=", 1)[1].split(",")}


def test_runs_on_different_gpus_never_share_a_port(tmp_path: Path) -> None:
    runs = [
        plan(
            "train.sh",
            "DrillLift",
            cwd=tmp_path,
            env={"GPU": "0", "TRAINER": "ppo", "NUM_ENVS": "8", "NUM_SERVERS": "2"},
        ),
        plan("train.sh", "DrillLift", cwd=tmp_path, env={"GPU": "1"}),
        plan("eval.sh", "DrillLift", "base", "2", cwd=tmp_path, env={"GPU": "2"}),
    ]
    ports = [launch_ports(text) for text in runs]
    assert [len(run) for run in ports] == [4, 2, 2]
    assert len(set().union(*ports)) == 8, ports


@pytest.mark.parametrize("script", ["train.sh", "eval.sh"])
def test_a_busy_port_refuses_the_launch_and_boots_nothing(
    script: str, tmp_path: Path
) -> None:
    marker = tmp_path / "simulator_started"
    sim_python = tmp_path / "sim_python"
    sim_python.write_text(f"#!/bin/bash\ntouch {marker}\nsleep 60\n", encoding="utf-8")
    sim_python.chmod(0o755)
    args = ("DrillLift",) if script == "train.sh" else ("DrillLift", "base", "2")
    with socket.socket() as other_run:
        other_run.bind(("127.0.0.1", 0))
        other_run.listen(1)
        busy = other_run.getsockname()[1]
        completed = run_dry(
            script,
            *args,
            cwd=tmp_path,
            env={
                "ISAAC_RL_POSTTRAINING_DRYRUN": "0",
                "PORT_BASE": str(busy),
                "ISAACLAB_ARENA_PATH": str(tmp_path),
                "SIM_PYTHON": str(sim_python),
                "POSTTRAIN_BASE_CKPT": str(tmp_path),
                "POSTTRAIN_DATA_ROOT": str(tmp_path),
                "LOG_DIR": str(tmp_path / "logs"),
            },
        )
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert f"port(s) {busy} already in use" in completed.stderr
    assert not marker.exists()


def stub_launch_env(
    tmp_path: Path, simulator: str, learner_exit: int = 0
) -> dict[str, str]:
    """A real (not dry-run) launch whose simulator and learner are shell stubs."""
    sim_python = tmp_path / "sim_python"
    sim_python.write_text(
        f"#!/bin/bash\necho $$ >> {tmp_path}/sim_pids\n{simulator}\n", encoding="utf-8"
    )
    learner = tmp_path / "learner_python"
    learner.write_text(
        f'#!/bin/bash\ncase "$*" in *scripts/train.py*) exit {learner_exit} ;; esac\nexec {sys.executable} "$@"\n',
        encoding="utf-8",
    )
    for stub in (sim_python, learner):
        stub.chmod(0o755)
    return {
        "ISAAC_RL_POSTTRAINING_DRYRUN": "0",
        "PORT_BASE": "7900",
        "ISAACLAB_ARENA_PATH": str(tmp_path),
        "SIM_PYTHON": str(sim_python),
        "LEARNER_PYTHON": str(learner),
        "POSTTRAIN_BASE_CKPT": str(tmp_path),
        "POSTTRAIN_DATA_ROOT": str(tmp_path),
        "LOG_DIR": str(tmp_path / "logs"),
        "ISAAC_RL_POSTTRAINING_OUTPUT_DIR": str(tmp_path / "out"),
        "WANDB_NAME": "run",
    }


def test_eval_reports_a_failed_learner_not_an_older_runs_result(tmp_path: Path) -> None:
    for old_run in ("run_2026-01-01_00-00-00", "run_b_2027-01-01_00-00-00"):
        summary = tmp_path / "out" / old_run / "checkpoints" / "evaluation_summary.json"
        summary.parent.mkdir(parents=True)
        summary.write_text('{"latest": {"success_rate": 1.0}}', encoding="utf-8")
    env = stub_launch_env(
        tmp_path, "echo 'listening on 127.0.0.1'; sleep 60", learner_exit=3
    )
    completed = run_dry("eval.sh", "DrillLift", "base", "2", cwd=tmp_path, env=env)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "FAILED run (exit 3" in completed.stdout and "SR=" not in completed.stdout


def test_eval_replay_mode_stops_its_simulator_without_a_training_one(
    tmp_path: Path,
) -> None:
    env = stub_launch_env(tmp_path, "echo 'Error: stub refused to boot'; sleep 60")
    completed = run_dry("eval.sh", "DrillLift", "replay", "1", cwd=tmp_path, env=env)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "unbound variable" not in completed.stderr, completed.stderr
    assert "eval sim failed" in completed.stdout


def test_an_interrupted_boot_stops_the_simulators(tmp_path: Path) -> None:
    env = stub_launch_env(
        tmp_path, "sleep 600"
    )  # never listens: the launcher waits at boot
    launcher = subprocess.Popen(
        ["bash", str(SCRIPTS / "train.sh"), "DrillLift"],
        env={key: value for key, value in os.environ.items() if key not in _CLEARED}
        | env,
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pids = tmp_path / "sim_pids"
    deadline = time.monotonic() + 60
    while not pids.exists() and time.monotonic() < deadline:
        time.sleep(0.2)
    launcher.terminate()
    launcher.wait(timeout=60)
    simulator = int(pids.read_text(encoding="utf-8").split()[0])
    deadline = time.monotonic() + 30
    while Path(f"/proc/{simulator}").exists() and time.monotonic() < deadline:
        time.sleep(0.2)
    alive = Path(f"/proc/{simulator}").exists()
    if alive:
        os.kill(simulator, 9)
    assert not alive, "the simulator outlived its interrupted launcher"


@pytest.mark.parametrize("script,args", [("train.sh", ()), ("eval.sh", ("base", "2"))])
def test_a_viewer_without_a_display_stops_before_booting(
    script: str, args: tuple[str, ...], tmp_path: Path
) -> None:
    env = stub_launch_env(tmp_path, "echo 'listening on 127.0.0.1'; sleep 60")
    completed = run_dry(
        script,
        "DrillLift",
        *args,
        cwd=tmp_path,
        env={**env, "SIM_VIZ": "kit", "DISPLAY": ""},
    )
    assert completed.returncode == 2, completed.stdout + completed.stderr
    assert "viewer requested (SIM_VIZ=kit) but no usable display" in completed.stderr
    assert not (tmp_path / "sim_pids").exists()


def test_a_viewer_that_cannot_open_a_window_is_named_as_the_boot_failure(
    tmp_path: Path,
) -> None:
    env = stub_launch_env(
        tmp_path,
        "echo '[Error] [omni.ui.python] AttributeError: no window'; sleep 60",
    )
    # Full-PPO evaluation opens the viewer on the first simulator it boots.
    completed = run_dry(
        "finetune_ppo.sh",
        "DrillLift",
        "ppo.eval_only=true",
        cwd=tmp_path,
        env={**env, "SIM_VIZ": "kit", "DISPLAY": ":99"},
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "no usable display: DISPLAY=:99 is unreachable" in completed.stdout


def test_offline_training_boots_no_simulator(tmp_path: Path) -> None:
    env = stub_launch_env(tmp_path, "echo 'listening on 127.0.0.1'; sleep 60")
    del env["ISAACLAB_ARENA_PATH"]  # offline training never starts Isaac Lab
    completed = run_dry(
        "train.sh", "DrillLift", cwd=tmp_path, env={**env, "ALGO": "sft"}
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not (tmp_path / "sim_pids").exists()


def test_offline_eval_boots_only_the_evaluation_simulator(tmp_path: Path) -> None:
    env = stub_launch_env(tmp_path, "echo 'listening on 127.0.0.1'; sleep 60")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    run_dry(
        "eval.sh",
        "DrillLift",
        "policy",
        "2",
        str(checkpoint),
        cwd=tmp_path,
        env={**env, "ALGO": "iql"},
    )
    assert len((tmp_path / "sim_pids").read_text(encoding="utf-8").split()) == 1


def test_launchers_default_to_the_workflow_environment(tmp_path: Path) -> None:
    """Both processes run under the workflow's own .venv once `uv sync` built it; otherwise nothing is printed."""
    helper = WORKFLOW.parent / "_common" / "launch.sh"
    python = tmp_path / ".venv" / "bin" / "python"

    def found() -> str:
        return subprocess.run(
            ["bash", "-c", f'source "{helper}"; workflow_python "{tmp_path}"'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    assert found() == ""
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    assert found() == str(python)
    for launcher in ("eval.sh", "train.sh", "preflight_batched.sh"):
        text = (WORKFLOW / "scripts" / launcher).read_text(encoding="utf-8")
        assert "LEARNER_PY=${LEARNER_PYTHON:-${WORKFLOW_PY:-python3}}" in text
        assert "SIM_PY=${SIM_PYTHON:-${WORKFLOW_PY:-" in text
