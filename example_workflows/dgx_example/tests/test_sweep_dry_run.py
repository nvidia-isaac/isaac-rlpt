# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The layout sweep's dry run resolves every launch without docker, a GPU or the simulator stack."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from example_workflows.conftest import dryrun_text, run_launcher_dryrun

WORKFLOW = Path(__file__).resolve().parent.parent
REPO_ROOT = WORKFLOW.parent.parent
SWEEP = WORKFLOW / "scripts" / "layout_sweep.sh"


def _station_env(sandbox) -> dict[str, str]:
    """A fake G1 Factory tree whose posttrain/ is this repo, and this interpreter first on PATH."""
    g1f = Path(sandbox["tmp"]) / "g1f"
    g1f.mkdir()
    (g1f / "posttrain").symlink_to(REPO_ROOT)
    return {
        "G1F": str(g1f),
        "HF_CACHE": str(Path(sandbox["tmp"]) / "hf_cache"),
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
    }


def _plans(text: str) -> list[str]:
    return [line for line in text.splitlines() if "plan layout=" in line]


def test_sweep_plans_one_launch_per_single_layout_and_two_per_split(
    launcher_sandbox,
) -> None:
    result = run_launcher_dryrun(
        SWEEP,
        [],
        launcher_sandbox,
        extra_env={
            **_station_env(launcher_sandbox),
            "LAYOUTS": "one_card:0:0 two_cards:1:0",
            "NS": "8 64",
        },
        timeout=240,
    )
    text = dryrun_text(result)
    assert result.returncode == 0, text
    plans = _plans(text)
    assert len(plans) == 6, text
    assert sum("role=single" in line for line in plans) == 2
    assert sum("role=simulator" in line for line in plans) == 2
    assert sum("role=learner" in line for line in plans) == 2
    # Every container is pinned to its card and runs without --privileged, which would undo the pin.
    assert text.count("launch.docker.gpus=device=") == 6
    assert text.count("launch.docker.privileged=none") == 6
    commands = [
        line for line in text.splitlines() if line.startswith("launch.container_cmd=")
    ]
    assert len(commands) == 6
    simulator_only = [command for command in commands if "SIMS_ONLY=1" in command]
    learners = [command for command in commands if "SIMS_ONLY=1" not in command]
    assert len(simulator_only) == 2 and len(learners) == 4
    # The two split-layout learners reach their simulator container over the train URL; the single-card ones do not.
    assert (
        sum("TRAIN_SIM_URL=tcp://127.0.0.1:7300" in command for command in learners)
        == 2
    )
    # The probe settings reach every learner: the budget max(1350 N, 6400), the burst cap, no eval sim, shrunk frames.
    assert any(
        "NUM_ROLLOUTS=10800 " in command and "N=8 " in command for command in learners
    )
    assert any(
        "NUM_ROLLOUTS=86400 " in command and "N=64 " in command for command in learners
    )
    for command in learners:
        assert "training.max_updates_per_burst=20" in command
        assert "schedule.kind=sync" in command
        assert "EVAL_SIM=0" in command and "IMAGE_SHORTEST_EDGE=256" in command
    # The frozen policy decodes the whole batch in one forward.
    assert any(
        "N=64 " in command and "simulator.decode_batch_size=64" in command
        for command in learners
    )
    assert all(
        "simulator.decode_batch_size" not in command for command in simulator_only
    )


def test_sweep_refuses_a_split_for_an_in_process_simulator(launcher_sandbox) -> None:
    result = run_launcher_dryrun(
        SWEEP,
        [],
        launcher_sandbox,
        extra_env={
            **_station_env(launcher_sandbox),
            "SIMULATOR": "isaaclab",
            "LAYOUTS": "two_cards:1:0",
            "NS": "8",
        },
        timeout=240,
    )
    assert result.returncode != 0
    assert "cannot be placed" in dryrun_text(result)
    assert not _plans(dryrun_text(result))


def test_sweep_needs_the_two_cards_or_explicit_layouts(launcher_sandbox) -> None:
    result = run_launcher_dryrun(
        SWEEP,
        [],
        launcher_sandbox,
        extra_env=_station_env(launcher_sandbox),
        timeout=60,
    )
    assert result.returncode != 0
    assert "set GB300_GPU and RTX_GPU" in dryrun_text(result)


TRAIN_EVAL = WORKFLOW / "scripts" / "train_eval.sh"


def test_train_eval_dry_run_trains_in_the_split_then_scores_base_and_checkpoint(
    launcher_sandbox,
) -> None:
    result = run_launcher_dryrun(
        TRAIN_EVAL,
        ["DrillLift"],
        launcher_sandbox,
        extra_env={
            **_station_env(launcher_sandbox),
            "SIMULATOR_GPU": "GPU-sim",
            "LEARNER_GPU": "GPU-learn",
            "NUM_ENVS": "64",
            "TRAIN_ENV_STEPS": "100000",
            "EVAL_EPISODES": "100",
            "OUT_DIR": str(Path(launcher_sandbox["tmp"]) / "out"),
        },
        timeout=240,
    )
    text = dryrun_text(result)
    assert result.returncode == 0, text
    commands = [
        line for line in text.splitlines() if line.startswith("launch.container_cmd=")
    ]
    assert len(commands) == 4, text
    simulators, learner, base, policy = commands
    assert (
        "SIMS_ONLY=1" in simulators
        and "EVAL_SIM=1" in simulators
        and " N=64 " in simulators
    )
    assert (
        "TRAIN_SIM_URL=tcp://127.0.0.1:7300" in learner
        and "EVAL_SIM_URL=tcp://127.0.0.1:7301" in learner
    )
    assert "NUM_ROLLOUTS=100000" in learner and "SIMS_ONLY" not in learner
    assert "simulator.decode_batch_size=64" in learner
    for scoring in (base, policy):
        assert " N=1 " in scoring
        for override in (
            "base_model.dataset_dir=null",
            "training.num_rollouts=0",
            "eval.eval_num_episodes=100",
        ):
            assert override in scoring, scoring
        assert (
            "base_model.task_instruction='Walk to the table and pick up the power drill'"
            in scoring
        )
    assert "eval.eval_base=true" in base and "training.load_dir" not in base
    assert "training.load_dir=" in policy and "checkpoints/final" in policy
    # The simulators sit on their card, the learner and both scoring runs on the other; every container is pinned.
    assert text.count("launch.docker.gpus=device=GPU-sim") == 1
    assert text.count("launch.docker.gpus=device=GPU-learn") == 3
    assert text.count("launch.docker.privileged=none") == 4


def test_train_eval_on_one_card_runs_everything_in_one_container(
    launcher_sandbox,
) -> None:
    result = run_launcher_dryrun(
        TRAIN_EVAL,
        ["BoxLift"],
        launcher_sandbox,
        extra_env={
            **_station_env(launcher_sandbox),
            "SIMULATOR_GPU": "0",
            "LEARNER_GPU": "0",
            "NUM_ENVS": "8",
        },
        timeout=240,
    )
    text = dryrun_text(result)
    assert result.returncode == 0, text
    commands = [
        line for line in text.splitlines() if line.startswith("launch.container_cmd=")
    ]
    assert len(commands) == 3 and "SIMS_ONLY" not in text
    assert " N=8 " in commands[0] and "TRAIN_SIM_URL" not in commands[0]
    assert "launch.task=LMBoxLift" in text


def test_sweep_runs_the_learner_on_the_rlinf_runtime_when_asked(
    launcher_sandbox,
) -> None:
    """RUNTIME=rlinf reaches every learner container (its own image, the runtime in its name); a simulator container stays native."""
    result = run_launcher_dryrun(
        SWEEP,
        [],
        launcher_sandbox,
        extra_env={
            **_station_env(launcher_sandbox),
            "LAYOUTS": "one_card:0:0 two_cards:1:0",
            "NS": "8",
            "RUNTIME": "rlinf",
        },
        timeout=240,
    )
    text = dryrun_text(result)
    assert result.returncode == 0, text
    commands = [
        line for line in text.splitlines() if line.startswith("launch.container_cmd=")
    ]
    assert len(commands) == 3, text
    simulator_only = [command for command in commands if "SIMS_ONLY=1" in command]
    learners = [command for command in commands if "SIMS_ONLY=1" not in command]
    assert len(simulator_only) == 1 and len(learners) == 2
    assert (
        "RUNTIME=native" in simulator_only[0]
        and "WANDB_NAME=probe_two_cards_rlinf_n8_sim" in simulator_only[0]
    )
    for command in learners:
        assert "RUNTIME=rlinf" in command, command
    assert any(
        "WANDB_NAME=probe_one_card_rlinf_n8 " in command for command in learners
    ), learners
    assert any(
        "WANDB_NAME=probe_two_cards_rlinf_n8 " in command for command in learners
    ), learners
    assert text.count("launch.image=g1_factory_learner_rlinf:") == 2, text
    assert text.count("launch.image=g1_factory_learner:") == 1, text
    assert "runtime=rlinf" in text


def test_sweep_spreads_the_split_over_several_servers_when_asked(
    launcher_sandbox,
) -> None:
    """NUM_SERVERS=2 reaches both containers of a split: the simulator's boots two servers on consecutive
    ports, the learner connects to both, and the run's name carries the server count."""
    result = run_launcher_dryrun(
        SWEEP,
        [],
        launcher_sandbox,
        extra_env={
            **_station_env(launcher_sandbox),
            "LAYOUTS": "two_cards:1:0",
            "NS": "8",
            "RUNTIME": "rlinf",
            "NUM_SERVERS": "2",
        },
        timeout=240,
    )
    text = dryrun_text(result)
    assert result.returncode == 0, text
    commands = [
        line for line in text.splitlines() if line.startswith("launch.container_cmd=")
    ]
    simulator_only = [command for command in commands if "SIMS_ONLY=1" in command]
    learners = [command for command in commands if "SIMS_ONLY=1" not in command]
    assert len(simulator_only) == 1 and len(learners) == 1, text
    assert (
        "NUM_SERVERS=2" in simulator_only[0]
        and "WANDB_NAME=probe_two_cards_rlinf_2srv_n8_sim" in simulator_only[0]
    )
    assert (
        "NUM_SERVERS=2" in learners[0]
        and "WANDB_NAME=probe_two_cards_rlinf_2srv_n8 " in learners[0]
    )
    assert "TRAIN_SIM_URL=tcp://127.0.0.1:7300,tcp://127.0.0.1:7301" in learners[0], (
        learners[0]
    )
    assert "servers=2" in text and "launch.num_servers=2" in text


def test_train_eval_on_the_rlinf_runtime_names_the_run_and_keeps_the_simulators_native(
    launcher_sandbox,
) -> None:
    result = run_launcher_dryrun(
        TRAIN_EVAL,
        ["DrillLift"],
        launcher_sandbox,
        extra_env={
            **_station_env(launcher_sandbox),
            "SIMULATOR_GPU": "GPU-sim",
            "LEARNER_GPU": "GPU-learn",
            "NUM_ENVS": "64",
            "RUNTIME": "rlinf",
            "OUT_DIR": str(Path(launcher_sandbox["tmp"]) / "out"),
        },
        timeout=240,
    )
    text = dryrun_text(result)
    assert result.returncode == 0, text
    commands = [
        line for line in text.splitlines() if line.startswith("launch.container_cmd=")
    ]
    assert len(commands) == 4, text
    simulators, learner, base, policy = commands
    assert (
        "RUNTIME=native" in simulators
        and "WANDB_NAME=dsrl_LMDrillLift_split_rlinf_n64_sim" in simulators
    )
    for command in (learner, base, policy):
        assert "RUNTIME=rlinf" in command, command
        assert "LEARNER_GPU=GPU-learn" not in command, command
    assert "WANDB_NAME=dsrl_LMDrillLift_split_rlinf_n64 " in learner, learner


def test_sweep_runs_the_overlapped_schedule_with_its_own_burst_cap_when_asked(
    launcher_sandbox,
) -> None:
    """SCHEDULE and MAX_UPDATES_PER_BURST reach every learner; the run's name is RUN_PREFIX, layout, runtime, schedule, N."""
    result = run_launcher_dryrun(
        SWEEP,
        [],
        launcher_sandbox,
        extra_env={
            **_station_env(launcher_sandbox),
            "LAYOUTS": "one_card:0:0 two_cards:1:0",
            "NS": "8",
            "RUNTIME": "rlinf",
            "SCHEDULE": "overlapped",
            "MAX_UPDATES_PER_BURST": "300",
            "RUN_PREFIX": "balanced",
        },
        timeout=240,
    )
    text = dryrun_text(result)
    assert result.returncode == 0, text
    commands = [
        line for line in text.splitlines() if line.startswith("launch.container_cmd=")
    ]
    learners = [command for command in commands if "SIMS_ONLY=1" not in command]
    assert len(learners) == 2, text
    for command in learners:
        assert "schedule.kind=overlapped" in command, command
        assert "training.max_updates_per_burst=300" in command, command
    assert any(
        "WANDB_NAME=balanced_one_card_rlinf_overlapped_n8 " in command
        for command in learners
    ), learners
    assert any(
        "WANDB_NAME=balanced_two_cards_rlinf_overlapped_n8 " in command
        for command in learners
    ), learners
    assert "WANDB_NAME=balanced_two_cards_rlinf_overlapped_n8_sim" in text
    assert "schedule=overlapped" in text


def test_train_eval_on_the_overlapped_schedule_names_the_run_and_trains_with_it(
    launcher_sandbox,
) -> None:
    result = run_launcher_dryrun(
        TRAIN_EVAL,
        ["DrillLift"],
        launcher_sandbox,
        extra_env={
            **_station_env(launcher_sandbox),
            "SIMULATOR_GPU": "GPU-sim",
            "LEARNER_GPU": "GPU-learn",
            "NUM_ENVS": "64",
            "SCHEDULE": "overlapped",
            "OUT_DIR": str(Path(launcher_sandbox["tmp"]) / "out"),
        },
        timeout=240,
    )
    text = dryrun_text(result)
    assert result.returncode == 0, text
    commands = [
        line for line in text.splitlines() if line.startswith("launch.container_cmd=")
    ]
    assert len(commands) == 4, text
    simulators, learner, base, policy = commands
    assert "WANDB_NAME=dsrl_LMDrillLift_split_overlapped_n64_sim" in simulators
    assert "WANDB_NAME=dsrl_LMDrillLift_split_overlapped_n64 " in learner, learner
    assert "schedule.kind=overlapped" in learner, learner
    for scoring in (base, policy):
        assert "schedule.kind" not in scoring, scoring


def test_train_eval_runs_the_full_recipe_on_several_servers_with_a_sized_burst_and_batched_scoring(
    launcher_sandbox,
) -> None:
    """NUM_SERVERS, MAX_UPDATES_PER_BURST, EVAL_FREQ and EVAL_NUM_ENVS reach the right containers:
    the servers and the evaluation width both containers of the split, the burst cap and the cadence
    the learner alone, the width also the scoring runs; the ports follow TRAIN_PORT."""
    result = run_launcher_dryrun(
        TRAIN_EVAL,
        ["DrillLift"],
        launcher_sandbox,
        extra_env={
            **_station_env(launcher_sandbox),
            "SIMULATOR_GPU": "GPU-sim",
            "LEARNER_GPU": "GPU-learn",
            "RUNTIME": "rlinf",
            "SCHEDULE": "overlapped",
            "NUM_SERVERS": "2",
            "NUM_ENVS": "256",
            "TRAIN_ENV_STEPS": "800000",
            "MAX_UPDATES_PER_BURST": "310",
            "EVAL_FREQ": "400000",
            "EVAL_NUM_ENVS": "25",
            "EVAL_EPISODES": "100",
            "TRAIN_PORT": "7400",
            "TRAIN_OVERRIDES": "algorithm.batch_size=256 algorithm.num_updates_per_train_step=5",
            "OUT_DIR": str(Path(launcher_sandbox["tmp"]) / "out"),
        },
        timeout=240,
    )
    text = dryrun_text(result)
    assert result.returncode == 0, text
    commands = [
        line for line in text.splitlines() if line.startswith("launch.container_cmd=")
    ]
    assert len(commands) == 4, text
    simulators, learner, base, policy = commands
    assert (
        " NUM_SERVERS=2 " in simulators
        and " EVAL_NUM_ENVS=25 " in simulators
        and " N=256 " in simulators
        and " TRAIN_PORT=7400 " in simulators
        and " EVAL_PORT=7402 " in simulators
    ), simulators
    assert (
        "TRAIN_SIM_URL=tcp://127.0.0.1:7400,tcp://127.0.0.1:7401" in learner
        and "EVAL_SIM_URL=tcp://127.0.0.1:7402" in learner
        and " NUM_SERVERS=2 " in learner
        and " EVAL_NUM_ENVS=25 " in learner
        and "NUM_ROLLOUTS=800000" in learner
    ), learner
    for override in (
        "schedule.kind=overlapped",
        "training.max_updates_per_burst=310",
        "eval.eval_freq=400000",
        "training.save_interval=400000",
        "algorithm.batch_size=256",
        "algorithm.num_updates_per_train_step=5",
    ):
        assert override in learner, learner
    assert "WANDB_NAME=dsrl_LMDrillLift_split_rlinf_overlapped_2srv_n256 " in learner, (
        learner
    )
    # The recipe's card names stay outside the containers: inside, the launcher indexes the pinned card itself.
    for command in commands:
        assert "LEARNER_GPU=GPU-learn" not in command and "GPU-sim" not in command, (
            command
        )
    # The scoring runs keep one environment on one server, score in parallel, and train nothing.
    for scoring in (base, policy):
        assert " N=1 " in scoring and " NUM_SERVERS=1 " in scoring, scoring
        assert " EVAL_NUM_ENVS=25 " in scoring, scoring
        assert "eval.eval_num_episodes=100" in scoring, scoring
        for absent in (
            "max_updates_per_burst",
            "eval.eval_freq",
            "save_interval",
            "algorithm.batch_size",
        ):
            assert absent not in scoring, scoring


def test_train_eval_refuses_several_servers_on_the_native_runtime(
    launcher_sandbox,
) -> None:
    result = run_launcher_dryrun(
        TRAIN_EVAL,
        ["DrillLift"],
        launcher_sandbox,
        extra_env={
            **_station_env(launcher_sandbox),
            "SIMULATOR_GPU": "GPU-sim",
            "LEARNER_GPU": "GPU-learn",
            "NUM_SERVERS": "2",
            "OUT_DIR": str(Path(launcher_sandbox["tmp"]) / "out"),
        },
        timeout=240,
    )
    assert result.returncode != 0
    assert "NUM_SERVERS=2 needs RUNTIME=rlinf" in dryrun_text(result)


@pytest.mark.parametrize(
    ("script", "old", "new"),
    [
        (SWEEP, "serial", "sync"),
        (SWEEP, "lookahead", "overlapped"),
        (TRAIN_EVAL, "serial", "sync"),
        (TRAIN_EVAL, "lookahead", "overlapped"),
    ],
)
def test_an_old_schedule_name_is_refused_with_its_new_name(
    launcher_sandbox, script: Path, old: str, new: str
) -> None:
    result = run_launcher_dryrun(
        script,
        ["DrillLift"] if script == TRAIN_EVAL else [],
        launcher_sandbox,
        extra_env={
            **_station_env(launcher_sandbox),
            "SIMULATOR_GPU": "GPU-sim",
            "LEARNER_GPU": "GPU-learn",
            "LAYOUTS": "two_cards:1:0",
            "NS": "8",
            "SCHEDULE": old,
            "OUT_DIR": str(Path(launcher_sandbox["tmp"]) / "out"),
        },
        timeout=240,
    )
    text = dryrun_text(result)
    assert result.returncode != 0, text
    assert f"FATAL: SCHEDULE={old} was renamed; use SCHEDULE={new}" in text, text
    assert "launch.container_cmd=" not in text, text


def test_sweep_goes_on_after_a_failed_run(launcher_sandbox, tmp_path) -> None:
    """A run whose launcher fails (here: no learner image, a log without the usual verdict lines) is reported and the next run starts."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "docker").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (fake_bin / "docker").chmod(0o755)
    env = _station_env(launcher_sandbox)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env.update(
        {
            "LAYOUTS": "one_card:0:0",
            "NS": "8 64",
            "COOLDOWN_S": "0",
            "ISAAC_RL_POSTTRAINING_DRYRUN": "0",
        }
    )
    result = run_launcher_dryrun(
        SWEEP, [], launcher_sandbox, extra_env=env, timeout=240
    )
    text = dryrun_text(result)
    assert result.returncode == 0, text
    assert (
        "N=8: the launcher exited with status" in text
        and "N=64: the launcher exited with status" in text
    )
    assert text.count("ended:") == 2, text
    assert any(line.endswith(" done") for line in text.splitlines())


def test_train_eval_stops_when_training_fails_instead_of_scoring_an_older_checkpoint(
    tmp_path: Path,
) -> None:
    (
        tmp_path
        / "out"
        / "dsrl_LMDrillLift_single_n64_2026-01-01_00-00-00"
        / "checkpoints"
        / "final"
    ).mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(
        "#!/bin/bash\nexit 3\n", encoding="utf-8"
    )  # the station's container fails
    (bin_dir / "docker").chmod(0o755)
    env = {
        key: value
        for key, value in os.environ.items()
        if key != "ISAAC_RL_POSTTRAINING_DRYRUN"
    }
    env |= {
        "PATH": f"{bin_dir}{os.pathsep}{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "G1F": str(tmp_path / "g1f"),
        "SIMULATOR_GPU": "0",
        "LEARNER_GPU": "0",
        "OUT_DIR": str(tmp_path / "out"),
        "LOG_DIR": str(tmp_path / "logs"),
    }
    completed = subprocess.run(
        ["bash", str(WORKFLOW / "scripts" / "train_eval.sh")],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    log = (tmp_path / "logs" / "train_eval.log").read_text(encoding="utf-8")
    assert completed.returncode == 1, log + completed.stderr
    assert "training failed" in log and "score" not in log, log
