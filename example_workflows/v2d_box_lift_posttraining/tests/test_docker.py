# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container launch boundaries: device mapping, mounts, credentials and fixed inputs."""

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from example_workflows.v2d_box_lift_posttraining import launch
from example_workflows.v2d_box_lift_posttraining.docker_entrypoint import (
    CHECKPOINT,
    V2D_ROOT,
    launch_arguments,
)
from example_workflows.v2d_box_lift_posttraining.launch import parser

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "example_workflows/v2d_box_lift_posttraining/scripts/docker_train.sh"


@pytest.mark.parametrize("image_id", [None, "sha256:test-image"])
def test_container_launch_records_baked_revision_without_posttrain_git(
    tmp_path, monkeypatch, image_id
):
    monkeypatch.setenv("POSTTRAIN_SOURCE_REVISION", "baked-source-revision")
    monkeypatch.delenv("POSTTRAIN_DOCKER_IMAGE_ID", raising=False)
    monkeypatch.delenv("ISAAC_RL_POSTTRAINING_DRYRUN", raising=False)
    if image_id is not None:
        monkeypatch.setenv("POSTTRAIN_DOCKER_IMAGE_ID", image_id)
    motion = tmp_path / "motion.parquet"
    motion.touch()
    sim_python = tmp_path / "sim-python"
    sim_python.touch()
    output = tmp_path / "output"
    arguments = parser().parse_args(
        [
            "--v2d-root",
            str(tmp_path / "v2d"),
            "--sim-python",
            str(sim_python),
            "--motion-file",
            str(motion),
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--output-dir",
            str(output),
        ]
    )
    monkeypatch.setattr(launch, "REPOSITORY", tmp_path / "posttrain-without-git")
    monkeypatch.setattr(launch, "validate_checkpoint", lambda path: {})
    plan = launch.resolve_plan(arguments)
    plan["servers"] = []
    plan["server_visible_devices"] = []
    plan["server_receipts"] = []

    def git_revision(command, **kwargs):
        assert command == ["git", "-C", str(arguments.v2d_root), "rev-parse", "HEAD"]
        return launch.V2D_REVISION

    monkeypatch.setattr(launch.subprocess, "check_output", git_revision)

    def run_learner(command, **kwargs):
        assert command == plan["learner"]
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch.subprocess, "run", run_learner)
    assert launch.execute(arguments, plan) == 0
    receipt = json.loads((output / "launch_plan.json").read_text())
    assert receipt["posttrain_revision"] == "baked-source-revision"
    if image_id is None:
        assert "container_image_id" not in receipt
    else:
        assert receipt["container_image_id"] == image_id
    assert not (output / "posttrain.patch").exists()


@pytest.mark.parametrize(
    ("count", "train"), [(1, "cuda:0"), (2, "cuda:1"), (4, "cuda:1")]
)
def test_container_uses_only_visible_gpu_indices(count, train):
    args = parser().parse_args(
        launch_arguments(["--training-steps", "1100"], gpu_count=count)
    )
    assert args.train_device == train
    assert args.training_steps == 1100
    assert args.evaluation_mode == "disabled"
    assert args.num_train_envs == 1
    assert args.eval_num_envs == 1
    assert args.checkpoint == CHECKPOINT
    assert args.v2d_root == V2D_ROOT
    assert args.output_dir == Path("/output/run")
    assert launch.resolve_plan(args)["server_visible_devices"] == [train.split(":")[1]]


def test_single_gpu_container_supports_separate_native_evaluation():
    args = parser().parse_args(
        launch_arguments(["--native", "--evaluation-mode", "enabled"], gpu_count=1)
    )
    plan = launch.resolve_plan(args)
    assert len(plan["servers"]) == 1
    assert plan["server_visible_devices"] == ["0"]
    assert plan["resolved_settings"]["eval.eval_base"] is True


def test_container_provenance_requires_nonempty_baked_revision(monkeypatch):
    monkeypatch.setenv("POSTTRAIN_DOCKER_IMAGE_ID", "sha256:test-image")
    monkeypatch.delenv("POSTTRAIN_SOURCE_REVISION", raising=False)
    with pytest.raises(ValueError, match="requires POSTTRAIN_SOURCE_REVISION"):
        launch.container_provenance()
    monkeypatch.setenv("POSTTRAIN_SOURCE_REVISION", "")
    with pytest.raises(ValueError, match="must not be empty"):
        launch.container_provenance()


def test_container_requires_gpu_before_download():
    with pytest.raises(ValueError, match="No CUDA GPU"):
        launch_arguments([], gpu_count=0)


@pytest.mark.parametrize(
    "option",
    [
        "--checkpoint",
        "--sim-python",
        "--v2d-root",
        "--motion-file",
        "--output-dir",
    ],
)
@pytest.mark.parametrize("equals", [True, False])
def test_container_rejects_overriding_managed_inputs(option, equals):
    arguments = [f"{option}=/wrong"] if equals else [option, "/wrong"]
    with pytest.raises(ValueError, match="container supplies"):
        launch_arguments(arguments, gpu_count=1)


def run_wrapper(arguments, *, environment):
    return subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_dry_run_needs_no_docker_and_preserves_paths_and_arguments(tmp_path):
    cache = tmp_path / "cache with spaces"
    output = tmp_path / "output with spaces"
    env = dict(
        os.environ,
        HF_TOKEN="test-token-never-print",
        WANDB_API_KEY="test-key-never-print",
    )
    result = run_wrapper(
        [
            "--gpus",
            "2,4",
            "--cache-dir",
            str(cache),
            "--output-dir",
            str(output),
            "--dry-run",
            "--",
            "--training-steps",
            "1100",
            "--override",
            "logging.wandb_name=smoke run",
        ],
        environment=env,
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert command[:4] == ["docker", "run", "--rm", "--init"]
    assert command[command.index("--gpus") + 1] == '"device=2,4"'
    assert f"type=bind,source={cache},target=/cache" in command
    assert f"type=bind,source={output},target=/output" in command
    assert command[-2:] == ["--override", "logging.wandb_name=smoke run"]
    assert "HF_TOKEN" in command
    assert "test-token-never-print" not in result.stdout + result.stderr
    assert "test-key-never-print" not in result.stdout + result.stderr
    assert not cache.exists()
    assert not output.exists()


@pytest.mark.parametrize("gpus", ["", "all", "0;echo bad", "0,", "-1"])
def test_invalid_gpu_selection_fails_before_docker(tmp_path, gpus):
    result = run_wrapper(
        ["--gpus", gpus, "--output-dir", str(tmp_path / "run"), "--dry-run"],
        environment=os.environ,
    )
    assert result.returncode == 2
    assert "GPU indices" in result.stderr


def test_existing_run_is_not_reused(tmp_path):
    (tmp_path / "run").mkdir()
    result = run_wrapper(
        ["--output-dir", str(tmp_path), "--dry-run"], environment=os.environ
    )
    assert result.returncode == 2
    assert "fresh output" in result.stderr


def test_wrapper_runs_inspected_image_without_installing_on_host(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text(
        "#!/bin/bash\n"
        'printf "%s\\0" "$@" >> "$DOCKER_TEST_LOG"\n'
        'if [[ "$1 $2" == "image inspect" ]]; then echo sha256:test-image; fi\n'
    )
    docker.chmod(0o755)
    log = tmp_path / "docker.log"
    env = dict(
        os.environ, PATH=f"{binary}:{os.environ['PATH']}", DOCKER_TEST_LOG=str(log)
    )
    result = run_wrapper(
        [
            "--image",
            "test/image:v1",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "output"),
            "--",
            "--training-steps",
            "1100",
        ],
        environment=env,
    )
    assert result.returncode == 0, result.stderr
    calls = log.read_text().split("\0")
    assert "POSTTRAIN_DOCKER_IMAGE_ID=sha256:test-image" in calls
    assert calls[-4:-1] == ["sha256:test-image", "--training-steps", "1100"]
    assert "buildx" not in calls
    assert "pull" not in calls
    assert (tmp_path / "cache/kit/cache").is_dir()
