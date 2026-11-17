# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve and launch the V2D box-lift servers and RLPT learner."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, TypedDict

# Run as a file from any directory, as scripts/train.py is.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from example_workflows.v2d_box_lift_posttraining.checkpoint_contract import (
    validate_checkpoint,
)
from example_workflows.v2d_box_lift_posttraining.config import TaskConfig, load_task
from simulators.v2d.contract import JsonValue

if TYPE_CHECKING:
    from isaac_rl_posttraining import Env
    from isaac_rl_posttraining.algorithms import DSRL

WORKFLOW = Path(__file__).resolve().parent
REPOSITORY = WORKFLOW.parents[1]
DEFAULT_TASK = (
    WORKFLOW / "configs" / "tasks" / "tissue_box_lift_hold_xy_1cm_550_fixed.json"
)
V2D_REVISION = "33129dd0f2d2dcfd1164d43fd076542660756ed2"


class LaunchPlan(TypedDict):
    output_dir: str
    runtime_dir: str
    train_url: str
    eval_url: str | None
    train_capability_path: str
    eval_capability_path: str | None
    train_capability: dict[str, str]
    eval_capability: dict[str, str] | None
    servers: list[list[str]]
    server_visible_devices: list[str]
    server_receipts: list[str]
    native_baseline_learner: list[str] | None
    learner: list[str]
    native_baseline_settings: dict[str, JsonValue] | None
    resolved_settings: dict[str, JsonValue]
    task: dict[str, JsonValue]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--v2d-root", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--motion-file", type=Path, required=True)
    result.add_argument("--sim-python", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--train-task-config", type=Path, default=DEFAULT_TASK)
    result.add_argument("--eval-task-config", type=Path, default=DEFAULT_TASK)
    result.add_argument("--num-train-envs", type=int, default=128)
    result.add_argument("--eval-num-envs", type=int, default=32)
    result.add_argument("--eval-num-episodes", type=int, default=32)
    result.add_argument(
        "--evaluation-mode",
        choices=("enabled", "disabled"),
        default="disabled",
        help="Whether to launch the evaluation simulator and run evaluations",
    )
    result.add_argument("--training-steps", type=int, default=704000)
    result.add_argument("--training-decode-batch-size", type=int, default=16)
    result.add_argument(
        "--training-cadence", choices=("step", "episode"), default="step"
    )
    result.add_argument("--updates-per-transition", type=int, default=1)
    result.add_argument("--replay-batch-size", type=int, default=64)
    result.add_argument(
        "--critic-target-reduction", choices=("mean", "min"), default="mean"
    )
    result.add_argument("--max-updates-per-burst", type=int, default=128)
    result.add_argument("--seed", type=int, default=2)
    result.add_argument("--eval-seed", type=int, default=10000)
    result.add_argument("--train-port", type=int, default=5555)
    result.add_argument("--eval-port", type=int, default=5556)
    result.add_argument("--train-device", default="cuda:1")
    result.add_argument("--eval-device", default="cuda:2")
    result.add_argument("--startup-timeout-seconds", type=int, default=900)
    result.add_argument("--run-name", default="v2d-box-lift-dsrl")
    result.add_argument("--steering-checkpoint", type=Path)
    result.add_argument(
        "--native",
        action="store_true",
        help="Run only the frozen-policy baseline on the evaluation server",
    )
    result.add_argument("--dry-run", action="store_true")
    result.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override; workflow contract keys cannot be replaced",
    )
    return result


def _validate_arguments(args: argparse.Namespace) -> None:
    for name in (
        "num_train_envs",
        "eval_num_envs",
        "eval_num_episodes",
        "training_decode_batch_size",
        "updates_per_transition",
        "replay_batch_size",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.eval_num_episodes < args.eval_num_envs:
        raise ValueError("eval_num_episodes must be at least eval_num_envs")
    if args.eval_num_episodes % args.eval_num_envs:
        raise ValueError("eval_num_episodes must be an exact multiple of eval_num_envs")
    if args.training_steps < 0:
        raise ValueError("training_steps must be non-negative")
    if args.max_updates_per_burst is not None and args.max_updates_per_burst < 1:
        raise ValueError("max_updates_per_burst must be positive")
    if args.startup_timeout_seconds <= 60:
        raise ValueError("startup_timeout_seconds must exceed 60")
    if args.native and args.steering_checkpoint is not None:
        raise ValueError("Native evaluation cannot load a steering checkpoint")
    if args.native and args.evaluation_mode == "disabled":
        raise ValueError("Native evaluation requires evaluation-mode=enabled")


def _server_command(
    args: argparse.Namespace,
    *,
    capability_config: Path,
    receipt: Path,
    port: int,
    seed: int,
    num_envs: int,
) -> list[str]:
    return [
        str(args.sim_python),
        "-m",
        "simulators.v2d.server",
        "--headless",
        "--device",
        "cuda:0",
        "--capability",
        "example_workflows.v2d_box_lift_posttraining.capability:create_capability",
        "--capability-config",
        str(capability_config),
        "--port",
        str(port),
        "--seed",
        str(seed),
        "--num-envs",
        str(num_envs),
        "--startup-timeout-seconds",
        str(args.startup_timeout_seconds),
        "--receipt",
        str(receipt),
    ]


def _physical_gpu(device: str) -> str:
    prefix, separator, index = device.partition(":")
    if prefix != "cuda" or separator != ":" or not index.isdigit():
        raise ValueError(f"Simulator devices must use cuda:<index>, got {device!r}")
    return str(int(index))


def _log_tail(path: Path, *, line_count: int = 80) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def _wait_for_server(
    process: subprocess.Popen[str],
    *,
    receipt: Path,
    log_path: Path,
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"V2D server exited with code {return_code} before writing {receipt}\n"
                f"Server log tail (full log: {log_path}):\n{_log_tail(log_path)}"
            )
        if receipt.is_file():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"V2D server did not write {receipt} within {timeout_seconds} seconds\n"
                f"Server log tail (full log: {log_path}):\n{_log_tail(log_path)}"
            )
        time.sleep(0.1)


def _load_eval_task(path: Path, train_task: TaskConfig) -> TaskConfig:
    eval_task = load_task(path)
    if (
        replace(
            eval_task,
            reset_perturbation=train_task.reset_perturbation,
            fixed_reset=train_task.fixed_reset,
        )
        != train_task
    ):
        raise ValueError(
            "Training and evaluation task configs may differ only in reset_perturbation and fixed_reset"
        )
    return eval_task


def _pinned_settings(
    args: argparse.Namespace,
    train_task: TaskConfig,
    eval_task: TaskConfig,
    *,
    evaluation_enabled: bool,
    output: Path,
) -> dict[str, JsonValue]:
    """The learner settings the workflow pins; overrides may not replace them."""
    return {
        "+simulator.reset_validation": None
        if train_task.reset_validation is None
        else train_task.reset_validation.as_dict(),
        "+simulator.training_fixed_reset": train_task.fixed_reset,
        "+simulator.evaluation_fixed_reset": eval_task.fixed_reset
        if evaluation_enabled
        else None,
        "env.env_name": f"v2d/{train_task.task_id}",
        "env.max_episode_steps": train_task.episode_horizon,
        "env.image_keys": [
            "video.front",
            "video.right_wrist_view",
            "video.left_wrist_view",
        ],
        "env.sparse_success_reward": True,
        "env.success_reward": 1.0,
        "env.failure_reward": 0.0,
        "env.step_reward": 0.0,
        "base_model.ckpt": str(args.checkpoint.resolve()),
        "base_model.dataset_adapter": "none",
        "base_model.dataset_dir": None,
        "base_model.embodiment_tag": "new_embodiment",
        "base_model.task_instruction": "lift the box and hold it",
        "training.num_envs": args.num_train_envs,
        "training.num_rollouts": 0 if args.native else args.training_steps,
        "training.chunk_size": 16,
        "training.seed": args.seed,
        "training.load_dir": (
            None
            if args.steering_checkpoint is None
            else str(args.steering_checkpoint.resolve())
        ),
        "training.rollout_count_by": args.training_cadence,
        "training.train_after_episode": args.training_cadence == "episode",
        "training.max_updates_per_burst": args.max_updates_per_burst,
        "training.policy_decode_batch_size": args.training_decode_batch_size,
        "training.checkpoint_retention": "best_last_final",
        "dsrl.noise_dim": 58,
        "dsrl.action_exec_len": train_task.execution_length,
        "dsrl.noise_action_bound": 1.0,
        "dsrl.warmup_exploration": "normal",
        "algorithm.learning_starts": args.num_train_envs * train_task.episode_horizon,
        "algorithm.gamma": 1.0,
        "algorithm.pooled_critic_features": False,
        "algorithm.critic_target_reduction": args.critic_target_reduction,
        "algorithm.num_updates_per_train_step": args.updates_per_transition,
        "algorithm.batch_size": args.replay_batch_size,
        "algorithm.num_critic_updates_per_actor_update": 4,
        "algorithm.bootstrap_at_truncation": False,
        "eval.eval_base": args.native,
        "eval.eval_on_first_step": evaluation_enabled,
        # A value below one complete collection round makes evaluation due at
        # every round boundary; it is not an exact step cadence.
        "eval.eval_freq": 1 if evaluation_enabled else 0,
        "eval.eval_at_episode_boundary": True,
        "eval.eval_num_envs": args.eval_num_envs,
        "eval.eval_num_episodes": args.eval_num_episodes,
        "eval.eval_max_steps": (
            eval_task.episode_horizon + eval_task.execution_length - 1
        )
        // eval_task.execution_length,
        "eval.eval_record_video": False,
        "eval.action_selection": "sample",
        "eval.reset_seed": args.eval_seed,
        "logging.wandb_project": "v2d_posttrain_pipeline",
        "logging.wandb_entity": "nvidia-isaac",
        "logging.wandb_name": args.run_name,
        "logging.wandb_log_dir_base": str(output / "wandb"),
        "hydra.run.dir": str(output / "hydra"),
    }


def _reject_pinned_overrides(overrides: list[str], protected: set[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Hydra override must contain '=': {override}")
        key = override.split("=", 1)[0].lstrip("+")
        if key in protected:
            raise ValueError(
                f"Override would replace a pinned workflow contract: {key}"
            )


def resolve_plan(args: argparse.Namespace) -> LaunchPlan:
    """Resolve the immutable execution contract without touching external inputs."""
    _validate_arguments(args)
    train_task = load_task(args.train_task_config)
    evaluation_enabled = args.evaluation_mode == "enabled"
    evaluation_only = args.native or (evaluation_enabled and args.training_steps == 0)
    eval_task = (
        _load_eval_task(args.eval_task_config, train_task)
        if evaluation_enabled
        else train_task
    )
    output = args.output_dir.resolve()
    runtime_dir = output / "runtime"
    train_capability = runtime_dir / "train_capability.json"
    eval_capability = runtime_dir / "eval_capability.json"
    train_url = f"tcp://127.0.0.1:{args.train_port}"
    eval_url = f"tcp://127.0.0.1:{args.eval_port}" if evaluation_enabled else None
    train_visible_device = _physical_gpu(args.train_device)
    eval_visible_device = _physical_gpu(args.eval_device)
    if (
        evaluation_enabled
        and not evaluation_only
        and train_visible_device == eval_visible_device
    ):
        raise ValueError("Training and evaluation simulators require distinct GPUs")
    settings = _pinned_settings(
        args,
        train_task,
        eval_task,
        evaluation_enabled=evaluation_enabled,
        output=output,
    )
    protected = {key.lstrip("+") for key in settings}
    # These recipe defaults remain overridable; the contract above is pinned.
    settings = {
        "algorithm.ent_coef": 0.005,
        "algorithm.td_target_clip": [0.0, 1.0],
        "training.save_interval": 10000,
        **settings,
    }
    _reject_pinned_overrides(args.override, protected)

    def learner_command(resolved_settings: dict[str, JsonValue]) -> list[str]:
        return [
            sys.executable,
            str(REPOSITORY / "scripts" / "train.py"),
            "algorithm=dsrl_sac",
            "base_model=gr00t",
            "simulator=v2d",
            *[f"{key}={json.dumps(value)}" for key, value in resolved_settings.items()],
            *args.override,
        ]

    native_baseline_settings = None
    native_baseline_learner = None
    if evaluation_enabled and not evaluation_only:
        native_baseline_settings = dict(settings)
        native_baseline_settings.update(
            {
                "training.num_rollouts": 0,
                "training.load_dir": None,
                "eval.eval_base": True,
                "logging.wandb_name": f"{args.run_name}-native-baseline",
                "logging.wandb_log_dir_base": str(output / "native_baseline/wandb"),
                "hydra.run.dir": str(output / "native_baseline/hydra"),
            }
        )
        native_baseline_learner = learner_command(native_baseline_settings)
    capability_values = {
        "v2d_root": str(args.v2d_root.resolve()),
        "motion_file": str(args.motion_file.resolve()),
        "expected_v2d_revision": V2D_REVISION,
    }
    train_capability_values = {
        **capability_values,
        "task_config": str(args.train_task_config.resolve()),
    }
    eval_capability_values = None
    if evaluation_enabled:
        eval_capability_values = {
            **capability_values,
            "task_config": str(args.eval_task_config.resolve()),
        }
    servers = []
    server_visible_devices = []
    server_receipts = []
    if not evaluation_only:
        train_receipt = runtime_dir / "train_server_receipt.json"
        servers.append(
            _server_command(
                args,
                capability_config=train_capability,
                receipt=train_receipt,
                port=args.train_port,
                seed=args.seed,
                num_envs=args.num_train_envs,
            )
        )
        server_visible_devices.append(train_visible_device)
        server_receipts.append(str(train_receipt))
    if evaluation_enabled:
        eval_receipt = runtime_dir / "eval_server_receipt.json"
        servers.append(
            _server_command(
                args,
                capability_config=eval_capability,
                receipt=eval_receipt,
                port=args.eval_port,
                seed=args.eval_seed,
                num_envs=args.eval_num_envs,
            )
        )
        server_visible_devices.append(eval_visible_device)
        server_receipts.append(str(eval_receipt))
    return {
        "output_dir": str(output),
        "runtime_dir": str(runtime_dir),
        "train_url": train_url,
        "eval_url": eval_url,
        "train_capability_path": str(train_capability),
        "eval_capability_path": str(eval_capability) if evaluation_enabled else None,
        "train_capability": train_capability_values,
        "eval_capability": eval_capability_values,
        "servers": servers,
        "server_visible_devices": server_visible_devices,
        "server_receipts": server_receipts,
        "native_baseline_learner": native_baseline_learner,
        "learner": learner_command(settings),
        "native_baseline_settings": native_baseline_settings,
        "resolved_settings": settings,
        "task": asdict(train_task),
    }


#: Variables `uv run` exports that would make a simulator wrapper pick this repository's environment.
_LAUNCHER_ONLY_VARIABLES = ("VIRTUAL_ENV", "UV", "UV_RUN_RECURSION_DEPTH")


def simulator_environment(
    environment: dict[str, str], visible_device: str
) -> dict[str, str]:
    """The simulator's environment: the launcher's, minus its virtual environment."""
    result = {
        key: value
        for key, value in environment.items()
        if key not in _LAUNCHER_ONLY_VARIABLES
    }
    virtual_env = environment.get("VIRTUAL_ENV")
    if virtual_env and "PATH" in result:
        venv_bin = str(Path(virtual_env) / "bin")
        result["PATH"] = os.pathsep.join(
            entry for entry in result["PATH"].split(os.pathsep) if entry != venv_bin
        )
    result["CUDA_VISIBLE_DEVICES"] = visible_device
    return result


def start_simulator(
    command: list[str], environment: dict[str, str], log: TextIO
) -> subprocess.Popen[str]:
    """Start a simulator in its own process group, so it can be stopped with every process it spawns."""
    return subprocess.Popen(
        command,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _signal_group(process: subprocess.Popen[str], signum: int) -> bool:
    """Signal the process group ``process`` leads; False when no member is left."""
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return False
    return True


def _terminate(
    children: list[subprocess.Popen[str]], grace_seconds: float = 30.0
) -> None:
    """Stop every simulator and the processes it started (Kit runs as a grandchild)."""
    for process in children:
        _signal_group(process, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    for process in children:
        while _signal_group(process, 0) and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.2)
        _signal_group(process, signal.SIGKILL)
        process.wait()


def container_provenance() -> dict[str, str]:
    """Read baked source metadata without requiring Git inside the image."""
    if "POSTTRAIN_SOURCE_REVISION" not in os.environ:
        if "POSTTRAIN_DOCKER_IMAGE_ID" in os.environ:
            raise ValueError(
                "POSTTRAIN_DOCKER_IMAGE_ID requires POSTTRAIN_SOURCE_REVISION"
            )
        return {}
    revision = os.environ["POSTTRAIN_SOURCE_REVISION"]
    if not revision:
        raise ValueError("POSTTRAIN_SOURCE_REVISION must not be empty")
    receipt = {"posttrain_revision": revision}
    if "POSTTRAIN_DOCKER_IMAGE_ID" in os.environ:
        receipt["container_image_id"] = os.environ["POSTTRAIN_DOCKER_IMAGE_ID"]
    return receipt


#: The learner command's fixed head: interpreter, entry point, and the algorithm, base model and simulator groups.
_LEARNER_HEAD = 5


def learner_recipe(
    plan: LaunchPlan, *, native_baseline: bool = False
) -> tuple[DSRL, Env, list[str]]:
    """The learner of ``plan`` as ``launch`` arguments: ``launch(algorithm, env=env, overrides=overrides)``.

    Exactly the run the launcher's learner command starts (``native_baseline`` picks the baseline
    scoring run of a launch with evaluation); the simulators must already listen at
    ``learner_environment(plan)``'s addresses.
    """
    from isaac_rl_posttraining import Env
    from isaac_rl_posttraining.algorithms import DSRL

    command = plan["native_baseline_learner" if native_baseline else "learner"]
    if command is None:
        raise ValueError("this launch has no native baseline run")
    overrides = [arg for arg in command[_LEARNER_HEAD:] if not arg.startswith("hydra.")]
    env = Env(str(plan["resolved_settings"]["env.env_name"]), simulator="v2d")
    return DSRL(base="gr00t"), env, overrides


def learner_environment(plan: LaunchPlan) -> dict[str, str]:
    """The learner's environment: the launcher's, plus the simulator addresses and the output root."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{REPOSITORY}:{environment.get('PYTHONPATH', '')}"
    environment["V2D_SERVER_URL"] = str(plan["train_url"])
    if plan["eval_url"] is not None:
        environment["V2D_EVAL_SERVER_URL"] = str(plan["eval_url"])
    else:
        environment.pop("V2D_EVAL_SERVER_URL", None)
    environment["ISAAC_RL_POSTTRAINING_OUTPUT_DIR"] = str(
        Path(plan["output_dir"]) / "run_outputs"
    )
    return environment


def library_dry_run(plan: LaunchPlan) -> int:
    """``scripts/train.py``'s own dry run of every learner command: its plan, with nothing loaded or written."""
    environment = {**learner_environment(plan), "ISAAC_RL_POSTTRAINING_DRYRUN": "1"}
    commands = [plan["native_baseline_learner"], plan["learner"]]
    with tempfile.TemporaryDirectory() as scratch:
        for command in (command for command in commands if command is not None):
            kept = [arg for arg in command if not arg.startswith("hydra.run.dir=")]
            dry = [*kept, f"hydra.run.dir={scratch}", "hydra.output_subdir=null"]
            completed = subprocess.run(dry, env=environment, check=False)
            if completed.returncode != 0:
                return completed.returncode
    return 0


def execute(args: argparse.Namespace, plan: LaunchPlan) -> int:
    """Validate external inputs, start only this launch's children, and wait."""
    source_receipt = container_provenance()
    revision = subprocess.check_output(
        ["git", "-C", str(args.v2d_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != V2D_REVISION:
        raise ValueError(f"Expected V2D revision {V2D_REVISION}, got {revision}")
    if not args.motion_file.is_file() or not args.sim_python.is_file():
        raise FileNotFoundError("motion file and simulator Python must exist")
    checkpoint_receipt = validate_checkpoint(args.checkpoint)
    output = Path(plan["output_dir"])
    if output.exists():
        raise FileExistsError(f"Refusing to reuse output directory {output}")
    runtime_dir = Path(plan["runtime_dir"])
    runtime_dir.mkdir(parents=True)
    Path(plan["train_capability_path"]).write_text(
        json.dumps(plan["train_capability"], indent=2, sort_keys=True) + "\n"
    )
    if plan["eval_capability_path"] is not None:
        Path(plan["eval_capability_path"]).write_text(
            json.dumps(plan["eval_capability"], indent=2, sort_keys=True) + "\n"
        )
    (output / "launch_plan.json").write_text(
        json.dumps(
            {**plan, "checkpoint": checkpoint_receipt, **source_receipt},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    environment = learner_environment(plan)
    children: list[subprocess.Popen[str]] = []
    logs: list[TextIO] = []
    try:
        for index, (command, visible_device, receipt) in enumerate(
            zip(
                plan["servers"],
                plan["server_visible_devices"],
                plan["server_receipts"],
                strict=True,
            )
        ):
            log_path = runtime_dir / f"server_{index}.log"
            log: TextIO = log_path.open("w")
            logs.append(log)
            process = start_simulator(
                command,
                simulator_environment(environment, str(visible_device)),
                log,
            )
            children.append(process)
            _wait_for_server(
                process,
                receipt=Path(receipt),
                log_path=log_path,
                timeout_seconds=args.startup_timeout_seconds,
            )
        if plan["native_baseline_learner"] is not None:
            native_baseline = subprocess.run(
                plan["native_baseline_learner"], env=environment, check=False
            )
            if native_baseline.returncode != 0:
                return native_baseline.returncode
        completed = subprocess.run(plan["learner"], env=environment, check=False)
        return completed.returncode
    finally:
        _terminate(children)
        for log in logs:
            log.close()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    plan = resolve_plan(args)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        print(
            "DRY RUN: no simulator, learner, output, or external input was touched",
            flush=True,
        )
        return library_dry_run(plan)
    return execute(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
