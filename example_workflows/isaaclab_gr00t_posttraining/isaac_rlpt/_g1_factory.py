# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G1 Factory tasks under ``isaaclab train/play --rl_library isaac_rlpt``.

The nine ``LM*`` tasks run through the G1 Factory workflow's launchers
(``example_workflows/g1_arena_posttraining/scripts``), which boot the IsaacLab-Arena
simulator servers next to the learner. This module maps the Isaac Lab flags onto
those launchers' arguments and environment, runs them, and returns their exit status.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path

#: The G1 Factory tasks (``example_workflows/g1_arena_posttraining/scripts/task_prompts.sh``).
TASKS = (
    "LMBoxLift",
    "LMBoxLiftFloor",
    "LMBoxTableToShelfPnP",
    "LMDrillLift",
    "LMDrillLiftObs",
    "LMDrillPnP",
    "LMPickDrillFromHolder",
    "LMPushButton",
    "LMPushShelfForward",
)
CHECKPOINT = "nvidia/G1-Factory-GR00T-1.7"
STEERING = ("dsrl_sac", "rfs", "sac_edit")
_PPO_EVAL_SCENES = 8  # finetune_ppo.sh's default NUM_ENVS


def format_task_table() -> str:
    """The G1 Factory tasks, for ``--list_tasks``."""
    return "\n".join(
        [
            "G1 Factory (example_workflows/g1_arena_posttraining):",
            f"  checkpoint:    {CHECKPOINT}",
            "  algorithms:    dsrl_sac rfs sac_edit dsrl_ppo ppo (play: dsrl_sac rfs sac_edit ppo)",
            *(f"  {task}" for task in TASKS),
        ]
    )


def _package_root(name: str) -> Path | None:
    """The checkout holding an installed package (found without importing it)."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parents[1]


def resolve_checkpoint(model_path: str, dry_run: bool) -> Path:
    """``--model_path`` as a local directory; a Hugging Face repo id downloads to ``checkpoints/<name>``."""
    local = Path(model_path).expanduser()
    if not local.is_dir():
        if model_path.count("/") != 1 or model_path.startswith((".", "/", "~")):
            raise SystemExit(
                f"--model_path {model_path} is neither a directory nor a Hugging Face repo id"
            )
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError

        local = Path("checkpoints", model_path.split("/")[1])
        print(f"isaac_rlpt: downloading {model_path} to {local.absolute()}", flush=True)
        try:
            # A dry run plans from the checkpoint's metadata only; the real run fetches the weights.
            snapshot_download(
                model_path,
                local_dir=local,
                allow_patterns=["*.json"] if dry_run else None,
            )
        except HfHubHTTPError as exc:
            raise SystemExit(
                f"cannot download {model_path} from Hugging Face ({exc.__class__.__name__}); "
                "check the name, and export HF_TOKEN for an account with access"
            ) from exc
    if not (local / "statistics.json").is_file():
        raise SystemExit(
            f"{local} has no statistics.json: not a G1 GR00T checkpoint (try {CHECKPOINT})"
        )
    return local.resolve()


def _reject(args: argparse.Namespace, names: tuple[str, ...], why: str) -> None:
    given = [
        f"--{name}" for name in names if getattr(args, name, None) not in (None, False)
    ]
    if given:
        raise SystemExit(f"{', '.join(given)}: {why}")


def _train_command(
    args: argparse.Namespace, scripts: Path
) -> tuple[list[str], dict[str, str], list[str]]:
    env, overrides = {}, []
    if args.algorithm == "ppo":
        script = scripts / "finetune_ppo.sh"
    else:
        script = scripts / "train.sh"
        env["ALGO"] = args.algorithm
    if args.algorithm in STEERING:
        _reject(args, ("iterations",), "applies to --algorithm ppo and dsrl_ppo")
        _reject(
            args, ("eval_num_envs",), "the G1 steering evaluation runs on one scene"
        )
        if args.num_steps is not None:
            env["NUM_ROLLOUTS"] = str(args.num_steps)
        if args.eval_freq is not None:
            overrides.append(f"eval.eval_freq={args.eval_freq}")
        if args.save_interval is not None:
            overrides.append(f"training.save_interval={args.save_interval}")
        if args.eval_episodes is not None:
            overrides.append(f"eval.eval_num_episodes={args.eval_episodes}")
        if args.skip_first_eval:
            overrides.append("eval.eval_on_first_step=false")
    else:
        _reject(
            args,
            (
                "num_steps",
                "eval_freq",
                "save_interval",
                "eval_num_envs",
                "skip_first_eval",
            ),
            "apply to the steering algorithms; use --iterations and ppo.* overrides",
        )
        _reject(
            args,
            ("eval_episodes",),
            "set ppo.eval_episodes_per_environment=<n> instead",
        )
        if args.iterations is not None:
            overrides.append(f"ppo.iterations={args.iterations}")
    if args.num_envs is not None:
        env["NUM_ENVS"] = str(args.num_envs)
    return [str(script), args.task], env, overrides


def _play_command(
    args: argparse.Namespace, scripts: Path
) -> tuple[list[str], dict[str, str], list[str]]:
    _reject(args, ("iterations",), "applies to training")
    checkpoint = (
        str(Path(args.checkpoint).expanduser().resolve()) if args.checkpoint else None
    )
    if args.algorithm == "ppo":
        # finetune_ppo.sh's evaluation-only mode: the policy snapshot (or the unchanged GR00T) without a learner.
        overrides = ["ppo.eval_only=true", "placement.learner.ranks=0"]
        if args.eval_episodes is not None:
            scenes = args.num_envs or _PPO_EVAL_SCENES
            overrides.append(
                f"ppo.eval_episodes_per_environment={-(-args.eval_episodes // scenes)}"
            )
        if checkpoint:
            overrides.append(f"ppo.resume_from={checkpoint}")
        env = {"NUM_ENVS": str(args.num_envs)} if args.num_envs is not None else {}
        return [str(scripts / "finetune_ppo.sh"), args.task], env, overrides
    if args.algorithm not in STEERING:
        raise SystemExit(
            f"play --algorithm {args.algorithm} has no G1 launcher; score the base policy with dsrl_sac or ppo"
        )
    episodes = str(args.eval_episodes or 20)
    cmd = [
        str(scripts / "eval.sh"),
        args.task,
        "policy" if checkpoint else "base",
        episodes,
    ]
    overrides = []
    if checkpoint:
        cmd.append(checkpoint)
        if args.algorithm != "dsrl_sac":
            overrides.append(f"algorithm={args.algorithm}")
    env = {"EVAL_NUM_ENVS": str(args.num_envs)} if args.num_envs is not None else {}
    return cmd, env, overrides


def command(
    args: argparse.Namespace, extra: list[str], mode: str, output_root: Path
) -> tuple[list[str], dict[str, str]]:
    """The launcher command and the environment variables it adds, for ``mode`` train or play."""
    if args.algorithm == "rfs_pc":
        raise SystemExit(
            "rfs_pc on G1 runs through example_workflows/g1_arena_posttraining/scripts/zoo_recipes.sh"
        )
    _reject(
        args,
        ("denoising_steps", "learner_gpus", "rollout_gpus"),
        "the G1 recipes fix these; set the launcher's LEARNER_GPUS, NUM_SERVERS or Hydra overrides",
    )
    unknown = [arg for arg in extra if arg.startswith("-")]
    if unknown:
        raise SystemExit(
            f"{' '.join(unknown)}: not an isaac_rlpt flag; extra arguments must be Hydra overrides"
        )
    viz = getattr(args, "visualizer", None)
    if viz and not (os.environ.get("DISPLAY") or args.dry_run):
        raise SystemExit("--viz needs a display: $DISPLAY is not set")

    repo = _package_root("isaac_rl_posttraining")
    if repo is None:
        raise SystemExit("isaac_rl_posttraining is not installed in this environment")
    scripts = repo / "example_workflows" / "g1_arena_posttraining" / "scripts"
    build = _train_command if mode == "train" else _play_command
    cmd, env, overrides = build(args, scripts)

    env.update(
        SIM_PYTHON=sys.executable,
        LEARNER_PYTHON=sys.executable,
        POSTTRAIN_BASE_CKPT=str(resolve_checkpoint(args.model_path, args.dry_run)),
        ISAAC_RL_POSTTRAINING_OUTPUT_DIR=str(output_root),
        LOG_DIR=os.environ.get("LOG_DIR", str(output_root / "simulator_logs")),
    )
    arena, g1_factory = _package_root("isaaclab_arena"), _package_root("g1_factory")
    if arena is None or g1_factory is None:
        if not args.dry_run:
            raise SystemExit(
                "G1 Factory tasks need isaaclab_arena and g1_factory installed in this environment "
                "(docs/pages/workflows/g1_arena.md, 'Run from Isaac Lab's command line')"
            )
    else:
        env.update(ISAACLAB_ARENA_PATH=str(arena), PYTHONPATH=str(g1_factory))
    if viz:
        env["SIM_VIZ"] = ",".join(viz)  # the launcher picks the viewer simulator
    if args.run_name:
        env["WANDB_NAME"] = args.run_name
    if args.runtime:
        env["RUNTIME"] = args.runtime
    if args.dry_run:
        env["ISAAC_RL_POSTTRAINING_DRYRUN"] = "1"

    overrides.append(f"training.seed={args.seed}")
    if args.wandb_offline:
        overrides.append("logging.wandb_offline=true")
    return ["bash", *cmd, *overrides, *extra], env


def run(
    args: argparse.Namespace, extra: list[str], mode: str, output_root: Path
) -> None:
    """Run the G1 launcher; a failed launch exits with its status."""
    cmd, env = command(args, extra, mode, output_root)
    shown = [f"{key}={value}" for key, value in env.items()]
    print(
        f"isaac_rlpt: G1 Factory task {args.task}: {shlex.join(shown + cmd)}",
        flush=True,
    )
    status = subprocess.run(cmd, env={**os.environ, **env}, check=False).returncode
    if status:
        raise SystemExit(status)
