#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native GR00T evaluation on the LIBERO benchmark suites.

Runs a GR00T checkpoint (e.g. ``nvidia/GR00T-N1.7-LIBERO``) in NATIVE mode
(no steering latents) over the tasks of a LIBERO suite and reports the
per-task success rate. The pipeline is the repository's own steering stack
end to end -- :func:`~simulators.libero.policies.gr00t.make_libero_gr00t_env`
(the checkpoint's observation/action conventions),
:class:`~isaac_rl_posttraining.policies.gr00t.Gr00tBaseModelPolicy` (the
frozen policy), and the rollout worker and evaluator training runs use, with
the frozen policy's own decode -- so its numbers are directly comparable to
DSRL steering runs on the same stack.

Protocol (the reference evaluation of the Isaac-GR00T repository, examples/
LIBERO): two 256x256 cameras, the 8-dim end-effector state, a 720-step
episode cap, and 8 actions executed per inference. Episodes walk the
benchmark's fixed initial states in order (episode ``k`` uses init state
``k``), so runs are reproducible without seed bookkeeping.

Example (single GPU)::

    MUJOCO_GL=egl LIBERO_CONFIG_PATH=... CUDA_VISIBLE_DEVICES=1 \
      python example_workflows/libero_posttraining/scripts/eval.py \
        --checkpoint /path/to/GR00T-N1.7-LIBERO/libero_10 \
        --suite libero_10 --episodes 20 \
        --output-dir logs/libero_eval

Results are appended to ``results.jsonl`` per task (crash-safe) and
summarized into ``results.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

if TYPE_CHECKING:
    from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
    from isaac_rl_posttraining.runtimes import PolicyState

__all__ = [
    "TaskResult",
    "evaluate_suite",
    "format_markdown",
    "pick_dsrl_target",
]

#: The DSRL post-training sweet spot: tasks the base policy neither fails
#: outright nor already saturates (success rate within these bounds).
DSRL_TARGET_BAND = (0.30, 0.60)

#: Below this success rate a task counts as "~0": steering has no successes
#: to bootstrap from, so it is a poor first DSRL target.
NEAR_ZERO_SR = 0.05


@dataclass
class TaskResult:
    """Success-rate record of one task's evaluation."""

    task_id: int
    task_name: str
    description: str
    episodes: int
    successes: int

    @property
    def success_rate(self) -> float:
        """Fraction of successful episodes (0.0 when no episodes ran)."""
        return self.successes / self.episodes if self.episodes else 0.0


def pick_dsrl_target(
    results: Sequence[TaskResult],
    band: tuple[float, float] = DSRL_TARGET_BAND,
    near_zero: float = NEAR_ZERO_SR,
) -> tuple[TaskResult | None, str]:
    """Select the DSRL post-training target task from a suite evaluation.

    The target is the WORST task whose success rate is not ~0 but inside
    ``band`` -- weak enough that steering has headroom, strong enough that
    successes exist to learn from. When the overall worst task is ~0 (or
    otherwise outside the band), it is called out in the note alongside the
    pick. With no task inside the band, the weakest non-~0 task is picked
    and the deviation is noted.

    Args:
        results: Per-task evaluation records.
        band: Inclusive success-rate band of good DSRL targets.
        near_zero: Threshold under which a task counts as ~0.

    Returns:
        ``(pick, note)``; ``pick`` is ``None`` only when every task is ~0.
    """
    if not results:
        return None, "no results"
    by_sr = sorted(results, key=lambda r: (r.success_rate, r.task_id))
    worst = by_sr[0]
    in_band = [r for r in by_sr if band[0] <= r.success_rate <= band[1]]
    if in_band:
        pick = in_band[0]
        if pick.task_name == worst.task_name:
            note = (
                f"the suite's worst task, inside the {band[0]:.0%}-{band[1]:.0%} band"
            )
        else:
            note = (
                f"worst task inside the {band[0]:.0%}-{band[1]:.0%} band; the overall "
                f"worst ({worst.task_name}, {worst.success_rate:.0%}) is outside it"
            )
        return pick, note
    viable = [r for r in by_sr if r.success_rate > near_zero]
    if viable:
        pick = viable[0]
        note = (
            f"no task inside the {band[0]:.0%}-{band[1]:.0%} band; picked the weakest "
            f"non-zero task instead (overall worst: {worst.task_name}, "
            f"{worst.success_rate:.0%})"
        )
        return pick, note
    return None, f"every task is ~0 (<= {near_zero:.0%}); no viable DSRL target"


def format_markdown(
    results: Sequence[TaskResult],
    checkpoint: str,
    suite: str,
    protocol: str,
    band: tuple[float, float] = DSRL_TARGET_BAND,
) -> str:
    """Render the per-task table and the DSRL-target pick as Markdown."""
    lines = [
        f"# {suite} -- native GR00T evaluation",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Protocol: {protocol}",
        "",
        "| Task | Description | Success rate |",
        "|---|---|---|",
    ]
    for record in sorted(results, key=lambda r: r.task_id):
        lines.append(
            f"| {record.task_id} | {record.description} | "
            f"{record.successes}/{record.episodes} ({record.success_rate:.0%}) |"
        )
    total_eps = sum(r.episodes for r in results)
    total_succ = sum(r.successes for r in results)
    mean_sr = sum(r.success_rate for r in results) / len(results) if results else 0.0
    lines += [
        "",
        f"Overall: {total_succ}/{total_eps} episodes; mean per-task SR {mean_sr:.1%}.",
        "",
    ]
    pick, note = pick_dsrl_target(results, band=band)
    if pick is not None:
        lines += [
            (
                f"**DSRL target: task {pick.task_id} -- {pick.task_name} "
                f"({pick.success_rate:.0%}).** {note}."
            ),
            "",
        ]
    else:
        lines += [f"**No DSRL target picked.** {note}.", ""]
    return "\n".join(lines)


class _NativeChunkPolicy:
    """The frozen policy's own decode as a rollout-worker policy (no steering latents)."""

    def __init__(self, base_model: Any, *, action_exec_len: int) -> None:
        self.base_model = base_model
        self.action_exec_len = int(action_exec_len)
        self._policy_version = 0

    def action_chunk(self, observation: dict[str, Any], *, mode: str) -> ActionChunk:
        from isaac_rl_posttraining.rollouts.action_chunk import ActionChunk
        from isaac_rl_posttraining.rollouts.steering_step import SteeringStep

        if mode != "eval":
            raise ValueError("the native evaluation policy accepts eval decisions only")
        batched = SteeringStep.vectorize_obs(observation)
        actions = np.asarray(
            self.base_model.infer(batched, None)["actions"], dtype=np.float32
        )
        return ActionChunk.from_arrays(
            learner_action=None,
            environment_actions=actions,
            execution_length=self.action_exec_len,
            source="native",
            policy_version=self._policy_version,
        )

    def set_policy_state(self, state: PolicyState) -> int:
        self._policy_version = state.version
        return self._policy_version


def evaluate_task(
    base_model: Any,
    env: Any,
    *,
    episodes: int,
    action_exec_len: int,
    max_decision_steps: int,
    record_video: bool = False,
    video_dir: str | None = None,
) -> int:
    """Run ``episodes`` native episodes on one task; returns the successes.

    The rollout worker and the evaluator are the ones training runs use, so
    the score means exactly what a steering run's evaluation means. The
    worker owns ``env`` and closes it.
    """
    from isaac_rl_posttraining.envs.simulator_entry import (
        EnvironmentSpec,
        SimulatorEntry,
    )
    from isaac_rl_posttraining.runtimes import (
        GymWorkerSpec,
        NativeRuntime,
        NativeRuntimeConfig,
        PolicyState,
    )
    from isaac_rl_posttraining.training import RolloutEvaluator

    class _PrebuiltTask(SimulatorEntry):
        """Hands the worker the task environment this script already built."""

        name = "libero_eval"

        def make_environment(self, spec, phase, context):
            del spec, phase, context
            return env

    def make_policy(context):
        del context
        return _NativeChunkPolicy(base_model, action_exec_len=action_exec_len)

    runtime = NativeRuntime(NativeRuntimeConfig(device="cpu"))
    try:
        group = runtime.open_rollout_group(
            GymWorkerSpec(
                entry=_PrebuiltTask(),
                environment=EnvironmentSpec(
                    env_name=f"libero_eval/{env.unwrapped.task_name}",
                    num_envs=1,
                    max_episode_steps=max_decision_steps * action_exec_len,
                    seed=None,
                ),
                phase="eval",
                policy=make_policy,
            ),
            num_rollout_workers=1,
            num_envs=1,
            name="libero_eval",
        )
        evaluator = RolloutEvaluator(
            group=group,
            num_episodes=episodes,
            max_decisions=max_decision_steps,
            record_video=record_video,
            video_dir=video_dir,
            image_keys=env.steering_key_mapping["image"],
        )
        metrics = evaluator.evaluate(PolicyState(version=1, tensors={}))
    finally:
        runtime.close()
    return round(metrics["success_rate"] * metrics["num_eval_episodes"])


def evaluate_suite(args: argparse.Namespace) -> list[TaskResult]:
    """Evaluate every requested task of the suite; returns the records."""
    import torch

    from isaac_rl_posttraining.policies.gr00t import Gr00tBaseModelPolicy
    from isaac_rl_posttraining.utils.torch_utils import require_cuda_kernels
    from simulators.libero.policies.gr00t import make_libero_gr00t_env

    device = "cuda" if torch.cuda.is_available() else "cpu"
    require_cuda_kernels(device)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.jsonl")

    print(f"[eval_libero] loading base model from {args.checkpoint} (device={device})")
    base_model = Gr00tBaseModelPolicy.from_checkpoint(
        base_ckpt=args.checkpoint,
        embodiment_tag=args.embodiment_tag,
        device=device,
    )
    print(
        f"[eval_libero] base model ready: action_horizon={base_model.action_horizon} "
        f"action_dim={base_model.action_dim} latent_horizon={base_model.latent_horizon} "
        f"latent_dim={base_model.latent_dim}"
    )
    max_decision_steps = math.ceil(args.max_episode_steps / args.n_action_steps) + 8

    results: list[TaskResult] = []
    task_ids = args.tasks if args.tasks else list(range(args.num_tasks))
    for task_id in task_ids:
        env = make_libero_gr00t_env(
            suite=args.suite,
            task_id=task_id,
            max_episode_steps=args.max_episode_steps,
            sequential_init_states=True,
            seed=args.seed,
        )
        inner = env.unwrapped
        print(
            f"[eval_libero] task {task_id}: {inner.task_name} -- "
            f"{inner.language_instruction!r}"
        )
        start = time.time()
        successes = evaluate_task(
            base_model,
            env,
            episodes=args.episodes,
            action_exec_len=args.n_action_steps,
            max_decision_steps=max_decision_steps,
            record_video=args.record_video,
            video_dir=os.path.join(args.output_dir, "videos")
            if args.record_video
            else None,
        )
        record = TaskResult(
            task_id=task_id,
            task_name=inner.task_name,
            description=inner.language_instruction,
            episodes=args.episodes,
            successes=successes,
        )
        results.append(record)
        with open(results_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")
        print(
            f"[eval_libero] task {task_id} done in {time.time() - start:.0f}s: "
            f"{record.successes}/{record.episodes} ({record.success_rate:.0%})"
        )

    protocol = (
        f"{args.episodes} episodes/task, native mode, sequential benchmark init "
        f"states, {args.max_episode_steps}-step cap, {args.n_action_steps} actions "
        f"per inference, seed {args.seed}"
    )
    markdown = format_markdown(results, args.checkpoint, args.suite, protocol)
    md_path = os.path.join(args.output_dir, "results.md")
    with open(md_path, "w") as f:
        f.write(markdown)
    print(markdown)
    print(f"[eval_libero] wrote {results_path} and {md_path}")
    return results


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--checkpoint", required=True, help="GR00T checkpoint directory (HF layout)"
    )
    parser.add_argument(
        "--embodiment-tag",
        default="libero_sim",
        help="Embodiment tag baked into the checkpoint (LIBERO checkpoints: libero_sim)",
    )
    parser.add_argument("--suite", default="libero_10", help="LIBERO benchmark suite")
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="*",
        default=None,
        help="Task ids to evaluate (default: all --num-tasks tasks)",
    )
    parser.add_argument("--num-tasks", type=int, default=10, help="Tasks in the suite")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per task")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=720,
        help="Per-episode simulator-step cap (reference protocol: 720)",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=8,
        help="Actions executed per inference call (reference protocol: 8)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Torch / env seed")
    parser.add_argument(
        "--record-video", action="store_true", help="Also record evaluation videos"
    )
    parser.add_argument(
        "--output-dir",
        default="logs/libero_eval",
        help="Directory for results.jsonl / results.md",
    )
    return parser


def main() -> None:
    """Console entry point."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    evaluate_suite(build_parser().parse_args())


if __name__ == "__main__":
    main()
