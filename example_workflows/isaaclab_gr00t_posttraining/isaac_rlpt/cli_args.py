# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line arguments and launch recipes for the ``isaac_rlpt`` Isaac Lab integration.

``isaac_rlpt`` post-trains a supervised-fine-tuned GR00T policy on an Isaac Lab
task with the algorithms of ``isaac_rl_posttraining``. The steering algorithms
keep the policy FROZEN and learn around it from live rollouts: DSRL
(``dsrl_sac``, steers the flow noise), RFS (flow noise plus a bounded action
residual) and SAC-edit (bounded edits of the proposed action chunk).
``dsrl_ppo`` trains the DSRL steering actor with PPO instead of SAC (the policy
stays frozen). ``ppo`` instead fine-tunes the checkpoint's action head with
RLinf's flow-SDE PPO recipe. Every algorithm's rollout workers host their own
Isaac Lab simulation (one child process per worker), so the launching process
starts no simulator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

STEERING_ALGORITHMS = ("dsrl_sac", "rfs", "rfs_pc", "sac_edit")
PPO_ALGORITHM = "ppo"
DSRL_PPO_ALGORITHM = "dsrl_ppo"
#: The on-policy algorithms: PPO iterations, ppo.* evaluation and the PPO placement flags.
ON_POLICY_ALGORITHMS = (DSRL_PPO_ALGORITHM, PPO_ALGORITHM)
ALGORITHM_CHOICES = (*STEERING_ALGORITHMS, *ON_POLICY_ALGORITHMS)
RUNTIME_CHOICES = (
    "native",
    "rlinf",
)

#: Defaults of the steering trainer's cadence flags (environment steps).
_STEERING_TRAIN_DEFAULTS = {
    "num_steps": 40_000,
    "eval_freq": 5_000,
    "save_interval": 5_000,
    "eval_num_envs": 0,
}
_PPO_FLAGS = ("runtime", "learner_gpus", "rollout_gpus", "iterations")
#: --eval_episodes and --num_envs default to None so a launcher can tell them from the recipe's default.
DEFAULT_EVAL_EPISODES = 20


@dataclass(frozen=True)
class TaskRecipe:
    """Per-task facts the launcher needs beyond the Isaac Lab task id.

    Attributes:
        simulator: ``simulator=<name>`` config group of ``isaac_rl_posttraining``.
        instruction: Language prompt the policy is conditioned on.
        data_config: GR00T data config (``module:Class``) describing the
            checkpoint's video/state/action keys and transforms.
        flow_width: Width of the flow-matching noise per step (the checkpoint's
            padded action width; the steering latent lives here).
        chunk_size: Steering latents per flow step (the checkpoint's action
            horizon, so every decoded step is steerable).
        exec_len: Decoded steps executed per decision (receding horizon).
        max_episode_steps: Episode cap in environment steps.
        embodiment_tag: Embodiment tag the checkpoint's statistics are stored
            under (``experiment_cfg/metadata.json``).
        checkpoint: Public SFT checkpoint (Hugging Face id) the recipe is built for.
        extra_overrides: Task-specific overrides of the steering algorithms.
    """

    simulator: str
    instruction: str
    data_config: str
    flow_width: int
    chunk_size: int
    exec_len: int
    max_episode_steps: int
    embodiment_tag: str = "new_embodiment"
    checkpoint: str = ""
    extra_overrides: tuple[str, ...] = field(default_factory=tuple)


_TROCAR = TaskRecipe(
    simulator="isaaclab_trocar",
    instruction="assemble trocar from tray",
    data_config=(
        "isaaclab_tasks.manager_based.manipulation.assemble_trocar.config.gr00t_config:IsaacLabDataConfig"
    ),
    flow_width=32,
    chunk_size=16,
    exec_len=8,
    max_episode_steps=1000,
    checkpoint="nvidia/GR00T-N1.5-3B_Assemble_Trocar",
)

# Franka cube stacking: RLinf's public SFT checkpoint (LIBERO-style 7-dim
# end-effector action). The RLinf protocol executes the whole 16-step chunk per
# policy call and caps episodes at 448 control steps; it reports SFT 0.654 and
# SFT+PPO 0.897 success (RLinf docs, examples/embodied/isaaclab).
_FRANKA_STACK = TaskRecipe(
    simulator="isaaclab_franka_stack",
    instruction="Stack the red block on the blue block, then stack the green block on the red block.",
    data_config="simulators.isaaclab.gr00t_data_configs:IsaacLabFrankaDataConfig",
    flow_width=32,
    chunk_size=16,
    exec_len=16,
    max_episode_steps=448,
    embodiment_tag="isaaclab_franka",
    checkpoint="RLinf/RLinf-Gr00t-SFT-Stack-cube",
    # RFS adds its residual in raw action units. The library default (0.01) is
    # tuned for joint targets in radians (about 0.2% of a G1 joint's span); the
    # stack-cube action is a pose delta in [-0.1, 0.1], so the bound is scaled
    # to 1% of that span. At 0.01 the residual alone dropped success to 8%.
    extra_overrides=("rfs.residual_action_bound=0.002",),
)

#: Isaac Lab task id -> launch recipe. Add a task here (and a simulator config in
#: ``simulators/isaaclab/configs/simulator/``) to post-train on it.
# The Eval variant is the "box" scene the published checkpoint was fine-tuned on
# (its Rheo name is install_trocar_sim_box); it carries the matching prompt.
_TROCAR_BOX = TaskRecipe(
    **{**_TROCAR.__dict__, "instruction": "install trocar from box"}
)

TASK_RECIPES: dict[str, TaskRecipe] = {
    "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0": _FRANKA_STACK,
    "Isaac-Assemble-Trocar-G129-Dex3-v0": _TROCAR,
    "Isaac-Assemble-Trocar-G129-Dex3-Eval-v0": _TROCAR_BOX,
}


def format_task_table() -> str:
    """The launch recipes as text, for ``--list_tasks``."""
    lines = []
    for task_id, recipe in TASK_RECIPES.items():
        lines.append(task_id)
        lines.append(f"  checkpoint:    {recipe.checkpoint}")
        lines.append(f"  simulator:     {recipe.simulator}")
        lines.append(f"  instruction:   {recipe.instruction}")
        lines.append(f"  data_config:   {recipe.data_config}")
        lines.append(f"  embodiment:    {recipe.embodiment_tag}")
        lines.append(
            f"  episode steps: {recipe.max_episode_steps} (action horizon {recipe.chunk_size}, "
            f"executed {recipe.exec_len}, flow width {recipe.flow_width})"
        )
        if recipe.extra_overrides:
            lines.append(f"  steering overrides: {' '.join(recipe.extra_overrides)}")
    return "\n".join(lines)


def add_isaac_rlpt_args(parser: argparse.ArgumentParser) -> None:
    """Add the arguments shared by ``train`` and ``play``."""
    group = parser.add_argument_group(
        "isaac_rlpt", description="Arguments for isaac_rl_posttraining."
    )
    group.add_argument(
        "--task",
        type=str,
        default=None,
        help="Isaac Lab task id with a launch recipe (see --list_tasks).",
    )
    group.add_argument(
        "--list_tasks",
        action="store_true",
        default=False,
        help="Print the tasks with a launch recipe and exit.",
    )
    group.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="SFT'd GR00T checkpoint directory (HuggingFace layout).",
    )
    group.add_argument(
        "--algorithm",
        type=str,
        default="dsrl_sac",
        choices=ALGORITHM_CHOICES,
        help="dsrl_sac | rfs | rfs_pc | sac_edit learn around the frozen policy; "
        "dsrl_ppo trains the DSRL steering actor with PPO; "
        "ppo fine-tunes its action head (RLinf's recipe).",
    )
    group.add_argument("--seed", type=int, default=0, help="Training seed.")
    group.add_argument(
        "--eval_episodes",
        type=int,
        default=None,
        help=f"Episodes per evaluation (default {DEFAULT_EVAL_EPISODES}; ppo, dsrl_ppo: rounded up to a multiple of the "
        "scenes, which evaluate in parallel).",
    )
    group.add_argument(
        "--denoising_steps",
        type=int,
        default=None,
        help="Flow denoising steps (default: the checkpoint's; ppo: 4).",
    )
    group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name (log directory and W&B name).",
    )
    group.add_argument(
        "--wandb_offline", action="store_true", default=False, help="Log W&B offline."
    )
    group.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Compose and print the plan, launch nothing.",
    )
    ppo = parser.add_argument_group(
        "isaac_rlpt-ppo",
        description="Placement of --algorithm ppo and dsrl_ppo. Under --runtime rlinf every learner rank and every rollout "
        "worker owns a GPU (ranks first); the default --runtime native keeps one learner and one rollout worker "
        "on a single GPU.",
    )
    ppo.add_argument(
        "--runtime",
        type=str,
        default=None,
        choices=RUNTIME_CHOICES,
        help="Execution runtime (default native).",
    )
    ppo.add_argument(
        "--learner_gpus",
        type=int,
        default=None,
        help="Learner ranks (default 1; train only).",
    )
    ppo.add_argument(
        "--rollout_gpus",
        type=int,
        default=None,
        help="Rollout workers, one Isaac Lab simulation each (default 1).",
    )
    ppo.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="PPO iterations (default: the recipe's; train only).",
    )


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Add training-only arguments."""
    group = parser.add_argument_group("isaac_rlpt-train")
    group.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Environment steps to train for (steering algorithms; default 40000).",
    )
    group.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Scenes collected in parallel (default 1): clones of the one simulation (steering) or all scenes over the "
        "rollout workers (ppo; a multiple of --rollout_gpus).",
    )
    group.add_argument(
        "--eval_freq",
        type=int,
        default=None,
        help="Evaluation period in environment steps (steering; default 5000).",
    )
    group.add_argument(
        "--save_interval",
        type=int,
        default=None,
        help="Checkpoint period in environment steps (steering; default 5000).",
    )
    group.add_argument(
        "--eval_num_envs",
        type=int,
        default=None,
        help="Scenes evaluated in parallel (steering; default: as many as --num_envs); the episode count rounds up.",
    )
    group.add_argument(
        "--skip_first_eval",
        action="store_true",
        default=False,
        help="Do not evaluate the untrained actor before training (steering; saves one evaluation when the base is known).",
    )


def add_play_args(parser: argparse.ArgumentParser) -> None:
    """Add evaluation-only arguments."""
    group = parser.add_argument_group("isaac_rlpt-play")
    group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Trained checkpoint to evaluate: the actor directory (steering) or an iteration_<n> directory (ppo); "
        "omit to evaluate the base policy.",
    )
    group.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Scenes evaluated in parallel (default 1; ppo; the steering evaluation uses one).",
    )


def require_task_and_model(args: argparse.Namespace) -> None:
    """``--task`` and ``--model_path`` are required for everything but ``--list_tasks``."""
    if not args.task:
        raise SystemExit(
            "--task is required (see --list_tasks for the tasks with a launch recipe)"
        )
    if not args.model_path:
        raise SystemExit(
            "--model_path is required: the SFT'd GR00T checkpoint directory"
        )


def resolve_recipe(task: str) -> TaskRecipe:
    """Return the launch recipe for ``task`` or raise with the supported list."""
    try:
        return TASK_RECIPES[task]
    except KeyError as exc:
        raise SystemExit(
            f"isaac_rlpt has no launch recipe for task {task!r}. Supported: {sorted(TASK_RECIPES)}. "
            "Add a TaskRecipe in cli_args.py and a simulator config under simulators/isaaclab/."
        ) from exc


def algorithm_overrides(algorithm: str, recipe: TaskRecipe) -> list[str]:
    """Hydra overrides selecting a steering algorithm and its action geometry."""
    steer = [f"training.chunk_size={recipe.chunk_size}"]
    if algorithm == "dsrl_sac":
        return ["algorithm=dsrl_sac", *steer]
    if algorithm == "rfs":
        return ["algorithm=rfs", *steer]
    if algorithm == "rfs_pc":
        return [
            "algorithm=rfs",
            *steer,
            "rfs.proposal_conditioned_residual=true",
            "rfs.reference_action_conditioning=true",
        ]
    if algorithm == "sac_edit":
        return ["algorithm=sac_edit"]
    raise SystemExit(
        f"unknown steering algorithm {algorithm!r}; choose from {STEERING_ALGORITHMS}"
    )


def task_overrides(
    args: argparse.Namespace, recipe: TaskRecipe, run_name: str
) -> list[str]:
    """Hydra overrides naming the task, the checkpoint, and the run; shared by every algorithm and mode."""
    # The recipes are GR00T N1.5 checkpoints: steering builds the frozen GR00T base_model
    # (which reads the generation from the checkpoint), fine-tuning must name it.
    base_model = "gr00t_n15" if args.algorithm == PPO_ALGORITHM else "gr00t"
    overrides = [
        f"simulator={recipe.simulator}",
        f"env.env_name=isaaclab/{args.task}",
        f"env.max_episode_steps={recipe.max_episode_steps}",
        f"base_model={base_model}",
        f"base_model.ckpt={args.model_path}",
        f"base_model.embodiment_tag={recipe.embodiment_tag}",
        f"base_model.data_config={recipe.data_config}",
        "base_model.dataset_dir=null",
        "base_model.dataset_adapter=none",
        f"base_model.task_instruction='{recipe.instruction}'",
        f"training.seed={args.seed}",
        "logging.wandb_project=isaac_rlpt",
        f"logging.wandb_name={run_name}",
    ]
    if args.denoising_steps is not None:
        overrides.append(f"base_model.denoising_steps={args.denoising_steps}")
    if args.wandb_offline:
        overrides.append("logging.wandb_offline=true")
    return overrides


def _eval_episodes(args: argparse.Namespace) -> int:
    return int(args.eval_episodes or DEFAULT_EVAL_EPISODES)


def steering_overrides(
    args: argparse.Namespace, recipe: TaskRecipe, eval_num_envs: int = 1
) -> list[str]:
    """Hydra overrides of the steering loop: latent geometry, evaluation protocol, algorithm.

    The evaluator gives every scene the same number of episodes, so ``--eval_episodes``
    rounds up to a multiple of the scenes it evaluates on (25 episodes over 8 scenes
    becomes 32), as the README documents.
    """
    episodes = -(-_eval_episodes(args) // eval_num_envs) * eval_num_envs
    return [
        f"dsrl.noise_dim={recipe.flow_width}",
        f"dsrl.action_exec_len={recipe.exec_len}",
        f"eval.eval_num_episodes={episodes}",
        "eval.action_selection=sample",
        *recipe.extra_overrides,
        *algorithm_overrides(args.algorithm, recipe),
    ]


@dataclass(frozen=True)
class PPOPlacement:
    """Where ``--algorithm ppo`` puts its learner ranks, rollout workers, and scenes."""

    runtime: str
    learner_ranks: int
    rollout_workers: int
    num_envs: int


def _learner_ranks(args: argparse.Namespace, *, mode: str) -> int:
    """Learner ranks for ``mode``: none when playing a checkpoint, ``--learner_gpus`` when training."""
    if mode == "play":
        if args.learner_gpus is not None:
            raise SystemExit(
                "play evaluates the checkpoint's policy snapshot without a learner; drop --learner_gpus"
            )
        if args.iterations is not None:
            raise SystemExit("--iterations applies to training")
        # DSRL-PPO restores the checkpoint into its learner; flow-SDE PPO scores the snapshot alone.
        return 1 if args.algorithm == DSRL_PPO_ALGORITHM else 0
    learner_ranks = 1 if args.learner_gpus is None else int(args.learner_gpus)
    if learner_ranks < 1:
        raise SystemExit("training needs --learner_gpus >= 1")
    return learner_ranks


def ppo_placement(args: argparse.Namespace, *, mode: str) -> PPOPlacement:
    """Resolve the PPO placement flags and reject combinations the runtimes cannot host."""
    runtime = "native" if args.runtime is None else args.runtime
    rollout_workers = 1 if args.rollout_gpus is None else int(args.rollout_gpus)
    learner_ranks = _learner_ranks(args, mode=mode)
    if rollout_workers < 1:
        raise SystemExit("--rollout_gpus must be >= 1")
    if runtime != "rlinf" and (learner_ranks > 1 or rollout_workers > 1):
        raise SystemExit(
            f"--runtime {runtime} runs one learner rank and one rollout worker on one GPU; "
            "use --runtime rlinf for --learner_gpus / --rollout_gpus above 1"
        )
    num_envs = int(args.num_envs or 1)
    if num_envs < 1 or num_envs % rollout_workers:
        raise SystemExit(
            f"--num_envs {num_envs} must be a positive multiple of --rollout_gpus {rollout_workers} "
            "(every rollout worker hosts one simulation with the same number of scenes)"
        )
    return PPOPlacement(runtime, learner_ranks, rollout_workers, num_envs)


def ppo_overrides(args: argparse.Namespace, placement: PPOPlacement) -> list[str]:
    """Hydra overrides of ``algorithm=ppo`` or ``dsrl_ppo``: placement and evaluation size."""
    episodes_per_scene = -(-_eval_episodes(args) // placement.num_envs)
    overrides = [
        f"algorithm={args.algorithm}",
        f"runtime={placement.runtime}",
        f"placement.learner.ranks={placement.learner_ranks}",
        f"placement.envs.workers={placement.rollout_workers}",
        f"training.num_envs={placement.num_envs}",
        f"ppo.eval_episodes_per_environment={episodes_per_scene}",
    ]
    if args.iterations is not None:
        overrides.append(f"ppo.iterations={args.iterations}")
    return overrides


def dsrl_ppo_geometry(args: argparse.Namespace, recipe: TaskRecipe) -> list[str]:
    """DSRL-PPO's steering geometry: DSRL's latent width, chunk and executed steps (none for ``ppo``)."""
    if args.algorithm != DSRL_PPO_ALGORITHM:
        return []
    return [
        f"dsrl.noise_dim={recipe.flow_width}",
        f"dsrl.action_exec_len={recipe.exec_len}",
        f"training.chunk_size={recipe.chunk_size}",
    ]


def _reject_ppo_flags(args: argparse.Namespace) -> None:
    given = [f"--{name}" for name in _PPO_FLAGS if getattr(args, name) is not None]
    if given:
        raise SystemExit(f"{', '.join(given)} apply to --algorithm ppo and dsrl_ppo")


def _reject_steering_train_flags(args: argparse.Namespace) -> None:
    given = [
        f"--{name}"
        for name in _STEERING_TRAIN_DEFAULTS
        if getattr(args, name) is not None
    ]
    if given:
        raise SystemExit(
            f"{', '.join(given)} set the steering trainer's cadence in environment steps; with "
            "--algorithm ppo or dsrl_ppo use --iterations and the ppo.eval_every / ppo.checkpoint_every overrides"
        )
    if args.skip_first_eval:
        raise SystemExit("--skip_first_eval applies to the steering algorithms")


def train_overrides(
    args: argparse.Namespace, recipe: TaskRecipe, run_name: str
) -> list[str]:
    """Full override list for a training run."""
    if args.algorithm in ON_POLICY_ALGORITHMS:
        _reject_steering_train_flags(args)
        return [
            *task_overrides(args, recipe, run_name),
            *dsrl_ppo_geometry(args, recipe),
            *ppo_overrides(args, ppo_placement(args, mode="train")),
        ]
    _reject_ppo_flags(args)
    cadence = {
        name: default if getattr(args, name) is None else int(getattr(args, name))
        for name, default in _STEERING_TRAIN_DEFAULTS.items()
    }
    num_envs = int(args.num_envs or 1)
    eval_envs = cadence["eval_num_envs"] or num_envs
    overrides = [
        *task_overrides(args, recipe, run_name),
        *steering_overrides(args, recipe, eval_envs),
        f"training.num_rollouts={cadence['num_steps']}",
        f"eval.eval_freq={cadence['eval_freq']}",
        f"training.save_interval={cadence['save_interval']}",
        "training.checkpoint_retention=best_last_final",
        f"eval.eval_on_first_step={'false' if args.skip_first_eval else 'true'}",
    ]
    if eval_envs > 1:
        overrides.append(f"eval.eval_num_envs={eval_envs}")
    # N scenes cloned in one simulation and stepped together (Isaac Lab's native
    # vectorization); a batch plays whole episodes, so the budget is collected per episode.
    overrides.append(f"training.num_envs={num_envs}")
    if num_envs > 1:
        overrides.append("training.rollout_count_by=episode")
    return overrides


def play_overrides(
    args: argparse.Namespace, recipe: TaskRecipe, run_name: str
) -> list[str]:
    """Full override list for an evaluation-only run (base policy or a trained checkpoint)."""
    if args.algorithm in ON_POLICY_ALGORITHMS:
        if args.algorithm == DSRL_PPO_ALGORITHM and not args.checkpoint:
            raise SystemExit(
                "play --algorithm dsrl_ppo scores a trained steering actor; pass --checkpoint "
                "(score the base policy with --algorithm dsrl_sac or ppo)"
            )
        overrides = [
            *task_overrides(args, recipe, run_name),
            *dsrl_ppo_geometry(args, recipe),
            *ppo_overrides(args, ppo_placement(args, mode="play")),
            "ppo.eval_only=true",
        ]
        if args.checkpoint:
            # The learner-less evaluation loads the checkpoint's policy snapshot (absolute path).
            overrides.append(
                f"ppo.resume_from={Path(args.checkpoint).expanduser().resolve()}"
            )
        return overrides
    _reject_ppo_flags(args)
    if int(args.num_envs or 1) != 1:
        raise SystemExit("the steering evaluation runs on one scene; drop --num_envs")
    overrides = [
        *task_overrides(args, recipe, run_name),
        *steering_overrides(args, recipe),
        "training.num_envs=1",
        "training.num_rollouts=0",
        "eval.eval_on_first_step=true",
    ]
    if args.checkpoint:
        overrides.append(
            f"training.load_dir={Path(args.checkpoint).expanduser().resolve()}"
        )
    else:
        overrides.append("eval.eval_base=true")
    return overrides
