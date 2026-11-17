#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Smoke check: under two minutes, no GPU, no Isaac Lab. Fails on the first broken step.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
WORKFLOW=example_workflows/isaaclab_gr00t_posttraining
TASK=Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0

# 1. The launch recipes and the launch logic import without Isaac Lab, and the stack cube has a recipe.
python -c "from example_workflows.isaaclab_gr00t_posttraining.isaac_rlpt import _launch, cli_args; print(cli_args.format_task_table())" > "$WORK/tasks.txt"
grep -q "$TASK" "$WORK/tasks.txt"
python "$WORKFLOW/scripts/install.py" --help > /dev/null

# 2. The PPO fine-tuning plan resolves on one GPU (the learner and one rollout worker of 16 scenes), from
#    the command line and from Python (run.py). The dry run stops before the checkpoint, Isaac Sim or
#    GR00T load, so the checkpoint path is a placeholder.
ISAAC_RL_POSTTRAINING_DRYRUN=1 python scripts/train.py algorithm=ppo base_model.ckpt=/path/to/RLinf-Gr00t-SFT-Stack-cube \
    runtime=native placement.learner.ranks=1 placement.envs.workers=1 training.num_envs=16 \
    hydra.run.dir="$WORK/dryrun_ppo" > "$WORK/plan_ppo.txt" 2>&1
grep -q "plan.algorithm=ppo" "$WORK/plan_ppo.txt"
grep -q "DRY RUN" "$WORK/plan_ppo.txt"
python -m example_workflows.isaaclab_gr00t_posttraining.run /path/to/RLinf-Gr00t-SFT-Stack-cube --dry-run > "$WORK/plan_ppo_python.txt" 2>&1
grep -q "plan.placement.learner=1 rank(s)" "$WORK/plan_ppo_python.txt"

# 3. The steering plan resolves: the overrides the launcher emits for --algorithm dsrl_sac on the stack cube,
#    with the mock base model standing in for the frozen GR00T policy.
ISAAC_RL_POSTTRAINING_DRYRUN=1 python scripts/train.py algorithm=dsrl_sac simulator=isaaclab_franka_stack base_model=mock base_model.action_dim=7 \
    "env.env_name=isaaclab/$TASK" env.max_episode_steps=448 \
    dsrl.noise_dim=7 dsrl.action_exec_len=16 training.chunk_size=16 \
    hydra.run.dir="$WORK/dryrun_dsrl" > "$WORK/plan_dsrl.txt" 2>&1
grep -q "plan.algorithm=dsrl_sac" "$WORK/plan_dsrl.txt"
grep -q "DRY RUN" "$WORK/plan_dsrl.txt"

if grep -h "is deprecated" "$WORK"/*.txt; then echo "smoke: a recipe passes a deprecated key" >&2; exit 1; fi

# 4. Every isaac_rlpt command line resolves to the config pinned in tests/golden_config.json.
python -m example_workflows._common.golden isaaclab_gr00t_posttraining

# 5. The reported numbers are well-formed and the README table matches them.
python -m example_workflows._common.results_table "$WORKFLOW/results/results.json" --check "$WORKFLOW/README.md"

echo "smoke check passed"
