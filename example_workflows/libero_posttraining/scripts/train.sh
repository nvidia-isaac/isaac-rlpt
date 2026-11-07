#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Steering launcher for one LIBERO task: SAC (DSRL) or PPO steers a FROZEN
# policy in its flow-latent space. ALGO=sft|iql instead trains the policy
# itself offline from a demonstration dataset and evaluates it in-process.
# The simulator is in-process (no env servers), so the learner is the only process.
#
# Usage:
#   example_workflows/libero_posttraining/scripts/train.sh TASK [extra Hydra overrides...]
#     TASK   LIBERO task name (e.g. KITCHEN_SCENE8_put_both_moka_pots_on_the_stove)
#            or its index in the suite (0-9)
#
# Required environment (not needed for a dry run):
#   POSTTRAIN_BASE_CKPT  GR00T-N1.7-LIBERO checkpoint directory (the per-suite
#                        subfolder, e.g. .../GR00T-N1.7-LIBERO/libero_10). The
#                        checkpoint is self-describing (baked libero_sim
#                        embodiment); no dataset directory is needed.
#                        BASE_MODEL=dreamzero_local: the bundled DreamZero LIBERO SFT
#                        checkpoint dir (config.json + model.safetensors +
#                        experiment_cfg/metadata.json).
#   LIBERO_CONFIG_PATH   LIBERO config dir (example_workflows/libero_posttraining/README.md)
# Optional:
#   BASE_MODEL (default gr00t)  base model: gr00t | pi0 | dreamzero_local
#                               (pi0 uses the DSRL-paper LIBERO recipe: MLP
#                               actor, 32-dim broadcast latent, exec 20 of the
#                               50-step chunk; dreamzero_local uses the RLinf-validated
#                               LIBERO recipe: one 32-dim latent per 16-step
#                               chunk, full-chunk execution)
#   SUITE (default libero_10)   LIBERO benchmark suite
#   SIMULATOR (default libero_10)  simulator config the episode cap is read from
#   GPU (default 0)             CUDA device for the learner AND the renderer
#   POSTTRAIN_BACKBONE_MODEL    optional local/HF override for GR00T's VLM backbone
#   LEARNER_PYTHON (default python3)  the combined policy-stack + LIBERO environment
#                               (example_workflows/libero_posttraining/README.md)
#   NUM_ROLLOUTS (default 100000)  TRAIN_MAXSTEPS  override the episode cap
#   ALGO (default dsrl_sac)  algorithm config group: dsrl_sac | rfs | sac_edit | dsrl_ppo | sft | iql
#   POSTTRAIN_DATASET_DIR  sft and iql: the LeRobot dataset of demonstrations (BASE_MODEL=gr00t
#                   trains the N1.7 adapter gr00t_n17; pi0 the pi0 adapter); NUM_ROLLOUTS counts
#                   updates (default 40000)
#   TRAINER         dsrl | ppo: one-release alias for ALGO=dsrl_sac | ALGO=dsrl_ppo
#   RUNTIME (default native)  native | rlinf
#                   rlinf on ONE GPU: append runtime.rlinf.first_accelerator=0 runtime.rlinf.exclusive_accelerators=false
#   PPO_LEARNER_RANKS (default 1)  learner ranks of dsrl_ppo (placement.learner.ranks); RLInf uses FSDP above 1
#   GPUS (default GPU)  comma-separated CUDA devices visible to the runtime
#   WANDB_NAME    wandb run name (default <algo>_libero_<task>_<timestamp>)
#   LOG_DIR (default ./dsrl_train_logs)   ISAAC_RL_POSTTRAINING_OUTPUT_DIR  checkpoint root
# Secrets are read from the environment only and are never printed:
#   HF_TOKEN, WANDB_API_KEY
#
# Dry run: ISAAC_RL_POSTTRAINING_DRYRUN=1 (or a --dry-run argument) prints this launcher's
# launch.* lines and every override it passes, then the library's own dry run (plan.* lines and
# the resolved config), and exits without starting anything.
set -uo pipefail

TASK=${1:?usage: example_workflows/libero_posttraining/scripts/train.sh TASK [hydra overrides...]}
shift || true

DRYRUN=${ISAAC_RL_POSTTRAINING_DRYRUN:-0}
EXTRA_OVERRIDES=()
for arg in "$@"; do
  case "$arg" in
    --dry-run|--print-config) DRYRUN=1 ;;
    *) EXTRA_OVERRIDES+=("$arg") ;;
  esac
done

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
# shellcheck source=../../_common/launch.sh
source "$REPO_ROOT/example_workflows/_common/launch.sh"
refuse_renamed_knob BACKEND BASE_MODEL

SUITE=${SUITE:-libero_10}
SIMULATOR=${SIMULATOR:-$SUITE}
SIMULATOR_CONFIG="$REPO_ROOT/simulators/libero/configs/simulator/$SIMULATOR.yaml"
if [[ ! -f "$SIMULATOR_CONFIG" ]]; then
  echo "[libero_train] no simulator config at $SIMULATOR_CONFIG" >&2
  exit 2
fi

# The documented override set: the canonical config (train) plus
# these overrides IS the LIBERO variant -- task variants are CLI overrides,
# not config files. The steered task's language instruction is read from the
# environment by the run builder, which syncs it onto the frozen policy.
BASE_MODEL=${BASE_MODEL:-gr00t}
# ALGO selects the algorithm config group; TRAINER=dsrl|ppo stays an alias for one release.
TRAINER=${TRAINER:-}
case "$TRAINER" in
  ""|dsrl) ALGO=${ALGO:-dsrl_sac} ;;
  ppo)     ALGO=${ALGO:-dsrl_ppo} ;;
  *) echo "[libero_train] TRAINER must be dsrl or ppo, got $TRAINER" >&2; exit 2 ;;
esac
ON_POLICY=0
if [[ "$ALGO" = "dsrl_ppo" ]]; then ON_POLICY=1; fi
OFFLINE=0
if [[ "$ALGO" = "sft" ]] || [[ "$ALGO" = "iql" ]]; then OFFLINE=1; fi
RUNTIME=${RUNTIME:-native}
PPO_LEARNER_RANKS=${PPO_LEARNER_RANKS:-1}
if [[ "$ON_POLICY" = "1" ]] && [[ "$BASE_MODEL" != "gr00t" ]]; then
  echo "[libero_train] dsrl_ppo currently supports BASE_MODEL=gr00t" >&2
  exit 2
fi
TRAIN_ENTRYPOINT="scripts/train.py"

# Episode cap. gr00t/pi0: the GR00T reference-evaluation protocol, read from the
# simulator config's gr00t_eval block when the config composes (the single source
# of truth). dreamzero: the RLinf-validated recipe's 240 steps (15 chunks of 16).
CAP_FROM_CONFIG='${simulator.gr00t_eval.max_episode_steps}'
if [[ "$BASE_MODEL" = "dreamzero_local" ]]; then
  MAXSTEPS=${TRAIN_MAXSTEPS:-240}
elif [[ "$OFFLINE" = "1" ]] && [[ "$BASE_MODEL" = "pi0" ]]; then
  MAXSTEPS=${TRAIN_MAXSTEPS:-400}
else
  MAXSTEPS=${TRAIN_MAXSTEPS:-$CAP_FROM_CONFIG}
fi

GPU=${GPU:-0}
GPUS=${GPUS:-$GPU}
if [[ "$OFFLINE" = "1" ]]; then NUM_ROLLOUTS=${NUM_ROLLOUTS:-40000}; else NUM_ROLLOUTS=${NUM_ROLLOUTS:-100000}; fi
# Parallel collection: one rollout worker per LIBERO sim. The native runtime steps
# its workers in turn; RUNTIME=rlinf places them as parallel processes. The UTD
# ratio is held constant by the training loop either way.
NUM_ENVS=${NUM_ENVS:-1}
NAME=${WANDB_NAME:-${ALGO}_libero_${TASK}_$(date +%m%d_%H%M%S)}
LEARNER_PY=${LEARNER_PYTHON:-python3}
LOGD=${LOG_DIR:-$PWD/dsrl_train_logs}

if [[ "$OFFLINE" = "1" ]]; then
  # Offline SFT/IQL: the policy's own action head trains on the demonstrations; evaluation
  # scores it in-process every 5000 updates.
  HYDRA_OVERRIDES=(
    "simulator=$SIMULATOR"
    "env.max_episode_steps=$MAXSTEPS"
    "base_model.ckpt=${POSTTRAIN_BASE_CKPT:-}"
    "base_model.dataset_dir=${POSTTRAIN_DATASET_DIR:-}"
    "base_model.denoising_steps=4"
    "training.num_rollouts=$NUM_ROLLOUTS"
    "training.save_interval=5000"
    "training.checkpoint_retention=best_last_final"
    "eval.eval_freq=5000"
    "eval.eval_num_episodes=50"
    "logging.wandb_prefix=$ALGO"
    "logging.wandb_name=$NAME"
  )
  case "$BASE_MODEL" in
    pi0)
      HYDRA_OVERRIDES+=(
        "base_model=pi0"
        "env.env_name=libero_pi0/$SUITE/$TASK"
        "env.image_keys=[observation/image]"
        "base_model.openpi_config=pi0_libero"
        "++base_model.observation_keys={video.image: observation/image, video.wrist_image: observation/wrist_image, state.q: observation/state}"
        "base_model.action_horizon=50"
        "base_model.action_dim=7"
        "eval.eval_max_steps=$MAXSTEPS"
      )
      ;;
    gr00t)
      HYDRA_OVERRIDES+=(
        "base_model=gr00t_n17"
        "env.env_name=libero/$SUITE/$TASK"
        "env.image_keys=[video.image]"
        "base_model.embodiment_tag=libero_sim"
        "base_model.dataset_adapter=none"
        'base_model.action_horizon=${simulator.gr00t_eval.action_horizon}'
        "eval.eval_max_steps=200"
      )
      ;;
    *)
      echo "[libero_train] ALGO=$ALGO trains BASE_MODEL=gr00t or pi0, got $BASE_MODEL" >&2
      exit 2
      ;;
  esac
else
case "$BASE_MODEL" in
  pi0)
    # The DSRL pi0-LIBERO recipe: a non-chunked MLP actor emits one 32-dim
    # latent per decision, broadcast across pi0's 50-step flow chunk; the
    # first 20 decoded actions execute. Per decision the trajwise loop runs
    # 20 full SAC updates (critic+actor+temperature), batch 256, gamma 0.999
    # (the run builder pre-folds it by the exec length to 0.980 per decision
    # for every actor architecture), 500 warmup episodes of N(0,1) latents
    # (near-base episodes seed the buffer), 400-step episodes. This variant's
    # critic reads frozen DINOv2 features of the 224px scene camera rather
    # than a from-scratch pixel CNN; without that CNN's per-update random-crop
    # regularization a mean-of-ensemble TD target tends to overestimate and
    # collapse, so the TD target uses REDQ pessimism instead (min over 2 of
    # 10 independent critics -- the REDQ configuration at UTD 20).
    HYDRA_OVERRIDES=(
      "base_model=pi0"
      "simulator=$SIMULATOR"
      "env.env_name=libero_pi0/$SUITE/$TASK"
      "env.max_episode_steps=$MAXSTEPS"
      "env.image_keys=[observation/image]"
      "env.sparse_success_reward=false"
      "env.use_init_states=false"
      "base_model.ckpt=${POSTTRAIN_BASE_CKPT:-}"
      "base_model.action_dim=7"
      "dsrl.noise_dim=32"
      "dsrl.noise_action_bound=1.0"
      "dsrl.action_exec_len=20"
      "dsrl.warmup_exploration=normal"
      "training.chunk_size=null"
      "training.train_after_episode=true"
      "training.num_rollouts=$NUM_ROLLOUTS"
      "training.save_interval=200000"
      "training.checkpoint_retention=best_last_final"
      "model.dinov2_model=facebook/dinov2-small"
      "algorithm.learning_starts=10000"
      "algorithm.target_entropy=-16.0"   # a float: the schema types it str | float
      "algorithm.gamma=0.999"
      "algorithm.batch_size=256"
      "algorithm.num_critics=10"
      "algorithm.n_critics_to_sample=2"
      "algorithm.num_critic_updates_per_actor_update=1"
      "algorithm.num_updates_per_train_step=20"
      "algorithm.critic_target_reduction=min"
      "algorithm.critic_reduction=min"
      "algorithm.pooled_critic_features=false"
      "algorithm.bootstrap_at_truncation=true"
      "algorithm.actor_optimizer_lr=1e-4"
      "algorithm.critic_optimizer_lr=3e-4"
      "algorithm.ent_coef_lr=3e-4"
      "buffer.capacity=25000"
      "eval.eval_freq=200000"
      "eval.eval_num_episodes=10"
      "eval.eval_max_steps=$MAXSTEPS"
      "logging.wandb_name=$NAME"
    )
    ;;
  dreamzero_local)
    # The RLinf-validated DreamZero-LIBERO DSRL recipe (5B libero_sim SFT
    # checkpoint, reference reproduction 81.25% -> 90.6% on LIBERO-Spatial):
    # SAC emits one 32-dim latent per decision in the WAN head's padded action
    # space, tiled across the checkpoint's 16-step chunk; all 16 decoded
    # actions execute, so one decision = one chunk and episodes cap at 240
    # steps (15 decisions). Update cadence: the trainer fires train_step once
    # per decision and each call runs num_updates_per_train_step gradient
    # steps, so 25 here is 25 updates PER DECISION (~15x the reference's
    # 1.67/decision) with heavy reuse over a small replay (1500 decisions);
    # the pixel recipe below overrides it to the reference cadence. Q-ensemble
    # of 10 aggregated by MEAN (the validated recipe's choice at this low UTD);
    # if the DINO-feature critic overestimates -- this repo's critic differs
    # from the reference's trainable 64x64 CNN -- switch to the pi0 block's
    # REDQ pessimism (n_critics_to_sample=2, critic_reduction=min). The
    # 160x320 target video is the 5B/LIBERO inference resolution the recipe pins.
    HYDRA_OVERRIDES=(
      "base_model=dreamzero_local"
      "simulator=$SIMULATOR"
      "env.env_name=libero_dreamzero/$SUITE/$TASK"
      "env.max_episode_steps=$MAXSTEPS"
      "env.image_keys=[main_image]"
      "env.sparse_success_reward=false"
      # The reference protocol draws episodes from the benchmark init-state
      # files (RLinf's 496-state suite eval and its training env both do).
      "env.use_init_states=true"
      "base_model.ckpt=${POSTTRAIN_BASE_CKPT:-}"
      "base_model.embodiment_tag=libero_sim"
      "base_model.target_video_height=160"
      "base_model.target_video_width=320"
      "dsrl.noise_dim=32"
      "dsrl.noise_action_bound=1.0"
      "dsrl.action_exec_len=16"
      "dsrl.warmup_exploration=normal"
      "training.chunk_size=null"
      "training.train_after_episode=true"
      "training.num_rollouts=$NUM_ROLLOUTS"
      # Save at the eval cadence: a save interval longer than the run would
      # otherwise leave a killed run with no checkpoint at all.
      "training.save_interval=20000"
      "training.checkpoint_retention=best_last_final"
      "model.dinov2_model=facebook/dinov2-small"
      "algorithm.learning_starts=1200"
      "algorithm.target_entropy=-16.0"   # a float: the schema types it str | float
      "algorithm.gamma=0.999"
      "algorithm.tau=0.005"
      "algorithm.batch_size=512"
      "algorithm.num_critics=10"
      "algorithm.n_critics_to_sample=10"
      "algorithm.num_critic_updates_per_actor_update=1"
      "algorithm.num_updates_per_train_step=25"
      "algorithm.critic_target_reduction=mean"
      "algorithm.critic_reduction=mean"
      "algorithm.bootstrap_at_truncation=true"
      "algorithm.actor_optimizer_lr=1e-4"
      "algorithm.critic_optimizer_lr=3e-4"
      "algorithm.ent_coef_lr=3e-4"
      "buffer.capacity=1500"
      "eval.eval_freq=24000"
      "eval.eval_num_episodes=20"
      "eval.eval_max_steps=$MAXSTEPS"
      "logging.wandb_name=$NAME"
    )
    # RECIPE=pixel swaps the perception and exploration setup for the
    # reference-validated configuration (LIBERO-Spatial 81.25% -> 90.6%):
    # actor and critic each train a small 64x64 conv encoder over the scene
    # camera instead of reading frozen DINO features; the actor objective is
    # max-entropy (alpha * logpi - Q, not the KL anchor); the temperature is
    # softplus-parameterized so exploration decays slowly from alpha = 1; the
    # replay window is effectively unbounded; and training pins one benchmark
    # init state (the reference trains a single scenario and generalizes).
    # Pair it with NUM_ENVS=8 NUM_ROLLOUTS=614400 for the full reference budget
    # (~1,920 episodes; RUNTIME=rlinf steps the eight sims in parallel).
    RECIPE=${RECIPE:-dino}
    if [[ "$RECIPE" = "pixel" ]]; then
      HYDRA_OVERRIDES+=(
        "model.dinov2_model=null"
        "model.image_encoder_type=small"
        "model.small_bottleneck_dim=64"
        "model.featurizer_dim=64"
        "env.critic_image_size=64"
        "env.fixed_init_state_id=0"
        # ++ (add-or-override) because the group nodes carry the dataclass
        # defaults; every field is spelled out so the recipe is self-contained.
        "++policy.mlp.hidden_dims=[128,128,128]"
        "++policy.mlp.activation=relu"
        "++policy.mlp.use_layer_norm=true"
        "++policy.mlp.dropout_rate=0.0"
        "++value_function.mlp.hidden_dims=[128,128,128]"
        "++value_function.mlp.activation=relu"
        "++value_function.mlp.use_layer_norm=true"
        "++value_function.mlp.dropout_rate=0.0"
        "algorithm.ent_coef_parameterization=softplus"
        "algorithm.temperature_loss_type=alpha"
        "algorithm.train_actor_with_kl_divergence=false"
        "algorithm.actor_optimizer_betas=[0.9,0.96]"
        "algorithm.critic_optimizer_betas=[0.9,0.999]"
        "algorithm.actor_optimizer_weight_decay=0.0"
        "algorithm.critic_optimizer_weight_decay=0.0"
        "algorithm.actor_clip_grad_norm=3.5"
        "algorithm.critic_clip_grad_norm=10.0"
        # learning_starts counts ENV steps; the reference gates training at
        # 1,200 decisions = 19,200 env steps (16-step chunks).
        "algorithm.learning_starts=19200"
        # The trainer calls train_step once per decision and each call runs
        # num_updates_per_train_step gradient steps, so this is the UPDATES-
        # PER-DECISION knob: 2 ~= the reference cadence of 1.67 (200 updates
        # per 120 collected decisions). The dino recipe's 25 runs 15x hotter.
        "algorithm.num_updates_per_train_step=2"
        "buffer.capacity=48000"
        "base_model.predict_micro_batch_size=8"
        "eval.eval_freq=61440"
      )
    elif [[ "$RECIPE" != "dino" ]]; then
      echo "[libero_train] unknown RECIPE '$RECIPE' (dino | pixel)" >&2
      exit 2
    fi
    ;;
  gr00t)
    HYDRA_OVERRIDES=(
      "base_model=gr00t"
      "simulator=$SIMULATOR"
      "env.env_name=libero/$SUITE/$TASK"
      "env.max_episode_steps=$MAXSTEPS"
      "env.image_keys=[video.image]"
      "base_model.ckpt=${POSTTRAIN_BASE_CKPT:-}"
      "base_model.embodiment_tag=libero_sim"
      "dsrl.noise_dim=7"
      "dsrl.action_exec_len=8"
      "training.num_rollouts=$NUM_ROLLOUTS"
      "logging.wandb_name=$NAME"
    )
    # The off-policy evaluator's protocol; dsrl_ppo evaluates through ppo.eval_* instead.
    if [[ "$ON_POLICY" != "1" ]]; then
      HYDRA_OVERRIDES+=("eval.eval_freq=5000" "eval.eval_num_episodes=50" "eval.eval_max_steps=200")
    fi
    ;;
  *)
    echo "[libero_train] BASE_MODEL must be gr00t, pi0 or dreamzero_local, got $BASE_MODEL" >&2
    exit 2
    ;;
esac
fi
# The steering actor is chunked (40 steps); SAC-edit runs one MLP over the whole chunk.
if [[ "$BASE_MODEL" = "gr00t" ]] && [[ "$ON_POLICY" != "1" ]] && [[ "$OFFLINE" != "1" ]]; then
  if [[ "$ALGO" = "sac_edit" ]]; then CHUNK=null; else CHUNK=40; fi
  HYDRA_OVERRIDES+=("training.chunk_size=$CHUNK")
fi
# Parallel-collection knobs apply to every base model block (default 1 / sync ==
# the unchanged single-sim path). EXTRA_OVERRIDES may still override them.
HYDRA_OVERRIDES+=(
  "algorithm=$ALGO"
  "runtime=$RUNTIME"
  "training.num_envs=$NUM_ENVS"
)
if [[ -n "${POSTTRAIN_BACKBONE_MODEL:-}" ]]; then
  HYDRA_OVERRIDES+=("base_model.backbone_model_name_or_path=$POSTTRAIN_BACKBONE_MODEL")
fi
if [[ "$ON_POLICY" = "1" ]]; then
  # The rest of the dsrl_ppo recipe (one decision = one whole decoded chunk of the simulator's
  # gr00t_eval.action_horizon, the actor's initialisation) lives in configs/algorithm/dsrl_ppo.yaml.
  HYDRA_OVERRIDES+=("placement.learner.ranks=$PPO_LEARNER_RANKS")
fi
if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
  HYDRA_OVERRIDES+=("${EXTRA_OVERRIDES[@]}")
fi

if [[ "$DRYRUN" = "1" ]]; then
  echo "[libero_train] DRY RUN: the launch plan, then the library's plan; nothing will be launched"
  echo "launch.suite=$SUITE"
  echo "launch.task=$TASK"
  echo "launch.gpus=$GPUS"
  echo "launch.log_dir=$LOGD"
  echo "launch.output_dir=${ISAAC_RL_POSTTRAINING_OUTPUT_DIR:-$HOME/isaac_rl_outputs}"
  echo "launch.env.learner.MUJOCO_GL=${MUJOCO_GL:-egl}"
  echo "launch.env.learner.LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-<unset>}"
  echo "launch.paths.POSTTRAIN_BASE_CKPT=${POSTTRAIN_BASE_CKPT:-<unset>}"
  echo "launch.secrets.HF_TOKEN=$([[ -n "${HF_TOKEN:-}" ]] && echo set || echo unset)"
  echo "launch.secrets.WANDB_API_KEY=$([[ -n "${WANDB_API_KEY:-}" ]] && echo set || echo unset)"
  echo "launch.cmd.learner=$TRAIN_ENTRYPOINT"
  print_overrides "${HYDRA_OVERRIDES[@]}"
  # The plan counts the GPUs the run will see, not every GPU of the machine.
  CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH=$REPO_ROOT${GR00T_REPO:+:$GR00T_REPO}:${PYTHONPATH:-} \
    library_dry_run "$LEARNER_PY" "$REPO_ROOT" "${HYDRA_OVERRIDES[@]}"
  exit $?
fi

# ---------------------------------------------------------------------------
# Real launch (single process: the LIBERO simulator steps in the learner)
# ---------------------------------------------------------------------------
: "${POSTTRAIN_BASE_CKPT:?set POSTTRAIN_BASE_CKPT}"
: "${LIBERO_CONFIG_PATH:?set LIBERO_CONFIG_PATH (example_workflows/libero_posttraining/README.md)}"
if [[ "$OFFLINE" = "1" ]]; then : "${POSTTRAIN_DATASET_DIR:?set POSTTRAIN_DATASET_DIR (a LeRobot dataset of demonstrations)}"; fi
mkdir -p "$LOGD"

if [[ "$ON_POLICY" = "1" ]]; then BUDGET="algorithm=$ALGO (budget: ppo.iterations)"; else BUDGET="rollouts=$NUM_ROLLOUTS"; fi
echo "[libero_train] $NAME task=$TASK $BUDGET maxsteps=$MAXSTEPS gpu=$GPU"
( export CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
  export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
  export PYTHONPATH=$REPO_ROOT${GR00T_REPO:+:$GR00T_REPO}:${PYTHONPATH:-}
  exec "$LEARNER_PY" -u "$REPO_ROOT/$TRAIN_ENTRYPOINT" \
    "${HYDRA_OVERRIDES[@]}" \
) 2>&1 | tee "$LOGD/$NAME.train.log"
