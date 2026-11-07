# Command-line reference

There are two ways to start a run, and this page covers both:

1. **The library entry point**, `scripts/train.py`, takes Hydra `key=value` overrides ([Hydra overrides](#hydra-overrides)).
2. **The Isaac Lab integration**, `uv run isaaclab train --rl_library isaac_rlpt` from the [Isaac Lab workflow](workflows/isaaclab_gr00t.md)'s folder (`./isaaclab.sh train` in its container), takes flags, turns them into overrides for the same entry point, and forwards anything it does not recognise unchanged ([Isaac Lab flags](#isaac-lab-flags)).

The workflow launchers (`example_workflows/*/scripts/*.sh`) wrap these; each launcher's header lists the environment variables it reads, and the [workflow pages](workflows/index.md) document them.

## Hydra overrides

```bash
ARENA_SERVER_URL=tcp://127.0.0.1:7300 ARENA_EVAL_SERVER_URL=tcp://127.0.0.1:7301 \
uv run python scripts/train.py algorithm=dsrl_sac simulator=g1_arena base_model=gr00t runtime=native \
    base_model.ckpt=/path/to/checkpoint training.num_rollouts=40000
```

G1 Factory needs the two addresses; a simulator that runs inside the training process does not.
An unknown key is refused at launch with a list of the keys that group does have.

### Config groups

| Override | Values | Meaning |
|---|---|---|
| `algorithm=` | `dsrl_sac` (default), `dsrl_ppo`, `rfs`, `sac_edit`, `ppo`, `iql`, `sft` (`iql` and `sft` untested) | the method |
| `simulator=` | `g1_arena` (default), `libero_10`, `libero_spatial`, `libero_object`, `libero_goal`, `isaaclab_franka_stack`, `isaaclab_trocar`, `v2d` | the simulator's settings (task list, episode caps, cameras); `env.env_name=<provider>/<task>` picks the task, and both must refer to the same simulator; the provider is `arena` for `g1_arena`, `libero`, `libero_pi0` or `libero_dreamzero` for the LIBERO groups, `isaaclab` for the Isaac Lab groups and `v2d` for `v2d` |
| `base_model=` | `gr00t` (default), `gr00t_n15`, `gr00t_n17`, `pi0`, `dreamzero_local`, `dreamzero_server`, `mock` | the frozen policy |
| `runtime=` | `native` (default), `rlinf`; [Runtimes](../runtimes.md) lists them all | where the rollout workers run |
| `schedule=` | `sync` (default), `overlapped`, `async` | when collection and updates happen; see [Schedule](#schedule) |

`algorithm=ppo` is composed last and pins its own simulator, runtime and base model, so the checkpoint is the only required field.

### Base model

| Key | Applies to | Meaning |
|---|---|---|
| `base_model.ckpt` | all | the checkpoint directory. Required |
| `base_model.embodiment_tag` | GR00T | the tag the checkpoint's statistics are stored under. Required for a baked checkpoint and for fine-tuning |
| `base_model.dataset_dir` | GR00T | a single-task LeRobot dataset directory supplying modality and statistics |
| `base_model.multitask_dirs`, `base_model.task_dir` | GR00T | a multi-task base: the parent directory of the task datasets, and the steered task's own directory |
| `base_model.task_instruction` | all | an explicit language override, used with `base_model.dataset_dir=null` for a self-describing checkpoint |
| `base_model.data_config` | GR00T N1.5 | the data config, a `DATA_CONFIG_MAP` key or `module:Class`. Required there |
| `base_model.dataset_adapter` | GR00T | `g1_wbc` or `none` |
| `base_model.denoising_steps` | GR00T N1.5, fine-tuning | flow denoising steps |
| `base_model.action_horizon` | fine-tuning | executed steps per decision |
| `base_model.action_dim` | pi0, mock | the action width where the policy does not expose one |
| `base_model.backbone_model_name_or_path` | GR00T | a local or Hugging Face override for the checkpoint's vision-language backbone |
| `base_model.openpi_config`, `base_model.observation_keys` | pi0 fine-tuning | the openpi training config and the observation-key mapping |
| `base_model.server_url` | served base models | the policy server address |

Exactly one source of normalization statistics must be given: `dataset_dir`, or `multitask_dirs` with `task_dir`, or the checkpoint's own statistics with `dataset_dir=null` and `task_instruction`.

### Environment and task

| Key | Default | Meaning |
|---|---|---|
| `env.env_name` | `arena/drill_lift` | `<simulator segment>/<task>`; the segment selects the simulator entry |
| `env.max_episode_steps` | from the simulator group | client-side episode cap |
| `env.sparse_success_reward` | `true` | replace the environment reward with a terminal-outcome signal |
| `env.success_reward` / `env.failure_reward` / `env.step_reward` | 0.0 / -1.0 / 0.0 | the outcome scheme |
| `env.critic_image_size` | `null` | set it, for example 64, to attach `critic_image` for a pixel critic |
| `env.image_keys` | `null` | cameras the learner embeds; `null` uses the simulator entry's mapped image keys |
| `env.use_init_states` | `true` | LIBERO: draw the initial scene from the benchmark's fixed init-state files |
| `env.fixed_init_state_id` | `null` | pin every reset to one benchmark init state |

### Training loop

| Key | Default | Meaning |
|---|---|---|
| `training.num_rollouts` | 40000 | the budget, in environment steps by default |
| `training.rollout_count_by` | `step` | `episode` counts the budget in episodes instead |
| `training.num_envs` | 1 | environments the rollout workers host |
| `training.chunk_size` | 40 | steering latents per flow step; `null` broadcasts one latent |
| `training.seed` | 0 | training seed |
| `training.save_interval` | 5000 | checkpoint period in environment steps |
| `training.checkpoint_retention` | `all` | `best_last_final` keeps the best-evaluated, newest and final checkpoints |
| `training.load_dir` | `null` | a checkpoint directory to load |
| `training.train_after_episode` | `true` | one training burst per completed episode |
| `training.max_updates_per_burst` | `null` | cap on updates per training phase |
| `training.policy_decode_batch_size` | 16 | training-only decode micro-batch cap |

### Steering geometry

| Key | Default | Applies to | Meaning |
|---|---|---|---|
| `dsrl.noise_dim` | 35 | `dsrl_sac`, `dsrl_ppo`, `rfs` | per-step steering-latent width |
| `dsrl.action_exec_len` | 8 | `dsrl_sac`, `dsrl_ppo`, `rfs` | actions executed per decision |
| `dsrl.noise_action_bound` | 1.0 | `dsrl_sac`, `dsrl_ppo`, `rfs` | half-width of the latent box |
| `dsrl.warmup_exploration` | `uniform` | same | latents before learning starts: `uniform` or `normal` |
| `rfs.residual_action_bound` | 0.01 | `rfs` only | residual half-width in raw action units |
| `rfs.proposal_conditioned_residual` | `false` | `rfs` only | condition the residual on the policy's proposal |
| `rfs.reference_action_conditioning` | `false` | `rfs` only | feed the reference action to the actor |
| `direct_action.actor_edit_scale` | 0.02 | `sac_edit` only | edit bound in normalized action units |
| `direct_action.action_horizon`, `direct_action.action_dim` | `null` | `sac_edit` only | `null` takes both from the frozen policy |
| `direct_action.reference_action_transform` | `identity` | `sac_edit` only | transform applied to the reference |

The `algorithm.*` hyperparameters differ per algorithm; each algorithm page lists them, and the defaults live in `isaac_rl_posttraining/configs/algorithm/`.

### Replay buffer

Applies to the off-policy algorithms (`dsrl_sac`, `rfs`, `sac_edit`).
The PPO paths train on fresh episodes and ignore `buffer.demonstrations` and `buffer.demonstration_fraction`.

| Key | Default | Meaning |
|---|---|---|
| `buffer.capacity` | 200000 | transitions kept; the oldest complete episodes are evicted whole |
| `buffer.reward_aggregation` | `last` | the decision's reward is its outcome step; `sum` adds every executed step |
| `buffer.success_fraction` | `null` | 0.5 draws half of every batch from successful episodes |
| `buffer.demonstrations`, `buffer.demonstration_fraction` | `null` | a saved buffer directory mixed into every batch at the given share |
| `buffer.save_buffer_on_exit` | `false` | write the buffer to disk when the run ends or is killed |
| `buffer.save_buffer_every` | 0 | also write it every N environment steps |

### Evaluation

| Key | Default | Meaning |
|---|---|---|
| `eval.eval_freq` | 5000 | evaluation period in environment steps |
| `eval.eval_num_episodes` | 50 | episodes per evaluation |
| `eval.eval_num_envs` | 1 | above 1 evaluates in parallel; `eval_num_episodes` must be an exact multiple |
| `eval.eval_base` | `false` | `true` evaluates the native frozen policy, with no steering latents |
| `eval.action_selection` | `sample` | `sample` or `mean`; the steering policy is stochastic, so `sample` evaluates it as it acts during training (was `eval.eval_deterministic`) |
| `eval.eval_on_first_step` | `true` | evaluate before training starts |
| `eval.eval_record_video` | `true` | record evaluation video |
| `eval.reset_seed` | `null` | seed evaluation resets per environment and episode |
| `eval.eval_max_steps` | 1000 | hard per-episode cap so a wedged simulator cannot hang an evaluation |

An evaluation-only off-policy run is `training.num_rollouts=0 eval.eval_on_first_step=true`, plus `eval.eval_base=true` for the frozen policy.
The on-policy equivalent is `ppo.eval_only=true`.

### Placement

| Key | Default | Meaning |
|---|---|---|
| `placement.envs.workers` | `null` | rollout workers; `null` uses the simulator entry's own layout (was `runtime.num_rollout_workers`) |
| `placement.envs.gpus`, `placement.learner.gpus` | `null` | a GPU count or a list such as `[0,1]` / `['node1:0']`; `null` lets the planner choose |
| `placement.policy.mode` | `auto` | `local` (a copy in every rollout worker), `served` (policy servers) or `auto` (was `runtime.rlinf.policy_serving`) |
| `placement.policy.replicas`, `max_batch`, `max_wait_ms` | `null` | policy-server replicas and batching; `null` batching is the runtime's (native: unbounded, 5 ms; rlinf: 32, 5 ms) (were `runtime.rlinf.num_policy_servers`, `policy_batch_size`, `policy_batch_wait_seconds`) |
| `placement.parts.base.mode` | `auto` | `served` serves the frozen base model alone (DSRL, RFS, SAC-edit) while every rollout worker keeps its own actor; `placement.parts.base.replicas`, `gpus`, `max_batch`, `max_wait_ms` size its servers; `auto` follows `placement.policy` |
| `placement.gpus_per_node` | `null` | GPUs per node the planner assumes; `null` uses the GPUs visible here |
| `placement.learner.ranks` | 1 | data-parallel learner processes; 0 evaluates an on-policy checkpoint without a learner |
| `runtime.native.processes` | `auto` | `auto` runs Ray actors when the placement asks for several processes (needs the `ray` extra); `single` keeps one process; `ray` always uses actors |
| `learner.device` | `auto` | device of a one-rank on-policy learner: `auto`, `cpu` or `gpu` |
| `runtime.rlinf.exclusive_accelerators` | `true` | set it `false` to let RLInf learner ranks share a card with the rollout workers or policy servers |
| `runtime.<native\|rlinf>.serve_evaluation_unbatched` | `true` | a policy server never batches evaluation requests of different workers, so batching cannot change a score (was `runtime.rlinf.deterministic_evaluation`) |
| `runtime.native.concurrent_collection` | `auto` | rollout workers collect on threads so a served policy batches their requests (auto: when the placement serves the policy and a group has several workers) |
| `runtime.rlinf.policy_request_timeout_seconds` | 600 | how long a rollout worker waits for its policy server |
| `dsrl.batched_actor` | `false` | a served steering policy runs the actor once per batch instead of once per session |

### Schedule

| Key | Default | Meaning |
|---|---|---|
| `schedule.kind` | `sync` | `sync` collects a round, trains, publishes; `overlapped` collects the next round while the learner trains (policy lag 1); `async` collects continuously. `schedule=<kind>` selects the same option |
| `schedule.max_policy_lag` | `null` | drop collected data whose policy lag (weight publishes since it was collected) exceeds this (`async`); `null` is the schedule's own bound (PPO: 1) |
| `schedule.publish_every` | `null` | `async`: publish every N learner updates; `null` publishes once per training burst |

The older names in parentheses in the Placement and Evaluation tables are still accepted, with a one-line deprecation note.

### Learner networks

| Key | Default | Meaning |
|---|---|---|
| `model.dinov2_model` | `facebook/dinov2-base` | the frozen actor and critic image encoder |
| `model.image_encoder_type` | `null` | `small` or `impala` selects a trainable pixel encoder instead |
| `model.impala_use_smaller`, `model.impala_bounded_head` | `false` | IMPALA encoder variants |
| `model.featurizer_dim` | `null` | per-key featurizer width; `null` means 384 |

### Logging

| Key | Default |
|---|---|
| `logging.wandb_project` | `isaac_rl_posttraining` |
| `logging.wandb_entity` | `null` |
| `logging.wandb_name` | `dsrl` |
| `logging.wandb_offline` | `false` |
| `logging.log_level` | `INFO` |

## Isaac Lab flags

`train` trains and `play` evaluates; they share most flags.

```bash
uv run isaaclab train --rl_library isaac_rlpt --task <TASK> --model_path <CKPT> [flags] [hydra.overrides=...]
uv run isaaclab play  --rl_library isaac_rlpt --task <TASK> --model_path <CKPT> [flags] [hydra.overrides=...]
```

The algorithm decides which flags are legal:

- **Steering**: `dsrl_sac` (default), `rfs`, `rfs_pc`, `sac_edit`. An off-policy learner steers the frozen policy, with a cadence in environment steps.
- **On-policy**: `ppo` fine-tunes the checkpoint's action head; `dsrl_ppo` trains the steering actor with PPO. Both run in PPO iterations and take the placement flags.

For every algorithm the rollout workers start their own simulations; the launching process starts none.
A steering-only flag with an on-policy algorithm, or a placement flag with a steering algorithm, is rejected with a message naming the flags.

### Common flags

| Flag | Default | Applies to | Meaning |
|---|---|---|---|
| `--task` | none, required | all | Isaac Lab task id with a launch recipe |
| `--model_path` | none, required | all | the supervised-fine-tuned GR00T checkpoint directory, in Hugging Face layout |
| `--list_tasks` | off | all | print the tasks with a launch recipe and exit; the only command that does not need `--task` and `--model_path` |
| `--algorithm` | `dsrl_sac` | all | `dsrl_sac`, `rfs`, `rfs_pc`, `sac_edit`, `dsrl_ppo`, `ppo` |
| `--seed` | 0 | all | training seed |
| `--eval_episodes` | 20 | all | episodes per evaluation, **rounded up to a multiple of the scenes that evaluate in parallel** (see below) |
| `--denoising_steps` | the checkpoint's; 4 under `ppo` | all | flow denoising steps |
| `--run_name` | generated | all | run name, used for the log directory and the Weights & Biases name |
| `--wandb_offline` | off | all | log Weights & Biases offline |
| `--dry_run` | off | all | compose and print the plan, launch nothing |

`--algorithm rfs_pc` is a launcher mode rather than a config group: it composes `algorithm=rfs` plus `rfs.proposal_conditioned_residual=true` and `rfs.reference_action_conditioning=true`.

### Training flags

| Flag | Default | Applies to | Meaning |
|---|---|---|---|
| `--num_steps` | 40000 | **steering only** | environment steps to train for |
| `--eval_freq` | 5000 | **steering only** | evaluation period in environment steps |
| `--save_interval` | 5000 | **steering only** | checkpoint period in environment steps |
| `--eval_num_envs` | as many as `--num_envs` | **steering only** | scenes evaluated in parallel. It also sets what `--eval_episodes` rounds up to, and no video is recorded above one scene |
| `--skip_first_eval` | off | **steering only** | do not evaluate the untrained actor before training, which saves one evaluation when the base is already known |
| `--num_envs` | 1 | all | scenes collected in parallel |

With `ppo` or `dsrl_ppo`, use `--iterations` and the `ppo.eval_every` / `ppo.checkpoint_every` overrides instead of the steering-only flags.

`--num_envs` means something different per family:

- **steering**: clones of the one simulation, stepped together.
- **`ppo`, `dsrl_ppo`**: all scenes over the rollout workers, and it must be a positive multiple of `--rollout_gpus`.

Training with steering also sets `training.checkpoint_retention=best_last_final`, so a run keeps the best-evaluated, last and final checkpoints.

### Rounding of `--eval_episodes`

Every scene that evaluates in parallel plays the same number of episodes, so the launcher rounds `--eval_episodes` **up** to a multiple of that scene count, and the rounded number is what lands in `evaluation_summary.json`.
For a steering `train`, the scene count is `--eval_num_envs` when given, and `--num_envs` otherwise:

| Flags | `eval.eval_num_episodes` composed |
|---|---|
| `--eval_episodes 25 --num_envs 1` | 25 |
| `--eval_episodes 25 --num_envs 8` | **32** |
| `--eval_episodes 25 --num_envs 8 --eval_num_envs 4` | **28** |
| `--eval_episodes 50 --num_envs 8 --eval_num_envs 1` | 50 |

For `ppo` and `dsrl_ppo` the same happens through `ppo.eval_episodes_per_environment`: `--eval_episodes 25 --num_envs 16` gives 2 episodes per scene, so 32 in total.
A steering `play` evaluates on one scene, so nothing is rounded there.

To evaluate exactly the number you asked for, make `--eval_episodes` a multiple of the parallel scene count, or pin `--eval_num_envs 1`.

### Evaluation flags

| Flag | Default | Applies to | Meaning |
|---|---|---|---|
| `--checkpoint` | none | all | the trained checkpoint to evaluate: the actor directory for steering, an `iteration_<n>` directory for `ppo`. Omit it to evaluate the base policy; `dsrl_ppo` requires it |
| `--num_envs` | 1 | `ppo`, `dsrl_ppo` | scenes evaluated in parallel. The steering evaluation runs on one scene and rejects any other value |

A steering `play` without `--checkpoint` composes `eval.eval_base=true`, the native frozen policy with no steering latents.
A `ppo` `play` without `--checkpoint` scores the base checkpoint.

### On-policy placement

All four are **`ppo` and `dsrl_ppo` only**; a steering algorithm refuses them even at their default value.

| Flag | Default | Meaning |
|---|---|---|
| `--runtime` | `native` | `native` or `rlinf` |
| `--learner_gpus` | 1 | learner ranks. Training only; `play` refuses it, because it evaluates a snapshot without a learner |
| `--rollout_gpus` | 1 | rollout workers, one Isaac Lab simulation each |
| `--iterations` | the recipe's (1000 for `ppo`, 100 for `dsrl_ppo`) | PPO iterations. Training only |

The launcher enforces before anything starts:

- `--runtime native` runs one learner rank and one rollout worker on one GPU. Use `--runtime rlinf` for `--learner_gpus` or `--rollout_gpus` above 1.
- Under `--runtime rlinf`, every learner rank and every rollout worker owns a GPU, ranks first.
- `--num_envs` must be a positive multiple of `--rollout_gpus`.
- Training needs `--learner_gpus` of at least 1.

### Isaac Lab's own launcher flags

`--headless`, `--device` and the rest of Isaac Lab's app arguments are accepted and unused: the launching process starts no simulator.
Every rollout worker starts its own headless Isaac Sim with cameras on.

### Other arguments

An argument the parser does not know is forwarded as a Hydra override, so every key under [Hydra overrides](#hydra-overrides) is reachable:

```bash
uv run isaaclab train --rl_library isaac_rlpt --task $TASK --model_path $CKPT \
    --algorithm dsrl_sac --num_envs 8 --headless \
    dsrl.noise_action_bound=2.0 algorithm.batch_size=128
```

## Environment variables

These are read by the entry point and the launchers, not composed from the config.

| Variable | Meaning |
|---|---|
| `ISAAC_RL_POSTTRAINING_DRYRUN=1` | print the launch plan and the resolved config, then exit before anything loads. The Video-to-Data (V2D) launchers take `--dry-run` instead |
| `ISAAC_RL_POSTTRAINING_OUTPUT_DIR` | the run-output root; default `~/isaac_rl_outputs` |
| `WANDB_API_KEY`, `HF_TOKEN` | read from the environment only; never printed by the launchers |
| `ARENA_SERVER_URL`, `ARENA_EVAL_SERVER_URL` | the two G1 Factory simulator addresses, exported by its launchers |
| `LIBERO_CONFIG_PATH`, `MUJOCO_GL` | required in every shell that touches LIBERO |
| `ISAACLAB_ARENA_PATH`, `POSTTRAIN_BASE_CKPT`, `POSTTRAIN_DATA_ROOT` | read by the G1 Factory launchers |
| `V2D_ROOT`, `V2D_MOTION_FILE`, `V2D_SIM_PYTHON`, `POSTTRAIN_OUTPUT_DIR` | read by the Video-to-Data launchers |

The library never reads tuning values from the environment; every knob is a config field or a constructor argument.
What the settings mean is in [concepts.md](concepts.md).
