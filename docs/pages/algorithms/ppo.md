# PPO

PPO fine-tuning, `algorithm=ppo`, trains the pretrained robot policy's own weights with Proximal Policy Optimization (Schulman et al., 2017), from rewards collected in simulation.
Unlike the steering methods ([DSRL](dsrl.md), [RFS](rfs.md), [SAC-edit](sac_edit.md)), which keep the policy frozen and train a small model around it, PPO changes the policy itself.
It updates the policy's action head, the part that turns an observation into actions.
The vision-language backbone, which encodes the camera images and the instruction, stays frozen.

## How it works

A flow-matching policy produces an action chunk by starting from random noise and refining it over a few denoising steps.
That process has no likelihood that PPO can use, so the method makes one of the denoising steps random, with noise of scale `ppo.noise_level`.
Each decoded chunk then has a probability under the current policy, and PPO's probability ratio between the old and new policy is defined.
This is the flow-SDE formulation that RLinf uses for GR00T.

Every training iteration has four steps:

1. **Rollout.** Every environment plays `ppo.episodes_per_environment` episodes with the current policy, and every decision is stored with its probability and the value function's estimate.
2. **Advantage estimation.** Generalized advantage estimation (GAE) scores each decision against the value function, with discount `ppo.gamma` and `ppo.gae_lambda`.
3. **Optimization.** `ppo.update_epochs` passes of clipped PPO updates train the action head and the value function on the batch. With several learner ranks the batch is split across them.
4. **Publication.** The new weights are sent to the rollout workers before the next rollout.

Every `ppo.eval_every` iterations the policy is evaluated with a deterministic decode on episodes that start from the fixed reset seed `ppo.eval_seed`.

## When to use it

- The policy is wrong in a way that a frozen policy cannot be steered out of, and changing its weights is acceptable.
- There are enough GPUs for a fine-tune. A single-GPU run fits on one 48 GB GPU; larger runs shard the learner over several GPUs with FSDP.
- Try a steering method first when the policy already succeeds sometimes. Steering is cheaper, and a failed run leaves the policy unchanged.

## Running it

A run needs a simulator, a trainable base model and its checkpoint.
The general form of the command, run in an environment that can start the simulator, is:

```bash
uv run python scripts/train.py algorithm=ppo \
    simulator=<simulator> env.env_name=<provider>/<task> \
    base_model=<gr00t_n15|gr00t_n17> base_model.ckpt=<checkpoint>
```

`ISAAC_RL_POSTTRAINING_DRYRUN=1` in front prints the run's plan (simulator, base model, placement, schedule and batch sizes) and the resolved configuration, and loads nothing.

The shipped `algorithm=ppo` recipe defaults to RLinf's Isaac Lab stack-cube setup, so the checkpoint is the only field it requires.
Its default placement fills one 8-GPU node with four learner ranks and four rollout workers.
On one GPU, put the learner and a single rollout worker in one process; run it from `example_workflows/isaaclab_gr00t_posttraining/`, whose environment can start Isaac Sim ([Isaac Lab](../workflows/isaaclab_gr00t.md#installation)):

```bash
uv run python ../../scripts/train.py algorithm=ppo base_model.ckpt=<checkpoint> \
    runtime=native placement.learner.ranks=1 placement.envs.workers=1 training.num_envs=16
```

Each workflow wraps these commands in its own launcher, sets up the simulator and names a checkpoint:

| Workflow | Example |
|---|---|
| [Isaac Lab](../workflows/isaaclab_gr00t.md#training) | GR00T N1.5 on Franka cube stacking, on one GPU or one 8-GPU node |
| [G1 Factory](../workflows/g1_arena.md#full-ppo-fine-tuning) | GR00T N1.7 on a Unitree G1 humanoid task, with eight scenes on one GPU and the learner on another |

### Served policy

By default every rollout worker keeps its own copy of the policy.
With many workers, that copy is repeated on every GPU, and each copy decodes only its own worker's scenes.
A **served policy** moves the policy into a few separate policy-server processes instead; every rollout worker sends its observations there, and a server decodes the requests of several workers in one batch.
`placement.policy.mode=served` turns it on, and `placement.policy.gpus` sets how many GPUs the servers use.

After every update the learner sends its new weights to the servers once.
A server stamps every decision with the weight version it used, and a rollout worker refuses a decision made with any other version, so training never mixes data from two policies.
A served decision has the same distribution as a local one, but its random draws come from the server, so a seeded run gives the same numbers as a local run only when the server runs in the worker's own process.
PPO fine-tuning supports served policies on `runtime=native` and `runtime=rlinf`.
The [Isaac Lab workflow](../workflows/isaaclab_gr00t.md#served-policy) has a worked example.

### Demonstration co-training

Reinforcement learning in simulation can raise the simulated success rate by learning behaviour that only works in simulation.
**Co-training** guards against this drift by adding a supervised loss on the policy's original demonstrations to every PPO update:

```text
loss = PPO loss + ppo.cotrain_beta * supervised loss on the demonstrations
```

The demonstrations do not change during training, so they keep pulling the policy back toward the behaviour it was originally taught.
`ppo.cotrain_beta` sets the weight of that pull; the default, 0, turns co-training off and costs nothing.
Above 0, `ppo.cotrain_dataset_path` must name a LeRobot dataset of the demonstrations, and the base model must be able to score them (GR00T N1.5 and N1.7 can).
`ppo.cotrain_batch_size` (32) sets how many demonstrations are scored per gradient step.
The [Isaac Lab workflow](../workflows/isaaclab_gr00t.md#demonstration-co-training) has a worked example.

## Evaluation

Training evaluates every `ppo.eval_every` iterations and logs `eval/success_rate`.
To score a saved checkpoint, run the training command again with `ppo.eval_only=true` and `ppo.resume_from=<checkpoint directory>`.
`placement.learner.ranks=0` evaluates on the rollout workers alone, without starting a learner.
Without `ppo.resume_from`, the same command scores the unchanged base checkpoint, which is the baseline to report next to the trained policy.

For a fair comparison, keep the task, the starting checkpoint, the number of environments and `ppo.eval_seed` the same, and score the base and the trained policy in the same session with the same number of episodes.
Simulators that render camera images may not be exactly reproducible across processes, so compare success rates over enough episodes.
`scripts/ppo/compare_evaluations.py` checks that two evaluation records scored the same policy on the same starting states and reports both success rates.

A run writes:

- `checkpoints/iteration_<n>/`, the policy snapshot of that iteration;
- `checkpoints/final/`, the learner state and policy snapshot at the end of the run;
- `ppo_metrics.jsonl`, with every iteration's metrics;
- `ppo_evaluation_episodes.jsonl`, with one line per evaluation episode;
- `checkpoints/evaluation_summary.json`.

## Configuration

The recipe is `isaac_rl_posttraining/configs/algorithm/ppo.yaml`, and the full schema is `PPOTrainingConfig` in `isaac_rl_posttraining/configs/schema.py`.

| Key | Default | Meaning |
|---|---|---|
| `ppo.iterations` | 1000 | training iterations |
| `ppo.episodes_per_environment` | 2 | episodes every environment plays per iteration |
| `ppo.eval_every`, `ppo.eval_episodes_per_environment` | 10, 1 | evaluation cadence and episodes per environment (0 = off) |
| `ppo.eval_seed` | 100000 | reset seed replayed by every evaluation |
| `ppo.checkpoint_every`, `ppo.checkpoint_retention` | 20, `best_last_final` | checkpoint cadence; `best_last_final` or `all` |
| `ppo.eval_only`, `ppo.resume_from` | false, null | evaluate once and exit; checkpoint directory to load |
| `ppo.learning_rate`, `ppo.value_learning_rate` | 1e-5, 2e-5 | action head and value function learning rates |
| `ppo.optimizer`, `ppo.adam_betas`, `ppo.optimizer_eps`, `ppo.weight_decay` | adamw, [0.9, 0.95], 1e-8, 0.01 | optimizer settings |
| `ppo.clip_ratio`, `ppo.value_clip` | 0.2, 0.2 | clipping of the probability ratio and of the value |
| `ppo.value_loss`, `ppo.huber_delta`, `ppo.value_loss_coef` | huber, 10.0, 1.0 | value loss (`mse` or `huber`) and its weight |
| `ppo.entropy_coef` | 0 | entropy bonus |
| `ppo.update_epochs`, `ppo.global_batch_size`, `ppo.micro_batch_size` | 3, 512, 32 | passes per iteration, batch over all learner ranks, gradient-accumulation slice |
| `ppo.max_grad_norm`, `ppo.target_kl` | 1.0, null | gradient clipping; stop the update when the KL divergence exceeds the target |
| `ppo.gamma`, `ppo.gae_lambda` | 0.99, 0.95 | discount and GAE parameter |
| `ppo.discount_per` | `decision` | `decision` discounts once per action chunk; `simulator_step` discounts every executed step |
| `ppo.bootstrap_at_time_limit` | false | bootstrap from the next value when the episode cap ends an episode |
| `ppo.noise_level` | 0.5 | scale of the random denoising step |
| `ppo.cache_features` | false | rollout workers send the frozen backbone's outputs instead of images, so the learner skips the backbone |
| `ppo.frozen_submodules` | null | the parts of the model kept frozen; null uses the base model's own choice |
| `ppo.cotrain_beta`, `ppo.cotrain_dataset_path`, `ppo.cotrain_batch_size` | 0, null, 32 | demonstration co-training |
| `base_model.denoising_steps`, `base_model.action_horizon` | 4, 16 | denoising steps per decode (one is random); executed steps per decision |

Batch sizes are global: each learner rank trains on `ppo.global_batch_size / placement.learner.ranks`, so an update is the same on one GPU and on four.
`training.num_envs` is the total number of environments and must be a multiple of `placement.envs.workers`.
The episode cap must be a whole number of action chunks (`base_model.action_horizon`); the launch checks this.

`schedule.kind=overlapped` collects the next iteration while the learner updates on the current one, so the data lags the policy by one version.
`schedule.kind=async` also publishes after every update; PPO keeps `schedule.max_policy_lag` at 1 and refuses larger values.

### Fine-tunable base models

| `base_model=` | Policy | Required fields |
|---|---|---|
| `gr00t_n15` | GR00T N1.5 | `ckpt`, `data_config`, `embodiment_tag`, `denoising_steps`, `action_horizon` |
| `gr00t_n17` | GR00T N1.7, experimental | `ckpt`, `embodiment_tag`, `denoising_steps`, `action_horizon`, plus `ppo.cache_features=true` |

pi0 has a trainable adapter, but no simulator that PPO runs on accepts it yet.
`mock`, `dreamzero_local` and `dreamzero_server` cannot be fine-tuned.

### Learner scaling over FSDP ranks

`placement.learner.ranks` above 1 shards the learner with PyTorch FSDP under `runtime=rlinf`.
It lowers the memory each GPU needs; at the model sizes measured here it does not make an update faster.

Identical PPO updates of a synthetic 85.2M-parameter transformer actor-critic: global batch 512, minibatch 128, 2 epochs, bfloat16, the same input batch in every condition, 5 timed updates after 2 warm-ups.
One node of NVIDIA L40S GPUs per condition; 2026-09-11, `scripts/bench_rlinf_ppo_fsdp.py`.
The `torchrun` rows launch the same learner directly with PyTorch, as a reference for the RLInf launch.

| Learner ranks | Launcher | Update (s) | Input samples/s | Peak allocated memory per rank | Startup |
|---|---|---|---|---|---|
| 1 | `torchrun` | 0.275 | 1,861 | 2.11 GiB | 3.7 s |
| 1 | RLInf | 0.282 | 1,818 | 2.11 GiB | 76.1 s |
| 2 | `torchrun` | 0.300 | 1,706 | 1.15 GiB | 8.7 s |
| 2 | RLInf | 0.365 | 1,401 | 1.15 GiB | 82.5 s |
| 4 | `torchrun` | 0.361 | 1,418 | 0.74 GiB | 17.7 s |
| 4 | RLInf | 0.359 | 1,425 | 0.74 GiB | 99.6 s |

Four ranks need 64.9% less memory per GPU than one.
The update gets slower as ranks are added: a model this small fits on one GPU, so sharding adds communication without removing work.
Use several ranks when the learner does not fit on one GPU.
The four-rank `torchrun` row is a repeat; the first run measured 0.965 s per update on a different node and was set aside as a machine outlier.

## Common issues

- **Not every evaluation has a checkpoint.** With the defaults, evaluations happen every 10 iterations and checkpoints every 20. Set `ppo.checkpoint_every` to a divisor of `ppo.eval_every` to re-score an evaluated policy.
- **Retention prunes at the end of the run.** `best_last_final` keeps the checkpoint nearest the best evaluation, the newest and `final`; `ppo.checkpoint_retention=all` keeps every one.
- **A gated backbone.** `base_model.backbone_model_name_or_path=<local directory>` points a checkpoint's gated vision-language backbone at a local copy.

## Results

Measured PPO results, each next to its base policy, are on the [G1 Factory](../workflows/g1_arena.md#results) and [Isaac Lab](../workflows/isaaclab_gr00t.md#results) workflow pages.

## Related pages

- Steering the policy without changing its weights: [DSRL](dsrl.md), including the PPO-trained steering variant
- Where it runs: [Isaac Lab](../workflows/isaaclab_gr00t.md), [G1 Factory](../workflows/g1_arena.md)
- Placement, runtimes and served policies in depth: `docs/runtimes.md`
- Internals (adapters, metric records, Python API, learner tools): `docs/ppo.md`
