# SFT

Supervised fine-tuning, `algorithm=sft`, trains a pretrained robot policy to imitate a fixed dataset of demonstrations.
It needs no simulator and no reward: it learns only from recorded observations and the actions taken in them.
Unlike the steering methods, it changes the policy's weights.
It updates the policy's action head and keeps the vision-language backbone frozen.
[IQL](iql.md) is the same update with a learned weight on each demonstration.
SFT is not yet tested end to end in this release.

## How it works

1. **Sampling.** A batch of `algorithm.batch_size` demonstration frames is drawn from the LeRobot dataset.
2. **Loss.** The frozen backbone encodes the observations, and the action head computes its usual training loss on the demonstrated actions (the flow-matching loss for GR00T N1.7 and pi0).
3. **Optimization.** One AdamW step on the mean loss, under GR00T's fine-tuning schedule (warmup, then cosine decay).
4. **Evaluation and checkpoints.** Every `eval.eval_freq` updates the policy is scored in a simulator, if one is configured; every `training.save_interval` updates its weights are saved.

## When to use it

- Demonstrations of the task exist, and the policy should imitate them more closely.
- A baseline for [IQL](iql.md): the two share the optimizer, the loop and the checkpoints, so they differ only in the loss.

## Running it

A run needs a trainable base model (`gr00t_n17` or `pi0`), its checkpoint and a LeRobot dataset of demonstrations:

```bash
uv run python scripts/train.py algorithm=sft \
    base_model=gr00t_n17 base_model.ckpt=<checkpoint> base_model.dataset_dir=<LeRobot dataset> \
    eval.eval_freq=0 eval.eval_on_first_step=false
```

The last two settings train without a simulator.
To evaluate during training, also name a simulator and task (`simulator=<simulator> env.env_name=<provider>/<task>`) and leave the evaluation on.
`ISAAC_RL_POSTTRAINING_DRYRUN=1` in front prints the plan (`plan.loop=offline`) and loads nothing.

Workflow examples:

| Workflow | Example |
|---|---|
| [G1 Factory](../workflows/g1_arena.md#offline-sft-and-iql) | the released GR00T N1.7 trained on one G1 humanoid task's demonstrations |
| [LIBERO](../workflows/libero.md#training) | GR00T N1.7 or pi0 on a LIBERO task, evaluated in the same process |

## Evaluation

Score the trained checkpoint and the starting checkpoint with the same decoding and the same number of episodes, in the same session.
`training.num_rollouts=0` with evaluation on scores `base_model.ckpt`, or the snapshot in `training.load_dir`, without opening the dataset.
Checkpoints are written to `checkpoints/<update>/`, `checkpoints/final/`, and `checkpoints/interrupted/` after Ctrl-C or SIGTERM.

## Configuration

From `isaac_rl_posttraining/configs/algorithm/sft.yaml`, GR00T's own fine-tuning recipe:

| Key | Default | Meaning |
|---|---|---|
| `algorithm.batch_size` | 256 | demonstrations per update; lower it (the G1 Factory workflow uses 8) to fit a 3B model on one 48 GB GPU |
| `algorithm.actor_optimizer_lr` | 1.0e-4 | peak AdamW learning rate |
| `algorithm.actor_optimizer_weight_decay` | 1.0e-5 | AdamW weight decay |
| `algorithm.actor_lr_schedule` / `algorithm.actor_lr_warmup_ratio` | `cosine` / 0.05 | 5% linear warmup, then cosine decay |
| `algorithm.actor_max_grad_norm` | 1.0 | gradient-norm clipping |
| `training.num_rollouts` | 40000 | the budget, counted in **updates**, not environment steps |
| `training.save_interval` | 5000 | updates between checkpoints |
| `eval.eval_freq` | 5000 | updates between evaluations; `eval.eval_freq=0` with `eval.eval_on_first_step=false` trains without a simulator |

## Common issues

- **A run cannot resume.** Checkpoints hold the trained weights only, not the optimizer state, which for a 1.6B-parameter head is about 13 GB.
- **SFT runs in one process.** SFT trains and evaluates in the launching process: it needs `runtime=native` and `placement.learner.ranks=1`, and refuses a batched evaluation simulator.
- **A checkpoint without the robot's layout needs the dataset to evaluate.** A general checkpoint (such as the released GR00T N1.7) does not know the robot's state and action layout, so evaluation reads it from the demonstrations; pass the same `base_model.dataset_dir`.

## Results

SFT is not yet tested end to end in this release, so no results are reported.

## Related pages

- The reward-weighted variant: [IQL](iql.md)
- Internals (code layout, optimizer, adding a base model): `docs/sft.md`
- Configuration file: `isaac_rl_posttraining/configs/algorithm/sft.yaml`
