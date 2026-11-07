# Getting started

This page covers the steps from a fresh clone to a first run.
Steps 1 and 2 need no GPU, simulator or checkpoint.
For other install variants, see [Installation](installation.md).

## 1. Install

You need Linux on x86_64, `git`, and `uv` ([Installation](installation.md#prerequisites) installs them).

```bash
git clone https://github.com/nvidia-isaac/isaac-rlpt.git
cd isaac-rlpt               # run every command from this repository root
uv sync
uv run python -m example_workflows._common.smoke
```

The last command takes a few minutes and ends with `5/5 smoke checks passed`.

## 2. Dry run

Every run goes through one entry point, `scripts/train.py`.
`ISAAC_RL_POSTTRAINING_DRYRUN=1` prints the resolved plan and config, then exits before anything loads:

```bash
ARENA_SERVER_URL=tcp://127.0.0.1:7300 ARENA_EVAL_SERVER_URL=tcp://127.0.0.1:7301 \
ISAAC_RL_POSTTRAINING_DRYRUN=1 uv run python scripts/train.py \
    algorithm=dsrl_sac base_model=mock simulator=g1_arena
```

The two addresses are placeholders that nothing connects to; G1 Factory's own launchers set them for you.
The output includes:

```
plan.simulator=g1_arena
plan.layout=1 rollout worker(s) x 1 environment(s)
plan.evaluation_mode=dedicated
plan.algorithm=dsrl_sac
plan.family=steering
plan.runtime=native
plan.schedule=40000 env steps, learning_starts=1200, eval_freq=5000 (50 episodes), save_interval=5000, retention=all
plan.schedule_kind=sync, max policy lag 0
plan.checkpoint_dir=/home/<you>/isaac_rl_outputs/dsrl_<timestamp>/checkpoints
plan.launch_inputs=ARENA_SERVER_URL (train) = tcp://127.0.0.1:7300; ARENA_EVAL_SERVER_URL (eval) = tcp://127.0.0.1:7301
plan.data_specs=...
plan.placement.envs=...
plan.placement.learner=...
plan.checks=... checked, 1 unverified, 0 error(s) | unverified: the policy's 'state' width binds at load: ...
```

- `plan.layout` and `plan.placement.*`: how many rollout workers and learner processes, and which GPUs each role uses.
- `plan.evaluation_mode`: `dedicated` is a second simulator, `shared` reuses the training workers, `fresh_workers` rebuilds them.
- `plan.schedule` and `plan.schedule_kind`: the budget and cadence, and whether collection overlaps training (`sync`, `overlapped`, `async`).
- `plan.data_specs`: what the simulator emits and accepts, and what the policy reads. A disagreement fails the launch before anything loads. For a GR00T checkpoint on local disk, the policy side is read from the checkpoint's JSON files, so widths and horizons are checked without loading weights.
- `plan.launch_inputs`: the simulator addresses the run needs; a real launch refuses to start without them.
- `plan.checks`: how many checks passed, what stays unverified until something loads, and every failed check, each naming the keys to change.

## 3. First real run

A real run needs an NVIDIA GPU, a simulator and a policy checkpoint.
Pick the workflow that matches your simulator; each page installs what it needs and runs it step by step.

| Workflow | Trains on |
|---|---|
| [G1 Factory](workflows/g1_arena.md) | nine Unitree G1 humanoid tasks in IsaacLab-Arena |
| [Isaac Lab + GR00T](workflows/isaaclab_gr00t.md) | an Isaac Lab 3.0 task |
| [LIBERO](workflows/libero.md) | the LIBERO tabletop suites |
| [DGX Station placement](workflows/dgx_example.md) | one run split across the GB300 and RTX GPUs of a DGX Station |
| [Video-to-Data (V2D) box lift](workflows/v2d_box_lift.md) | a Video-to-Data tissue-box task with batched simulators |

[Concepts](concepts.md) explains the terms the workflow pages use, and the [command-line reference](cli_reference.md) lists every setting.

## Python API

The same run, from Python:

```python
from isaac_rl_posttraining import launch
from isaac_rl_posttraining.algorithms import DSRL

run = launch(DSRL(base="gr00t_n17", base_ckpt="/ckpt/n17"), env="arena/drill_lift")
run.loop()
```

`launch` composes the config `scripts/train.py` would, checks it, and returns a run handle; `run.plan` holds the plan above, and `launch(..., dry_run=True)` stops there.
`train(...)` is `launch(...).loop()`.

- **Algorithms**: `DSRL`, `RFS`, `SACEdit`, `PPO` and `DSRLPPO`. Keyword hyperparameters such as `DSRL(..., batch_size=128)` become `algorithm.batch_size=128`; other settings go in `overrides=["training.num_rollouts=1000"]`.
- **Base model**: `base=` takes a `base_model=` option such as `"gr00t_n17"`, a `BaseModelPolicy` subclass, or `BaseModel("gr00t_n17", ckpt=..., embodiment_tag=...)` when it needs `base_model.*` settings (`from isaac_rl_posttraining.algorithms import BaseModel`).
- **Placement and schedule**:

  ```python
  from isaac_rl_posttraining.runtimes.placement import Learner, Placement, Served

  launch(DSRLPPO(base="gr00t_n17", base_ckpt=ckpt), env="arena/drill_lift", runtime="rlinf",
         placement=Placement(policy=Served(replicas=2), learner=Learner(ranks=2)), schedule="overlapped")
  ```

  On the command line this is `placement.policy.mode=served placement.policy.replicas=2 placement.learner.ranks=2 schedule.kind=overlapped` (add `schedule.max_policy_lag=<n>` with `schedule.kind=async`).

  This example uses two policy servers and two learner ranks, so a real launch needs four GPUs; its dry run passes on any machine.
  On one GPU use one replica and one learner rank, with `runtime.rlinf.first_accelerator=0 runtime.rlinf.exclusive_accelerators=false` appended.
- **One step at a time**, for every algorithm on the native and RLInf runtimes: `run.collect()`, `run.train()`, `run.evaluate()`, `run.publish()`, `run.save(path)`, `run.load(path)`, `run.close()` (or `with launch(...) as run:`). For the off-policy algorithms, `run.collect(rounds=3)` fills the replay buffer (`run.buffer`) and `run.train(steps=10)` runs ten updates on it, returning the last update's metrics averaged over learner ranks. For PPO and DSRL-PPO, one `collect()` and one `train()` are one iteration of the `sync` schedule.
- **Saving collected data**: `launch(..., persist="collected")` saves every completed training episode under `<run dir>/collected` (config key `data.persist.path`; see [the data model](../data_model.md)).
