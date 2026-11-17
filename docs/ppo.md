# PPO internals

How the two PPO paths (`algorithm=dsrl_ppo`, steering a frozen policy, and `algorithm=ppo`, fine-tuning a GR00T action head) are put together in code.
What each method trains, how to run and evaluate it and its options are on the user pages: `docs/pages/algorithms/ppo.md` for fine-tuning, with the full `ppo:` option table, and `docs/pages/algorithms/dsrl.md` for DSRL-PPO.

## Code layout

| Piece | Where |
|---|---|
| The `ppo:` section | `PPOTrainingConfig` in `isaac_rl_posttraining/configs/schema.py` |
| Recipes | `isaac_rl_posttraining/configs/algorithm/dsrl_ppo.yaml`, `isaac_rl_posttraining/configs/algorithm/ppo.yaml` |
| PPO learner | `isaac_rl_posttraining/algorithms/ppo/` |
| Trainable adapters | `isaac_rl_posttraining/policies/trainable.py` |
| Shared sampling and likelihood | `FlowSDEActorCritic` in `isaac_rl_posttraining/policies/flow_sde.py` |
| Rollout runtimes and learner placement | [runtimes.md](runtimes.md) |

`dsrl_ppo` sizes its steering actor through the SAC schema composed at `algorithm` (that SAC is never trained); `ppo:` holds the update and the schedule.

## Trainable adapters

`base_model=` selects the trainable adapter; the learner ranks and the rollout workers each load their own copy.

| `base_model=` | Adapter | Frozen part |
|---|---|---|
| `gr00t_n15` | `isaac_rl_posttraining/policies/gr00t/flow_sde.py` | `model.backbone` |
| `gr00t_n17` | `isaac_rl_posttraining/policies/gr00t/flow_sde_n17.py` | `model.backbone` |
| `pi0` | `isaac_rl_posttraining/policies/pi0/flow_sde.py` (fields in `isaac_rl_posttraining/configs/base_model/pi0.yaml`) | `model.paligemma_with_expert.paligemma` |

`ppo.frozen_submodules=null` keeps the adapter's own frozen part; a list overrides it.
GR00T N1.7 requires `ppo.cache_features=true` because Qwen3-VL's flattened image patches cannot travel per sample, so the rollout workers ship the frozen backbone's outputs and the learner skips the backbone.
A GR00T N1.7 checkpoint with a baked embodiment, on the library entry point:

```bash
$ISAAC_RLPT_PYTHON scripts/train.py algorithm=ppo base_model=gr00t_n17 base_model.ckpt=/path/to/n17 \
  base_model.embodiment_tag=<tag> base_model.denoising_steps=4 base_model.action_horizon=16 ppo.cache_features=true
```

[adding_a_base_model.md](adding_a_base_model.md) describes how to add an adapter.

## Metrics and records

Metrics use four namespaces, each logged with its own `<namespace>/step`: `train/*` (training rollouts, `train/success_rate`), `eval/*` (`eval/success_rate`), `learner/*` (losses, KL, clip fraction, entropy) and `system/*` (timings, throughput, `system/policy_lag`).
Every iteration's metrics are appended to `ppo_metrics.jsonl`.
`ppo_evaluation_episodes.jsonl` records, for every evaluation episode, the environment index, the reset seed and its scope (`environment` for one-environment workers, `worker_batch` for batched workers), success, length, return, the policy's SHA-256 (of the published weights, or for a learner-less evaluation of the base checkpoint, of its weight files; `policy_source` says which), and hashes of the initial observation, of every learner observation and of every executed action chunk.
`scripts/ppo/compare_evaluations.py` compares two such records episode by episode.
A checkpoint directory holds the learner state and a `policy_state.pt` snapshot of the published weights.

## Python API

The `PPO` object (`isaac_rl_posttraining/algorithms/ppo/`) supplies a learner worker from a model builder and a `PPOConfig`; the runtime places the ranks ([runtimes.md](runtimes.md)).

```python
from isaac_rl_posttraining.algorithms.ppo import PPO, PPOConfig
from isaac_rl_posttraining.runtimes import open_distributed_learner_group

algorithm = PPO(PPOConfig(epochs=4, minibatch_size=256, clip_ratio=0.2, target_kl=0.02))
learner = open_distributed_learner_group(runtime, algorithm.learner(build_model), num_learner_ranks=4, name="ppo_learner")
metrics_by_rank = learner.train_step(rank_local_batches)
rollout_group.set_policy_state(learner.policy_state(version=next_policy_version))
```

Invariants:

- Every rank must receive the same number of samples.
- A new policy version is published only after the whole update has finished.
- Batch sizes are global: the rank-local minibatch is `ppo.global_batch_size / placement.learner.ranks`, and `ppo.micro_batch_size` splits it for gradient accumulation.
- `num_learner_ranks` is the number of learner processes; inside the PyTorch process group the same number is the `world_size`, and it never means the number of environments.

## Learner tools

Two tools exercise the RLInf learner on its own.
`uv run --extra rlinf python scripts/smoke_rlinf_fsdp_learner.py --num-learner-ranks 2 --steps 4` runs a small checkpoint round trip on a machine with two GPUs.
`uv run --extra rlinf python scripts/bench_rlinf_ppo_fsdp.py --launcher rlinf --num-learner-ranks 4` times identical PPO updates under RLInf-managed FSDP; `--launcher torchrun` under `torchrun --standalone --nproc-per-node 4` is the direct PyTorch control condition.
Its measured results are on the user page; per-phase timing of a 128-environment G1 Factory run is in [runtimes.md](runtimes.md#measured-performance).
