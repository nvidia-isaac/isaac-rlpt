# SFT internals

How an SFT run (`algorithm=sft`) is put together in code; [IQL](iql.md) shares all of it except the loss.
What the method does, how to run and evaluate it, its options and its pitfalls are on the user page, `docs/pages/algorithms/sft.md`.

## Code layout

| Piece | Where | Role |
|---|---|---|
| Trainable base model | `isaac_rl_posttraining/policies/` (`TrainablePolicySpec`) | per-sample action loss `actor_bc_loss`; backbone frozen |
| Demonstration source | `isaac_rl_posttraining/training/offline.py` | `sample(B) -> {inputs_cur, action, ...}` |
| Learner | `isaac_rl_posttraining/algorithms/sft/sft.py` | mean `actor_bc_loss`, then one optimizer step |
| Loop | `isaac_rl_posttraining/runners/offline_launch.py` | updates, evaluation, checkpoints; no collection workers |

On GR00T N1.7 `actor_bc_loss` is the model's flow-matching reconstruction (`supervised_loss`); on pi0 it is the shared flow-SDE reconstruction (`FlowSDEActorCritic.actor_bc_loss`).
The evaluation workers drive the live policy object, so they must share the training process; this is why the plan refuses any runtime other than a one-process `runtime=native` (refused when `runtime.native.processes` resolves to `ray`).

## Optimizer

The optimizer (`isaac_rl_posttraining/algorithms/actor_optimizer.py`) is shared with IQL.
It keeps float32 master weights for the bfloat16 action head, because AdamW steps on bfloat16 weights round most updates away.
A step whose gradient is not finite is skipped.

## Checkpoints

A checkpoint holds the trained weights (`policy_state.pt`) and a manifest, with no optimizer state.

## Extension points

A trainable base model (a `FlowSDEActorCritic` subclass) is SFT-ready once it can turn a dataset directory into these batches: register its opener in `SOURCE_BUILDERS` in `isaac_rl_posttraining/training/offline.py`.
A new offline method is a `BaseAlgorithm` that reads `actor_bc_loss` on the injected policy, a `isaac_rl_posttraining/configs/algorithm/<name>.yaml`, and one registry entry with `loop="offline"`.
