# IQL internals

How an IQL run (`algorithm=iql`, Kostrikov et al., 2022) is put together in code.
What the method does, how to run and evaluate it, its options and its pitfalls are on the user page, `docs/pages/algorithms/iql.md`.
The loop, the demonstration source, the actor optimizer and the checkpoints are SFT's ([sft.md](sft.md)).

| Piece | Where | Role |
|---|---|---|
| Token critics | `isaac_rl_posttraining/algorithms/iql/critics.py` | expectile V, TD Q, advantage weights |
| Learner | `isaac_rl_posttraining/algorithms/iql/iql.py` | tokens, critic losses, weighted `actor_bc_loss` |

The critics read the backbone's observation tokens through `feature_tokens` on the trainable base model.
The Q target is the chunk return plus `gamma ** bootstrap_horizon` times V of the next observation, zero after a terminal; the target critic follows by Polyak averaging at `tau`.
The critics are not saved: a checkpoint holds the policy only.
