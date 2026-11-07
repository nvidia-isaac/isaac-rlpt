# IQL

Implicit Q-Learning (Kostrikov et al., 2022), `algorithm=iql`, trains a pretrained robot policy from a fixed dataset of demonstrations, like [SFT](sft.md), but weights each demonstration by how much it leads toward success.
It learns that weight from rewards recorded in the dataset, so demonstrations that succeed count more than those that fail.
It needs no simulator, and it changes the policy's weights.
IQL is not yet tested end to end in this release.

## How it works

1. **Sampling.** A batch of transitions (an observation, its action, the reward and the next observation) is drawn from the LeRobot dataset.
2. **Critics.** Two small networks read the frozen backbone's encoding of each observation. One estimates the value of a state (V) and the other the value of a state and action (Q); V learns an optimistic estimate of Q (its `algorithm.expectile` expectile), and Q learns from the next state's V.
3. **Weighting.** The advantage, Q minus V, says how much better a demonstrated action was than usual in that state. Each sample's weight is `exp(advantage_temp * advantage)`, clipped at `clip_score`.
4. **Optimization.** The action head minimizes SFT's per-sample action loss under those weights, with SFT's optimizer and schedule.

## When to use it

- The demonstrations mix successes and failures, and the policy should imitate the successful ones more.
- A comparison against [SFT](sft.md) on the same data: the two share the optimizer, the loop and the checkpoints, so they differ only in the loss.

## Running it

The command is [SFT's](sft.md#running-it) with `algorithm=iql`:

```bash
uv run python scripts/train.py algorithm=iql \
    base_model=gr00t_n17 base_model.ckpt=<checkpoint> base_model.dataset_dir=<LeRobot dataset> \
    eval.eval_freq=0 eval.eval_on_first_step=false
```

The dataset must record the reward: every frame is a transition, and the last frame of an episode carries the success reward in `next.reward` and `next.done` (or `rl_info.next.*`).

Workflow examples:

| Workflow | Example |
|---|---|
| [G1 Factory](../workflows/g1_arena.md#offline-sft-and-iql) | the released GR00T N1.7 trained on one G1 humanoid task's demonstrations |
| [LIBERO](../workflows/libero.md#training) | GR00T N1.7 on a LIBERO task, evaluated in the same process |

## Evaluation

As for [SFT](sft.md#evaluation): score the trained and the starting checkpoint with the same decoding and number of episodes, in the same session.

## Configuration

From `isaac_rl_posttraining/configs/algorithm/iql.yaml`; the actor's optimizer and schedule are SFT's ([SFT](sft.md#configuration)).

| Key | Default | Meaning |
|---|---|---|
| `algorithm.expectile` | 0.7 | expectile of Q that V regresses |
| `algorithm.advantage_temp` | 10.0 | temperature of the advantage weight |
| `algorithm.clip_score` | 20.0 | upper clip on the weight |
| `algorithm.gamma` / `algorithm.tau` | 0.99 / 0.005 | discount; Polyak rate of the target critic |
| `algorithm.num_q` / `algorithm.n_q_to_sample` / `algorithm.q_reduce` | 5 / 2 / `min` | Q heads; the TD and value targets use the minimum over a random 2 of them, and every head trains on the Q loss |
| `algorithm.critic_hidden` / `algorithm.critic_n_heads` | 1024 / 8 | token-critic width and attention heads |
| `algorithm.critic_optimizer_lr` | 3.0e-4 | critic learning rate |
| `algorithm.batch_size` | 256 | demonstrations per update |

## Common issues

- **Keep `advantage_temp` high.** With a sparse terminal reward the advantages are small, and a temperature of 1 weights every sample almost equally, which reduces IQL to SFT.
- **The critics are not saved.** A checkpoint holds the policy only, so a run cannot resume; the other limits are SFT's ([SFT](sft.md#common-issues)).

## Results

IQL is not yet tested end to end in this release, so no results are reported.

## Related pages

- The unweighted update: [SFT](sft.md)
- Internals (critics, learner): `docs/iql.md`
- Configuration file: `isaac_rl_posttraining/configs/algorithm/iql.yaml`
