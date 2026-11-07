# DSRL internals

How a DSRL run (`algorithm=dsrl_sac`) is put together in code.
What the method does, how to run it, its options and its pitfalls (including the temperature collapse beyond 40,000 steps and `algorithm.ent_coef=0.05`) are on the user page, `docs/pages/algorithms/dsrl.md`.
Everything policy-specific sits behind the `BaseModelPolicy` contract ([adding_a_base_model.md](adding_a_base_model.md)); everything learner-specific is the generic SAC core ([adding_an_algorithm.md](adding_an_algorithm.md)).

## Code layout

One training run is the off-policy loop (`isaac_rl_posttraining/training/off_policy.py`) over these pieces:

| Piece | Where | Role |
|---|---|---|
| Frozen policy | `isaac_rl_posttraining/policies/gr00t/` | `infer(obs, noise)` decodes a noise sample into an action chunk; never trained |
| Steering action space | `isaac_rl_posttraining/envs/steering_wrappers.py` | a `[chunk_size, noise_dim]` box bounded by `dsrl.noise_action_bound` |
| Actor | `isaac_rl_posttraining/modules/transformer/actor.py` | one token per flow step, a transformer, a tanh-squashed Gaussian over the box |
| Critic ensemble | `isaac_rl_posttraining/modules/transformer/critic.py` | 4 transformer critics |
| SAC core | `isaac_rl_posttraining/algorithms/sac/` | ensemble SAC; the steering sample is its action |
| DSRL layer | `isaac_rl_posttraining/algorithms/dsrl/` | the `dsrl:` config section and the chunk-discount glue |
| Collection and evaluation | `isaac_rl_posttraining/runtimes/steering_policy.py`, `isaac_rl_posttraining/training/evaluation.py` | the steering rollout policy and the evaluator every steering method shares |

## Training loop

**Discounting.** One transition per executed chunk, so `gamma ** action_exec_len` is folded into the SAC discount once at setup; a time-out counts as a terminal state (the objective is "succeed within the episode cap").
**Updates.** One burst per completed episode: one training call per stored decision (a short final chunk counts as one), each running 10 gradient steps (4 critic updates and 1 actor plus temperature update per step).
**Critic target.** A random 2-of-4 subset of the critics reduced by `algorithm.critic_target_reduction`.
**Schedule.** `schedule.kind=sync` collects a round, trains on it, then publishes the new actor to the workers.
`overlapped` starts the next round before the burst, so the burst runs while the workers collect on the actor of the previous publish (a policy lag of one), and the publish after the burst waits for that round; the update rule and the budget are the same, only an iteration's wall time changes (`system/iteration_env_steps_per_s` against the collection-only `system/env_steps_per_s`).
`async` re-submits a round as soon as the previous one returns, publishes once after each burst (or every `schedule.publish_every` learner updates when set), and drops collected decisions whose policy lag, the number of publishes between the weights that collected them and the learner's current weights, exceeds `schedule.max_policy_lag` (`system/dropped_stale_decisions`).
Rounds that end in an evaluation, and the last one of the budget, are never overlapped.

## Latent injection

The GR00T flow head denoises a 40-step noise window (`latent_horizon`); the checkpoint's processor decodes a 16-step action chunk from it (`action_horizon`); the rollout executes the first 8.
Steering samples are sized against the noise window, execution against the decoded chunk.
How a sample becomes the head's initial noise is handled by the base model (`LatentInjectionConfig` in `isaac_rl_posttraining/policies/gr00t/model.py`): injected entries replace the sampled noise, clipped to plus or minus 3; a sample covering fewer flow steps than the window is tiled across it; entries the sample does not cover are drawn fresh from N(0, 1) on every call.

## Evaluation and checkpoints

The evaluator (`RolloutEvaluator`) scores the same rollout policy the workers train with, so an evaluation runs exactly the decode-and-execute path of training.
Base mode (`eval.eval_base=true`) runs the frozen policy alone through the identical pipeline.
Checkpoints are written at evaluation boundaries with no training in between, so a checkpoint holds exactly the weights of the evaluation sharing its step; `fingerprints.jsonl` records the actor's SHA-256 at every evaluation, save and load.

## Config validation

A config that still sets a removed replay key (`buffer.split`, `buffer.fractions`) or an unknown `buffer.*` key is refused at launch with the replacement named.

## Metric names

Shared with the PPO trainers: `train/*` are the training rollouts, `eval/*` the evaluations, `learner/*` the learner's losses, Q statistics, KL and temperature, `buffer/*` the replay statistics, `system/*` timings and throughput.
Every row's `<namespace>/step` is the run's environment step count; `learner/*` is logged once per burst (the burst's last update) with `learner/updates`, the gradient updates done so far, and the evaluation video at the step of its evaluation.

## References

- A. Wagenmaker et al. *DSRL: Diffusion Steering via Reinforcement Learning*, 2025.
- X. Chen et al. *Randomized Ensembled Double Q-Learning* (REDQ), ICLR 2021.
