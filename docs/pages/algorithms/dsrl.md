# DSRL

DSRL (Diffusion Steering via Reinforcement Learning, Wagenmaker et al., 2025), `algorithm=dsrl_sac`, improves a pretrained flow-matching or diffusion policy without changing its weights.
Such a policy turns a random noise sample into an action chunk.
DSRL trains a small reinforcement-learning agent that chooses that noise sample instead of drawing it at random, and so steers the frozen policy toward the actions that succeed.
The policy's weights never change, so a failed run leaves the policy unchanged.

## How it works

1. **Steering.** A small actor network looks at the robot's state and a frozen image embedding of the scene, and emits a noise sample of shape `[training.chunk_size, dsrl.noise_dim]`. Before `algorithm.learning_starts` environment steps the sample is random.
2. **Decoding.** The frozen policy, which still receives its usual observations, decodes the sample into an action chunk.
3. **Execution.** The first `dsrl.action_exec_len` actions run in the simulator. The reward is sparse: 0 on success, -1 when the episode ends without success, and 0 otherwise.
4. **Optimization.** Soft Actor-Critic (SAC) trains the actor and an ensemble of critics on the stored decisions, in a burst of updates after each round of collection.

Because the actor acts in the policy's noise space, every action it produces is one the frozen policy could have produced.
DSRL can therefore only pick better behaviour out of what the policy already does.

## When to use it

- The default starting point for any task.
- The policy already succeeds sometimes, and reinforcement learning should make the successful behaviour more frequent.
- Alternatives: [RFS](rfs.md) adds a small correction on top of the decoded actions; [SAC-edit](sac_edit.md) edits the decoded chunk directly; [PPO](ppo.md) fine-tunes the policy's own weights; [SFT](sft.md) and [IQL](iql.md) learn from demonstrations without a simulator.

## Running it

The general form of the command, run in an environment that can start the simulator, is:

```bash
uv run python scripts/train.py algorithm=dsrl_sac \
    simulator=<simulator> env.env_name=<provider>/<task> \
    base_model=<policy> base_model.ckpt=<checkpoint> training.num_rollouts=40000
```

`ISAAC_RL_POSTTRAINING_DRYRUN=1` in front prints the run's plan and the resolved configuration, and loads nothing.
The steering geometry (`dsrl.noise_dim`, `training.chunk_size`, `dsrl.action_exec_len`) must match the policy; see [Configuration](#configuration).

Each workflow wraps this command in a launcher that starts the simulator and sets the geometry for its policy:

| Workflow | Example |
|---|---|
| [G1 Factory](../workflows/g1_arena.md#training) | GR00T N1.7 on nine Unitree G1 humanoid tasks |
| [Isaac Lab](../workflows/isaaclab_gr00t.md#training) | GR00T N1.5 on Franka cube stacking, through Isaac Lab's `train` command |
| [LIBERO](../workflows/libero.md#training) | GR00T N1.7 and pi0 on LIBERO tabletop tasks |
| [Video-to-Data (V2D) tissue box](../workflows/v2d_box_lift.md) | a 58-joint GR00T policy learned from human videos |

### DSRL with PPO

`algorithm=dsrl_ppo` trains the same steering actor in the same noise space, with PPO instead of SAC.
The policy stays frozen, as in DSRL; only the optimizer differs.
It is useful for comparing an on-policy optimizer against DSRL's off-policy SAC without changing anything else.

In DSRL-PPO one decision is one whole decoded chunk, so every decoded action executes.
The actor starts as an unsquashed Gaussian with unit standard deviation.
The optimizer settings live in the `ppo:` section, whose keys are described on the [PPO page](ppo.md#configuration); the recipe (`dsrl_ppo.yaml`) sets the main values below.

| Key | `dsrl_ppo` | Meaning |
|---|---|---|
| `ppo.iterations`, `ppo.checkpoint_every` | 100, 10 | training iterations; a checkpoint every 10 iterations |
| `ppo.episodes_per_environment` | 1 | episodes every environment plays per iteration |
| `ppo.learning_rate`, `ppo.value_learning_rate` | 3e-4, null | actor learning rate; null trains the value function at the same rate |
| `ppo.optimizer`, `ppo.adam_betas`, `ppo.optimizer_eps`, `ppo.weight_decay` | adam, [0.9, 0.999], 1e-5, 0 | optimizer settings |
| `ppo.value_loss`, `ppo.huber_delta`, `ppo.value_loss_coef` | mse, 1.0, 0.5 | value loss and its weight |
| `ppo.update_epochs`, `ppo.global_batch_size` | 4, 64 | passes per iteration, batch over all learner ranks |
| `ppo.max_grad_norm` | 0.5 | gradient clipping |
| `ppo.discount_per`, `ppo.bootstrap_at_time_limit` | `simulator_step`, true | discount every executed step; bootstrap when the episode cap ends an episode |
| `ppo.isolate_evaluation_workers`, `ppo.sample_evaluation_actions` | true, true | evaluate on fresh workers, and sample the actor rather than take its mean |

The [G1 Factory](../workflows/g1_arena.md#training), [Isaac Lab](../workflows/isaaclab_gr00t.md#training) and [LIBERO](../workflows/libero.md#training) workflows each have a DSRL-PPO example.

## Evaluation

The actor is a distribution, so evaluation samples from it (`eval.action_selection=sample`, the default).
Report a trained actor next to the frozen policy scored in the same session with the same number of episodes; `eval.eval_base=true` scores the frozen policy through the same evaluator.
Confirm a result on 100 episodes or more before adopting it.

A checkpoint holds exactly the weights of the evaluation that shares its step.
`fingerprints.jsonl` in the run directory records the actor's hash at every evaluation, save and load.

## Configuration

The recipe is `isaac_rl_posttraining/configs/algorithm/dsrl_sac.yaml`, on top of `isaac_rl_posttraining/configs/train.yaml`.
Replay, budget, evaluation and schedule keys shared by every off-policy method are in the [command-line reference](../cli_reference.md).

| Key | Default | Meaning |
|---|---|---|
| `training.chunk_size` | 40 | noise rows per decision, one per action step of the policy's horizon |
| `dsrl.noise_dim` | 35 | width of one noise sample, equal to the policy's action width |
| `dsrl.action_exec_len` | 8 | actions executed per decision |
| `dsrl.noise_action_bound` | 1.0 | half-width of the box the actor samples noise in |
| `dsrl.warmup_exploration` | `uniform` | noise before learning starts: `uniform` over the box, or `normal` for the policy's own noise |
| `algorithm.learning_starts` | 1200 | environment steps of random noise before learning begins |
| `algorithm.gamma` | 0.995 | discount per environment step |
| `algorithm.batch_size` | 64 | batch per gradient step |
| `algorithm.num_critics` / `algorithm.n_critics_to_sample` | 4 / 2 | ensemble size, and the subset sampled per target |
| `algorithm.critic_target_reduction` | `mean` | reduction in the TD target; `min` is the pessimistic variant |
| `algorithm.critic_reduction` | `min` | reduction in the actor objective |
| `algorithm.actor_optimizer_lr` / `algorithm.critic_optimizer_lr` | 5e-5 / 1e-4 | learning rates |
| `algorithm.ent_coef` | `auto` | SAC temperature; a number fixes it |
| `algorithm.num_updates_per_train_step` | 10 | gradient steps per training burst |
| `training.num_rollouts` | 40000 | environment-step budget |
| `schedule.kind` | `sync` | `overlapped` collects the next round during the training burst; `async` collects continuously |

The defaults are GR00T's geometry on the G1 humanoid.
The geometry follows the policy, not the task: `dsrl.noise_dim` is the policy's action width, and `training.chunk_size` is its action horizon (a shorter chunk is tiled across the horizon and steers less).
Each workflow launcher sets the values for its policy.

`dsrl.noise_action_bound` is the one geometry setting with a measured effect.
At 1.0 the box excludes about a third of the noise the frozen policy samples by itself.
At 2.0 it covers that noise and steers the policy further, which helped on some tasks but also showed up as an early peak followed by a decline.

## Common issues

**Budgets beyond about 40,000 environment steps need a fixed temperature.**
With `algorithm.ent_coef=auto` the actor's entropy never reaches its target, so the temperature keeps falling.
Below about 0.01 the temperature no longer limits the actor: the noise samples leave the range the policy was trained on, and success collapses within a few evaluations.
`algorithm.ent_coef=0.05` fixes the temperature and runs long budgets without the collapse.

**The frozen policy needs the inputs it was trained with.**
DSRL does not correct for wrong inputs to the frozen policy.
Give it the camera frames at the resolution it was trained with, and exactly one source of action normalization statistics (its training dataset or its checkpoint); the wrong statistics decode wrong actions everywhere.
Score the frozen policy alone first: if it fails on tasks where it is known to succeed, its inputs are wrong.

## Results

Measured DSRL results, each next to its base policy, are on the [G1 Factory](../workflows/g1_arena.md#results), [Isaac Lab](../workflows/isaaclab_gr00t.md#results) and [LIBERO](../workflows/libero.md#results) workflow pages.

## Related pages

- Internals (code layout, training loop, noise injection): `docs/dsrl.md`
- Configuration files: `isaac_rl_posttraining/configs/algorithm/dsrl_sac.yaml`, `isaac_rl_posttraining/configs/algorithm/dsrl_ppo.yaml`, `isaac_rl_posttraining/configs/train.yaml`
