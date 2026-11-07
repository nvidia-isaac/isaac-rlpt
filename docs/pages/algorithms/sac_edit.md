# SAC-edit

SAC-edit, `algorithm=sac_edit`, lets the frozen policy propose an action chunk in its ordinary way, and trains a small agent that looks at the proposal and makes a small, bounded edit to it.
Unlike [DSRL](dsrl.md) and [RFS](rfs.md), it does not choose the policy's noise; it works directly on the decoded actions.
Unlike the default RFS residual, the edit is chosen after seeing the proposal it corrects, and the learning algorithm (Soft Actor-Critic, SAC) explores and scores the edit itself.
The policy's weights never change.
Read [Common issues](#common-issues) before you interpret a run.

## How it works

1. **Proposal.** The frozen policy decodes an action chunk from its own noise.
2. **Edit.** The actor sees the proposal and emits an edit bounded by `direct_action.actor_edit_scale`.
3. **Execution.** The edited chunk runs in the simulator.
4. **Optimization.** SAC trains the actor and two critics on the edit, with an L2 pull of the edit back toward the proposal.

## When to use it

- The frozen policy's proposal is close to right and what is needed is a small, state-dependent adjustment of it.

## Running it

The general form of the command matches [DSRL](dsrl.md#running-it), with `algorithm=sac_edit`:

```bash
uv run python scripts/train.py algorithm=sac_edit \
    simulator=<simulator> env.env_name=<provider>/<task> \
    base_model=<policy> base_model.ckpt=<checkpoint>
```

Workflow examples:

| Workflow | Example |
|---|---|
| [G1 Factory](../workflows/g1_arena.md#training) | SAC-edit on the G1 humanoid tasks, under DSRL's 40,000-step budget |
| [Isaac Lab](../workflows/isaaclab_gr00t.md#training) | `--algorithm sac_edit` on Franka cube stacking |
| [LIBERO](../workflows/libero.md#training) | `ALGO=sac_edit` on a LIBERO task |

## Evaluation

Evaluation works as for [DSRL](dsrl.md#evaluation): the actor is sampled every `eval.eval_freq` environment steps, and `eval.eval_base=true` scores the frozen policy through the same evaluator.
Adopt a checkpoint only after a re-score on 100 episodes or more beats the frozen policy scored in the same session.

## Configuration

The edit geometry (`direct_action:` section of `isaac_rl_posttraining/configs/train.yaml`):

| Key | Default | Meaning |
|---|---|---|
| `direct_action.actor_edit_scale` | 0.02 | the edit bound in normalized action units. `null` means an unbounded edit |
| `direct_action.action_horizon` | `null` | `null` uses the frozen policy's decoded chunk length |
| `direct_action.action_dim` | `null` | `null` uses the frozen policy's action width |
| `direct_action.action_bound` | 1.0 | bound of the underlying action box |
| `direct_action.relative_action_scale` | 0.025 | scale of the relative-action parameterization |
| `direct_action.reference_action_transform` | `identity` | transform applied to the proposal before conditioning |
| `direct_action.reference_image_transform_mode` | `eval` | frozen-policy image preprocessing during collection |
| `direct_action.include_sim_state` | `false` | include the simulator state in the actor's observation |
| `direct_action.actor_reference_centered_full_support` | `false` | center the edit distribution on the proposal with full support |

`sac_edit.yaml` also sets `training.chunk_size=null`, because there is no per-flow-step latent to size.

Learning (`isaac_rl_posttraining/configs/algorithm/sac_edit.yaml`) differs from DSRL in several places:

| Key | DSRL | SAC-edit |
|---|---|---|
| `algorithm.learning_starts` | 1200 | 4000 |
| `algorithm.gamma` | 0.995 | 0.997 |
| `algorithm.tau` | 0.005 | 0.125 |
| `algorithm.batch_size` | 64 | 2048 |
| `algorithm.num_critics` | 4 | 2 |
| `algorithm.num_updates_per_train_step` | 10 | 1 |
| `algorithm.actor_optimizer_lr` | 5e-5 | 3.0e-4 |
| `algorithm.critic_optimizer_lr` | 1e-4 | 3.0e-4 |
| `algorithm.use_amp` | not used | `false`, that is fp32 |
| `algorithm.clip_grad_norm` | unset | 1.0 |
| `algorithm.reference_action_dropout` | not used | 0.0; **must stay 0** with a bounded or proposal-centered edit |
| `algorithm.reference_action_l2_coef` | not used | 0.1, the pull of the edit back toward the proposal |

## Common issues

**Do not quote a SAC-edit number measured before the entropy fix.**
The edit's log-density used to carry a constant term that made the SAC entropy target unreachable, so the auto-tuned temperature grew without bound: in fp32 it grew from 1 to about 6e5 over a 40,000-step run, against 0.008 for DSRL and RFS.
The actor degenerated to near-uniform noise inside its bound, and in mixed precision the entropy gradient overflowed at 35,000 to 36,000 steps.
The density is now measured over the normalized edit, which removes the term.

**Keep the safe defaults.**
An unbounded edit, learning from 20,000 steps and mixed precision together stayed at zero percent success on every measured task and crashed with a NaN on one.
The shipped defaults are fp32, a 0.02 edit bound, learning from 4,000 steps, and gradient clipping at 1.0, and the actor raises a `FloatingPointError` naming the offending tensor instead of training on NaNs.
If you change `use_amp`, `actor_edit_scale` or `learning_starts`, change one at a time and watch the temperature.

**Give a short run enough budget.**
Learning starts at 4,000 steps, so a smoke run needs `training.num_rollouts` above that, for example 5000.

## Results

Measured SAC-edit results, next to DSRL and the base policy, are on the [Isaac Lab](../workflows/isaaclab_gr00t.md#results) workflow page.

## Related pages

- The steering alternatives: [DSRL](dsrl.md), [RFS](rfs.md)
- How the method is wired in: `docs/adding_an_algorithm.md`
- Configuration file: `isaac_rl_posttraining/configs/algorithm/sac_edit.yaml`
