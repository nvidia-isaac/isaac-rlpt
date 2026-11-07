# RFS

Residual Flow Steering, `algorithm=rfs`, steers a frozen policy's noise exactly as [DSRL](dsrl.md) does, and also adds a small, bounded correction to the actions the policy decodes.
DSRL can only choose among actions the frozen policy could already produce.
The correction, called a residual, lets RFS reach actions just outside that set.
A residual moves the actions outside the policy's own behaviour, so its bound is kept small.

## How it works

1. **Steering.** The actor emits a noise sample, as in DSRL, and the frozen policy decodes it into an action chunk.
2. **Correction.** The actor also emits a residual, bounded by `rfs.residual_action_bound`, which is added to the decoded actions.
3. **Execution.** The corrected actions run in the simulator, with DSRL's reward and execution length.
4. **Optimization.** The same SAC learner as DSRL trains the actor on the joint action (noise sample and residual).

Two variants change what the residual sees.
`rfs_pc`, the proposal-conditioned variant, shows the actor the action the frozen policy proposed before it chooses the residual (`rfs.proposal_conditioned_residual=true`, `rfs.reference_action_conditioning=true`).
`rfs_nb2` widens the noise bound from DSRL's default of 1.0 to 2.0.

## When to use it

- DSRL reaches the frozen policy's level, and the behaviour the task needs seems to lie just outside what the policy produces.
- There is budget to try more than one residual bound, because the right bound depends on the robot's action units.
- Try `rfs_pc` first among the variants; it was the most consistent across the measured tasks.

## Running it

The general form of the command matches [DSRL](dsrl.md#running-it), with `algorithm=rfs`:

```bash
uv run python scripts/train.py algorithm=rfs \
    simulator=<simulator> env.env_name=<provider>/<task> \
    base_model=<policy> base_model.ckpt=<checkpoint> rfs.residual_action_bound=<bound>
```

Workflow examples:

| Workflow | Example |
|---|---|
| [G1 Factory](../workflows/g1_arena.md#training) | `rfs`, `rfs_pc` and `rfs_nb2` on the G1 humanoid tasks, under DSRL's 40,000-step budget |
| [Isaac Lab](../workflows/isaaclab_gr00t.md#training) | `--algorithm rfs` or `rfs_pc` on Franka cube stacking |
| [LIBERO](../workflows/libero.md#training) | `ALGO=rfs` on a LIBERO task |

## Evaluation

Evaluation works as for [DSRL](dsrl.md#evaluation): the actor is sampled, and `eval.eval_base=true` scores the frozen policy through the same evaluator.
Adopt a checkpoint only after a re-score on 100 episodes or more beats the frozen policy scored in the same session.

## Configuration

The residual is set in the `rfs:` section of `isaac_rl_posttraining/configs/train.yaml`:

| Key | Default | Meaning |
|---|---|---|
| `rfs.residual_action_bound` | 0.01 | half-width of the residual, **in the policy's raw action units** |
| `rfs.proposal_conditioned_residual` | `false` | condition the residual on the frozen policy's proposal |
| `rfs.reference_action_conditioning` | `false` | show the proposal to the actor |
| `rfs.reference_action_key` | `reference_action` | observation key the proposal arrives on |
| `rfs.initial_residual_log_std` | -5.0 | initial log standard deviation of the residual |
| `rfs.residual_exploration_std` | 0.0 | extra exploration noise on the residual |

The steering half uses DSRL's geometry keys ([DSRL](dsrl.md#configuration)).

Learning (`isaac_rl_posttraining/configs/algorithm/rfs.yaml`) matches DSRL except for two settings:

| Key | DSRL | RFS |
|---|---|---|
| `algorithm.critic_target_reduction` | `mean` | `min` |
| `algorithm.pooled_critic_features` | `true` | `false` |

Comparing RFS and DSRL at their defaults therefore compares more than the residual; that two-setting ablation has not been run.

## Common issues

**Residual bound.**
It is in raw action units, so the right value depends on what an action means for the robot.
The default, 0.01, suits joint targets in radians, where it is a fraction of a percent of a joint's range.
For other action spaces, start at about one percent of the action's actual range.
When the bound was too large, the residual alone dropped success sharply in the measured tasks; check the bound first when an RFS run underperforms.

**Keep DSRL's step budget.**
RFS has DSRL's temperature decay ([DSRL](dsrl.md#common-issues)), and in longer runs it collapsed in the same way.

## Results

Measured RFS results, next to DSRL and the base policy, are on the [Isaac Lab](../workflows/isaaclab_gr00t.md#results) and [LIBERO](../workflows/libero.md#results) workflow pages.

## Related pages

- The steering half: [DSRL](dsrl.md)
- The other method that edits decoded actions: [SAC-edit](sac_edit.md)
- How the method is wired in: `docs/adding_an_algorithm.md`
- Configuration file: `isaac_rl_posttraining/configs/algorithm/rfs.yaml`
