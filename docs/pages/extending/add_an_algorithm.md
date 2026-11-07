# Adding an algorithm

Most methods here reuse one shared SAC core: a new method passes the core its own decision variable as the action.
This page is the procedure; the code layering, what a learner owns and what the training harness provides are in [the developer reference](../../adding_an_algorithm.md).

## Intervention types

Three already exist on the frozen policy, each with its own environment wrapper and rollout policy.

| Acts on | `action_space` | Wrapper | Rollout policy | Existing methods |
|---|---|---|---|---|
| the steering latent | `latent` | `LatentSpaceEnv` | steering | `dsrl_sac`, `dsrl_ppo` |
| the latent plus a bounded residual on the decoded actions | `latent_residual` | `ResidualFlowSpaceEnv` | steering | `rfs` |
| a bounded edit of the decoded chunk | `edit` | `ReferenceActionEnv` | reference-edit | `sac_edit` |

The first two wrappers are in `isaac_rl_posttraining/envs/steering_wrappers.py`, the third in `isaac_rl_posttraining/envs/reference_action.py`.
The smallest template to copy is `isaac_rl_posttraining/algorithms/dsrl/`, which holds a config and a small amount of glue code.
`algorithms/rfs/` adds a bounded residual head next to the latent, and `algorithms/sac_edit/` adds an actor conditioned on the chunk it edits.

Methods that train the policy itself use `action_space="full_chunk"`: `ppo` on the on-policy loop, and `sft` and `iql` on the offline loop; a new offline method follows `docs/sft.md`.

## Steps

1. **Create the package** `isaac_rl_posttraining/algorithms/<name>/` with `__init__.py` and `config.py`.
   Add `<name>.py` **only if** the SAC core's hooks (`isaac_rl_posttraining/algorithms/sac/`) do not fit.

2. **Write the config dataclass.** `<Name>Config` subclasses `BaseAlgorithmConfig` (`isaac_rl_posttraining/algorithms/base.py`), or `SACConfig` when you reuse the core, and its `algorithm_class` property names the learner.
   A new learner subclasses `BaseAlgorithm`; what it must provide is in [the developer reference](../../adding_an_algorithm.md#algorithm-responsibilities).

3. **Register it** with one `AlgorithmSpec` in `isaac_rl_posttraining/algorithms/registry.py` (or `register_algorithm(...)` from your own package): its `loop` (`off_policy`, `on_policy` or `offline`), its `action_space` (`latent`, `latent_residual`, `edit` or `full_chunk`), its config class and its launcher, all as lazy `"module:attr"` strings.
   The trainer, the acting policy, the learner networks of the off-policy action spaces and the data-spec check all follow from these fields, and no `cs.store` call is needed.

4. **Add the config group file** `isaac_rl_posttraining/configs/algorithm/<name>.yaml`, shaped like `dsrl_sac.yaml`.
   It lists `base_<name>` first in its `defaults`, then `name: <name>`; the registry declares the rest.
   Every hyperparameter in it must also be a field of your config dataclass (step 2): the YAML is checked against a schema built from it, and an undeclared key fails the dry run.
   Extend `isaac_rl_posttraining/configs/schema.py` only for a new top-level section, as `dsrl:`, `rfs:` and `direct_action:` are.

5. **Pick the launcher.** An off-policy algorithm in an existing action space uses the off-policy launcher and needs nothing more.
   A new loop brings its own launcher (`plan(cfg)` and `build(cfg)`, see `isaac_rl_posttraining/runners/launchers.py`); a new off-policy action space adds its networks in `isaac_rl_posttraining/runners/off_policy_models.py`.
   From Python, `Algorithm.named("<name>", base=..., **hyperparameters)` builds it like the shipped `DSRL(...)`.

6. **Add tests**, in the shape of the existing ones:
   - `tests/unit/test_algorithm_zoo.py`: registry round-trip, a finite `train_step` after collection through the rollout workers, and the full off-policy loop on the mock base model;
   - `tests/unit/test_sac_core.py`: the learner with stub critics and a scripted buffer, so every TD quantity is exact;
   - `tests/unit/test_runner_semantics.py`: the algorithm-to-loop contract;
   - a hyperparameter pin, as `tests/unit/test_recipe_pins.py` pins DSRL and `tests/unit/test_sac_edit_guards.py` pins RFS and SAC-edit.

7. **Benchmark it** under the same protocol as the method you compare it against, with same-session base comparisons (`eval.eval_base=true`) at matched episode counts.
   On G1 Factory that is `example_workflows/g1_arena_posttraining/scripts/zoo_recipes.sh`, and the numbers go in that workflow's `results/results.json`.

## Verification

```bash
uv run pytest tests/unit/test_algorithm_zoo.py tests/unit/test_sac_core.py tests/unit/test_runner_semantics.py -q
uv run python -c "from isaac_rl_posttraining.configs import register_configs; register_configs()"
ARENA_SERVER_URL=tcp://127.0.0.1:7300 ARENA_EVAL_SERVER_URL=tcp://127.0.0.1:7301 \
ISAAC_RL_POSTTRAINING_DRYRUN=1 uv run python scripts/train.py algorithm=<name> base_model=mock simulator=g1_arena
```

This G1 Factory dry run covers the steering, edit and DSRL-PPO families.
`algorithm=ppo` pins an Isaac Lab task, so dry-run it as `ISAAC_RL_POSTTRAINING_DRYRUN=1 uv run python scripts/train.py algorithm=ppo base_model.ckpt=/path/to/checkpoint` instead ([PPO](../algorithms/ppo.md)); an offline method needs a trainable base model, as in `algorithm=sft base_model=gr00t_n17` ([SFT](../algorithms/sft.md)).

Then run the real loop once on a small budget, so collect, train and evaluate all run:

```bash
TRAIN_SIM_DEVICE=cpu ALGO=<name> bash example_workflows/g1_arena_posttraining/scripts/train.sh DrillLift training.num_rollouts=2000
```

Use `train_ppo.sh` for an on-policy method.
The budget must exceed your algorithm's `learning_starts`, or the run collects and evaluates but never trains: SAC-edit starts learning at 4,000 steps, so give it `training.num_rollouts=5000`.

## Rules

- **One top-level training config.** Variants are command-line overrides, never new config files.
- **The learner uses config values as given.** Never override a field's value inside the learner. Invalid combinations raise in `__post_init__` or the constructor, and a key the schema does not declare fails the dry run with the settings it does have.
- **Reuse the rollout machinery**: `isaac_rl_posttraining/runtimes/steering_policy.py`, `isaac_rl_posttraining/runtimes/edit_policy.py`, `isaac_rl_posttraining/rollouts/steering_step.py`, `isaac_rl_posttraining/training/off_policy.py`, `isaac_rl_posttraining/training/evaluation.py`.
- **Nothing in the SAC core imports a base model.**
- **The learner draws batches through samplers** and never touches the buffer layout.
- **No tuning values from `os.environ`** in algorithm code (`tests/unit/test_repo_hygiene.py`).
- **Add pins, never relax them.**
- **Give each supported mode its own explicit branch**; the `else` branch raises.

## Related pages

- Developer reference: [adding_an_algorithm.md](../../adding_an_algorithm.md), `docs/dsrl.md`, `docs/ppo.md`, `docs/sft.md`, `docs/data_model.md`
- What the existing methods do: [DSRL](../algorithms/dsrl.md), [RFS](../algorithms/rfs.md), [SAC-edit](../algorithms/sac_edit.md), [PPO](../algorithms/ppo.md), [SFT](../algorithms/sft.md), [IQL](../algorithms/iql.md)
- How to report what it does: [Evaluation](../concepts.md#evaluation)
