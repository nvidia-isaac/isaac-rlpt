---
name: isaac-rlpt-add-algorithm
description: Adds a new RL post-training algorithm to Isaac RLPT, usually a thin variant over the shared SAC core in one of the latent, latent_residual or edit action spaces, registered with one AlgorithmSpec plus a config dataclass, a config group and tests. Use when the user wants a new steering, residual, edit, on-policy or offline method selectable as algorithm=<name>, writes an AlgorithmSpec or BaseAlgorithmConfig, or hits an undeclared-key or registry error for a new algorithm. Not for tuning an existing algorithm's hyperparameters, adding a simulator, or adding a base model.
license: CC-BY-4.0 AND Apache-2.0
---

# Add an algorithm to Isaac RLPT

Most methods here are small variations on one SAC core: the method hands the core its own decision variable as the action.
The training loop owns collection, evaluation, checkpointing and buffers; a learner owns only its learning state.
`docs/pages/extending/add_an_algorithm.md` is the full guide and `docs/adding_an_algorithm.md` the developer reference; this skill is the checklist.

## When to Use

- A user wants a new method selectable as `algorithm=<name>` on the shipped simulators.
- A user is writing an `AlgorithmSpec`, a `BaseAlgorithmConfig` or `SACConfig` subclass, or a new launcher.
- A new algorithm's dry run fails on an undeclared config key or an unknown algorithm name.

Not for: changing an existing method's hyperparameters (pass overrides on the command line), a new simulator (`isaac-rlpt-add-simulator`), or a new base model (`isaac-rlpt-add-base-model`).

## Workflow

1. **Choose the action space**; joining an existing one is much less work than adding a fourth:

   | `action_space` | Acts on | Template |
   |---|---|---|
   | `latent` | the steering latent | `algorithms/dsrl/` (`dsrl_sac`, `dsrl_ppo`) |
   | `latent_residual` | the latent plus a bounded residual on decoded actions | `algorithms/rfs/` |
   | `edit` | a bounded edit of the decoded chunk | `algorithms/sac_edit/` |
   | `full_chunk` | the policy itself | `ppo` (on-policy), `sft` and `iql` (offline) |

2. **Create** `isaac_rl_posttraining/algorithms/<name>/` with `__init__.py` and `config.py`; add `<name>.py` only if the SAC core's hooks (`algorithms/sac/`) do not fit.
3. **Config dataclass**: `<Name>Config` subclasses `BaseAlgorithmConfig` (`algorithms/base.py`), or `SACConfig` to reuse the core, and its `algorithm_class` property names the learner.
   A new learner subclasses `BaseAlgorithm`, registers every stateful piece in `component_names`, and makes `train_step` a no-op below `learning_starts`.
4. **Register** one `AlgorithmSpec` in `isaac_rl_posttraining/algorithms/registry.py` (or `register_algorithm(...)` from an outside package) with its `loop` (`off_policy`, `on_policy`, `offline`), `action_space`, config class and launcher as lazy `"module:attr"` strings.
   The trainer, rollout policy, learner networks and data-spec check follow from these fields; no `cs.store` call is needed.
5. **Config group** `isaac_rl_posttraining/configs/algorithm/<name>.yaml`, shaped like `dsrl_sac.yaml`: `base_<name>` first in `defaults`, then `name: <name>`.
   Every key must be a field of the dataclass; an undeclared key fails the dry run.
   Extend `isaac_rl_posttraining/configs/schema.py` only for a new top-level section.
6. **Launcher**: an off-policy method in an existing action space needs nothing more.
   A new loop brings `plan(cfg)` and `build(cfg)` (`isaac_rl_posttraining/runners/launchers.py`); a new off-policy action space adds its networks in `isaac_rl_posttraining/runners/off_policy_models.py`.
7. **Tests**: extend `tests/unit/test_algorithm_zoo.py`, `tests/unit/test_sac_core.py` and `tests/unit/test_runner_semantics.py`, and pin the canonical hyperparameters as `tests/unit/test_recipe_pins.py` does (add pins, never relax them).
8. **Benchmark** under the protocol of the method it is compared with: same-session base scores (`eval.eval_base=true`) and matched episode counts; record results in the workflow's `results/results.json`.

Rules: one top-level training config (variants are overrides); the config is the source of truth (never override a field inside the learner); reuse the rollout machinery; nothing in the SAC core imports a base model; no tuning from `os.environ`; each real mode gets its own branch and `else` raises.

## Verify

```bash
uv run --extra cpu pytest tests/unit/test_algorithm_zoo.py tests/unit/test_sac_core.py tests/unit/test_runner_semantics.py -q
uv run --extra cpu python -c "from isaac_rl_posttraining.configs import register_configs; register_configs()"
ARENA_SERVER_URL=tcp://127.0.0.1:7300 ARENA_EVAL_SERVER_URL=tcp://127.0.0.1:7301 \
ISAAC_RL_POSTTRAINING_DRYRUN=1 uv run --extra cpu python scripts/train.py algorithm=<name> base_model=mock simulator=g1_arena
```

Done when the dry run prints `plan.algorithm=<name>` and ends with `plan.checks=... 0 error(s)`.
Then train once on a small budget above `learning_starts` so collect, train and evaluate all run, for example `TRAIN_SIM_DEVICE=cpu ALGO=<name> bash example_workflows/g1_arena_posttraining/scripts/train.sh DrillLift training.num_rollouts=2000`.
