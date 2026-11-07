---
name: isaac-rlpt-add-base-model
description: Adds a new frozen base model (a pretrained robot policy such as a VLA) to Isaac RLPT behind the BaseModelPolicy contract, with its registry entry, config group, contract tests and optional PPO/SFT fine-tuning adapter. Use when the user wants to steer or fine-tune a policy that is not GR00T, pi0 or DreamZero, writes a BaseModelPolicy or from_dsrl_config, or hits "unknown base model" or a base-model contract test failure. Not for adding a simulator or robot, adding an RL algorithm, or loading a new checkpoint of an already supported model.
license: CC-BY-4.0 AND Apache-2.0
---

# Add a base model to Isaac RLPT

A base model is the frozen policy a run steers.
It maps observations, plus an optional injected latent, to a chunk of raw actions; nothing in `isaac_rl_posttraining/algorithms/` touches its internals.
`docs/pages/extending/add_a_base_model.md` is the full guide and `docs/adding_a_base_model.md` the developer reference; this skill is the checklist.

## When to Use

- A user wants DSRL, RFS or SAC-edit to steer a policy the repository does not ship.
- A user wants PPO, SFT or IQL to fine-tune such a policy (the optional adapter below).
- `get_base_model('<name>')` raises `unknown base model`, or `tests/unit/test_base_model_policy.py` fails for a new model.

Not for: a new checkpoint of GR00T, pi0 or DreamZero (set `base_model.ckpt`), a new simulator (`isaac-rlpt-add-simulator`), or a new algorithm (`isaac-rlpt-add-algorithm`).

## Workflow

1. **Pick the closest sibling** and mirror it: `isaac_rl_posttraining/policies/gr00t/` (steerable reference), `policies/pi0/` (loaded checkpoint, `policy.py`), `policies/dreamzero/` (served model, `client.py`, with `local_base_model.py` and `server_base_model.py`).
   `MockBaseModelPolicy` in `isaac_rl_posttraining/policies/base.py` is the executable specification.
2. **Create** `isaac_rl_posttraining/policies/<name>/` with `__init__.py` and `base_model.py`.
3. **Implement `BaseModelPolicy`**: `action_horizon`, `action_dim`, `latent_dim` (greater than 0, at most `action_dim`), optional `latent_horizon` and `supports_steering`, `infer(observations, noise) -> {"actions": [B, H, A]}`, and the classmethod `from_dsrl_config(cfg, device, dataset_adapter=None)`.
   Hold the five rules:
   - `noise=None` is the un-steered policy (what `eval.eval_base=true` scores);
   - decode per dimension, with no mixing after the head;
   - clip injected latents to `DEFAULT_LATENT_CLIP` (3) at injection;
   - draw uncovered entries fresh from N(0, 1) on every call;
   - a model that cannot steer sets `supports_steering = False` and raises `NotImplementedError` for a steered call.
   Any other `noise` shape than `None`, `[B, latent_dim]` or `[B, K, latent_dim]` with `K <= latent_horizon` raises `ValueError`.
4. **Keep it embodiment-agnostic**: never import the environment layer or name a simulator; the launcher passes in any dataset adapter.
   Raise on a missing required config field; no `getattr(..., default)`.
5. **Guard optional imports** inside methods so `import isaac_rl_posttraining` works without the model stack; read no tuning from `os.environ`.
6. **Register** in `isaac_rl_posttraining/policies/__init__.py`:

   ```python
   _BASE_MODELS["<name>"] = ("isaac_rl_posttraining.policies.<name>.base_model", "<Name>BaseModelPolicy")
   _OPTIONAL_REQUIREMENTS["<name>"] = ("<optional_dep>",)   # only if construction needs it
   ```

   Out-of-tree code calls `register_base_model("<name>", "pkg.base_model:<Name>BaseModelPolicy", requires=(...))` during its own setup instead.
7. **Config group** `isaac_rl_posttraining/configs/base_model/<name>.yaml`: `base_base_model` in `defaults`, then `name: <name>` and the fields the factory reads.
   Add every new field to `BaseModelConfig` in `isaac_rl_posttraining/configs/base_model_schema.py`; a field missing there is silently dropped.
8. **Tests**: add the name to the `any_base_model` fixture of `tests/unit/test_base_model_policy.py` (skip when the dependency or checkpoint is absent) and teach `_contract_obs()` its observations; add a model test shaped like `tests/unit/test_pi0_base_model.py`.
9. **Docs**: add a row to the base models table in `docs/pages/installation.md`.

### Optional: fine-tuning adapter (PPO, SFT, IQL)

1. Subclass `FlowSDEActorCritic` (`isaac_rl_posttraining/policies/flow_sde.py`) in `policies/<name>/flow_sde.py`, satisfying `TrainablePolicy` in `isaac_rl_posttraining/policies/trainable.py`.
2. Register it as `_TRAINABLE_POLICIES["<name>"]` in `policies/trainable.py`; the key must equal the config group's `base_model.name`.
3. If the simulator entry lists `base_models` (`simulators/isaaclab/runtime.py`), add the name there.
4. Test it as `tests/unit/test_trainable_policies.py` does.

## Verify

```bash
uv run --extra cpu python -c "from isaac_rl_posttraining.policies import get_base_model; print(get_base_model('<name>'))"
uv run --extra cpu pytest tests/unit/test_repo_hygiene.py tests/unit/test_base_model_policy.py -q
ARENA_SERVER_URL=tcp://127.0.0.1:7300 ARENA_EVAL_SERVER_URL=tcp://127.0.0.1:7301 \
ISAAC_RL_POSTTRAINING_DRYRUN=1 uv run --extra cpu python scripts/train.py \
    base_model=<name> base_model.ckpt=/path/to/checkpoint
```

Done when the dry run ends with `plan.checks=... 0 error(s)` and its `base_model:` section shows `name: <name>`.
The default simulator, G1 Factory, holds the model to 35-wide actions in 40-step chunks; a model built for another geometry fails with a spec mismatch, so dry-run it on an entry that accepts it (the guide's LIBERO example).
