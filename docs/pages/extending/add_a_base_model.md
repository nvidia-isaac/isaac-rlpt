# Add a base model

A **base model** is the frozen policy a run steers: GR00T, pi0, DreamZero, or your own.
Adding one takes a class with a factory, one line in a registry table and one config file; nothing else in the core changes.
How the registry and the run builder use these pieces is in [the developer reference](../../adding_a_base_model.md).

## Requirements

- The library installed as in [installation](../installation.md), so `uv run --no-sync pytest` works from the repository root.
- Your model's own package importable in the same environment, and a checkpoint for the integration test.

## Contract

Your class subclasses `BaseModelPolicy` (`isaac_rl_posttraining/policies/base.py`).

| Member | Meaning |
|---|---|
| `action_horizon` | `H`, environment steps produced per `infer` call |
| `action_dim` | `A`, the width of one decoded action step |
| `latent_dim` | steerable latent dimensions per step; greater than 0 and at most `action_dim` |
| `latent_horizon` | the maximum number of per-step latents accepted. Defaults to `action_horizon`; override only when the denoising window is longer than the decoded chunk |
| `supports_steering` | defaults to `True` |
| `infer(observations, noise)` | one inference call, optionally steered |
| `from_dsrl_config(cls, cfg, device, dataset_adapter=None)` | the classmethod factory the run builder calls after resolving `base_model.name` |

`observations` is a base-model-specific dict with a leading batch dimension.
`infer` returns `{"actions": np.ndarray}` of shape `[B, H, A]`.
The `noise` argument takes three shapes and raises `ValueError` for anything else:

- `None`: native, the un-steered policy;
- `[B, latent_dim]`: one latent broadcast across every step;
- `[B, K, latent_dim]` with `K <= latent_horizon`: per-step latents.

Five rules make base models interchangeable:

1. **`noise=None` must be the un-steered policy**, because that is what `eval.eval_base=true` scores.
2. **Decode per dimension.** Each action dimension comes from its own head output, with no mixing, so latent dimensions stay aligned with action dimensions.
3. **Clip injected latents** to plus or minus 3 (`DEFAULT_LATENT_CLIP`) at injection time.
4. **Draw padding noise fresh.** Entries the caller does not cover (steps beyond `K`, dimensions beyond `latent_dim`) come from a standard normal on every call, never zeroed and never cached.
5. **Declare when steering is unsupported.** A base model that cannot accept an external latent sets `supports_steering=False` and raises `NotImplementedError` for a steered call; the entry point then restricts it to evaluating the frozen policy.

Document any deviation in the class docstring; pi0's tiling and padded latent width are the precedent.
`MockBaseModelPolicy`, in the same file, is the executable specification: a test can recover any injected value from its output, and it runs the whole pipeline with no GPU.

## Steps

1. **Create the package** `isaac_rl_posttraining/policies/<name>/` with `__init__.py` and `base_model.py`.
   Add `policy.py` for a loaded checkpoint (as `isaac_rl_posttraining/policies/pi0/policy.py` does) or `client.py` for a served model (as `isaac_rl_posttraining/policies/dreamzero/client.py` does).

2. **Implement the class and its factory** in `base_model.py`:

   ```python
   from isaac_rl_posttraining.policies.base import BaseModelPolicy

   class MyBaseModelPolicy(BaseModelPolicy):
       @classmethod
       def from_dsrl_config(cls, cfg, device, dataset_adapter=None):
           # Read base_model.<field> from the composed config; raise on a missing one.
           ...

       @property
       def action_horizon(self) -> int: ...
       @property
       def action_dim(self) -> int: ...
       @property
       def latent_dim(self) -> int: ...

       def infer(self, observations: dict, noise=None) -> dict:
           # validate noise shapes -> ValueError; clip at injection;
           # fresh N(0, 1) for everything not steered; decode per dimension.
           return {"actions": actions}  # float [B, H, A]
   ```

   Keep tiling, clipping and the rest of the injection mechanics with the model they steer; `LatentInjectionConfig` in `isaac_rl_posttraining/policies/gr00t/model.py` is the worked example.
   A base model never imports the environment layer or names a simulator; the launcher passes in any dataset adapter.
   A missing required field raises; no `getattr(..., default)` masks it.

3. **Guard the optional imports.** `import isaac_rl_posttraining` must succeed without your model stack, so put heavy imports inside the methods that need them.
   Every knob is a constructor argument or a config field, never an environment variable.
   `tests/unit/test_repo_hygiene.py` enforces both.

4. **Register it** with one line in `isaac_rl_posttraining/policies/__init__.py`:

   ```python
   _BASE_MODELS["myname"] = ("isaac_rl_posttraining.policies.myname.base_model", "MyBaseModelPolicy")
   _OPTIONAL_REQUIREMENTS["myname"] = ("my_optional_dep",)   # when it needs one
   ```

   Code outside the repository calls `register_base_model("myname", "my_pkg.base_model:MyBaseModelPolicy", requires=("my_optional_dep",))` during its own setup instead.

5. **Pass the contract tests.** Add your name to the `any_base_model` fixture in `tests/unit/test_base_model_policy.py`, with a builder that skips when the dependency or the checkpoint is absent, and teach `_contract_obs()` your observation format.
   Then add a base-model-specific test shaped like `tests/unit/test_gr00t_base_model.py` or `tests/unit/test_pi0_base_model.py`.

6. **Add the config group file** `isaac_rl_posttraining/configs/base_model/<name>.yaml`: `base_base_model` in its `defaults`, then `name: <name>` and the checkpoint and geometry fields your factory reads.
   Leave unused fields `null`.
   Add the same fields to `BaseModelConfig` in `isaac_rl_posttraining/configs/base_model_schema.py`; it is the union of every base model's fields, and a field missing from it is silently dropped.

7. **Add a row** to the base models table in [installation](../installation.md).

## Fine-tuning adapter (optional)

PPO fine-tuning (`algorithm=ppo`) and offline fine-tuning (`algorithm=sft`, `algorithm=iql`) need a trainable adapter, a second, separate implementation.

1. Write `isaac_rl_posttraining/policies/<name>/flow_sde.py`: a subclass of `FlowSDEActorCritic` (`isaac_rl_posttraining/policies/flow_sde.py`) that satisfies the `TrainablePolicy` protocol of `isaac_rl_posttraining/policies/trainable.py`.
   The members it supplies are listed in [the developer reference](../../adding_a_base_model.md#trainable-adapters).

2. Register it in `isaac_rl_posttraining/policies/trainable.py`:

   ```python
   _TRAINABLE_POLICIES["myname"] = ("isaac_rl_posttraining.policies.myname.flow_sde", "MyFlowSDEActorCritic")
   ```

   The key must equal the `base_model.name` of your config group, so one `base_model=` selection serves both steering and fine-tuning.

3. Add the adapter's required fields to your config group and to `BaseModelConfig`.

4. If the simulator entry you fine-tune on lists its `base_models` (`IsaacLab` in `simulators/isaaclab/runtime.py` does), add your name there; an entry refuses any base model it does not list.

5. Test it as `tests/unit/test_trainable_policies.py` does: the learner's forward reproduces the behavior log-probability from a stored chain, gradients reach the head and not the frozen part, and the deterministic decode matches the model's own sampler.

Trainable adapters exist today for `gr00t_n15`, `gr00t_n17` and `pi0`.

## Verify

Run from the repository root:

```bash
uv run python -c "from isaac_rl_posttraining.policies import get_base_model; print(get_base_model('<name>'))"
uv run pytest tests/unit/test_repo_hygiene.py -q
uv run pytest tests/unit/test_base_model_policy.py -q
uv run pytest tests/unit/test_trainable_policies.py -q          # only if you added the trainable adapter

ARENA_SERVER_URL=tcp://127.0.0.1:7300 ARENA_EVAL_SERVER_URL=tcp://127.0.0.1:7301 \
ISAAC_RL_POSTTRAINING_DRYRUN=1 uv run python scripts/train.py \
    base_model=<name> base_model.ckpt=/path/to/checkpoint
```

The dry run prints the `plan.*` lines and the resolved configuration, whose `base_model:` section shows `name: <name>`, without loading the model; there is no `plan.base_model=` line.
It should end with `plan.checks=... 0 error(s)`.
The default simulator is G1 Factory, which needs the two Arena addresses and holds your base model to G1's geometry (35-wide actions, 40-step chunks).
A base model built for another geometry fails with a spec mismatch; dry-run it on a simulator entry that accepts it.
For example, `dreamzero_local` on the LIBERO entry built for DreamZero:

```bash
ISAAC_RL_POSTTRAINING_DRYRUN=1 uv run python scripts/train.py base_model=dreamzero_local \
    simulator=libero_spatial env.env_name=libero_dreamzero/libero_spatial/8 \
    'env.max_episode_steps=${simulator.gr00t_eval.max_episode_steps}' training.chunk_size=null
```

The first segment of `env.env_name` names the simulator entry (`libero`, `libero_pi0` or `libero_dreamzero` on LIBERO), and each entry lists the base models it accepts.
`training.chunk_size=null` lets the base model's action horizon decide the chunk; a chunk longer than the horizon fails the check.
