# Add a simulator

Bringing a new simulator, task suite or real robot into the repository means writing one environment class, one simulator entry and one config file, and registering them.
Every existing algorithm then works on it, because none of them imports a simulator.
The shipped integrations under `simulators/` are the worked examples; `simulators/README.md` maps them.
`simulators/_template/` is a stand-in robot that passes the dry run and trains on CPU with no simulator installed; copy it to start.

## Requirements

- The library installed as in [installation](../installation.md), so `uv run --no-sync pytest` works from the repository root.
- Your simulator's Python package, in this environment for an in-process simulator or in its own environment for a remote one.

## Execution models

| Model | Meaning | Copy from |
|---|---|---|
| **in-process** | the simulator steps inside the learner process; the client keeps the episode-cap bookkeeping and mirrors the chunked-action semantics of `LiberoEnv.step` | `simulators/libero/` |
| **in-process, many scenes** | one process, N scenes cloned inside it | `simulators/isaaclab/` |
| **remote** | the simulator runs behind an environment server speaking the wire schema of `isaac_rl_posttraining/envs/transport.py`, with a thin client like `ArenaEnv` | `simulators/g1_arena/` (server: `simulators/g1_arena/env_server.py`) |
| **remote, natively batched** | a dedicated server that steps a whole batch | `simulators/v2d/` (server: `simulators/v2d/server.py`) |

Choose remote when the simulator needs its own Python environment or its own machine.

**A real robot uses the same remote interface.**
It is a remote provider: a bridge process on the robot's network speaking the transport's wire schema (`RESET`, `STEP`, `SPEC` and `CLOSE` messages, with `need_obs=False` to skip observation capture mid-chunk), a thin client class and a config entry.
The transport addresses endpoints by URI (`tcp://host:port`), and its `SPEC` handshake is self-describing, so a client can size its models against an endpoint it cannot introspect.

## Environment contract

Your environment is a `gym.Env` whose observations, actions and `info` follow one contract, pinned by the steering wrappers (`isaac_rl_posttraining/envs/steering_wrappers.py`) and the rollout workers.

- **Observations** are a dict with a proprioceptive `"state"` vector (`float32 [state_dim]`), one `uint8 [H, W, 3]` entry per camera, and a `"prompt"` string where the simulator carries language.
- **Actions** are a flat `float32 [action_dim]` box (35-dim whole-body joint targets for `g1_arena`, 7-dim end-effector deltas for `libero`). A `[chunk, action_dim]` action executes sequentially, stops early on termination, and consumes one episode-cap step per row.
- **Outcome**: `info["success"]` and `info["is_success"]` carry the task's success predicate on every step, and `terminated` reflects it. Episodes that never succeed end as truncations at the `max_episode_steps` cap.
- **Key mapping**: the environment exposes `steering_key_mapping`, a dict naming which observation entries are the state, the language embedding (`None` for single-task environments) and the image list (`ARENA_KEY_MAPPING` in `simulators/g1_arena/env.py` is an example).

`ArenaEnv` in `simulators/g1_arena/env.py` and `LiberoEnv` in `simulators/libero/env.py` are the worked examples.

## Steps

1. **Create the package** `simulators/<name>/` with `__init__.py`, `env.py`, `runtime.py`, `configs/__init__.py` and `configs/simulator/<name>.yaml`; copying `simulators/_template/` gives the first three.
   A remote simulator also gets an environment server; a GR00T-style state and action layout also gets `embodiment.py`.

2. **Write the environment** in `simulators/<name>/env.py`, adapting the simulator to the contract above.

3. **Guard the optional imports.** `import isaac_rl_posttraining` must succeed without your simulator stack: import it inside the methods that need it, fail with a message pointing at your README, and declare the requirement in the registry (step 5).
   Every knob is a constructor argument or a config field, never read from `os.environ`.
   `tests/unit/test_repo_hygiene.py` enforces both.

4. **Write the simulator entry** in `simulators/<name>/runtime.py`: a subclass of `SimulatorEntry` (`isaac_rl_posttraining/envs/simulator_entry.py`).
   Its instance travels to every rollout worker, which calls `make_environment` there.

   ```python
   from isaac_rl_posttraining.envs.simulator_entry import SimulatorEntry
   from simulators.myname.env import MyEnv


   class MyName(SimulatorEntry):
       name = "myname"

       def make_environment(self, spec, phase, context):
           # Runs on the rollout worker. `spec` describes the phase's environments
           # (env_name, num_envs, max_episode_steps, seed, server_url, ...) and
           # `context.environment_indices` says which of them this worker owns.
           return MyEnv(task=spec.env_name.split("/", 1)[1], max_episode_steps=spec.max_episode_steps)
   ```

   `name` and `make_environment` are the only required members.
   The defaults: one environment per worker, evaluation on the training workers at episode boundaries (`evaluation = "shared"`), any base model, no server addresses, no decode cap, one `step` per action, and observation keys derived from `observation_spec` when declared, else `{"state": "state", "image": ["image"], "language": None}`.
   A missing required config value raises from `make_environment`, naming the key to set; no `dict.get(key, default)` masks it.
   Override what differs:

   | Override | When | In-tree example |
   |---|---|---|
   | `evaluation = "dedicated"` | a second persistent server evaluates | `G1Arena`, `V2D` |
   | `evaluation = "fresh_workers"` | rebuilt, independently seeded simulators evaluate | `Libero` |
   | `worker_layout(num_envs, num_servers=1)` | one server hosts many environments behind one client; return `WorkerLayout(workers=num_servers, batched=num_envs > 1)` and a batched environment from `make_environment`. `num_servers` counts the comma-separated addresses in the server variable; the default layout rejects more than one | `G1Arena` |
   | `server_url_variables = ("MY_SERVER_URL", "MY_EVAL_SERVER_URL")` | a remote simulator whose addresses the launcher exports (`spec.server_url`) | `G1Arena`, `V2D` |
   | `base_models = ("gr00t",)` | the environment follows one base model's conventions | `Libero`, `V2D`, `IsaacLab` |
   | `steering_key_mapping(spec)` | the derived keys do not fit: several state-like arrays, a language embedding, cameras keyed per task | `IsaacLab` |
   | `supports_chunked_step = True`, `decode_batch_size = 16` | a chunk-executing server behind a shared policy | `G1Arena` |
   | `make_scenes(spec, context)`, `instruction(spec)` | flow-SDE PPO (`algorithm=ppo`), which plays all of a worker's scenes in one batched simulation | `IsaacLab`, `G1Arena` |
   | `observation_spec(spec)`, `action_spec(spec)` | the launch should check the policy against a typed declaration ([data model](../../data_model.md#specs-and-checks)) | `G1Arena`, `Libero`, `IsaacLab` |
   | `sized_from_declaration = True` | environments are slow to start, so the learner is sized from the declared specs without building one | `IsaacLab` |
   | `simulates_on_gpu = True` | each rollout worker simulates or renders on a GPU, so the placement gives every worker one | `Libero`, `IsaacLab` |

   A malformed declaration is rejected when the class is defined: an empty `name`, an `evaluation` mode other than `shared`, `dedicated` or `fresh_workers`, or one server variable instead of two.
   `skills/isaac-rlpt-add-simulator/SKILL.md` carries the same table with the reason behind each default.

5. **Register it** in `isaac_rl_posttraining/envs/providers.py`, keyed by the simulator segment of `env.env_name` (`"<simulator>/<task>"`):

   ```python
   _PROVIDERS["myname"] = ("simulators.myname.env", "MyEnv")
   _OPTIONAL_REQUIREMENTS["myname"] = ("my_sim_package",)
   _ENV_BUILDERS["myname"] = "simulators.myname.env:build_dsrl_env"     # optional
   _SIMULATOR_ENTRIES["myname"] = "simulators.myname.runtime:MyName"
   ```

   All of them import lazily, and segments match exactly, so `libero_pi0` and `libero` never shadow each other.
   Register an env builder only when sizing the learner's models needs more than `make_environment` behind the standard wrappers.
   A GR00T-backed simulator with its own state and action layout also registers a dataset adapter in `_ADAPTERS` of `isaac_rl_posttraining/envs/embodiments/__init__.py`.

   Code outside the repository registers at run time with `register_provider(...)`, `register_env_builder(...)`, `register_simulator_entry(...)` and `register_dataset_adapter(...)` during its own setup, names its entry in the config as `+simulator.entry=my_package.runtime:MyName`, or registers the class with `register_simulator_entry("myname", MyName)` and calls `launch(...)` from `isaac_rl_posttraining/train.py`.

6. **Add the config group entry** at `simulators/<name>/configs/simulator/<name>.yaml`, carrying `provider:`, the task list and the episode caps: a per-task map, or one suite-wide cap where the benchmark defines one.
   This file is the one place the task list and the caps are defined.
   In the shipped integrations a failed episode ends only at the cap, so set each cap above the longest demonstration, or it cuts successes short and lowers the success rate.
   Then wire it in:
   - add `pkg://simulators.<name>.configs` to `hydra.searchpath` in `isaac_rl_posttraining/configs/train.yaml` so `simulator=<name>` resolves;
   - add `"simulators.<name>*"` under `[tool.setuptools.packages.find] include` and `"simulators.<name>" = ["configs/**/*.yaml"]` under `[tool.setuptools.package-data]` in `pyproject.toml`; a built wheel leaves out a simulator missing from them.

7. **Write `simulators/<name>/README.md`**: what the package provides, which workflow README carries the commands, the setup that workflow does not cover, the environment variables, config keys, task names and caps, the observation and action shapes, where the demonstrations and the policy's normalization statistics come from (wrong statistics fail silently), the reference results with their evaluation protocol, and the tests.
   Add a row to the table in `simulators/README.md`.

8. **Test it**, in three layers under `simulators/<name>/tests/`:
   1. registry and config tests, needing no simulator;
   2. contract tests against a fake simulator (observation mapping, success plumbing, chunk semantics, close), as `simulators/isaaclab/tests/` does without Isaac Sim;
   3. one integration test that boots a real task, marked `arena`, `libero` or `gpu`; `tests/conftest.py` turns the missing dependency into a visible skip.

## Verify

Run from the repository root:

```bash
uv run pytest tests/unit/test_repo_hygiene.py -q
uv run pytest tests/unit/test_doc_references.py tests/unit/test_simulators.py tests/unit/test_simulator_entry.py -q
uv run pytest simulators/<name>/tests -m "not gpu and not arena and not libero" -q
ISAAC_RL_POSTTRAINING_DRYRUN=1 uv run python scripts/train.py simulator=<config> base_model=mock env.env_name=<entry>/<task> \
    'env.max_episode_steps=${simulator.max_episode_steps.<task>}'
```

`simulator=` names the config file from step 6, and the first segment of `env.env_name` selects the entry registered in step 5; for a new simulator both are usually `<name>`, but the shipped ones differ (`simulator=g1_arena` with `env.env_name=arena/...`).
The entry's `name` only labels the run and its messages (`G1Arena` is named `g1_arena` and serves `arena/...`), so keep it equal to the segment unless one entry serves several.
`env.max_episode_steps` is read from the selected simulator group's cap for the task; a task that group does not list (an out-of-tree entry under the default `simulator=g1_arena`) stops the dry run with `env.max_episode_steps is unset and task '<task>' of env.env_name=... names none of the simulator's tasks`.
Set it explicitly (`env.max_episode_steps=50`) or point it at your cap (`${simulator.max_episode_steps.<task>}` for a per-task map, or the suite-wide key you defined).
`base_model=mock` defaults to G1's geometry (35-wide actions, 40-step chunks); a simulator with other action widths also sets `base_model.action_dim`, `base_model.action_horizon`, `dsrl.noise_dim` and `training.chunk_size` to match.
A remote simulator also needs the two server addresses its entry reads (`server_url_variables`); nothing connects to them in a dry run.

The dry run should print lines of the shape:

```
plan.entry=simulators.<name>.runtime.<Name>(name='<name>')
plan.image_keys=image from <name>.steering_key_mapping
plan.layout=1 rollout worker(s) x 1 environment(s)
plan.evaluation_mode=shared
...
plan.checks=15 checked, 1 unverified, 0 error(s) | unverified: ...
```

A simulator without a registered entry fails there, before anything loads.
An entry that declares `observation_spec` and `action_spec` gets them compared with the policy in `plan.data_specs`; one that does not is listed as unverified in `plan.checks`.
The same check on the shipped simulators:

```bash
export ISAAC_RL_POSTTRAINING_DRYRUN=1
ARENA_SERVER_URL=tcp://127.0.0.1:7300 ARENA_EVAL_SERVER_URL=tcp://127.0.0.1:7301 \
  uv run python scripts/train.py simulator=g1_arena base_model=mock env.env_name=arena/LMDrillLift \
    'env.max_episode_steps=${simulator.max_episode_steps.LMDrillLift}'
uv run python scripts/train.py simulator=isaaclab_franka_stack base_model=mock \
    env.env_name=isaaclab/Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 \
    'env.max_episode_steps=${simulator.max_episode_steps.Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0}' \
    base_model.action_dim=7 dsrl.noise_dim=7
uv run python scripts/train.py simulator=libero_10 base_model=mock env.env_name=libero/libero_10/0 \
    'env.max_episode_steps=${simulator.max_episode_steps}' base_model.action_dim=7 dsrl.noise_dim=7
V2D_SERVER_URL=tcp://127.0.0.1:7400 V2D_EVAL_SERVER_URL=tcp://127.0.0.1:7401 \
  uv run python scripts/train.py simulator=v2d base_model=mock env.env_name=v2d/tissue_box_lift_hold \
    'env.max_episode_steps=${simulator.max_episode_steps.tissue_box_lift_hold}' \
    base_model.action_dim=58 base_model.action_horizon=16 dsrl.noise_dim=58 training.chunk_size=16
```

Once it runs, [the workflow template](../workflows/template.md) turns it into a workflow with results.
