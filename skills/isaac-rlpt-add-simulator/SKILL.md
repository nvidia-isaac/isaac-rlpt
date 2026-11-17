---
name: isaac-rlpt-add-simulator
description: Brings a new simulator, task suite or real robot into Isaac RLPT, in-tree under simulators/<name>/ or out-of-tree through +simulator.entry=<module>:<Class>. Use when the user wants to train or evaluate on their own simulator, robot or environment server, writes a SimulatorEntry or Observation declaration, or hits a SpecMismatch, "no simulator entry is registered", "env.max_episode_steps is unset" or server-address error on a custom simulator. Not for running the bundled G1 Factory, LIBERO, Isaac Lab or Video-to-Data (V2D) workflows, adding a base model, or adding an RL algorithm.
license: CC-BY-4.0 AND Apache-2.0
---

# Add a simulator or robot to Isaac RLPT

The core library (`isaac_rl_posttraining/`) imports no simulator.
A simulator plugs in through one environment class, one `SimulatorEntry` subclass and, for in-tree integrations, a few registry and config entries.
Every algorithm (DSRL, RFS, SAC-edit, DSRL-PPO) then runs on it unchanged.
`docs/pages/extending/add_a_simulator.md` is the full guide; this skill is the executable checklist.

## When to Use

- A user has their own simulator, robot bridge or Gymnasium environment and wants to post-train a frozen policy on it.
- A user's custom simulator fails the dry run: `SpecMismatch` or `data specs ... disagree`, `no simulator entry is registered for environment`, `env.max_episode_steps is unset`, `<name> needs <VAR> for its train environments`, or `is not a complete simulator entry`.
- A user wants to promote an out-of-tree simulator into `simulators/<name>/`.

Not for: running a shipped workflow (follow its README under `example_workflows/`), adding a base model (`isaac-rlpt-add-base-model`), or adding an algorithm (`isaac-rlpt-add-algorithm`).

## Workflow

### 1. Start from the stand-in

Copy `simulators/_template/` (a seven-joint stand-in robot) to the user's package or to `simulators/<name>/`.
It holds `env.py` (the environment and its `Observation` declaration) and `runtime.py` (the entry).
Confirm the copy passes the dry run before changing the dynamics:

```bash
ISAAC_RL_POSTTRAINING_DRYRUN=1 uv run --extra cpu python scripts/train.py base_model=mock \
    +simulator.entry=<package>.runtime:MyRobot env.env_name=my_robot/reach \
    env.max_episode_steps=50 base_model.action_dim=7 base_model.action_horizon=16 \
    dsrl.noise_dim=7 training.chunk_size=16
#   plan.entry=<package>.runtime.MyRobot(name='my_robot')
#   plan.checks=20 checked, 1 unverified, 0 error(s) | ...
```

Always pass `--extra cpu` (or `UV_NO_SYNC=1`) on a CPU machine; a plain `uv run` re-syncs to the CUDA build.

### 2. Choose the execution model

| Model | Use when | Copy from |
|---|---|---|
| in-process | the simulator imports into the learner's Python | `simulators/libero/` |
| in-process, many scenes | one process clones N scenes | `simulators/isaaclab/` |
| remote | the simulator needs its own Python or machine; a server speaks `isaac_rl_posttraining/envs/transport.py` (`RESET`, `STEP`, `SPEC`, `CLOSE`) | `simulators/g1_arena/` with `simulators/g1_arena/env_server.py` |
| remote, natively batched | one server steps a whole batch | `simulators/v2d/` with `simulators/v2d/server.py` |

A real robot is a remote provider: a bridge process speaking the transport's wire schema, a thin client and an entry.

### 3. Implement the environment contract (`env.py`)

- Observations: a dict with a `float32 [state_dim]` state, one `uint8 [H, W, 3]` entry per camera, and a `"prompt"` string where the task carries language.
- Actions: a flat `float32 [action_dim]` box; a `[chunk, action_dim]` action executes row by row, stops early on termination, and consumes one episode-cap step per row.
- Outcome: set `info["success"]` and `info["is_success"]` on every step; `terminated` reflects success; failures end as truncations at `max_episode_steps`.
- Import the simulator stack inside the methods that need it, so `import isaac_rl_posttraining` succeeds without it.
- Read no tuning value from `os.environ`; every knob is a constructor argument or config field (`tests/unit/test_repo_hygiene.py` enforces this in-tree).

### 4. Write the entry (`runtime.py`)

`name` and `make_environment(spec, phase, context)` are the only required members.
`make_environment` runs on each rollout worker; `spec` carries `env_name`, `num_envs`, `max_episode_steps`, `seed`, `server_url`, `action_dim` and `image_resolution`, and `context.environment_indices` names the environments the worker owns.
Raise from `make_environment` with the config key to set when a required value is missing; never mask it with `dict.get(key, default)`.

Declare what the environment emits so the launch checks the policy before any worker starts:

```python
@dataclass(frozen=True)
class MyRobotObservation(Observation):
    state: Array[np.float32, dim.state_dim]
    image: Array[np.uint8, dim.H, dim.W, 3]

class MyRobot(SimulatorEntry):
    name = "my_robot"
    def observation_spec(self, spec):
        return MyRobotObservation.declared_spec({"H": 16, "W": 16, "state_dim": 7})
    def action_spec(self, spec):
        return ActionSpec(width=7)
```

The default `steering_key_mapping` is derived from that declaration (image arrays are cameras, the one other array is the state).
Override only what differs:

| Member | Default | Override when |
|---|---|---|
| `evaluation` | `"shared"`: evaluate on the training workers at episode boundaries | a second persistent server evaluates (`"dedicated"`, G1 Factory, Video-to-Data) or seeded simulators are rebuilt per evaluation (`"fresh_workers"`, LIBERO) |
| `worker_layout(num_envs, num_servers=1)` | one environment per worker | one server hosts many environments: `WorkerLayout(workers=num_servers, batched=num_envs > 1)` and a batched environment from `make_environment` (G1 Factory) |
| `server_url_variables` | `None`: in-process | a remote simulator: `("MY_SERVER_URL", "MY_EVAL_SERVER_URL")`, delivered per phase as `spec.server_url` |
| `base_models` | `None`: any base model | the environment fits specific base models: `("gr00t",)` |
| `steering_key_mapping(spec)` | derived from `observation_spec` | several state-like arrays, a language embedding, or cameras that depend on the task (`IsaacLab`) |
| `supports_chunked_step`, `decode_batch_size` | `False`, `None` | a server executes a whole chunk per call behind a shared policy (G1 Factory: `True`, `16`) |
| `make_scenes(spec, context)`, `instruction(spec)` | not implemented | `algorithm=ppo` plays all of a worker's scenes in one batched simulation (`IsaacLab`, `G1Arena`) |
| `sized_from_declaration` | `False` | environments are slow to start, so the learner is sized from the declared specs (`IsaacLab`) |
| `simulates_on_gpu` | `False` | each rollout worker simulates or renders on a GPU, so placement gives each one a GPU (`Libero`, `IsaacLab`) |

A malformed declaration is rejected when the class is defined: an empty `name`, an unknown `evaluation` mode, or one server variable instead of two.

### 5a. Out-of-tree: attach per run

Nothing in the repository changes.
Attach the entry with `+simulator.entry=<module>:<Class>` and `env.env_name=<name>/<task>`, or from Python:

```python
from isaac_rl_posttraining import launch
from isaac_rl_posttraining.envs.providers import register_simulator_entry

register_simulator_entry("my_robot", MyRobot)
run = launch("dsrl_sac", env="my_robot/reach", overrides=[...])
run.loop()
```

The first segment of `env.env_name` selects the entry; `name` labels the run and its messages.
Keep them equal unless one entry serves several segments (`G1Arena` is named `g1_arena` and serves `arena` and `isaac`).

### 5b. In-tree: register and package

1. In `isaac_rl_posttraining/envs/providers.py`, add sibling-shaped entries keyed by the `env.env_name` segment:

   ```python
   _PROVIDERS["<name>"] = ("simulators.<name>.env", "<Name>Env")
   _OPTIONAL_REQUIREMENTS["<name>"] = ("<sim_pkg>",)                # only if construction needs it
   _ENV_BUILDERS["<name>"] = "simulators.<name>.env:build_dsrl_env"  # optional
   _SIMULATOR_ENTRIES["<name>"] = "simulators.<name>.runtime:<Name>"
   ```

2. Add `simulators/<name>/configs/__init__.py` and `simulators/<name>/configs/simulator/<name>.yaml` with `provider:`, `tasks:` and `max_episode_steps:` (a per-task map or one suite-wide cap).
   Shipped tasks end only on success, so set each cap above the longest demonstration.
3. Add `pkg://simulators.<name>.configs` to `hydra.searchpath` in `isaac_rl_posttraining/configs/train.yaml`.
4. In `pyproject.toml`, add `"simulators.<name>*"` to `[tool.setuptools.packages.find] include` and `"simulators.<name>" = ["configs/**/*.yaml"]` to `[tool.setuptools.package-data]`; a built wheel silently drops a simulator missing from either.
5. A GR00T base model with its own state and action layout also needs a `DatasetAdapter` in `simulators/<name>/embodiment.py`, registered in `_ADAPTERS` of `isaac_rl_posttraining/envs/embodiments/__init__.py`.
6. Launchers live with the workflow, under `example_workflows/<workflow>/scripts/` (start from `example_workflows/_template/`), not in the simulator package.
7. Add tests under `simulators/<name>/tests/` (copy `simulators/g1_arena/tests/conftest.py`): registry and config tests with no simulator, contract tests against a fake simulator, and one integration test marked `gpu`, `arena` or `libero`.
8. Write `simulators/<name>/README.md` and add a row to `simulators/README.md`.

## Verify

```bash
uv run --extra cpu pytest tests/unit/test_repo_hygiene.py tests/unit/test_doc_references.py \
    tests/unit/test_simulators.py tests/unit/test_simulator_entry.py -q
uv run --extra cpu pytest simulators/<name>/tests -m "not gpu and not arena and not libero" -q
ISAAC_RL_POSTTRAINING_DRYRUN=1 uv run --extra cpu python scripts/train.py simulator=<config> base_model=mock \
    env.env_name=<name>/<task> 'env.max_episode_steps=${simulator.max_episode_steps.<task>}'
```

Done when the dry run ends with `plan.checks=... 0 error(s)` and `plan.data_specs` lists the observation and action checks.

## Common errors

| Message | Fix |
|---|---|
| `data specs of simulator <name> and the policy disagree: action: emitted float32[35] x 40 steps vs accepted float32[7]` | `base_model=mock` defaults to G1's geometry; set all four of `base_model.action_dim`, `dsrl.noise_dim`, `base_model.action_horizon` and `training.chunk_size` to the simulator's |
| `env.max_episode_steps is unset and task '<task>' ... names none of the simulator's tasks` | the selected `simulator=` group does not list the task (out-of-tree entries run under `g1_arena`); pass `env.max_episode_steps=<cap>` |
| `no simulator entry is registered for environment '<env_name>'` | pass `+simulator.entry=<module>:<Class>`, or register the entry under the first segment of `env.env_name` |
| `<name> needs <VAR> for its train environments` | export both `server_url_variables`; placeholders such as `tcp://127.0.0.1:7300` suffice for a dry run |
| `... is not a complete simulator entry: ...` | add the missing `name` or `make_environment` the message names |
