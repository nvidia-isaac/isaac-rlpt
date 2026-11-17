# Stand-in robot (copy this to start a simulator)

A minimal simulator package that trains and evaluates on CPU with no external simulator installed.
It is a seven-joint arm whose joints move toward their action targets, with a 16x16 camera image and success when the joints reach a random goal.
Copy the folder, rename it, and replace the dynamics in `env.py` with your simulator or robot bridge; [`docs/pages/extending/add_a_simulator.md`](../../docs/pages/extending/add_a_simulator.md) is the full guide.

## Files

| File | Contents |
|---|---|
| `env.py` | `MyRobotEnv`, a Gymnasium environment following the environment contract, and `MyRobotObservation`, the typed declaration of what it emits |
| `runtime.py` | `MyRobot`, the `SimulatorEntry` the rollout workers build the environment from, with its observation and action declarations |

## Check it

Attach the entry for one run with `+simulator.entry`; nothing in the repository needs editing:

```bash
ISAAC_RL_POSTTRAINING_DRYRUN=1 uv run --extra cpu python scripts/train.py base_model=mock \
    +simulator.entry=simulators._template.runtime:MyRobot env.env_name=my_robot/reach \
    env.max_episode_steps=50 base_model.action_dim=7 base_model.action_horizon=16 \
    dsrl.noise_dim=7 training.chunk_size=16
```

It prints `plan.entry=simulators._template.runtime.MyRobot(name='my_robot')` and ends with `plan.checks=20 checked, 1 unverified, 0 error(s)`.
The four geometry overrides match the mock base model to the seven-wide action; without them the plan reports the action width mismatch as an error.
Drop `ISAAC_RL_POSTTRAINING_DRYRUN=1` to train against the mock base model; `tests/integration/test_out_of_tree_entry_launch.py` runs this package end to end from a copy outside the repository.

## After copying

- Keep `name` in `runtime.py` equal to the first segment of `env.env_name`; the segment selects the entry and `name` labels the run and its messages.
- In a package of your own, point `+simulator.entry` at `<your_package>.runtime:<YourClass>`.
- To ship it inside this repository, follow the registration and packaging steps of the guide.
