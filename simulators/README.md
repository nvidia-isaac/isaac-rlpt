# Simulators

Each directory here is one simulator integration: the environments the algorithms train and evaluate on, behind one contract, so nothing under `isaac_rl_posttraining/algorithms/` imports a simulator.
Install, train and evaluate commands live in the example workflows; each simulator README is the integration reference and points to its workflow.
To add a simulator or a real robot, follow [`docs/pages/extending/add_a_simulator.md`](../docs/pages/extending/add_a_simulator.md), starting from the stand-in robot in [`_template/`](_template/README.md), which passes the dry run and trains on CPU with no simulator installed.

## Shipped integrations

| Name | Reference | Workflow | Environments | Execution |
|---|---|---|---|---|
| `g1_arena` | [`g1_arena/README.md`](g1_arena/README.md) | [`example_workflows/g1_arena_posttraining/README.md`](../example_workflows/g1_arena_posttraining/README.md) | 9 Unitree G1 locomanipulation tasks (IsaacLab-Arena) | remote: a separate simulator process behind the `tcp://` env-server transport |
| `libero` | [`libero/README.md`](libero/README.md) | [`example_workflows/libero_posttraining/README.md`](../example_workflows/libero_posttraining/README.md) | 4 suites x 10 Franka tabletop tasks (MuJoCo/robosuite) | in-process: stepped inside the learner |
| `isaaclab` | [`isaaclab/README.md`](isaaclab/README.md) | [`example_workflows/isaaclab_gr00t_posttraining/README.md`](../example_workflows/isaaclab_gr00t_posttraining/README.md) | Isaac Lab 3.0 manager-based tasks with GR00T-keyed observations (Franka cube stacking, G1 trocar assembly) | one Isaac Sim child process per rollout worker; the learner is sized from the task mapping, so the launching process starts none |
| `v2d` | [`v2d/README.md`](v2d/README.md) | [`example_workflows/v2d_box_lift_posttraining/README.md`](../example_workflows/v2d_box_lift_posttraining/README.md) | workflow-supplied Video-to-Data (V2D) tasks with a 58-D Vega/Sharpa joint contract | remote: dedicated natively batched training and evaluation servers |

[`robolab/README.md`](robolab/README.md) is an exploratory single-environment DROID integration in Isaac Sim that drives served policy servers; it has no reproduction suite and is not part of the benchmarked release.

## Integration contents

1. **An environment provider**: the Gymnasium environment class, registered lazily by name in `isaac_rl_posttraining/envs/providers.py`, so the base package imports without any simulator stack.
2. **A simulator entry**: a `SimulatorEntry` subclass in `simulators/<name>/runtime.py` that builds the environment on each rollout worker (`G1Arena`, `Libero`, `IsaacLab`, `V2D`).
3. **A task list with episode caps**: one entry in the Hydra `simulator` config group under `simulators/<name>/configs/simulator/`, reached through the `pkg://` search path of `isaac_rl_posttraining/configs/train.yaml`. It is the only place tasks and caps are defined; the launchers read it and `tests/unit/test_simulators.py` pins the agreement.
4. **A README** with the dataset and normalization-statistics pointers, the integration facts and the reference results with their evaluation protocol.

The core `isaac_rl_posttraining/` package owns the simulator-neutral algorithms, policy interfaces, transports, rollout machinery and runtime contracts.
Each `simulators/<name>/` directory owns the code that imports its simulator or encodes its observation schema, task registry, artifact format, policy bridge, launch commands and tests.
Model-family code reused across simulators stays under `isaac_rl_posttraining/policies/`, execution backends such as RLInf under `isaac_rl_posttraining/runtimes/`, and the RLInf 0.2 compatibility adapters under `isaac_rl_posttraining/compat/`.
