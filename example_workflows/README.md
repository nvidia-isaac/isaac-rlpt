# Example workflows

A workflow is one thing a person wants to do with this repository, end to end: post-train a policy on a simulator, score it against the frozen policy, place a run on a machine.
The library (`isaac_rl_posttraining/`) and the simulator packages (`simulators/`) are the parts; a workflow is the recipe that uses them, with its own commands, glue code and measured results.
The folder layout and the rules every workflow follows are on [Start a new workflow](../docs/pages/workflows/template.md).

## Workflows

| Workflow | What it shows | Status | Hardware |
|---|---|---|---|
| [`g1_arena_posttraining/`](g1_arena_posttraining/README.md) | DSRL, RFS, SAC-edit and PPO steering of a frozen GR00T N1.7 policy on nine Unitree G1 humanoid tasks in IsaacLab-Arena; retained-checkpoint results for DSRL | validated (DSRL); RFS, SAC-edit and PPO exploratory | 2 GPUs (48 GB) per training run, 1 for evaluation |
| [`isaaclab_gr00t_posttraining/`](isaaclab_gr00t_posttraining/README.md) | steering or PPO fine-tuning of a GR00T checkpoint on an Isaac Lab task, from Isaac Lab's own `train.py` / `play.py` | exploratory | 1 GPU (48 GB); 8 for the distributed PPO placement |
| [`libero_posttraining/`](libero_posttraining/README.md) | steering a frozen GR00T N1.7, pi0 or DreamZero policy on the LIBERO tabletop suites with DSRL-SAC or DSRL-PPO | exploratory (no sourced results yet) | 1 GPU (48 GB) |
| [`dgx_example/`](dgx_example/README.md) | placing one run across the two GPUs of a DGX Station and what each placement collects per second, measured on G1 Factory | exploratory | a DGX Station; any 48 GB card for the single-card layout |
| [`v2d_box_lift_posttraining/`](v2d_box_lift_posttraining/README.md) | DSRL over a frozen 58-D joint policy with batched Video-to-Data (V2D) training and evaluation simulators | exploratory (hardware validation pending) | 1 GPU (48 GB) to start; 3 with evaluation during training |

Status values: **validated** (numbers with provenance in `results/`), **exploratory** (runs, but has no validated results), **stale** (smoke check failing, or last validated more than a quarter ago), **migrating** (being moved into this layout).
The experimental DROID integration with served robot foundation policies has not moved into this layout yet; it lives at `simulators/robolab/`.

## Checks

```bash
uv run python -m example_workflows._common.smoke                       # every workflow's smoke check, no GPU
uv run python -m example_workflows._common.smoke --only libero_posttraining
uv run python -m example_workflows._common.golden --all                # every workflow's recipes against their pinned configs
uv run pytest example_workflows/*/tests -m "not gpu and not arena and not libero and not slow" -q
```

## Internal material

Cluster job templates, node names, container registry paths, shared-filesystem paths and anything else that only works inside the company go under a workflow's `internal/` directory (or `_common/internal/` when shared, or a simulator package's `internal/`).
Nothing outside those directories may reference them; `tools/strip_internal.py` removes them for a release and a test keeps the rest of the tree free of internal references.

## New and migrated workflows

Start from `_template/` as described on [Start a new workflow](../docs/pages/workflows/template.md); to move an existing example here, follow [`MIGRATION.md`](MIGRATION.md).
