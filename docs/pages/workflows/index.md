# Workflows

A **workflow** is one complete recipe: post-train a policy on one simulator, score it against the frozen policy, and record the numbers.
Each one lives in its own folder under `example_workflows/`.
Pick one by what it trains and the hardware you have.

| Workflow | What it trains | Hardware |
|---|---|---|
| [Start a new workflow](template.md) | a skeleton to copy for a new workflow | none |
| [G1 Factory RL Post-Training](g1_arena.md) | steering of a frozen GR00T N1.7 policy on nine Unitree G1 humanoid Factory tasks in IsaacLab-Arena | 2 GPUs (48 GB) per training run, 1 for evaluation |
| [Isaac Lab Post-Training](isaaclab_gr00t.md) | steering or PPO fine-tuning of a GR00T checkpoint on an Isaac Lab 3.0 task | 1 GPU (48 GB); 8 for distributed PPO |
| [LIBERO Post-Training](libero.md) | steering of a frozen GR00T N1.7, pi0 or DreamZero policy on the LIBERO tabletop suites | 1 GPU (48 GB) |
| [Video-to-Data (V2D) Tissue Box Post-Training](v2d_box_lift.md) | steering of a frozen 58-joint GR00T policy, learned from human videos, to lift a tissue box | 1 GPU (48 GB) to start; 3 with evaluation during training |
| [DGX Station Example](dgx_example.md) | the G1 Factory recipe placed across the two GPUs of a DGX Station, with measured throughput per placement | a DGX Station; any 48 GB card for the single-card layout |

## Smoke checks

From the repository root, after `uv sync`:

```bash
uv run python -m example_workflows._common.smoke                       # every workflow
uv run python -m example_workflows._common.smoke --list                # list them
uv run python -m example_workflows._common.smoke --only libero_posttraining
```

Each check finishes in under two minutes and needs no GPU or external repository.
