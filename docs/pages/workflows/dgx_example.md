# DGX Station Example

Place one post-training run on the two GPUs of a DGX Station, and measure how many environment steps per second each placement collects.
The recipe is the [G1 Factory workflow](g1_arena.md)'s: DSRL on a frozen GR00T N1.7 Unitree G1 policy in IsaacLab-Arena.
Only the placement varies; the workflow lives in `example_workflows/dgx_example/`.

## Overview

A run has two GPU consumers: the simulator stepping N environments, and the learner process, which holds both the frozen policy and the steering learner.
They can share one card (the **single** layout) or take one card each and talk over loopback TCP (the **split** layout).
A split needs a simulator that runs behind a server; one that steps inside the learner process can only run single.
To see which simulators support which layout:

```bash
uv run python -m example_workflows.dgx_example.layouts capabilities
```

G1 Factory is the measured case; Isaac Lab and LIBERO go through the same placement code in the single layout and are not measured here.

## Requirements

| | |
|---|---|
| Hardware | a GB300 DGX Station (one GB300 and one RTX Pro 6000 Blackwell with 96 GB) for the station layouts; any NVIDIA card with 48 GB or more runs the single layout up to N = 128 |
| Station software | Linux with Docker, Docker Buildx and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html); no Python, since every step runs in containers |
| Staging machine | a network-connected Linux machine with Git, rsync, [uv](https://docs.astral.sh/uv/getting-started/installation/) and SSH access to the station; it fetches everything (G1 Factory from GitHub, its task assets from Hugging Face), because the station may have no internet access |
| Checkpoint and data | the GR00T N1.7 G1 checkpoint and the task's LeRobot dataset, obtained as on the [G1 Factory page](g1_arena.md) |
| Weights & Biases | an account and `WANDB_API_KEY` for the throughput sweep, whose tables are read back from the W&B runs; the training-and-scoring recipe prints its own success rates |
| Time | under two minutes for the CPU check; 15 minutes to 1.5 hours per probe run; about 4 hours per layout at five environment counts; about 5 hours for the training-and-scoring recipe |

## Installation

Steps 1 to 3 run on the staging machine, steps 4 to 7 on the station.

1. Clone this repository and install it:

   ```bash
   git clone https://github.com/nvidia-isaac/isaac-rlpt.git
   cd isaac-rlpt
   uv sync
   ```

2. Check the workflow without a GPU (placement, launcher plans and tables):

   ```bash
   uv run bash example_workflows/dgx_example/smoke.sh
   ```

   It ends with `smoke check passed`.

3. Stage the G1 Factory stack and deploy it, replacing `user@station-host` with your SSH login on the station:

   ```bash
   STATION=user@station-host bash example_workflows/g1_arena_posttraining/scripts/station/stage_and_deploy_station.sh
   ```

   The script clones G1 Factory with its submodules and task assets under `~/gb300-staging/g1-factory`, applies the whole-body-controller patch from `simulators/g1_arena/patches/`, places this repository at `posttrain/` inside that tree, and copies the tree to `/data/src/g1-factory` on the station.
   Re-running it transfers only what changed.
   The stack is pinned in `example_workflows/dgx_example/externals.lock.yaml`.

4. Copy the checkpoint and datasets to the station so `/data` has this layout:

   | Path | Holds |
   |---|---|
   | `/data/src/g1-factory` | the deployed stack; `posttrain/` inside it is this repository |
   | `/data/ckpts/<checkpoint>` | the GR00T N1.7 G1 checkpoint, the value of `POSTTRAIN_BASE_CKPT` |
   | `/data/datasets/<root>/<LMTask>/lerobot` | the task datasets; `/data/datasets/<root>` is the value of `POSTTRAIN_DATA_ROOT` |
   | `/data/secrets/wandb_key`, `/data/secrets/hf_token` | optional, mode 0600; read when `WANDB_API_KEY` or `HF_TOKEN` is unset |
   | `/data/dsrl_out_g1arena` | created by runs: checkpoints, evaluation summaries and `util/` telemetry |

5. On the station, set up the shell every later station command runs in, replacing `<checkpoint>` and `<root>` with your directory names:

   ```bash
   cd /data/src/g1-factory/posttrain
   export PYTHON=$PWD/example_workflows/g1_arena_posttraining/scripts/station/station_python.sh
   export POSTTRAIN_BASE_CKPT=/data/ckpts/<checkpoint> POSTTRAIN_DATA_ROOT=/data/datasets/<root>
   nvidia-smi -L
   ```

   `station_python.sh` runs the placement module inside the learner image, because the host has no Python.
   `nvidia-smi -L` lists each card with its UUID (`GPU-...`); every `<uuid>` below is one of those.
   Pin cards by UUID, because the container runtime's device indices need not match `nvidia-smi`'s.

6. Build the G1 Factory image (`g1_factory:latest`), which the learner image builds on; the first build is long and cached afterwards:

   ```bash
   docker build --pull --progress=plain --build-arg WORKDIR=/workspaces/isaaclab_arena \
     -t g1_factory:latest -f /data/src/g1-factory/docker/Dockerfile.isaaclab_arena /data/src/g1-factory
   docker images g1_factory:latest
   ```

   The second command lists the image.

7. Build the learner image, which `station_python.sh` and every launch run in, then check that the placement module runs:

   ```bash
   STATION_DIR=example_workflows/g1_arena_posttraining/scripts/station
   docker build --build-arg BASE_IMAGE=g1_factory:latest \
     -t g1_factory_learner:$(md5sum $STATION_DIR/Dockerfile.learner | cut -c1-12) \
     -f $STATION_DIR/Dockerfile.learner $STATION_DIR
   "$PYTHON" -m example_workflows.dgx_example.layouts capabilities
   ```

   The second command prints the table of simulators and their layouts.
   For `RUNTIME=rlinf`, the launcher builds the RLInf learner image from `example_workflows/g1_arena_posttraining/scripts/station/Dockerfile.learner_rlinf` on first use.

## Training

Put `ISAAC_RL_POSTTRAINING_DRYRUN=1` in front of either script below to print every launch and start nothing.

### Dry run

```bash
"$PYTHON" -m example_workflows.dgx_example.layouts plan --simulator arena --num-envs 64 \
    --simulator-gpu <RTX Pro 6000 uuid> --learner-gpu <GB300 uuid>
```

`plan` prints where each process goes and the config overrides the learner needs; `--format shell` emits something a script can evaluate.
The same card for both roles is the single layout; two different cards are the split.
On the staging machine, use `uv run python` instead of `"$PYTHON"`.

### Layout sweep

```bash
GB300_GPU=<GB300 uuid> RTX_GPU=<RTX Pro 6000 uuid> bash example_workflows/dgx_example/scripts/layout_sweep.sh
```

This runs the three station layouts at every N (`NS`, default `8 64 128 256 512`) with the probe settings of `configs/probe.env`, one log and one W&B run each, named `probe_<layout>_n<N>`.
Prefix `RUNTIME=rlinf` to place the rollout worker with the RLInf runtime (runs named `probe_<layout>_rlinf_n<N>`).
On a station with one card, name the layout instead: `LAYOUTS="gb300_only:<uuid>:<uuid>"`.
The other knobs (`SCHEDULE`, `NUM_SERVERS`, `MAX_UPDATES_PER_BURST`, `RUN_PREFIX`, `LOG_DIR`) are listed in the script's header and in the workflow README.

### Training and scoring

```bash
SIMULATOR_GPU=<RTX Pro 6000 uuid> LEARNER_GPU=<GB300 uuid> \
  bash example_workflows/dgx_example/scripts/train_eval.sh DrillLift
```

This trains DSRL in the split layout (`NUM_ENVS`, default 64; `TRAIN_ENV_STEPS`, default 100000).
It then scores the frozen policy and the retained checkpoint on `EVAL_EPISODES` (default 100) episodes each, and prints both success rates.
To score an existing checkpoint without training, prefix `CKPT_DIR=/data/dsrl_out_g1arena/<run>/checkpoints/<step>`, with `<run>` and `<step>` from a previous run's output directory.

### Throughput tables and figures

On the staging machine, read the numbers back from W&B and render them into the workflow README:

```bash
uv run python -m example_workflows.dgx_example.fetch_wandb \
    example_workflows/dgx_example/results/throughput.json \
    --wandb-path "$WANDB_ENTITY/isaac_rl_posttraining" \
    --util-dir /data/dsrl_out_g1arena/util --log-dir "$HOME/dgx_example_logs"

uv run python -m example_workflows.dgx_example.throughput \
    example_workflows/dgx_example/results/throughput.json \
    --write example_workflows/dgx_example/README.md

uv run python -m example_workflows.dgx_example.figures \
    example_workflows/dgx_example/results/throughput.json \
    --out example_workflows/dgx_example/figures
```

`WANDB_ENTITY` is the W&B entity the runs logged to, and `isaac_rl_posttraining` the launcher's default project (`WANDB_PROJECT` changes it).
`--util-dir` and `--log-dir` take the run's `util/` directory under `OUT_DIR` and the sweep's `LOG_DIR`, shown here at their defaults.
`--block throughput_dgx_station` and `--block throughput_schedules` render the other two results files into their own README blocks; `--prefix` does the same for figure names.

## Outputs

| Output | Where |
|---|---|
| Sweep logs | `$HOME/dgx_example_logs`; `LOG_DIR` moves them |
| Checkpoints and evaluation summaries | under `OUT_DIR`, default `/data/dsrl_out_g1arena` |
| Throughput rows | `results/throughput.json`, with the W&B run ids and the recipe |
| Figures | `example_workflows/dgx_example/figures/` |

The trainer logs the collection rate and the collect-versus-train wall split every iteration under `system/`, and the collector's per-step and per-decision costs under `train/collector/`.

## Results

### Single-card collection throughput

The tables below are environment steps per second, not success rates.
Each entry is the mean over the run's collection cycles, followed by the spread across cycles.

Recipe: task DrillLift, 1350 environment steps per environment, 8 simulator steps per decision, SAC bursts capped at 20 updates of batch 64, frames shrunk to a 256 px shortest edge, the policy decoding all N environments in one forward, no evaluation simulator.

| N envs | GB300 GPU only | L40 only | GB300 GPU only, RLInf runtime | RTX 5880 only |
|---|---|---|---|---|
| 8 | 29 ± 1 | 52 ± 3 | 33 ± 0 | 51 ± 3 |
| 64 | 91 ± 1 | 139 ± 2 | 94 ± 2 | 145 ± 8 |
| 128 | 106 ± 1 | 126 ± 2 | 108 ± 2 | 151 ± 5 |
| 256 | 112 ± 0 | — | 113 ± 0 | out of memory |

From `results/throughput.json`, 2026-09-18.

### The DGX Station's layouts

Same recipe; the split also on the RLInf runtime and with its training environments spread over two Arena servers.

| N envs | GB300 GPU only | RTX Pro 6000 only | Both GPUs | Both GPUs, RLInf runtime | Both GPUs, RLInf runtime, two servers |
|---|---|---|---|---|---|
| 8 | 44 ± 2 | 63 ± 2 | 57 ± 4 | 59 ± 5 | — |
| 64 | 113 ± 1 | 208 ± 4 | 226 ± 5 | 218 ± 1 | — |
| 128 | 120 ± 0 | 216 ± 1 | 245 ± 5 | 234 ± 2 | — |
| 256 | 126 ± 1 | 227 ± 2 | 255 ± 1 | 252 ± 2 | 336 ± 5 |
| 512 | — | — | — | 256 ± 1 | 358 ± 5 |

From `results/throughput_dgx_station.json`, 2026-09-18.

A single card flattens above N = 64, because one card serves both the simulator's rendering and the policy's decode.
Both cards together collect twice what the GB300 alone does at N = 256 and 13 % more than the RTX Pro 6000 alone.
The RTX Pro 6000 steps the simulator faster (0.54 s against 1.46 s at N = 256) and the GB300 decodes GR00T faster (2.26 s against 3.07 s), so the split puts the simulator on the RTX Pro 6000 and the policy and learner on the GB300.
The whole stack needs 52 to 53 GiB at N = 256 on one card; the split divides that into 27 GiB on the simulator card and 25 GiB on the learner card.
With one server the split also flattens from N = 256, because the simulator card waits for the decode part of every cycle.
Two servers on the RTX Pro 6000, each with half the environments, raise its rate 1.33× at N = 256 and 1.40× at N = 512.

Per-layout breakdowns of cycle time and peak memory, and the overlapped-schedule comparison, are in `example_workflows/dgx_example/README.md`.

## Related pages

- The recipe this places: [G1 Factory](g1_arena.md) and [DSRL](../algorithms/dsrl.md)
- When something fails: [troubleshooting.md](../troubleshooting.md)
