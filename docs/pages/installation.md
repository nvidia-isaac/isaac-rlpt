# Installation

This page covers every install variant: the base library, the CPU-only build, the optional extras, the base models and the simulators.
The base install needs no GPU.
For the steps to a first run, see [Getting started](getting_started.md).

## Prerequisites

| | |
|---|---|
| OS | Linux on x86_64 |
| Python | 3.11 or newer; `uv` downloads it for you (`.python-version` pins 3.12) |
| `git` and Git LFS | `sudo apt install git git-lfs && git lfs install`; Git LFS is needed only by workflows that pull large assets |
| GPU | none for this page; each workflow page lists the GPU, driver and disk it needs |

## Install the library

1. Install `uv` (0.11.19 is the version we test with), then open a new shell so it is on your `PATH` ([uv docs](https://docs.astral.sh/uv/getting-started/installation/)):

   ```bash
   curl -LsSf https://astral.sh/uv/0.11.19/install.sh | sh
   ```

   Check: `uv --version` prints a version.

2. Clone the repository and enter it.
   Every later command on this page runs from this directory, `isaac-rlpt/`.

   ```bash
   git clone https://github.com/nvidia-isaac/isaac-rlpt.git
   cd isaac-rlpt
   ```

3. Create the environment.
   This installs the library, its dependencies, a CUDA build of PyTorch, and the test tools (pytest, ruff, mypy).

   ```bash
   uv sync
   ```

   On a machine without a GPU you can use the smaller CPU build instead; see [PyTorch build](#pytorch-build).
   For pip, `pip install -e ".[test]"` is the equivalent.

4. Check the install.
   This resolves every example workflow's launch plan without a GPU, simulator or checkpoint, and takes a few minutes.

   ```bash
   uv run python -m example_workflows._common.smoke
   ```

   It ends with `5/5 smoke checks passed`.

5. Optional: run the CPU-safe unit tests.

   ```bash
   uv run pytest tests/unit -m "not gpu and not arena and not libero and not slow" -q
   ```

## PyTorch build

A plain `uv sync`, or `uv sync --extra cuda`, takes `torch` from PyPI, whose Linux wheels bundle CUDA.
`uv sync --extra cpu` takes `torch` and `torchvision` from the PyTorch CPU index.
The two extras are mutually exclusive.

`uv run` syncs the environment to the extras on its own command line, so a plain `uv run` after `uv sync --extra cpu` swaps the CPU build for the CUDA one.
Name the extra on every command, or turn the sync off for the shell:

```bash
uv sync --extra cpu
uv run --extra cpu python -m example_workflows._common.smoke
# or
export UV_NO_SYNC=1
uv run python -m example_workflows._common.smoke
```

## Optional extras

Install one with `uv sync --extra <name>`.

| Extra | Needed for |
|---|---|
| `ray` | `runtime=native` with a placement that asks for several processes (Ray actors) |
| `rlinf` | `runtime=rlinf`: parallel rollout workers and multi-rank learners |
| `libero` | the LIBERO simulator's pins; see [Simulators](#simulators) |
| `zmq` | the ZeroMQ environment transport |
| `dreamzero` | `base_model=dreamzero_server` |
| `dreamzero-local` | `base_model=dreamzero_local` |
| `docs` | building this site; `docs/README.md` has the commands |

`ray` and `rlinf` share one Ray pin and can share an environment.
The setup of each runtime is in [Runtimes](../runtimes.md#setup).

### Packages installed outside the lock

The base model stacks below are installed with `uv pip install`, outside `uv.lock`.
`uv sync` makes the environment match the lock exactly, so it uninstalls them.
Once you have installed one, add extras with `--inexact`, which removes nothing:

```bash
uv sync --inexact --extra rlinf
```

## Base models

`base_model=` selects the frozen policy, and each one needs its own package in the same environment.

| `base_model=` | Install |
|---|---|
| `gr00t` (default), `gr00t_n15`, `gr00t_n17` | `uv pip install -e /path/to/Isaac-GR00T` |
| `pi0` | the `dsrl_openpi` fork from source, plus its `openpi-client` package |
| `dreamzero_local` | `uv sync --extra dreamzero-local`, then `uv pip install -e /path/to/dreamzero --no-deps`, in its own environment (the `uv sync` removes GR00T and openpi) |
| `dreamzero_server` | `uv sync --inexact --extra dreamzero` |
| `mock` | nothing; runs the whole pipeline with no GPU |

`/path/to/...` is your own checkout of that repository.
Each workflow page gives the clone URL and pinned commit, also recorded in that workflow's `externals.lock.yaml`.

A GR00T checkpoint directory must carry the normalization statistics the policy decodes with: `experiment_cfg/metadata.json` for GR00T N1.5, and `statistics.json` (with `processor_config.json` and `embodiment_id.json`) for GR00T N1.7.
Without them, or with another robot's, the policy produces silently wrong actions.

## Simulators

Each simulator is installed on its workflow page.

| Simulator | Setup |
|---|---|
| IsaacLab-Arena (G1 Factory) | [G1 Factory](workflows/g1_arena.md) |
| Isaac Lab 3.0 | [Isaac Lab + GR00T](workflows/isaaclab_gr00t.md) |
| LIBERO | [LIBERO](workflows/libero.md) |
| Video-to-Data (V2D) | [Video-to-Data box lift](workflows/v2d_box_lift.md) |

LIBERO needs its own Python 3.11 environment, because its `mujoco==2.3.7` has no wheel for Python 3.12; its workflow page builds that environment.

## Outputs

Runs write under `~/isaac_rl_outputs/<run name>_<timestamp>/`; `ISAAC_RL_POSTTRAINING_OUTPUT_DIR` moves the root.
The Isaac Lab entry point writes under `logs/isaac_rlpt/` instead, relative to where `uv run isaaclab` runs (the workflow folder).

Runs log to Weights & Biases with the account in `WANDB_API_KEY`.
Without an account, `export WANDB_MODE=offline` writes the logs to local files; with neither, a run stops at start with `No API key configured`.
`HF_TOKEN` and `WANDB_API_KEY` are read from the environment only and are never printed.
