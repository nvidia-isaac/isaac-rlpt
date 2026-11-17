# G1 Factory RL Post-Training

Raise the success rate of a frozen GR00T N1.7 humanoid policy on nine IsaacLab-Arena tasks, and measure the change against the frozen policy under one evaluator.
DSRL is the reference method; RFS, SAC-edit and DSRL-PPO run under the same protocol, full PPO fine-tuning trains GR00T's action head itself, and offline SFT and IQL (untested) train GR00T N1.7 from demonstrations.
The workflow lives in `example_workflows/g1_arena_posttraining/`.

<video src="../../_static/g1_arena_tasks.mp4" autoplay loop muted playsinline width="100%" title="The nine G1 Factory tasks, each a successful episode of GR00T N1.7 after full PPO fine-tuning"></video>

Successful episodes of GR00T N1.7 after full PPO fine-tuning, in real time; each holds its last frame until the longest finishes.
Rows, left to right: `LMBoxLift`, `LMBoxLiftFloor`, `LMBoxTableToShelfPnP`; `LMDrillLift`, `LMDrillLiftObs`, `LMDrillPnP`; `LMPickDrillFromHolder`, `LMPushButton`, `LMPushShelfForward`.

The simulator is IsaacLab-Arena with the G1 Factory tasks, served to the learner over TCP by `simulators/g1_arena/env_server.py`.
The base policy is GR00T N1.7 fine-tuned on Unitree G1 teleoperation demonstrations ([`nvidia/G1-Factory-Benchmark`](https://huggingface.co/datasets/nvidia/G1-Factory-Benchmark)) and kept frozen.
It maps the head camera, the joint state and a task instruction to chunks of 35-dimensional whole-body actions; a separate balance controller moves the legs.

## Requirements

| | |
|---|---|
| Hardware | one NVIDIA RTX-class GPU with 48 GB per training run and one per evaluation; about 120 GB of host memory per run |
| OS and driver | Linux on x86_64 with an NVIDIA driver that Isaac Sim 6.2 supports ([Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)) |
| Software | `git`, Git LFS and `uv`, installed as on the [installation page](../installation.md#prerequisites) |
| Accounts | a Hugging Face account with access to the gated `nvidia/Cosmos-Reason2-2B` backbone (request it on the model page); Weights & Biases is optional |
| Disk | about 15 GB for G1 Factory with its submodules and assets, about 6.5 GB for the checkpoint, plus the Isaac Sim environment; about 1.3 GB more for the demonstrations, which only offline training reads |
| Time | smoke check under two minutes; a 100-episode evaluation about two hours; one 40,000-step training run about ten hours |

## Installation

Every command runs from the root of this repository, `isaac-rlpt/`, unless it changes directory itself.

1. Clone and install this repository:

   ```bash
   git clone https://github.com/nvidia-isaac/isaac-rlpt.git
   cd isaac-rlpt
   uv sync
   ```

2. Log in to Hugging Face with an account that has access to `nvidia/Cosmos-Reason2-2B` (or export `HF_TOKEN`):

   ```bash
   uv run hf auth login
   ```

3. Install [G1 Factory](https://github.com/NVlabs/G1-Factory) next to this repository, at `../g1-factory`, by following its own installation instructions, including its submodules.
   G1 Factory holds the nine `LM*` tasks and pins IsaacLab-Arena and Isaac-GR00T as submodules.
   Download its scene assets, [`nvidia/G1-Factory`](https://huggingface.co/datasets/nvidia/G1-Factory) on Hugging Face, into `../g1-factory/datasets/assets`; without them no scene loads:

   ```bash
   uv run hf download --repo-type dataset nvidia/G1-Factory --local-dir ../g1-factory/datasets/assets
   ```

   To keep G1 Factory elsewhere, set `G1_FACTORY` to its path in step 6 instead.

4. Check that the checkout has its submodules and assets:

   ```bash
   ls ../g1-factory/submodules/IsaacLab-Arena ../g1-factory/submodules/Isaac-GR00T ../g1-factory/datasets/assets
   ```

5. Download the base checkpoint, `nvidia/G1-Factory-GR00T-1.7` (about 6.5 GB): GR00T N1.7 fine-tuned for 20,000 steps on the nine G1 Factory teleoperation datasets ([`nvidia/G1-Factory-Benchmark`](https://huggingface.co/datasets/nvidia/G1-Factory-Benchmark)), with the vision encoder, projector and action head trained and the language backbone frozen; 35-dimensional actions in 40-step chunks.
   It matches the G1 Factory scene and head camera at commit `a9c20e7`, the revision this workflow pins.

   ```bash
   uv run hf download nvidia/G1-Factory-GR00T-1.7 --local-dir checkpoints/G1-Factory-GR00T-1.7
   ls checkpoints/G1-Factory-GR00T-1.7/{statistics.json,processor_config.json,embodiment_id.json}
   ```

   The three files hold the normalization statistics the policy decodes with; without them, or with another robot's, the actions are silently wrong.
   `example_workflows/g1_arena_posttraining/finetune_configs/README.md` is the recipe that produced the checkpoint.

6. Point the launchers at these pieces; repeat this block in every new shell:

   ```bash
   export G1_FACTORY=$(realpath ../g1-factory)
   export ISAACLAB_ARENA_PATH=$G1_FACTORY/submodules/IsaacLab-Arena     # the Arena checkout the simulator loads its embodiments from
   export PYTHONPATH=$G1_FACTORY                                         # only G1 Factory: the nine LM tasks (g1_factory); assets load from $G1_FACTORY/datasets/assets; replaces any earlier value
   export POSTTRAIN_BASE_CKPT=$PWD/checkpoints/G1-Factory-GR00T-1.7
   export POSTTRAIN_DATA_ROOT=$PWD/datasets                             # holds <LMTask>/lerobot; ALGO=sft|iql and dataset replay only
   ```

7. Install the workflow's environment: Isaac Sim 6.2, Isaac Lab, IsaacLab-Arena, G1 Factory, GR00T N1.7 and this repository, in one Python 3.12 environment.
   `fetch_externals.sh` links the G1 Factory checkout into the workflow folder, checks its IsaacLab-Arena and Isaac Lab submodules out at the commits in `example_workflows/g1_arena_posttraining/externals.lock.yaml`, and applies the patches in `simulators/g1_arena/patches/`.
   Isaac Lab is pinned to a commit of its `release/3.0.0` branch until that release is on PyPI.

   ```bash
   ( cd example_workflows/g1_arena_posttraining && bash ./scripts/fetch_externals.sh && uv sync )
   example_workflows/g1_arena_posttraining/.venv/bin/python -c "import gr00t, isaaclab_arena, g1_factory; print('ok')"   # ok
   ls "$ISAACLAB_ARENA_PATH/isaaclab_arena_gr00t/embodiments/g1"        # the G1 embodiment files the simulator loads
   ```

   The launchers run both the learner and the simulator with this environment's Python; `LEARNER_PYTHON` and `SIM_PYTHON` override it.

8. Optional: export `WANDB_API_KEY` for metrics ([running without a Weights & Biases account](../installation.md#outputs)).
   `HF_TOKEN` and `WANDB_API_KEY` are read from the environment and never printed.

Only offline training (`ALGO=sft|iql`, untested) and `eval.sh ... replay` read one LeRobot dataset per task, from `$POSTTRAIN_DATA_ROOT/<LMTask>/lerobot`.
They are the checkpoint's training demonstrations, [`nvidia/G1-Factory-Benchmark`](https://huggingface.co/datasets/nvidia/G1-Factory-Benchmark) (100 episodes per task, about 1.3 GB in all).
From the repository root, this places all nine at `$PWD/datasets/<LMTask>/lerobot`:

```bash
uv run hf download --repo-type dataset nvidia/G1-Factory-Benchmark --include 'datasets/*' --local-dir .
```

DSRL and PPO training and evaluation need only the checkpoint.

## Smoke check

1. Resolve every launcher's plan against a mock policy (no GPU, no G1 Factory and no checkpoint; under two minutes):

   ```bash
   uv run bash example_workflows/g1_arena_posttraining/smoke.sh
   ```

   It passes when the last line is `smoke check passed`.

2. Print one launcher's plan without booting anything:

   ```bash
   ISAAC_RL_POSTTRAINING_DRYRUN=1 uv run bash example_workflows/g1_arena_posttraining/scripts/eval.sh DrillLift base 50
   ```

   The plan must show `launch.override=env.max_episode_steps=450` and `env.learner.ARENA_IMAGE_RESOLUTION=0` (the native camera frame), and end with the library's `plan.*` lines.

3. Evaluate the frozen policy on 10 episodes, the cheapest check that boots the simulator:

   ```bash
   uv run bash example_workflows/g1_arena_posttraining/scripts/eval.sh DrillLift base 10
   ```

   It ends with a line carrying `SR=<percent>` (see [Outputs](#outputs)).

## Training

`reproduce_results.sh` launches one 80,000-step DSRL run with seed 0 (the third argument), or one 100-iteration full PPO fine-tuning run (hardware in [Full PPO fine-tuning](#full-ppo-fine-tuning)):

```bash
uv run bash example_workflows/g1_arena_posttraining/scripts/reproduce_results.sh dsrl DrillLift 0
uv run bash example_workflows/g1_arena_posttraining/scripts/reproduce_results.sh ppo  DrillLift 0
```

The DSRL run uses noise bound 2.0, an actor initialized at the frozen policy's own noise, a reward of 1 on success and 0 otherwise, and a fixed entropy coefficient of 0.05.
It evaluates 20 episodes every 5,000 steps and keeps a checkpoint every 2,500 steps.
`example_workflows/g1_arena_posttraining/scripts/repro_sep2026.sh train DrillLift 0` is the older 40,000-step protocol; `VARIANT=b2` in front selects `dsrl.noise_action_bound=2.0`, and `VARIANT=b2_succ` adds the +1-on-success reward of the released PushButton policy.

The plain launcher runs the default DSRL recipe plus any Hydra overrides you append, such as `dsrl.noise_action_bound=2.0` or `algorithm.ent_coef=0.05`; its header lists every environment variable it reads:

```bash
uv run bash example_workflows/g1_arena_posttraining/scripts/train.sh DrillLift training.num_rollouts=40000
```

The other steering methods, under the same 40,000-step protocol:

```bash
uv run bash example_workflows/g1_arena_posttraining/scripts/zoo_recipes.sh rfs      DrillPnP 0
uv run bash example_workflows/g1_arena_posttraining/scripts/zoo_recipes.sh rfs_pc   DrillPnP 0
uv run bash example_workflows/g1_arena_posttraining/scripts/zoo_recipes.sh rfs_nb2  PushButton 0
uv run bash example_workflows/g1_arena_posttraining/scripts/zoo_recipes.sh sac_edit DrillPnP 0
uv run bash example_workflows/g1_arena_posttraining/scripts/train_ppo.sh DrillLift
```

`rfs` is Residual Flow Steering at its defaults, `rfs_pc` conditions the residual on the frozen policy's proposal, and `rfs_nb2` widens the latent bound to 2.0.
`sac_edit` edits the decoded chunk directly.
`train_ppo.sh` is `train.sh` with `ALGO=dsrl_ppo`.
It takes `RUNTIME` (`native`, `rlinf`) and `PPO_LEARNER_RANKS` (`placement.learner.ranks`); `LEARNER_GPUS` lists the learner's GPUs, and `GPU` stays the simulator's.
With `RUNTIME=rlinf`, launch it as `uv run --extra rlinf bash ...`; under plain `uv run`, Ray starts its workers without RLInf and they fail with `No module named ray`.
Thirty-two scenes in one batched simulator, two FSDP learner ranks on GPUs 0 and 1, and the rollout workers on GPU 2:

```bash
NUM_ENVS=32 RUNTIME=rlinf PPO_LEARNER_RANKS=2 LEARNER_GPUS=0,1,2 \
  uv run --extra rlinf bash example_workflows/g1_arena_posttraining/scripts/train_ppo.sh DrillLift
```

With the default auto-tuned temperature, keep the budget at 40,000 steps; longer runs collapse, and `algorithm.ent_coef=0.05` fixes the temperature ([DSRL](../algorithms/dsrl.md)).

### Full PPO fine-tuning

Full PPO fine-tuning trains GR00T N1.7's flow-matching action head (1.6B of its 3.1B parameters), with the vision-language backbone frozen.
It needs the batched-scene patch of [More than one scene per simulator](#more-than-one-scene-per-simulator): eight scenes share one simulator on GPU 0, and the policy and learner run on GPU 1.

```bash
GPU=0 LEARNER_GPUS=1 uv run bash example_workflows/g1_arena_posttraining/scripts/finetune_ppo.sh PushButton \
  ppo.iterations=50 ppo.eval_every=5 ppo.eval_episodes_per_environment=4 \
  ppo.checkpoint_every=5 ppo.checkpoint_retention=all
```

Fifty iterations take about 2.5 hours on two L40 GPUs and keep `checkpoints/iteration_<n>/` every five iterations.
`finetune_ppo.sh` is `train.sh` with `ALGO=ppo` and the settings the N1.7 checkpoint needs (`simulator=g1_arena`, `base_model.embodiment_tag=new_embodiment`, `base_model.denoising_steps=4`, `base_model.action_horizon=10`, `ppo.cache_features=true`).
[Evaluation](#evaluation) has the scoring command.
[PPO](../algorithms/ppo.md) describes the method and its settings.

### Offline SFT and IQL

`ALGO=iql` (or `ALGO=sft`) trains the released GR00T N1.7 3B from the task's LeRobot demonstrations ([IQL](../algorithms/iql.md), [SFT](../algorithms/sft.md)); no simulator starts:

```bash
ALGO=iql POSTTRAIN_BASE_CKPT=nvidia/GR00T-N1.7-3B \
  uv run bash example_workflows/g1_arena_posttraining/scripts/train.sh DrillLift
```

`NUM_ROLLOUTS` counts updates (40,000 by default); `BATCH_SIZE` (8), `SAVE_INTERVAL` (5000), `ACTION_HORIZON` (16) and `DENOISING_STEPS` (4) are the other knobs.
The run does not evaluate; score its checkpoint with `eval.sh` as in [Evaluation](#evaluation).
SFT and IQL are not yet tested end to end in this release.

### More than one scene per simulator

`NUM_ENVS=8` clones the scene eight times inside one Isaac Lab instance; `EVAL_NUM_ENVS` does the same for evaluation.
Both need a patch to the Arena checkout, applied once; the server refuses to boot batched without it.

1. Apply the patch, from this repository's root:

   ```bash
   git -C "$ISAACLAB_ARENA_PATH" apply "$PWD/simulators/g1_arena/patches/arena-homie-per-env-wbc.patch"
   ```

2. Check the batched setup in seconds, without booting a simulator:

   ```bash
   uv run bash example_workflows/g1_arena_posttraining/scripts/preflight_batched.sh DrillLift
   ```

3. Boot a two-scene server in one terminal (with the block of install step 6 exported) and run the batched smoke test against it from a second terminal:

   ```bash
   "${SIM_PYTHON:-example_workflows/g1_arena_posttraining/.venv/bin/python}" simulators/g1_arena/env_server.py --task LMDrillLift --num-envs 2 --port 7300 --image-resolution 0
   ```

   ```bash
   uv run python example_workflows/g1_arena_posttraining/scripts/smoke_batched.py --url tcp://127.0.0.1:7300 --num-envs 2
   ```

4. Stop the server, then run a short batched training:

   ```bash
   NUM_ENVS=8 uv run bash example_workflows/g1_arena_posttraining/scripts/train.sh DrillLift training.num_rollouts=2000
   ```

Confirm that scenes other than 0 log successes before raising `NUM_ENVS` further.
`NUM_SERVERS=2` with `NUM_ENVS=8` splits the scenes over two training simulators of four each, with one rollout worker per simulator (at least two scenes per simulator, split equally).
DSRL-PPO then also evaluates on two simulators of that shape, and `runtime.native.processes=ray` makes the workers collect at the same time.
The launcher waits `600 + 120 * NUM_ENVS / NUM_SERVERS` seconds for each simulator to start; `SIM_BOOT_TIMEOUT` overrides that.

### Python API

The DSRL run of `train.sh` as a Python call, once the two simulators listen.
The dry run of `train.sh` prints their commands as `launch.cmd.*` lines, and `ARENA_SERVER_URL` and `ARENA_EVAL_SERVER_URL` must name them:

```python
from isaac_rl_posttraining import Env, launch
from isaac_rl_posttraining.algorithms import DSRL, BaseModel

ckpt = "/path/to/G1-Factory-GR00T-1.7"
base = BaseModel(
    "gr00t", ckpt, bc_ckpt=ckpt, dataset_dir="/path/to/datasets/LMDrillLift/lerobot"
)
run = launch(
    DSRL(base=base),
    env=Env("arena/drill_lift"),
    overrides=["training.num_rollouts=40000"],
)
print("\n".join(run.plan.lines()))  # what the dry run prints
run.loop()  # the whole run; run.collect(), run.train(steps=...), run.evaluate() step it
```

`run.py` is this recipe for any task (`uv run python -m example_workflows.g1_arena_posttraining.run DrillLift --dry-run`); `DSRLPPO`, `RFS` or `SACEdit` in place of `DSRL` select the algorithms of `train_ppo.sh` and `zoo_recipes.sh`.

## Evaluation

`base` scores the downloaded GR00T checkpoint (`POSTTRAIN_BASE_CKPT`) unchanged.
`policy` scores it steered by a checkpoint that [Training](#training) wrote.
Each command below runs 100 episodes:

```bash
uv run bash example_workflows/g1_arena_posttraining/scripts/eval.sh DrillLift base 100
uv run bash example_workflows/g1_arena_posttraining/scripts/eval.sh DrillLift policy 100 \
  ~/isaac_rl_outputs/<run name>_<timestamp>/checkpoints/final
```

A training run prints its checkpoint folder at start as a `plan.checkpoint_dir=` line; the default root is `~/isaac_rl_outputs`, and `ISAAC_RL_POSTTRAINING_OUTPUT_DIR` moves it.
The run writes `final` when it ends, plus a folder named by the environment-step count (such as `checkpoints/24087`) about every 5,000 environment steps (`training.save_interval`); any of them can follow `policy`.
The only published steering checkpoint is PushButton's ([Released PushButton checkpoints](#released-pushbutton-checkpoints)); train any other task before scoring its `policy`.

`reproduce_results.sh` scores 50 episodes seeded from 5000, so the frozen policy and every checkpoint meet the same initial scenes:

```bash
uv run bash example_workflows/g1_arena_posttraining/scripts/reproduce_results.sh base        DrillLift
uv run bash example_workflows/g1_arena_posttraining/scripts/reproduce_results.sh rescore     DrillLift \
  ~/isaac_rl_outputs/<run name>_<timestamp>/checkpoints/<step>
uv run bash example_workflows/g1_arena_posttraining/scripts/reproduce_results.sh rescore_ppo DrillLift \
  ~/isaac_rl_outputs/<run name>_<timestamp>/iteration_<n>
```

The GPU simulation is not deterministic, so seeded runs start from the same scenes but can have different outcomes.
`example_workflows/g1_arena_posttraining/scripts/repro_sep2026.sh base|rescore` are the older unseeded 100-episode scores (`EVAL_SEED=<int>` in front seeds them).
`EVAL_RESET_STATE_DIR=<dir>` with `SIM_STRICT_REPRODUCIBILITY=1` restores the full scene state; it reproduces the low-dimensional state exactly but not the camera image (`simulators/g1_arena/README.md`).

`EVAL_NUM_ENVS=10` in front of `eval.sh` scores ten scenes in parallel in one batched simulator; it needs the patch above, and the episode count must be a multiple of it.
A 100-episode evaluation then takes about 25 minutes on one GPU instead of about two hours.
The batched simulator scores a given policy about ten points lower than the one-scene simulator, so compare a before and an after only within one setting.

An offline SFT or IQL checkpoint is scored by `eval.sh` with the same `ALGO` and the policy's deterministic decode; `base` in place of `policy` scores the starting weights.
It reads the task's dataset to register the G1 layout, so `POSTTRAIN_DATA_ROOT` must be set:

```bash
ALGO=iql POSTTRAIN_BASE_CKPT=nvidia/GR00T-N1.7-3B \
  uv run bash example_workflows/g1_arena_posttraining/scripts/eval.sh DrillLift policy 100 <run>/checkpoints/final
```

A full-PPO checkpoint is scored by its own launcher on 104 episodes (13 in each of 8 scenes) from fixed starting states, with deterministic decoding.
`ppo.resume_from` is a checkpoint directory (`iteration_<n>`, `final` or `aborted`); without it the command scores the unchanged GR00T:

```bash
GPU=0 LEARNER_GPUS=1 uv run bash example_workflows/g1_arena_posttraining/scripts/finetune_ppo.sh PushButton \
  placement.learner.ranks=0 ppo.eval_only=true ppo.eval_episodes_per_environment=13 ppo.eval_seed=200000 \
  ppo.resume_from=$PWD/<run>/checkpoints/iteration_45
```

`eval.sh ... policy <checkpoint>` scores any trained checkpoint, whichever method trained it.

A third `eval.sh` mode replays a recorded demonstration through the simulator, as a diagnostic; it reads the task's dataset:

```bash
uv run bash example_workflows/g1_arena_posttraining/scripts/eval.sh DrillLift replay 10
```

### Released PushButton checkpoints

`nvidia/Isaac-RLPT-GR00T-1.7-G1-Factory-PushButton-DSRL` on Hugging Face is a DSRL steering policy trained on PushButton on top of the base checkpoint, with the settings of `reproduce_results.sh dsrl PushButton 0` plus `algorithm.train_actor_with_prior_kl=true algorithm.latent_prior_std=0.63`, saved at 72,488 of 80,000 environment steps.
`reproduce_results.sh` takes no extra overrides, so to retrain it add those settings to the script's `RECIPE` list.
These commands score the base checkpoint alone and then steered:

```bash
uv run hf download nvidia/Isaac-RLPT-GR00T-1.7-G1-Factory-PushButton-DSRL --local-dir checkpoints/Isaac-RLPT-GR00T-1.7-G1-Factory-PushButton-DSRL

uv run bash example_workflows/g1_arena_posttraining/scripts/reproduce_results.sh base PushButton
uv run bash example_workflows/g1_arena_posttraining/scripts/reproduce_results.sh rescore PushButton \
  checkpoints/Isaac-RLPT-GR00T-1.7-G1-Factory-PushButton-DSRL/72488
```

`rescore` applies the training settings; a different `dsrl.noise_action_bound` runs without error but rescales every steering action and gives a wrong result.
On 300 held-out episodes (seeds 5050–5199 and 6000–6149) of 1000 steps (20 s, G1 Factory's episode length), the base checkpoint succeeded in 61.3% (184/300) and the steered policy in 82.0% (246/300).
This measurement uses this repository's DSRL evaluator with GPU physics, so it differs from G1 Factory's evaluation, which reports 42% for the base checkpoint on PushButton.
Each command above scores 50 episodes (seeds 5000–5049) with the workflow's 350-step PushButton cap, so expect a different number.

## Run from Isaac Lab's command line

The nine tasks also run through Isaac Lab's own `train` and `play` commands, from one environment that holds Isaac Sim 6.2, Isaac Lab (the `release/3.0.0` pin of step 7), IsaacLab-Arena, G1 Factory, GR00T N1.7 and this repository.

1. Clone G1 Factory and its assets next to this repository, as in steps 3 and 4 of [Installation](#installation).

2. Build the environment, as in step 7 of [Installation](#installation) (skip it if done).
   `example_workflows/g1_arena_posttraining/scripts/fetch_externals.sh` links the G1 Factory checkout (default `../g1-factory`; `G1_FACTORY=<path>` in front picks another) to `.externals/g1-factory`, which `uv sync` installs from.
   It also changes that checkout, idempotently: it applies `simulators/g1_arena/patches/arena-homie-per-env-wbc.patch` to IsaacLab-Arena, so the balance controller resets each scene on its own, and registers `--rl_library isaac_rlpt` in Isaac Lab's `scripts/reinforcement_learning/{train,play}.py`.
   `example_workflows/isaaclab_gr00t_posttraining/scripts/install.py --isaaclab <path> --uninstall` reverts the registration.

   ```bash
   cd example_workflows/g1_arena_posttraining
   bash ./scripts/fetch_externals.sh
   uv sync
   export OMNI_KIT_ACCEPT_EULA=yes HF_TOKEN=<token with access to nvidia/Cosmos-Reason2-2B>
   ```

3. Train and evaluate from that folder:

   ```bash
   uv run isaaclab train --rl_library isaac_rlpt --task LMDrillPnP --algorithm ppo \
     --model_path nvidia/G1-Factory-GR00T-1.7
   uv run isaaclab play --rl_library isaac_rlpt --task LMDrillPnP --algorithm ppo \
     --model_path nvidia/G1-Factory-GR00T-1.7 --checkpoint logs/isaac_rlpt/<run>/checkpoints/iteration_<n>
   uv run isaaclab train --rl_library isaac_rlpt --list_tasks
   ```

The plain `train` command runs the full recipe (1,000 PPO iterations, many hours).
`--iterations 1` is a short first try that downloads the checkpoint, boots the simulators and trains once (about 20 minutes on one L40S).
A Hugging Face id for `--model_path` downloads the checkpoint to `checkpoints/<name>`; a local directory is used as it is.
`--algorithm` takes `ppo` (full PPO fine-tuning), `dsrl_sac`, `rfs`, `sac_edit` or `dsrl_ppo`, and `--dry_run` prints the launch plan without starting anything.
Each command calls the matching launcher above (`finetune_ppo.sh`, `train.sh` or `eval.sh`) with the same settings, and writes under `logs/isaac_rlpt/`.
`--num_envs`, `--num_steps`, `--iterations`, `--eval_episodes`, `--seed` and `--run_name` map onto the launcher's knobs; further `key=value` arguments pass on as Hydra overrides.
The simulators run headless by default.
`--viz kit` opens Isaac Sim's viewer on the simulator whose rollouts you watch: the training simulator for `play --algorithm ppo`, the evaluation simulator otherwise (the launchers take `SIM_VIZ=kit`).
It needs a display (`$DISPLAY`), so run it on a machine with a monitor, not over plain SSH.
A simulator that exits before it listens stops the launch, and the message names its log under `logs/isaac_rlpt/simulator_logs/`.

### Watch a trained policy

This command opens Isaac Sim's viewer and runs two episodes of a trained PPO checkpoint in two scenes:

```bash
uv run isaaclab play --rl_library isaac_rlpt --task LMDrillPnP --algorithm ppo \
  --model_path nvidia/G1-Factory-GR00T-1.7 --checkpoint logs/isaac_rlpt/<run>/checkpoints/final \
  --num_envs 2 --eval_episodes 2 --viz kit
```

The viewer shows a still scene for several minutes while the simulators boot and the policy loads; the robot moves once the episodes start.

## Outputs

Each launcher boots its simulator processes, runs the learner or the evaluator, and shuts everything down.
Training starts two simulators, one for collection and one for evaluation.

| Output | Where |
|---|---|
| Checkpoints | `$ISAAC_RL_POSTTRAINING_OUTPUT_DIR/<run name>_<timestamp>/checkpoints/<step>`, default root `~/isaac_rl_outputs`; `<step>` is the environment-step count, and a run also leaves `checkpoints/final` |
| Every evaluation | `checkpoints/evaluation_summary.json`: `latest` plus the full `history` |
| Actor fingerprints | `fingerprints.jsonl`: the actor's hash at every evaluation, save and load |
| Logs | `./dsrl_train_logs` and `./dsrl_eval_logs`; `LOG_DIR` moves them |
| Metrics | Weights & Biases when `WANDB_API_KEY` is set |

An evaluation ends with one summary line, for example:

```
[dsrl_eval] DONE eval_DrillLift_base_1007_101500 SR=87.0%  (summary: <run dir>/checkpoints/evaluation_summary.json; log: ./dsrl_eval_logs/eval_DrillLift_base_1007_101500.eval.log)
```

Training logs each in-training evaluation as a block of `[EVAL]` lines.
A run that is working boots both simulators within the launcher's timeout, logs an `[EVAL]` block every 5,000 steps, and writes a checkpoint at every evaluation.

## Results

### G1 Factory base policy

The base checkpoint `nvidia/G1-Factory-GR00T-1.7` in G1 Factory's own evaluation (100 episodes per task, CPU physics, 20-second episodes), as reported by [G1 Factory](https://github.com/NVlabs/G1-Factory):

| Task | Success rate |
|---|---|
| BoxLift | 98% |
| BoxLiftFloor | 96% |
| BoxTableToShelfPnP | 77% |
| DrillLift | 66% |
| DrillLiftObs | 75% |
| DrillPnP | 69% |
| PickDrillFromHolder | 44% |
| PushButton | 42% |
| PushShelfForward | 58% |
| Overall | 69.4% |

### DSRL and PPO

The DSRL and PPO tables come from this repository's evaluators, each scoring its own method and the base checkpoint.
These use GPU physics (the physics used for training), 20-second episodes and 50 held-out episodes per task.
G1 Factory's evaluation uses CPU physics and 100 episodes, so these base scores are not directly comparable with the G1 Factory table.
Compare each method only with the base in its own table.

#### DSRL

| Task | Base | Peak | Final |
|---|---|---|---|
| BoxLift | 96% | 98% | 100% |
| BoxLiftFloor | 84% | 96% | 92% |
| BoxTableToShelfPnP | 74% | 80% | 56% |
| DrillLift | 62% | 90% | 70% |
| DrillLiftObs | 82% | 62% | 60% |
| DrillPnP | 54% | 86% | 74% |
| PickDrillFromHolder | 30% | 40% | 20% |
| PushButton | 74% | 66% | 64% |
| PushShelfForward | 0% | 8% | 6% |
| Overall | 61.8% | 69.6% | 60.2% |

#### PPO

| Task | Base | Peak | Final |
|---|---|---|---|
| BoxLift | 94% | 98% | 92% |
| BoxLiftFloor | 72% | 96% | 90% |
| BoxTableToShelfPnP | 66% | 76% | 74% |
| DrillLift | 66% | 84% | 84% |
| DrillLiftObs | 58% | 88% | 70% |
| DrillPnP | 54% | 84% | 82% |
| PickDrillFromHolder | 18% | 52% | 36% |
| PushButton | 48% | 90% | 78% |
| PushShelfForward | 0% | 20% | 10% |
| Overall | 52.9% | 76.4% | 68.4% |

In both tables, Base is the frozen base checkpoint scored by that method's evaluator.
Peak is the run's checkpoint with the best in-training evaluation (ties go to the earliest), re-scored on the held-out episodes.
Final is the run's last checkpoint, re-scored the same way.
Each row is one of three training seeds: the seed with the highest Peak, or on a tie the one with the higher Final.
Both methods execute all 40 actions of each policy chunk.

The released PushButton steering policy ([Released PushButton checkpoints](#released-pushbutton-checkpoints)), scored by this workflow against the base checkpoint:

<!-- results: g1_arena_posttraining -->

## Configuration

- `GPU=<index>` selects the card. A run's simulators listen on consecutive ports from `PORT_BASE + 16*GPU` (`PORT_BASE` defaults to 7300), and a launch refuses ports another run already holds, so keep one run per GPU index or set `PORT_BASE`.
- `RUNTIME=rlinf` (training and evaluation) runs on one GPU: the launchers put RLInf's rollout and policy-server groups on the learner's GPU; `LEARNER_GPUS=0,1` gives them the second one.
- `SIM_STARTUP_LOCK=/tmp/sim_boot.lock` starts simulators one at a time on a shared machine.
- `TRAIN_SIM_DEVICE=cpu` and `EVAL_SIM_DEVICE=cpu` select CPU physics.
- `ISAAC_RL_POSTTRAINING_DRYRUN=1` in front of any launcher prints its resolved plan and exits.

**Tasks.** Nine, accepted with or without the `LM` prefix, each with its own episode cap in simulator steps, defined in `simulators/g1_arena/configs/simulator/g1_arena.yaml`:

| Task | Cap |
|---|---|
| `PushButton` | 350 |
| `BoxLift` | 450 |
| `DrillLift` | 450 |
| `PickDrillFromHolder` | 500 |
| `DrillLiftObs` | 650 |
| `DrillPnP` | 650 |
| `BoxLiftFloor` | 750 |
| `BoxTableToShelfPnP` | 900 |
| `PushShelfForward` | 1000 |

The environments end only on success, so the cap is what ends a failed episode.

**Reward.** 0 on success, -1 when an episode ends without success.

## Common issues

Rules that keep a run correct:

1. Keep `training.chunk_size` at 40, GR00T N1.7's action horizon, so there is one steering row per action step.
2. Give the policy the native camera frame. The launchers set `ARENA_IMAGE_RESOLUTION=0`.
3. Give exactly one source of normalization statistics: `base_model.dataset_dir` for a single task, `base_model.multitask_dirs` with `base_model.task_dir` for a multi-task base, or the checkpoint's own with `base_model.dataset_dir=null` and `base_model.task_instruction='<prompt>'`.
4. Leave the per-task episode caps where they are, in the simulator config.
5. Give training and evaluation their own simulator; each server takes one client.

For RFS on these joint-target actions, keep the residual bound at its 0.01 default; 0.05 fell to near zero success on every task.

When something fails (a simulator that never prints `listening on`, 0% on a task the policy handles), see [troubleshooting](../troubleshooting.md#g1-factory).

## Related pages

- Algorithms: [DSRL](../algorithms/dsrl.md), [RFS](../algorithms/rfs.md), [SAC-edit](../algorithms/sac_edit.md), [PPO](../algorithms/ppo.md), [IQL](../algorithms/iql.md), [SFT](../algorithms/sft.md)
- Integration reference: `simulators/g1_arena/README.md`
- When something fails: [troubleshooting.md](../troubleshooting.md)
