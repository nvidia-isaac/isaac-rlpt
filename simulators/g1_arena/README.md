# G1 Factory

Nine Unitree G1 locomanipulation tasks in IsaacLab-Arena, served to the learner over TCP: provider `g1_arena` (`simulators/g1_arena/env.py`), entry `G1Arena` (`simulators/g1_arena/runtime.py`), config group `simulator/g1_arena.yaml`.
The user guide with the maintained results is `docs/pages/workflows/g1_arena.md`; this page is the integration reference.

## Installation

Each run has two kinds of process: the simulator (IsaacLab-Arena under its own Python, served by `simulators/g1_arena/env_server.py`) and the learner (this repository plus the `gr00t` package).
Install both, the base checkpoint and the environment variables by following the install steps of `docs/pages/workflows/g1_arena.md` to the end.
Those steps install [G1 Factory](https://github.com/NVlabs/G1-Factory) with its submodules by following its own instructions, download its scene assets ([`nvidia/G1-Factory`](https://huggingface.co/datasets/nvidia/G1-Factory)) into `$G1_FACTORY/datasets/assets` (the nine `LM*` tasks and the `g1_factory_wbc_joint` embodiment, with IsaacLab-Arena and Isaac-GR00T as submodules), build the workflow's one environment (`example_workflows/g1_arena_posttraining/scripts/fetch_externals.sh`, then `uv sync` in that folder; Isaac Sim 6.2), and download `nvidia/G1-Factory-GR00T-1.7`.
`example_workflows/g1_arena_posttraining/finetune_configs/README.md` is the recipe that produced the checkpoint.

Verify the wiring without booting a simulator:

```bash
ISAAC_RL_POSTTRAINING_DRYRUN=1 uv run bash example_workflows/g1_arena_posttraining/scripts/eval.sh DrillLift base 50
```

The plan must show `launch.override=env.max_episode_steps=450` and `env.learner.ARENA_IMAGE_RESOLUTION=0` (the policy must receive the native camera frame; a pre-resized frame sharply lowers manipulation success rates).

## Usage

```bash
uv run bash example_workflows/g1_arena_posttraining/scripts/eval.sh DrillLift base 50                            # the frozen policy, 50 episodes
uv run bash example_workflows/g1_arena_posttraining/scripts/train.sh DrillLift training.num_rollouts=40000       # one 40k-step DSRL run
uv run bash example_workflows/g1_arena_posttraining/scripts/eval.sh DrillLift policy 50 <run>/checkpoints/<step> # a trained steering checkpoint
```

An evaluation ends with a line carrying `SR=<percent>`; the full summary is `checkpoints/evaluation_summary.json` under the run's output directory.
Training boots a training and an evaluation simulator, because the server accepts a single client.
Any trailing `key=value` is a Hydra override, and `ISAAC_RL_POSTTRAINING_DRYRUN=1` in front of any launcher prints its plan and exits.
`example_workflows/g1_arena_posttraining/scripts/station/` runs the stack in containers on one multi-GPU workstation; `docs/pages/workflows/dgx_example.md` describes it.

## Integration facts

- **Tasks and caps** (`simulators/g1_arena/configs/simulator/g1_arena.yaml`, the only definition; `tests/unit/test_simulators.py` pins the launchers to it): `LMPushButton` 350, `LMBoxLift` 450, `LMDrillLift` 450, `LMPickDrillFromHolder` 500, `LMDrillLiftObs` 650, `LMDrillPnP` 650, `LMBoxLiftFloor` 750, `LMBoxTableToShelfPnP` 900, `LMPushShelfForward` 1000 simulator steps.
  The launchers accept the names with or without the `LM` prefix.
  The server enforces no horizon; the client-side cap ends a failed episode, and a cap below the longest demonstration truncates successes.
- **Other config keys**: `provider: g1_arena`, `action_dim: 35`, `image_resolution: 0` (the native frame), `decode_batch_size: null` (the entry's cap of 16 environments per policy forward; set it to `training.num_envs` on a card with room for the whole batch).
- **Observations**: `ego_view_image` (`uint8 [480, 640, 3]`) and `state` (`float32 [31]`, arm, hand and waist proprioception); no language embedding (`ARENA_KEY_MAPPING`).
  **Actions**: `float32 [35]` whole-body-control joint targets; a `[chunk, 35]` action executes in one server round trip (`supports_chunked_step`).
  `info["success"]` and `info["is_success"]` carry the task's success term; episodes end only on success or at the cap.
- **Entry** (`G1Arena`): evaluation on a dedicated server; addresses from `ARENA_SERVER_URL` and `ARENA_EVAL_SERVER_URL` (`tcp://host:port`, exported by the launchers); one worker per training server, batched (`ArenaBatchedEnv`, payloads with a leading `[N]` axis) when `training.num_envs > 1`, in which case the launcher also sets `training.rollout_count_by=episode`.
  The client checks the server's declared observation and action against its own before the first episode.
- **Dataset adapter**: `g1_wbc` (`simulators/g1_arena/embodiment.py`) turns the LeRobot datasets at `$POSTTRAIN_DATA_ROOT/<LMTask>/lerobot` into buffer arrays.
  Evaluation of `base` and `policy` reads only the checkpoint; `eval.sh TASK replay N` replays a demonstration through the simulator, and `ALGO=sft|iql` reads the dataset to register the G1 layout.
- **Env server** (run under the Arena's Python): `--task LM<Task>`, `--num-envs`, `--port`, `--image-resolution 0`, `--embodiment g1_factory_wbc_joint`, `--env-spacing` (200 m above one environment, the factory floor's size; closer clones overlay their floors), `--render-once-prims` (above one environment only env_0's copy of the factory's ambient dome light renders; `none` disables this for a task without the factory), `--image-shortest-edge`; `ARENA_SIM_DEVICE=cpu` selects CPU physics.
- **Launcher knobs** (each header lists all of them): `GPU` (default 0), `PORT_BASE` (default 7300; a run's simulators listen on consecutive ports from `PORT_BASE + 16*GPU`, and a launch refuses ports another run holds), `LOG_DIR`, `NUM_ENVS`, `NUM_SERVERS` (training simulators, `NUM_ENVS` split equally over them, at least two each), `ENV_SPACING`, `RENDER_ONCE_PRIMS`, `TRAIN_SIM_DEVICE` and `EVAL_SIM_DEVICE` (`cpu|gpu`), `SIM_PYTHON`, `LEARNER_PYTHON`, `SIM_BOOT_TIMEOUT`, `SIM_STARTUP_LOCK`, `NUM_ROLLOUTS`, `TRAIN_MAXSTEPS`, `EVAL_MAXSTEPS`, `ALGO` (`dsrl_sac|rfs|sac_edit|dsrl_ppo|ppo|sft|iql` for `train.sh`; `sft|iql` for `eval.sh`), `BASE_MODEL`, `RUNTIME` (`native|rlinf`; `rlinf` needs `uv sync --extra rlinf`, see `docs/runtimes.md`), `ISAAC_RL_POSTTRAINING_OUTPUT_DIR`.
  `HF_TOKEN` and `WANDB_API_KEY` are read from the environment only and never printed.
- **Parallel environments** (`NUM_ENVS>1`): apply the whole-body-controller patch to the Arena checkout once, `git -C "$ISAACLAB_ARENA_PATH" apply "$PWD/simulators/g1_arena/patches/arena-homie-per-env-wbc.patch"`; the server refuses to boot batched without it (`ARENA_SKIP_WBC_PATCH_CHECK=1` skips the check for an embodiment without the homie controller).
  `example_workflows/g1_arena_posttraining/scripts/preflight_batched.sh DrillLift` checks the batched setup without a simulator, and `example_workflows/g1_arena_posttraining/scripts/smoke_batched.py` checks a running batched server, including that the controller's `stand.onnx` and `walk.onnx` accept a batch above one.
  Startup grows with the environments per simulator; the launcher waits 600 seconds plus 120 per environment for each simulator (`SIM_BOOT_TIMEOUT` overrides).
  Success terms that compare world-frame positions against fixed goals fire only for environment 0 above one environment, so a task's success term must subtract `env.scene.env_origins` (the `LM*` terms all do).
- **Tests**: `uv run pytest simulators/g1_arena/tests -q -m "not gpu and not arena"` runs the launcher, config, client and adapter tests without a simulator; the tests marked `arena` and `gpu` need `ISAACLAB_ARENA_PATH` and a GPU.

Failures and fixes are in `docs/pages/troubleshooting.md` (G1 Factory section).

## Batched GR00T decode

With `NUM_ENVS>1` the frozen GR00T N1.7 policy decodes the action chunks of several environments in one forward pass.
`example_workflows/g1_arena_posttraining/scripts/smoke_batched_decode.py --sweep 8,16,32,64` times one batched decode on the real G1 checkpoint, with no simulator.
Measured on one NVIDIA L40, mean of 3 repeats; 2026-09-12:

| Environments per decode | Time per decode | Per environment | Peak allocated GPU memory |
|---|---|---|---|
| 8 | 0.23 s | 29.1 ms | 6.3 GiB |
| 16 | 0.32 s | 19.8 ms | 6.7 GiB |
| 32 | 0.70 s | 21.8 ms | 7.4 GiB |
| 64 | 1.40 s | 21.8 ms | 8.8 GiB |

Decoding 8 environments in one batch took 0.24 s against 1.36 s for eight one-environment decodes, 5.7x faster.
The batched decode matches the one-environment decode of environment 0 to within 0.0042 (tolerance 0.01), and different steering latents give different action chunks.
From 16 environments on, the time per environment stays at 20 to 22 ms, so the entry's default cap of 16 (`decode_batch_size`) already reaches the batched rate on this card.

## Reset reproducibility

`EVAL_RESET_STATE_DIR=<dir>` records the evaluation simulator's full scene state at each seeded reset and restores it at later evaluations, so retained checkpoints are scored from the same physical starts.
`SIM_STRICT_REPRODUCIBILITY=1` turns on deterministic RTX rendering without frame-varying effects, PhysX enhanced determinism and deterministic torch kernels in both simulators; it may lower throughput.
`SIM_ANTIALIASING_MODE` (`off`, `taa`, `fxaa`, `dlss`, `rtxaa`) and `EVAL_RESET_RERENDERS` (default 1) change the renderer for a diagnostic.

Restoring the state does not reproduce the camera image.
Two consecutive full resets with the same seed in one simulator process, LMDrillLift, 8 environments, compared with `example_workflows/g1_arena_posttraining/scripts/capture_reset_replay.py` and `scripts/ppo/compare_reset_observations.py`; all measured 2026-09-13 on one NVIDIA L40:

| Renderer setting | Environments with identical state | Environments with identical image | RGB values that changed | Mean abs. error (0-255) | RMS error | Max error |
|---|---|---|---|---|---|---|
| default, earlier reset path | 8/8 | 0/8 | 64.2% | 1.303 | 2.369 | 105 |
| FXAA, earlier reset path | 8/8 | 0/8 | 63.7% | 1.354 | 2.541 | 112 |
| FXAA | 8/8 | 0/8 | 61.2% | 1.213 | 2.312 | 89 |
| strict, deterministic RTX only | 8/8 | 0/8 | 62.5% | 1.124 | 1.853 | 82 |
| strict, 10 rerenders per reset | 8/8 | 0/8 | 61.5% | 1.062 | 1.796 | 81 |
| strict, tone-map dithering off | 8/8 | 0/8 | 61.8% | 1.109 | 1.870 | 106 |
| strict, all frame-varying RTX effects off (current `SIM_STRICT_REPRODUCIBILITY=1`) | 8/8 | 0/8 | 60.1% | 0.913 | 1.421 | 50 |

Every setting restores the physical state exactly, and none reproduces a raw camera frame exactly, so two evaluations of one checkpoint from the same restored starts can still differ in actions and outcomes.
Without state restore, two fresh 128-environment evaluations of the previous base checkpoint on DrillLift with the same reset seed (100000) scored 69/128 and 80/128 (two NVIDIA L40 GPUs, 2026-09-13).
Compare checkpoints over enough episodes against a base scored in the same session.

## Reference results

The maintained results are in the workflow README; the tables below are this integration's reference numbers, each read against the protocol stated with it.
They were measured in September 2026 with the previous base checkpoint (100,000 steps) in an older G1 Factory scene, not with the published `nvidia/G1-Factory-GR00T-1.7`, so they do not predict its scores.

### Base policy

Mean of three independent N=50 samples, GPU physics; at N=50 a per-task deviation within ±0.15 is consistent.
The CPU-physics column (`EVAL_SIM_DEVICE=cpu`, N=100) mostly agrees within noise, but reads higher on the contact-rich DrillPnP and PushShelfForward.
The last column is the base measured in the same container and session as the DSRL results below, one 100-episode run per task.

| Task | GPU physics | CPU physics | GPU physics, N=100 (Sep 2026) |
|---|:---:|:---:|:---:|
| BoxLift | 0.93 | 0.99 | 0.94 |
| BoxLiftFloor | 0.91 | 0.95 | 0.92 |
| DrillLift | 0.83 | 0.85 | 0.86 |
| DrillLiftObs | 0.89 | 0.86 | 0.92 |
| PushButton | 0.53 | 0.50 | 0.59 |
| DrillPnP | 0.75 | 0.86 | 0.79 |
| PickDrillFromHolder | 0.22 | 0.19 | 0.19 |
| BoxTableToShelfPnP | 0.71 | 0.77 | 0.73 |
| PushShelfForward | 0.39 | 0.45 | 0.41 |

### Reference DSRL results

The recipe behind these numbers is the branch default, pinned by `tests/unit/test_recipe_pins.py`: noise bound 1.0, `training.chunk_size=40`, `dsrl.action_exec_len=8`, a synchronous learner, a 4-member critic with pooled features and mean target reduction, actor learning rate 5e-5, critic learning rate 1e-4, batch 64, discount 0.995 and an auto-tuned temperature.
`example_workflows/g1_arena_posttraining/scripts/repro_sep2026.sh` adds only the run length, evaluation cadence, checkpoint retention, prompt and N=100 scoring settings:

```bash
S=example_workflows/g1_arena_posttraining/scripts/repro_sep2026.sh
$S train   DrillPnP 1                               # 40k steps, seed 1
VARIANT=b2 $S train DrillLift 0                     # the noise-bound-2.0 cells
$S base    DrillPnP                                 # frozen base, N=100
$S rescore DrillPnP <run>/checkpoints/16088         # trained checkpoint, N=100
```

Retention keeps the best-evaluated step, the last periodic save and `final`; re-score both of the first two.
One 40k run is about ten hours on one L40; a base or re-score is about two hours.
Keep the DSRL budget at 40k steps: longer runs collapse once the auto-tuned temperature decays, and `algorithm.ent_coef=0.05` fixes the temperature (`docs/dsrl.md`).

Best in-training 50-episode evaluation per task, over every run of that task listed, against the same-session N=100 base; September 2026, default recipe unless a variant is named:

| Task | base (N=100) | DSRL (best in-training eval) | run · eval step | differs from recipe |
|---|:---:|:---:|---|---|
| BoxLift | 0.94 | 1.00 | 40k run, seed 0 · step 15,280 | — |
| BoxLiftFloor | 0.92 | 1.00 | 200k run, seed 0 · step 47,736 | — |
| DrillLift | 0.86 | 0.94 | 40k run, seed 0 · step 10,136 | `dsrl.noise_action_bound=2.0` |
| DrillLiftObs | 0.92 | 1.00 | 200k run, seed 0 · step 5,280 | — |
| DrillPnP | 0.79 | 0.86 | 40k run, seed 1 · step 16,080 | — |
| PushButton | 0.59 | 0.78 | 40k run, seed 1 · step 25,824 | `dsrl.noise_action_bound=2.0` |
| PickDrillFromHolder | 0.19 | 0.42 | 200k run, seed 0 · step 46,560 | — |
| BoxTableToShelfPnP | 0.73 | 0.86 | 200k run, seed 0 · step 10,288 | — |
| PushShelfForward | 0.41 | 0.44 | 40k run, seed 0 · step 5,688 | `dsrl.noise_action_bound=2.0` |

A best-in-training number is the maximum over many noisy 50-episode draws, so it overestimates the checkpoint's success rate.
Re-scoring the retained checkpoints of the default-recipe 40k runs at N=100 against the same-session base:

| Task | base | retained-best ckpt (step) | last periodic save (step) | adopt? |
|---|:---:|:---:|:---:|---|
| BoxLift | 0.94 | 0.94 (35.7k) | 0.97 (35.7k, same weights) | tie — the two numbers are one checkpoint scored twice |
| BoxLiftFloor | 0.92 | 0.97 (15.6k) | 0.96 (36.5k) | yes, retained-best (+5) |
| DrillLift | 0.86 | — | 0.76 (36.3k) | no (base ships) |
| DrillLiftObs | 0.92 | 0.95 (31.5k) | 0.89 (36.7k) | retained-best only (+3) |
| DrillPnP, seed 0 | 0.79 | 0.73 (30.9k) | 0.73 (36.0k) | no |
| DrillPnP, seed 1 | 0.79 | 0.84 (16.1k) | 0.67 (37.3k) | yes, retained-best (+5) |
| PushButton | 0.59 | — | 0.42 (35.7k) | no (base ships) |
| PickDrillFromHolder | 0.19 | — | 0.17 (36.3k) | no |
| BoxTableToShelfPnP | 0.73 | 0.66 (16.2k) | 0.80 (37.3k) | yes, last save (+7) |
| PushShelfForward | 0.41 | — | 0.09 (37.9k) | no (base ships) |

Adopt a trained checkpoint only when its N=100 re-score beats the same-session base; which of the two retained checkpoints wins varies by task.
`fingerprints.jsonl` records the actor's SHA-256 at every evaluation, save and load, so a checkpoint holds exactly the weights of the evaluation sharing its step.
Evaluation scenes are unseeded, so two 100-episode scores of one checkpoint differ by about three points.

### Algorithm zoo on G1 Factory

Two more algorithms under `isaac_rl_posttraining/algorithms/`, selected with `algorithm=<name>`, were run on the nine tasks in September 2026 with the protocol above (40k steps, 50-episode evaluations every 5k, same-session N=100 bases, one seed):

- **Residual Flow Steering (`rfs`)** steers the flow noise like DSRL and adds a bounded residual (default ±0.01 in normalized action units) to the decoded chunk.
- **SAC-edit (`sac_edit`)** skips the latent: the actor emits a bounded edit (`direct_action.actor_edit_scale`, default 0.02) around the frozen policy's 35-dimensional proposal.

`example_workflows/g1_arena_posttraining/scripts/zoo_recipes.sh` launches the conditions that reached or beat the base; everything it does not set is the branch default, pinned by `tests/unit/test_sac_edit_guards.py` (for SAC-edit: fp32, a 0.02 edit, learning from 4k steps, gradient clipping at 1.0).

```bash
S=example_workflows/g1_arena_posttraining/scripts/zoo_recipes.sh
$S rfs      DrillPnP 0     # RFS, branch defaults
$S rfs_pc   DrillPnP 0     # RFS, residual conditioned on the frozen policy's proposal
$S rfs_nb2  PushButton 0   # RFS, flow-noise bound widened to 2.0
$S sac_edit DrillPnP 0     # SAC-edit
```

Best in-training 50-episode evaluation per task (steps ≥ 5k, seed 0), against the same-session N=100 base; selected the same way as the DSRL table above.
Each cell is the peak, the gain over the base in points and in percent of the base, and the step of that evaluation.

| Task | base (N=100) | DSRL, Table A§ | `rfs` | `rfs_pc` | `rfs_nb2` | `sac_edit`¶ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| BoxLift | 0.93 | 1.00 (+0.06, +6%) 15.3k vs 0.94 | 1.00 (+0.07, +8%) 35.9k | 1.00 (+0.07, +8%) 15.3k | 0.96 (+0.03, +3%) 36.4k | 1.00 (+0.07, +8%) 10.5k |
| BoxLiftFloor | 0.91 | 1.00 (+0.08, +9%) 47.7k vs 0.92 | 1.00 (+0.09, +10%) 35.8k‡ | 1.00 (+0.09, +10%) 36.9k | 0.96 (+0.05, +5%) 5.2k | 1.00 (+0.09, +10%) 26.8k |
| BoxTableToShelfPnP | 0.71 | 0.86 (+0.13, +18%) 10.3k vs 0.73 | 0.80 (+0.09, +13%) 11.4k | 0.60 (-0.11, -15%) 5.4k | 0.68 (-0.03, -4%) 27.5k | 0.74 (+0.03, +4%) 15.9k |
| DrillLift | 0.83 | 0.94 (+0.08, +9%) 10.1k vs 0.86 | 0.98 (+0.15, +18%) 15.4k‡ | 0.90 (+0.07, +8%) 26.2k | 0.96 (+0.13, +16%) 25.7k | 0.88 (+0.05, +6%) 5.3k |
| DrillLiftObs | 0.89 | 1.00 (+0.08, +9%) 5.3k vs 0.92 | 0.92 (+0.03, +3%) 10.3k | 1.00 (+0.11, +12%) 10.3k | 0.86 (-0.03, -3%) 36.7k | 0.92 (+0.03, +3%) 16.0k |
| DrillPnP | 0.75 | 0.86 (+0.07, +9%) 16.1k vs 0.79 | 0.94 (+0.19, +25%) 25.9k | 0.92 (+0.17, +23%) 15.7k | 0.78 (+0.03, +4%) 16.2k | 0.88 (+0.13, +17%) 36.8k |
| PickDrillFromHolder | 0.22 | 0.42 (+0.23, +121%) 46.6k vs 0.19 | 0.42 (+0.20, +91%) 15.6k | 0.30 (+0.08, +36%) 26.2k | 0.32 (+0.10, +45%) 15.2k | 0.32 (+0.10, +45%) 5.1k |
| PushButton | 0.53 | 0.78 (+0.19, +32%) 25.8k vs 0.59 | 0.54 (+0.01, +2%) 10.2k | 0.60 (+0.07, +13%) 10.5k | 0.78 (+0.25, +47%) 20.7k | 0.72 (+0.19, +36%) 35.6k |
| PushShelfForward | 0.39 | 0.44 (+0.03, +7%) 5.7k vs 0.41 | 0.32 (-0.07, -18%) 22.3k | 0.52 (+0.13, +33%) 33.0k | 0.40 (+0.01, +3%) 22.3k | 0.24 (-0.15, -38%) 10.7k |
| **mean gain · tasks at or above base** | | +0.11, +24% · ≥ base on 9/9 | +0.08, +17% · ≥ base on 8/9 | +0.08, +14% · ≥ base on 8/9 | +0.06, +13% · ≥ base on 7/9 | +0.06, +10% · ≥ base on 8/9 |

§ The peaks of the first DSRL table, with the gain over their own same-session base (after "vs"); that table takes the best over all runs of a task, including 200k-step runs, seed 1 and the noise-bound-2.0 variant, so two of its peaks lie beyond 40k steps.
The DSRL control arm of this campaign reproduced it on the three tasks it ran (BoxLift 1.00 against 0.93, DrillPnP 0.82 against 0.75, PushButton 0.64 against 0.53).
¶ Measured before the SAC-edit entropy fix, with a runaway temperature; treat as bounded-noise injection until it is re-validated.
‡ The 40k run lost its evaluations to an evaluator hang after a simulator frame timeout (`Arena step failed: failed to receive frame`); the cell comes from the first 40k steps of the 100k run of the same condition and seed.

RFS's `algorithm/rfs.yaml` differs from the DSRL recipe in two critic settings (`critic_target_reduction: min`, `pooled_critic_features: false`), so the RFS-versus-DSRL comparison is not fully controlled.
What the campaign established about each algorithm (bounds that collapse, the SAC-edit temperature, the 40k budget) is on `docs/pages/algorithms/rfs.md` and `docs/pages/algorithms/sac_edit.md`.
Adopt a checkpoint from any of these runs only after the N=100 re-score of the retained checkpoint beats the same-session base.
