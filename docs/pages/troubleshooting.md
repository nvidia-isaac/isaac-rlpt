# Troubleshooting

Start with the dry run: it reproduces most configuration failures in seconds, with no GPU, simulator or checkpoint.
The Video-to-Data (V2D) launchers take `--dry-run` instead.

```bash
ISAAC_RL_POSTTRAINING_DRYRUN=1 <your launcher or scripts/train.py command>
```

## Configuration and launch

| What you see | What to do |
|---|---|
| `g1_arena needs ARENA_SERVER_URL for its train environments` | Set `ARENA_SERVER_URL` and `ARENA_EVAL_SERVER_URL`. The workflow launchers set them; running `scripts/train.py` directly, set them yourself, and placeholders such as `tcp://127.0.0.1:7300` are enough for a dry run |
| A config key is refused | The message lists the keys that group does have, and names the replacement when a key was renamed |
| `--num_steps ... set the steering trainer's cadence`, or `--runtime ... apply to --algorithm ppo and dsrl_ppo` | The cadence flags (`--num_steps`, `--eval_freq`, `--save_interval`, `--eval_num_envs`, `--skip_first_eval`) are steering-only; with `ppo` or `dsrl_ppo` use `--iterations` and the `ppo.eval_every` / `ppo.checkpoint_every` overrides. The placement flags (`--runtime`, `--learner_gpus`, `--rollout_gpus`, `--iterations`) are for `ppo` and `dsrl_ppo` only, and are refused even at their default value. [cli_reference.md](cli_reference.md) has the full split |
| `--num_envs N must be a positive multiple of --rollout_gpus M` | Raise `--num_envs` to a multiple of `--rollout_gpus`; every PPO rollout worker hosts the same number of scenes |
| `--runtime native runs one learner rank and one rollout worker on one GPU` | Only `--runtime rlinf` places more than one of either. Install it with `uv sync --extra rlinf` |
| A dry run fails with a specification mismatch | `plan.data_specs` compares what the simulator emits and accepts against what the policy reads. The usual causes are a checkpoint whose action width does not match the simulator's, and a camera key the environment does not emit |
| A dry run fails with "the launch plan found N problems" | Several checks failed; each numbered item names the keys to change. `plan.checks` lists the same problems and what stayed unverified |
| A simulator has no registered entry | The first segment of `env.env_name` selects the simulator. See [extending/add_a_simulator.md](extending/add_a_simulator.md) |

## Custom simulators

`simulators/_template/` is a stand-in that passes every check below; compare your package with it when one fails.

| What you see | What to do |
|---|---|
| `no simulator entry is registered for environment 'my_robot/...'` | Attach the entry with `+simulator.entry=<module>:<Class>`, or register it under the first segment of `env.env_name` |
| `... is not a complete simulator entry: ...` | The entry class lacks `make_environment` or a non-empty `name`; the message names the member |
| `env.max_episode_steps is unset and task '<task>' of env.env_name=... names none of the simulator's tasks` | The selected `simulator=` group does not list the task; an out-of-tree entry runs under the default `g1_arena` group. Set `env.max_episode_steps=<cap>`, or add the task to your own simulator group |
| `data specs of simulator <name> and the policy disagree: action: emitted float32[35] x 40 steps vs accepted float32[7]` | `base_model=mock` defaults to G1's geometry, 35-wide actions in 40-step chunks. Match it to your simulator with `base_model.action_dim`, `dsrl.noise_dim`, `base_model.action_horizon` and `training.chunk_size` |
| `<name> needs MY_SERVER_URL for its train environments` | The entry declares `server_url_variables`; export both addresses. Placeholders are enough for a dry run |
| `Environment provider ... requires optional dependencies that are not installed` | Install the listed packages in the environment that runs the simulator, or move the import inside the method that needs it |
| Run directories and messages carry a different name than `env.env_name` | The entry's `name` labels the run; the first segment of `env.env_name` only selects the entry. Set `name` to the segment unless one entry serves several (`G1Arena` serves `arena` and `isaac`) |

## Checkpoints and policies

### The policy loads but scores zero everywhere

The most common cause is an **observation mismatch**: frame orientation, or the layout of the state vector.
On LIBERO the adapter in `simulators/libero/policies/gr00t.py` owns both.

The next is **the wrong normalization statistics**, which produce silently wrong actions.
Give exactly one source: `base_model.dataset_dir`, or `base_model.multitask_dirs` with `base_model.task_dir`, or the checkpoint's own with `base_model.dataset_dir=null` and `base_model.task_instruction='<prompt>'`.

| What you see | What to do |
|---|---|
| `no baked modality config for embodiment tag ...` | Point `base_model.ckpt` at the suite subfolder, for example `.../GR00T-N1.7-LIBERO/libero_10`, not at the snapshot root |
| `flash_attn is not installed` | Nothing. Attention falls back to scaled dot-product attention, which is correct |
| The GR00T editable install rejects your Python version | Relax the exact `requires-python` pin of your Isaac-GR00T clone to `>=3.10,<3.12` |
| A model fails to load with a `transformers` error | Check `transformers.__version__` at run time. GR00T N1.7 needs `transformers>=4.57.3`, 5.x breaks its config subclass, and a stale editable install earlier on `sys.path` can supply an older one |
| The checkpoint's backbone is gated on Hugging Face | Point it at a local copy with `base_model.backbone_model_name_or_path=/local/dir`, or `POSTTRAIN_BACKBONE_MODEL=/local/dir` in front of a launcher |
| `base_model=<name> has no cacheable features; set ppo.cache_features=false` | Only some adapters can ship the frozen backbone's outputs instead of pixels. GR00T N1.7 requires `ppo.cache_features=true`; a base model without the feature requires it off |
| A steered call raises `NotImplementedError` | That base model cannot be steered — `dreamzero_server` is the shipped example. It can only evaluate the frozen policy (`training.num_rollouts=0 eval.eval_base=true`); use `base_model=dreamzero_local` to steer |

## G1 Factory

| What you see | What to do |
|---|---|
| `ImportError: cannot import name 'set_control_rate_50hz'` | The simulator's IsaacLab-Arena or Isaac Lab is not at the commit in `example_workflows/g1_arena_posttraining/externals.lock.yaml`, for example Arena's own environment at G1 Factory's pin. Run step 7 of the G1 Factory installation, `( cd example_workflows/g1_arena_posttraining && bash ./scripts/fetch_externals.sh && uv sync )`, and leave `SIM_PYTHON` unset |
| The simulator never prints `listening on` | Read the simulator log the launcher names. The usual causes are missing scene assets (download [`nvidia/G1-Factory`](https://huggingface.co/datasets/nvidia/G1-Factory) into `$G1_FACTORY/datasets/assets`) and an unaccepted Omniverse EULA; containers may need `OMNI_KIT_ALLOW_ROOT=1`. Startup also grows with `NUM_ENVS`: the launcher waits `600 + 120 * NUM_ENVS` seconds, and `SIM_BOOT_TIMEOUT` overrides that |
| A task the policy handles scores 0% | Something is overriding `ARENA_IMAGE_RESOLUTION`. The policy must receive the native camera frame; the dry run's plan should show `ARENA_IMAGE_RESOLUTION=0` |
| Two runs collide on a port | A run's two simulators listen on consecutive ports from `PORT_BASE + 16*GPU`, `PORT_BASE` defaulting to 7300, and a launch refuses ports another run already holds. Keep one run per GPU index, or set `PORT_BASE`. On a shared machine, `SIM_STARTUP_LOCK=/tmp/sim_boot.lock` starts them one at a time |
| The batched server refuses to boot | `NUM_ENVS>1` needs the patch below applied once to the Arena checkout. `ARENA_SKIP_WBC_PATCH_CHECK=1` skips the check for an embodiment without the homie controller |
| Only scene 0 ever logs a success at `NUM_ENVS>1` | A success term that compares world-frame positions against fixed goals fires only for scene 0 unless it subtracts `env.scene.env_origins`. The shipped `LM*` terms all do. Check with the two commands below before raising `NUM_ENVS` |

```bash
git -C "$ISAACLAB_ARENA_PATH" apply "$PWD/simulators/g1_arena/patches/arena-homie-per-env-wbc.patch"
uv run bash example_workflows/g1_arena_posttraining/scripts/preflight_batched.sh DrillLift
uv run python example_workflows/g1_arena_posttraining/scripts/smoke_batched.py --url tcp://127.0.0.1:7300 --num-envs 2
```

### Training collapses after about 45,000 steps

With `algorithm.ent_coef=auto` the SAC temperature keeps falling until it no longer limits the actor; the latents leave the distribution the policy was trained on and success collapses.
RFS shows the same failure in long runs.
Keep the budget at 40,000 steps, or set `algorithm.ent_coef=0.05` to fix the temperature.

## LIBERO

| What you see | What to do |
|---|---|
| `import libero` fails after the editable install | `touch "$LIBERO_ROOT/libero/__init__.py"` |
| The first import hangs, or raises `EOFError` | Seed `LIBERO_CONFIG_PATH` with a `config.yaml` as in [the LIBERO setup](workflows/libero.md#install-libero), and export it in every shell |
| The wrong tasks or assets resolve | A stale global `~/.libero/config.yaml` from another checkout. Set `LIBERO_CONFIG_PATH` |
| Frames are black, or `MUJOCO_GL` errors | No usable GL backend. Check `ldconfig -p \| grep libEGL_nvidia`, or install `libosmesa6` and use `MUJOCO_GL=osmesa`. EGL errors at interpreter exit are harmless, and `env.close()` avoids them |
| `TypeError: mj_fullM(): incompatible function arguments` | MuJoCo 3.x with robosuite 1.4.x on a utility path. Rollouts themselves run on either version: pi0 needs `mujoco==3.3.1`, and the GR00T reference numbers were measured on `2.3.7` |
| `No module named 'gym'` | Install `gym==0.25.2`. LIBERO imports the legacy `gym` even though this integration uses gymnasium |
| pi0 scores about 2 to 4 percent instead of about 33 | Install `mujoco==3.3.1`, **after** the LIBERO pins |
| DreamZero scores near zero | `SUITE` must be the checkpoint's own fine-tuning suite, `libero_spatial` for the reference 5B checkpoint |

## Isaac Lab

| What you see | What to do |
|---|---|
| The Isaac Lab scripts directory is not writable (container path) | Use the command `setup_env.sh` prints, `$ISAAC_RLPT_PYTHON $ISAAC_RLPT_RL_SCRIPTS/train.py --rl_library isaac_rlpt ...`, rather than `uv run isaaclab`, which assumes a writable checkout |
| An episode cap is rejected | For PPO fine-tuning the cap must be a whole number of action chunks (`base_model.action_horizon`) |

## Video-to-Data

The launcher refuses, before any GPU work:

| Refusal | Fix |
|---|---|
| an existing output directory | use a new `POSTTRAIN_OUTPUT_DIR` per run |
| an incomplete, end-effector, mixed or differently ordered base checkpoint | use the 58-dimensional absolute-joint checkpoint with the complete joint contract |
| a Video-to-Data revision that differs from the pin | check out the commit in `externals.lock.yaml` |
| evaluation counts with `N < B` or `N % B != 0` | make `--eval-num-episodes` a multiple of `--eval-num-envs` |

## Runtimes and placement

| What you see | What to do |
|---|---|
| Ray groups run out of GPUs | Roles get GPUs of their own while there are enough and share them otherwise; the dry run's `plan.placement.*` lines show where each role runs. Under `rlinf`, learner ranks refuse a card the rollout workers use: move `runtime.rlinf.first_accelerator` past them or set `runtime.rlinf.exclusive_accelerators=false` |
| A single-simulator run gains nothing from Ray | Expected. `runtime=native` is the reference, and Ray costs about 4 GiB resident at idle plus a few seconds of boot per run |
| RLInf ignores `CUDA_VISIBLE_DEVICES` | It builds NCCL groups from detected hardware. Place work with the runtime's own settings instead of device masking |
| A test is skipped | A test whose dependency or environment is missing skips with a stated reason. The markers are `arena`, `gpu`, `libero`, `rlinf`, `rlinf_cluster`, `native_ray` and `slow` |

Flags and overrides are in [cli_reference.md](cli_reference.md); per-workflow setup is in [the workflow pages](workflows/index.md).
