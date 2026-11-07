# Isaac Lab

This integration steps [Isaac Lab 3.0](https://github.com/isaac-sim/IsaacLab) manager-based tasks in a simulator child process of each rollout worker (`simulators/isaaclab/subprocess_env.py`) and presents their observations in the keys of the GR00T checkpoint fine-tuned for each task.
Provider `isaaclab` (`simulators/isaaclab/env.py`), entry `IsaacLab` (`simulators/isaaclab/runtime.py`), config groups `simulator/isaaclab_franka_stack.yaml` and `simulator/isaaclab_trocar.yaml`.

Install, training and evaluation are in the workflow guide, [docs/pages/workflows/isaaclab_gr00t.md](../../docs/pages/workflows/isaaclab_gr00t.md); the `isaac_rlpt` launcher module is in [`example_workflows/isaaclab_gr00t_posttraining/`](../../example_workflows/isaaclab_gr00t_posttraining/README.md).
This page is the simulator reference.

Each task id has a mapping (`TASK_MAPPINGS`) from the simulator to the checkpoint's conventions:

* `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0`: a Franka Panda stacking three cubes, in the layout of RLinf's `RLinf-Gr00t-SFT-Stack-cube` checkpoint (`FRANKA_STACK_GR00T_MAPPING`, data config `simulators/isaaclab/gr00t_data_configs.py::IsaacLabFrankaDataConfig`). This is the task v0.1 supports.
* `Isaac-Assemble-Trocar-G129-Dex3-v0` and `-Eval-v0` (fixed tray yaw): a Unitree G1 29-DoF with Dex3 hands, in the layout of Isaac Lab's `IsaacLabDataConfig` (`TROCAR_GR00T_MAPPING`). The mapping is kept in code but not supported in v0.1.

## Environment contract (stack cube)

| Entry | Source | Shape |
|---|---|---|
| `video.image`, `video.wrist_image` | `policy.table_cam`, `policy.wrist_cam` (256x256 via `env_cfg_overrides`) | `uint8 [256, 256, 3]` each |
| `state.x`, `state.y`, `state.z` | `policy.eef_pos[0:1]`, `[1:2]`, `[2:3]` | `float32 [1]` each |
| `state.roll`, `state.pitch`, `state.yaw` | `policy.eef_quat` (w, x, y, z) -> axis-angle (`quat_wxyz_to_axis_angle`), `[0:1]`, `[1:2]`, `[2:3]` | `float32 [1]` each |
| `state.gripper` | `policy.gripper_pos[0:2]` (finger joints) | `float32 [2]` |
| `state` | the seven parts concatenated in that order | `float32 [8]` |
| `prompt` | `"Stack the red block on the blue block, then stack the green block on the red block."` | `str` |

Actions are `float32 [7]`: a relative end-effector pose delta (6) and a gripper command (1) passed straight to Isaac Lab's IK-Rel action (`prefix_pad` 0), the gripper as its sign (`sign_dims: [6]`).
Bounds are the fine-tuning data's action range (`action_bounds`: +-0.1 for the pose deltas, +-1 for the gripper).
`terminated` is Isaac Lab's `success` termination (`cubes_stacked`); the client-side cap is 448 control steps at 20 Hz (RLinf's protocol; Isaac Lab's own horizon is 600).
`info["task_stage"]` counts the `subtask_terms` flags that hold (`grasp_1`, `stack_1`, `grasp_2`).

## Environment contract (trocar)

| Entry | Source | Shape |
|---|---|---|
| `video.room_view`, `video.left_wrist_view`, `video.right_wrist_view` | `camera_images` group: `front_camera`, `left_wrist_camera`, `right_wrist_camera` | `uint8 [224, 224, 3]` each |
| `state.left_arm`, `state.right_arm` | `policy.robot_joint_state[15:22]`, `[22:29]` (joint positions) | `float32 [7]` each |
| `state.left_hand`, `state.right_hand` | `policy.robot_dex3_joint_state[0:7]`, `[7:14]` | `float32 [7]` each |
| `state` | the four groups concatenated in that order | `float32 [28]` |
| `prompt` | `"assemble trocar from tray"` | `str` |

Actions are `float32 [28]` joint-position targets in the model's order (left arm, right arm, left hand, right hand); the wrapper zero-pads them to the simulator's 43 joints (15 uncontrolled leg and waist joints first) and executes `[chunk, 28]` chunks row by row at 50 Hz.
Per-joint bounds are the soft joint-position limits of the controlled joints (`action_term: joint_pos`).
`terminated` is Isaac Lab's `task_success` termination; time-out, object drop and the client-side cap (1000 steps, Isaac Lab's own 20 s horizon) are truncations.
`info["task_stage"]` comes from the task's `get_task_stage` (`stage_fn`).

For both tasks the learner sees `state` plus the DINO embeddings of the views (`steering_remove_obs_keys` drops the raw views and per-part states); the frozen policy receives the full dict.

## Mapping keys

A mapping is a plain dict, mirrored by the `gr00t_mapping` block of the task's yaml (the tests pin the two equal) and passed to `IsaacLabGr00tEnv(mapping=...)`:

| Key | Meaning |
|---|---|
| `cameras` | GR00T video key -> camera term (looked up in `camera_images`, then `policy`) |
| `states` | GR00T state key -> `{term, slice, transform?}`; the optional `transform` (`quat_wxyz_to_axis_angle`) is applied to the whole term before the slice |
| `action_term`, `prefix_pad`, `suffix_pad` | joint-position action term whose order the model follows (bounds from its soft limits) and the zero padding around the model's dims |
| `action_bounds` | explicit `{low, high}` instead of joint limits (IK or delta actions) |
| `sign_dims` | action dims sent as their sign (binary gripper) |
| `task_description` | the language prompt |
| `env_cfg_overrides` | dotted env-config paths assigned before construction (camera resolution, robot USD); string values may use `{ISAACLAB_NUCLEUS_DIR}`-style asset-root placeholders |
| `stage_fn` / `stage_terms` | per-episode progress diagnostic: a `module:function` of the env, or boolean `subtask_terms` to count |

Another Isaac Lab task needs only its own block plus `register_task_mapping(task_id, mapping)`.

## One simulation per process, N scenes inside it

Isaac Sim allows one app and one simulation per process.
A rollout worker's child process starts the headless app (`ISAAC_RLPT_APP_LAUNCHED=1` then stops `launch_app()` from starting a second one), and its views share one Isaac Lab environment (`get_or_create_isaaclab_env`).
The launching process starts no app: the `IsaacLab` entry sets `sized_from_declaration`, so the learner is sized from the task mapping's declared observation and action layout.
`training.num_envs=N` clones the scene N times in that simulation, each clone with its own cameras.
`IsaacLabBatchedGr00tEnv` presents the N scenes as one batched environment (`[N, ...]` observations, `[N, A]` actions) behind the library's `BatchedEnvAdapter`; for DSRL the collector decodes all N scenes in one GR00T forward.
Evaluation uses either the scene-0 view (`IsaacLabGr00tEnv`, `eval.eval_num_envs=1`, which sends zero actions to the other scenes) or a second batched view over the same N scenes (`eval.eval_num_envs=N`), scoring N episodes in parallel.
An evaluation invalidates the running training episodes: the batched view's next step returns them as truncated `disconnected` transitions (not stored) and every scene restarts.
Isaac Lab resets a finished scene inside its own `step`; the adapter's masked reset on the following step restarts it once more (one spent step per episode) and also restarts scenes cut by the client-side cap.

For PPO (`docs/pages/algorithms/ppo.md`), `simulators/isaaclab/runtime.py` hosts all N scenes of one simulation behind the batched environment surface (the `IsaacLab` entry's `make_scenes`), driven by the generic batched rollout worker with the GR00T flow-SDE policy (`isaac_rl_posttraining/policies/gr00t/flow_sde_policy.py`).
A runtime (`docs/runtimes.md`) places that worker outside the learner process; it records one decision per scene and chunk with the denoising chain, behavior log-probability and value the learner needs.

## Direct training

The `--rl_library isaac_rlpt` launcher composes exactly these overrides.
To run without it, inside the Isaac Lab environment the workflow sets up, from the repository root, with `CKPT` set to the downloaded checkpoint:

```bash
$ISAAC_RLPT_PYTHON scripts/train.py algorithm=dsrl_sac \
  simulator=isaaclab_franka_stack base_model=gr00t \
  env.env_name=isaaclab/Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 \
  'env.max_episode_steps=${simulator.max_episode_steps.Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0}' \
  dsrl.noise_dim=32 dsrl.action_exec_len=16 training.chunk_size=16 training.num_envs=1 \
  base_model.ckpt="$CKPT" base_model.embodiment_tag=isaaclab_franka \
  base_model.data_config=simulators.isaaclab.gr00t_data_configs:IsaacLabFrankaDataConfig \
  base_model.dataset_dir=null base_model.dataset_adapter=none \
  "base_model.task_instruction='Stack the red block on the blue block, then stack the green block on the red block.'"
```

This is DSRL on the stack cube in one scene: the steering latent is the checkpoint's 32-wide flow noise over its 16-step chunk, and all 16 decoded steps execute per decision.
`ISAAC_RL_POSTTRAINING_DRYRUN=1` in front prints the resolved plan and exits without starting Isaac Sim.

## Tests

```bash
uv run pytest simulators/isaaclab/tests -q
```

These run without Isaac Sim: fake Isaac Lab environments for both tasks pin the observation mappings, action padding, bounds and sign dims, chunk and cap semantics, stage diagnostics, the shared-instance cache, the batched view, and the Hydra composition of both simulator configs.
