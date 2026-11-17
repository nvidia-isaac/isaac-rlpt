# Video-to-Data (V2D) simulator integration

This package connects the simulator-neutral rollout runtime to a separately served Video-to-Data Isaac Sim process: provider `v2d` (`simulators/v2d/env.py`), entry `V2D` (`simulators/v2d/runtime.py`), config group `simulator/v2d.yaml`.
It contains no tissue-box scene logic; a workflow supplies a capability module that builds and scores the task.
To install, train and evaluate, follow `docs/pages/workflows/v2d_box_lift.md`.

## Contract

- Action and state width: 58 joint dimensions (`JOINT_DIM` in `simulators/v2d/contract.py`: 7 per arm, 22 per hand).
- State order: right arm, left arm, right fingers, left fingers.
  Action order: right arm, right fingers, left arm, left fingers.
- Cameras: `video.front`, `video.right_wrist_view` and `video.left_wrist_view` RGB images (the front camera and two wrist cameras) at native resolution.
- One simulator process owns one environment or a native batch; the runtime uses one rollout worker for that process (`WorkerLayout(workers=1, batched=num_envs > 1)`), and a `[chunk, 58]` action executes in one server round trip.
- Training and evaluation have separate server endpoints, and evaluation runs on its dedicated server.
  Training decoder micro-batching (`training.policy_decode_batch_size`) is independent of evaluation, which decodes all active environments in one call.
- The simulator returns terminal observations and task metrics before autoreset; finished environments stay excluded from inference and scoring until the next evaluation round.

The transport handshake validates the protocol version, task identity, episode horizon, action width, environment count, image shapes, and the typed observation and action declarations before rollout starts.

## Server

The workflow launcher writes the capability configuration and starts this server for you; run it by hand only to debug a server.
It runs under Video-to-Data's Isaac Sim Python (`V2D_SIM_PYTHON` in the workflow) from the repository root; `--headless` and `--device` are Isaac Lab's app-launcher flags.
`--dry-run` on the launcher prints the exact command, with real paths for the two files below:

```bash
"$V2D_SIM_PYTHON" -m simulators.v2d.server \
  --headless \
  --device cuda:0 \
  --capability example_workflows.v2d_box_lift_posttraining.capability:create_capability \
  --capability-config /path/to/capability.json \
  --num-envs 32 \
  --port 5556 \
  --seed 10000 \
  --startup-timeout-seconds 900 \
  --receipt /path/to/server_receipt.json
```

The capability owns the simulator imports, scene construction, reset randomization, reward and terminal success; the server owns only the wire lifecycle, writes the receipt (the capability name and the endpoint spec) before it listens, and prints `[v2d] listening on <address>`.
`--startup-timeout-seconds` must exceed 60; if the server is still starting 60 seconds before the timeout, it dumps a traceback.

## Client configuration

Select `simulator=v2d` (`provider: v2d`, the task `tissue_box_lift_hold` with a 550-step cap, `action_dim: 58`, `image_resolution: 0`), export `V2D_SERVER_URL` for training and `V2D_EVAL_SERVER_URL` for evaluation (`tcp://host:port`), and use a GR00T base model whose checkpoint matches the 58-D absolute-joint contract; the entry rejects any other action width or a non-native camera resolution.
Other task behavior stays workflow-owned.

Run the CPU contract tests with:

```bash
uv run pytest simulators/v2d/tests -q
```
