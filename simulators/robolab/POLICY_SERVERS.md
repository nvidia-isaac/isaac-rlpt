# GR00T N1.7 policy server

Start the native `nvidia/GR00T-N1.7-DROID` server and run a RoboLab smoke evaluation against it.
No harness monitoring, steering, recovery or planner behavior is enabled here.
This assumes [`SETUP.md`](SETUP.md) is done.

The RoboLab client sends the left exterior and wrist RGB images as HWC `uint8` at 180x320 without letterboxing, receives seven joint-position actions and one gripper action (binarized at 0.5), and executes eight actions before querying the server again.

1. Authenticate to the gated Hugging Face models through the host credential store; never put a token in a script, config, shell history or report.
   On a remote runner whose credentials are not in the default location, point `HF_HOME` at the authorized cache.

2. Start the server from the GR00T checkout.
   `--use-sim-policy-wrapper` is required because the RoboLab client uses GR00T's flat simulation-modality keys.

   ```bash
   cd "$EXT_DEPS_ROOT/isaac-gr00t"
   export HF_HOME=/path/to/authorized/huggingface-cache
   CUDA_VISIBLE_DEVICES=0 uv run python gr00t/eval/run_gr00t_server.py \
     --model-path nvidia/GR00T-N1.7-DROID \
     --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
     --device cuda \
     --host 127.0.0.1 \
     --port 5555 \
     --use-sim-policy-wrapper
   ```

3. In a second terminal, run the native smoke evaluation from the RoboLab checkout.

   ```bash
   cd "$EXT_DEPS_ROOT/robolab"
   uv run python policies/gr00t/run.py \
     --headless \
     --task BananaOnPlateTask \
     --num-envs 10 \
     --num-runs 1 \
     --remote-host 127.0.0.1 \
     --remote-port 5555 \
     --open-loop-horizon 8 \
     --enable-subtask \
     --video-mode all \
     --output-folder-name gr00t_n17_smoke
   ```
