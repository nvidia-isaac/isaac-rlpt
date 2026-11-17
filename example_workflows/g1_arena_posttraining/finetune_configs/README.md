# The base checkpoint: GR00T modality configs and the fine-tuning recipe

The workflow's base policy is GR00T-N1.7 fine-tuned on the nine G1 Factory teleoperation datasets, published as [`nvidia/G1-Factory-Benchmark`](https://huggingface.co/datasets/nvidia/G1-Factory-Benchmark); this page holds what is needed to reproduce that checkpoint.
Fine-tuning runs entirely in the upstream Isaac-GR00T repository and its own environment, independent of the simulator; the resulting `checkpoint-<step>/` directory is the only hand-off.
Do not use the Arena's own fine-tune workflow: it targets GR00T-N1.5 and cannot load an N1.7 config.

## Modality configs

The modality configs tell the GR00T fine-tuner which columns of a LeRobot dataset are the robot's state and which are the actions to predict; they are not part of upstream Isaac-GR00T.
Pass one to `launch_finetune.py --modality-config-path`; each registers itself under `EmbodimentTag.NEW_EMBODIMENT`.

- `g1_35dim_config.py`, the benchmark base: a 35-number action (both arms, both hands, the waist, plus a standing-height and a walk command); the robot's balance controller moves the legs from those two commands, and the state is the upper body only.
- `g1_43dof_config.py`: predicts every joint including the legs (43 numbers).
- `g1_noleg_config.py`: the 43-DoF layout with the leg actions dropped (31-number action, legs left to the balance controller); the state still includes the legs.

## Recipe

1. Build the policy environment, once.
   It pins newer dependencies than the learner environment (torch 2.7.1, transformers 4.57.3, flash-attn, torchcodec), so it lives inside the Isaac-GR00T checkout.
   Pull the Git LFS wheels before `uv sync`, or the flash-attn extraction fails with "Invalid zip file structure".

   ```bash
   RLPT_ROOT=$PWD                          # start from this repository's root, isaac-rlpt/
   git clone https://github.com/NVIDIA/Isaac-GR00T ../Isaac-GR00T && cd ../Isaac-GR00T
   git lfs install && git lfs pull --include="scripts/deployment/dgpu/wheels/*"  # prebuilt flash-attn/torchcodec wheels
   sudo apt-get install -y ffmpeg          # torchcodec video backend
   export CUDA_HOME=/usr/local/cuda
   uv sync --python 3.10
   ```

2. Download the base model:

   ```bash
   huggingface-cli download nvidia/GR00T-N1.7-3B --local-dir <BASE_DIR>   # ungated
   ```

3. Place one LeRobot dataset per task at `$POSTTRAIN_DATA_ROOT/<LMTask>/lerobot`; [`nvidia/G1-Factory-Benchmark`](https://huggingface.co/datasets/nvidia/G1-Factory-Benchmark) holds all nine under `datasets/<LMTask>/lerobot` (download it as in the [guide](../../../docs/pages/workflows/g1_arena.md#installation)).

4. Fine-tune.
   `--dataset-path` is split on `:`, so a multitask run colon-joins all nine `lerobot` directories, and `--num-gpus N` launches `torchrun` itself.
   The recipe of the published checkpoint freezes the language backbone and trains the vision encoder, the projector and the diffusion head, for 20,000 steps on 8 GPUs at a global batch of 32:

   ```bash
   cd "$RLPT_ROOT/../Isaac-GR00T"
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run python gr00t/experiment/launch_finetune.py \
     --base-model-path <BASE_DIR> \
     --dataset-path "$POSTTRAIN_DATA_ROOT/LMBoxLift/lerobot:$POSTTRAIN_DATA_ROOT/LMDrillLift/lerobot:...:$POSTTRAIN_DATA_ROOT/LMPushShelfForward/lerobot" \
     --embodiment-tag NEW_EMBODIMENT \
     --modality-config-path "$RLPT_ROOT"/example_workflows/g1_arena_posttraining/finetune_configs/g1_35dim_config.py \
     --num-gpus 8 --output-dir <OUT> \
     --max-steps 20000 --save-steps 5000 --save-total-limit 8 \
     --global-batch-size 32 --learning-rate 1e-4 --warmup-ratio 0.05 \
     --tune-visual --tune-projector --tune-diffusion-model --no-tune-llm \
     --dataloader-num-workers 8 --use-wandb
   ```

   The other settings are the fine-tuner's defaults: AdamW with a cosine schedule, weight decay 1e-5, state dropout 0.2, action horizon 40 and seed 42.
   On a managed node, run this in a held foreground session; a detached `nohup` or `setsid` process is reaped when the launching session ends.

5. Hand off.
   Each `checkpoint-<step>/` holds the weights plus `statistics.json`, `processor_config.json` and `embodiment_id.json`; `statistics.json` holds the normalization statistics the policy decodes with, and without it, or with another embodiment's statistics, the policy produces silently wrong actions.
   Point `POSTTRAIN_BASE_CKPT` at the checkpoint to evaluate or steer.
