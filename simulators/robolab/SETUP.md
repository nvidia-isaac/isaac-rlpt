# RoboLab setup

RoboLab and the policy repositories are installed outside this repository, at revisions pinned in [`externals.lock.yaml`](externals.lock.yaml); never evaluate from a moving branch.

## Requirements

- An RTX NVIDIA GPU with a 580-series driver (Isaac Sim 5.1.0 camera rendering crashes on other drivers; changing the driver needs an administrator and a reboot).
- Python 3.11, `ffmpeg` and `vulkan-tools`.
- About 80 GB of free disk.
- This repository cloned and `uv sync` done.

## Installation

1. Choose where the external checkouts go (any writable location).

   ```bash
   export POSTTRAIN_ROOT=/path/to/posttrain
   export EXT_DEPS_ROOT=/path/to/external/robot-runtime-deps
   mkdir -p "$EXT_DEPS_ROOT"
   ```

2. Clone the pinned checkouts.

   ```bash
   git clone https://github.com/NVLabs/RoboLab.git "$EXT_DEPS_ROOT/robolab"
   git -C "$EXT_DEPS_ROOT/robolab" checkout 92313e06dd90d2eeedf48da567755e0b1a8e32d1

   git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git "$EXT_DEPS_ROOT/isaac-gr00t"
   git -C "$EXT_DEPS_ROOT/isaac-gr00t" checkout 9c7e746b2cd37a810070a98ef41d290a07e806c2
   ```

3. Without `git-lfs` on the host, fetch the Git LFS object the GR00T checkout needs before its `uv sync` (an aarch64 `torchcodec` wheel).

   ```bash
   python "$POSTTRAIN_ROOT/simulators/robolab/scripts/hydrate_lfs.py" \
     --checkout-root "$EXT_DEPS_ROOT/isaac-gr00t" \
     --revision 9c7e746b2cd37a810070a98ef41d290a07e806c2
   ```

4. Fetch the RoboLab scenes, USD assets and textures for `BananaOnPlateTask` and `BananaInBowlTask` (only these, not the full asset corpus).
   A texture left as an LFS pointer can make Isaac Sim retry the material load forever without advancing the episode.

   ```bash
   python "$POSTTRAIN_ROOT/simulators/robolab/scripts/hydrate_lfs.py" \
     --checkout-root "$EXT_DEPS_ROOT/robolab" \
     --repository NVLabs/RoboLab \
     --revision 92313e06dd90d2eeedf48da567755e0b1a8e32d1 \
     --path assets/scenes/bagel_plate_banana_bowl.usda \
     --path assets/scenes/banana_bowl.usda \
     --path assets/fixtures/franka_table.usd \
     --path assets/fixtures/table_oak.usd \
     --path assets/fixtures/Props/instaceable_meshes.usd \
     --path assets/objects/objaverse/bagel_00.usd \
     --path assets/objects/objaverse/bagel_06.usd \
     --path assets/objects/vomp/plate_large/plate_large.usd \
     --path assets/objects/ycb/banana.usd \
     --path assets/objects/ycb/bowl.usd \
     --path assets/materials/Base/Wood/Oak/Oak_N.png \
     --path assets/materials/Base/Wood/Oak/Oak_BaseColor.png \
     --path assets/materials/Base/Wood/Oak/Oak_ORM.png \
     --path assets/materials/2023_1/vMaterials_2/Paint/Carpaint/textures/smudges_scratches_A_rough.jpg \
     --path assets/materials/Base/Metals/RustedMetal/RustedMetal_ORM.png \
     --path assets/materials/Base/Metals/RustedMetal/RustedMetal_BaseColor.png \
     --path assets/materials/Base/Metals/RustedMetal/RustedMetal_N.png \
     --path assets/backgrounds/default/home_office.exr \
     --path assets/objects/objaverse/textures/bagel_00.png \
     --path assets/objects/objaverse/textures/bagel_06.png \
     --path assets/objects/vomp/plate_large/textures/Plates_Large_BaseColor.png \
     --path assets/objects/vomp/plate_large/textures/Plates_Large_Metallic.png \
     --path assets/objects/vomp/plate_large/textures/Plates_Large_Normal.png \
     --path assets/objects/vomp/plate_large/textures/Plates_Large_Roughness.png \
     --path assets/objects/ycb/textures/obj_000010.png \
     --path assets/objects/ycb/textures/obj_000013.png \
     --path assets/robots/franka_robotiq_2f_85_flattened.usd
   ```

5. Install the RoboLab environment (Isaac Sim 5.1.0, Isaac Lab 2.3.2.post1).
   `--python 3.11` is required: without it `uv` picks the newest interpreter, for which the pinned Isaac Lab wheel does not exist.

   ```bash
   sudo apt install ffmpeg vulkan-tools
   cd "$EXT_DEPS_ROOT/robolab"
   uv sync --python 3.11 --extra isaac51
   export OMNI_KIT_ACCEPT_EULA=Y
   ```

6. Run the preflight before downloading model weights or starting a policy server.
   It changes nothing; it fails on less than 80 GB of free disk, a driver other than the 580 series, or a missing NVIDIA-backed Vulkan renderer (CUDA alone is not enough for camera rendering).

   ```bash
   uv run --project "$POSTTRAIN_ROOT" python "$POSTTRAIN_ROOT/simulators/robolab/scripts/preflight.py" \
     --robolab-root "$EXT_DEPS_ROOT/robolab" \
     --gr00t-root "$EXT_DEPS_ROOT/isaac-gr00t" \
     --require-vulkan
   ```

Next, start the GR00T policy server: [`POLICY_SERVERS.md`](POLICY_SERVERS.md).

## Artifacts

Write run outputs outside the RoboLab source tree through its output option, for example under `posttrain/outputs/robolab/` (untracked).
Commit only the report and the trace metadata needed to reproduce it.
