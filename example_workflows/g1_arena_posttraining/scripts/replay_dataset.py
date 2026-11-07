# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open-loop replay of a recorded lerobot episode through the Arena env-server.

Streams a demonstration episode's *recorded* actions to a live env-server (the
same simulator the DSRL eval drives) and writes the head-camera frames to an
mp4. This is a ground-truth reference -- what the human demonstrations look like
in the current build, with no learned policy involved -- useful for separating
setup/geometry issues from policy-behavior issues.

A live env-server must already be listening at ``--server-url`` (e.g. one started
by ``example_workflows/g1_arena_posttraining/scripts/eval.sh`` in ``replay`` mode). Example::

    python example_workflows/g1_arena_posttraining/scripts/replay_dataset.py \
        --dataset-dir $POSTTRAIN_DATA_ROOT/LMDrillLift/lerobot --episode 0 \
        --server-url tcp://127.0.0.1:7301 --out replay.mp4
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from simulators.g1_arena.embodiment import assemble_wbc_arrays as _assemble_wbc_arrays
from simulators.g1_arena.env import ArenaEnv


def _episode_actions(dataset_dir: str, episode: int) -> np.ndarray:
    """The recorded 35-dim policy action for one episode, in the trained order.

    Reuses :func:`_assemble_wbc_arrays` so the replayed action layout is byte
    identical to what the checkpoint was trained on (arms, hands, waist, then
    the base-height and navigate commands).
    """
    direct = os.path.join(
        dataset_dir, "data", "chunk-000", f"episode_{episode:06d}.parquet"
    )
    if os.path.exists(direct):
        path = direct
    else:
        candidates = sorted(
            glob.glob(os.path.join(dataset_dir, "data", "*", "episode_*.parquet"))
        )
        if not candidates:
            raise FileNotFoundError(f"no episode parquets under {dataset_dir}/data")
        path = candidates[episode % len(candidates)]
    _, actions = _assemble_wbc_arrays(pd.read_parquet(path))
    return np.asarray(actions, dtype=np.float32)


def _write_video(path: str, frames: list[np.ndarray], fps: int = 20) -> None:
    """Encode frames to mp4, tolerating whichever writer the venv ships."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
        return
    # Fall through to the next writer.
    except Exception as exc:  # noqa: BLE001
        print(f"[replay] imageio unavailable ({exc}); trying cv2", flush=True)
    try:
        import cv2

        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height)
        )
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
        writer.release()
        return
    # Last resort: raw frames for host-side encode.
    except Exception as exc:  # noqa: BLE001
        print(f"[replay] cv2 unavailable ({exc}); dumping frames to .npz", flush=True)
    np.savez_compressed(
        os.path.splitext(path)[0] + "_frames.npz", frames=np.asarray(frames)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a recorded episode through the Arena env-server."
    )
    parser.add_argument(
        "--dataset-dir", required=True, help="Lerobot dataset root for the task."
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--server-url", default=os.environ.get("ARENA_EVAL_SERVER_URL"))
    parser.add_argument("--out", required=True, help="Output mp4 path.")
    parser.add_argument("--max-steps", type=int, default=450)
    args = parser.parse_args()
    if not args.server_url:
        raise SystemExit("no --server-url given and ARENA_EVAL_SERVER_URL is unset")

    actions = _episode_actions(args.dataset_dir, args.episode)
    print(
        f"[replay] episode {args.episode}: {len(actions)} recorded actions, dim={actions.shape[1]}",
        flush=True,
    )

    env = ArenaEnv(
        server_url=args.server_url,
        image_resolution=0,
        action_dim=int(actions.shape[1]),
        state_dim=31,
        max_episode_steps=args.max_steps,
    )
    obs, _ = env.reset()
    frames = [np.asarray(obs["ego_view_image"])]
    for step, action in enumerate(actions[: args.max_steps]):
        obs, _reward, terminated, truncated, _info = env.step(
            np.asarray(action, dtype=np.float32)
        )
        frames.append(np.asarray(obs["ego_view_image"]))
        if terminated or truncated:
            print(
                f"[replay] episode ended at step {step} (term={terminated} trunc={truncated})",
                flush=True,
            )
            break
    _write_video(args.out, frames, fps=20)
    print(f"[replay] wrote {args.out} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
