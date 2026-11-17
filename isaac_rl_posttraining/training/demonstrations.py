# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a run's co-training demonstrations come from.

A policy family supplies the builder that turns a LeRobot dataset into batches
its own model reads, registered here by base model name. That indirection is the
point: reimplementing a checkpoint's processor risks a silent normalization
mismatch, so the family that owns the processor owns the encoding, and a
base model with no builder is refused rather than handed demonstrations it cannot
score.

:func:`open_demonstration_source` is the seam between "co-training is
configured" and "co-training has demonstrations": an entry point calls it and
hands the result to ``PPO.learner``. See
:mod:`isaac_rl_posttraining.training.cotrain` for what the anchor is for.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "LeRobotDemonstrationReader",
    "read_reward_done",
]

_STATE_KEY = "observation.state"
_ACTION_KEY = "action"


def read_reward_done(df: Any, num_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-step ``(reward, done)`` float32 columns of one episode.

    RL-annotated datasets store ``rl_info.next.*`` columns and plain exports
    ``next.*``; a dataset with neither gets zeros, which behaviour cloning
    never reads.
    """

    def _column(names: list[str]) -> np.ndarray:
        for name in names:
            if name in df.columns:
                return np.asarray(df[name].to_list(), np.float32).reshape(-1)
        return np.zeros(num_steps, np.float32)

    return (
        _column(["rl_info.next.reward", "next.reward"]),
        _column(["rl_info.next.done", "next.done"]),
    )


class LeRobotDemonstrationReader:
    """Read a LeRobot dataset's episodes, in env space.

    ``cameras`` are the GR00T video keys to decode. ``action_horizon`` is
    checked against the episodes here, so a dataset too short for the policy's
    window fails before anything is decoded. ``assemble`` maps an episode's
    dataframe to its flat ``(state, action)`` when the dataset does not store
    them as the ``observation.state`` and ``action`` columns.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        action_horizon: int = 16,
        cameras: Sequence[str] = (),
        episodes: Sequence[int] | None = None,
        assemble: Callable[[Any], tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self.root = Path(root)
        self._assemble = assemble
        meta = self.root / "meta"
        if not (meta / "info.json").is_file():
            raise FileNotFoundError(f"no LeRobot metadata at {meta / 'info.json'}")

        self.info = json.loads((meta / "info.json").read_text())
        self.modality = json.loads((meta / "modality.json").read_text())
        self.tasks = {
            record["task_index"]: record["task"]
            for record in (
                json.loads(line)
                for line in (meta / "tasks.jsonl").read_text().splitlines()
                if line
            )
        }
        self.action_horizon = int(action_horizon)
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")

        video_modality: dict[str, dict] = self.modality.get("video", {})
        unknown = [name for name in cameras if name not in video_modality]
        if unknown:
            raise ValueError(
                f"unknown camera(s) {unknown}; dataset has {sorted(video_modality)}"
            )
        self.cameras = tuple(cameras)

        total = int(self.info.get("total_episodes", 0))
        self.episodes = tuple(episodes) if episodes is not None else tuple(range(total))
        if not self.episodes:
            raise ValueError(f"no episodes selected from {self.root}")

        self._frames: dict[int, tuple[np.ndarray, ...]] = {}

        if not any(
            len(self._episode(episode)[0]) > self.action_horizon
            for episode in self.episodes
        ):
            raise ValueError(
                f"action_horizon={self.action_horizon} exceeds every episode in {self.root}"
            )

    def _episode_path(self, episode: int) -> Path:
        chunk = episode // int(self.info.get("chunks_size", 1000))
        return self.root / self.info["data_path"].format(
            episode_chunk=chunk, episode_index=episode
        )

    def _video_path(self, episode: int, camera: str) -> Path:
        chunk = episode // int(self.info.get("chunks_size", 1000))
        original = self.modality["video"][camera]["original_key"]
        return self.root / self.info["video_path"].format(
            episode_chunk=chunk, video_key=original, episode_index=episode
        )

    def _episode(self, episode: int) -> tuple[np.ndarray, ...]:
        """``(state [T, S], action [T, A], task_index, reward [T], done [T])``, cached per episode."""
        if episode not in self._frames:
            import pandas as pd

            frame = pd.read_parquet(self._episode_path(episode))
            if self._assemble is not None:
                state, action = self._assemble(frame)
            else:
                state = np.stack(frame[_STATE_KEY].to_numpy())
                action = np.stack(frame[_ACTION_KEY].to_numpy())
            task_index = (
                int(frame["task_index"].iloc[0]) if "task_index" in frame else 0
            )
            reward, done = read_reward_done(frame, len(state))
            self._frames[episode] = (
                np.asarray(state, np.float32),
                np.asarray(action, np.float32),
                task_index,
                reward,
                done,
            )
        return self._frames[episode]

    def episode_arrays(self, episode: int) -> tuple[np.ndarray, np.ndarray, str]:
        """One episode's ``state``, ``action`` and task string."""
        state, action, task_index, _, _ = self._episode(episode)
        return state, action, self.tasks.get(task_index, "")

    def episode_outcomes(self, episode: int) -> tuple[np.ndarray, np.ndarray]:
        """One episode's per-step ``reward`` and ``done``, zeros where the dataset has none."""
        return self._episode(episode)[3:]

    def decode_episode(self, episode: int, camera: str) -> np.ndarray:
        """Every frame of one episode's camera, ``[T, H, W, 3]`` uint8."""
        path = str(self._video_path(episode, camera))
        # N1.7 ships this helper as gr00t.utils.video_utils and returns the
        # frames with their timestamps; N1.5 ships gr00t.utils.video and
        # returns the frames alone.
        if importlib.util.find_spec("gr00t.utils.video_utils") is not None:
            from gr00t.utils.video_utils import get_all_frames

            try:
                return get_all_frames(path)[0]
            except ImportError:
                # N1.7 decodes with torchcodec, which needs FFmpeg < 8.
                import decord

                reader = decord.VideoReader(path)
                return reader.get_batch(list(range(len(reader)))).asnumpy()[..., :3]
        from gr00t.utils.video import get_all_frames

        return get_all_frames(path)


def open_demonstration_source(
    config: Any,
    *,
    action_horizon: int,
    trainable: Any = None,
) -> tuple[Any, Any]:
    """Build the co-training source a PPO config asks for, or ``None``.

    The seam between "co-training is configured" and "co-training has
    demonstrations": an entry point calls this and hands the result to
    ``PPO.learner``. Returning ``None`` when the weight is zero keeps the
    disabled path free of dataset reads.

    Args:
        config: A PPO config carrying ``cotrain_beta``,
            ``cotrain_dataset_path``, ``cotrain_batch_size`` and ``seed``.
        action_horizon: The run's executed action window. The supervised
            target spans the same window the policy emits, so this comes from
            the run rather than from config -- a target of a different length
            teaches a different action geometry than the model can emit.
        trainable: The run's trainable-policy spec. When its base model ships a
            demonstration-source builder, that builder is used so the anchor is
            encoded exactly as the checkpoint expects.

    Returns:
        ``(sampler, source)``: the controller's draw over the demonstrations
        and the encoding a rank applies to the ones it is shipped. Both are
        ``None`` when ``cotrain_beta`` is 0.

    Raises:
        ValueError: If co-training is enabled with no dataset path, or the
            run's action window is not positive.
    """
    beta = float(getattr(config, "cotrain_beta", 0.0) or 0.0)
    path = getattr(config, "cotrain_dataset_path", None)
    if beta <= 0.0:
        if path:
            raise ValueError(
                "cotrain_dataset_path is set but cotrain_beta is 0, so the "
                "demonstrations would never be read; set a positive beta or "
                "clear the path"
            )
        return None, None
    if not path:
        raise ValueError(
            f"cotrain_beta={beta} enables demonstration co-training, but "
            "cotrain_dataset_path is unset; point it at a LeRobot dataset or "
            "set cotrain_beta=0"
        )

    if action_horizon < 1:
        raise ValueError(
            f"co-training needs the run's action window, got {action_horizon}"
        )
    seed = int(config.seed)

    # The family that owns the processor owns the encoding: reimplementing it
    # here would risk a silent normalization mismatch.
    if trainable is not None:
        builder = _base_model_source_builder(getattr(trainable, "base_model", ""))
        if builder is not None:
            return builder(
                path,
                checkpoint=trainable.checkpoint,
                action_horizon=int(action_horizon),
                seed=seed,
                **_base_model_source_options(trainable),
            )

    raise ValueError(
        f"base model {getattr(trainable, 'base_model', None)!r} has no demonstration "
        "source builder, so co-training has no way to encode demonstrations "
        f"it can score; registered: {sorted(_SOURCE_BUILDERS)}"
    )


#: Base model name prefix -> "module:function" building a source in that policy
#: family's own input format. Imported lazily, so resolving this module never
#: pulls a model stack.
_SOURCE_BUILDERS: dict[str, str] = {
    "gr00t_n15": "isaac_rl_posttraining.policies.gr00t.demonstrations:open_gr00t_n15_demonstration_source",
    "gr00t_n17": "isaac_rl_posttraining.policies.gr00t.demonstrations:open_gr00t_n17_demonstration_source",
}


def _base_model_source_builder(base_model: str) -> Callable[..., Any] | None:
    """Resolve a base model's demonstration-source builder, or ``None``."""
    target = _SOURCE_BUILDERS.get(base_model)
    if target is None:
        return None
    import importlib

    module_name, function_name = target.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)


def _base_model_source_options(trainable: Any) -> dict[str, Any]:
    """The loading fields a base model's builder needs, from its spec options."""
    options = dict(getattr(trainable, "options", {}) or {})
    passed = {}
    for name in ("embodiment_tag", "backbone_model_name_or_path", "data_config"):
        if name in options:
            passed[name] = options[name]
    return passed
