# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.7 offline batches from a LeRobot dataset, encoded as the checkpoint expects."""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from isaac_rl_posttraining.training.offline import WorkerSampler

__all__ = ["open_gr00t_offline_source"]


def open_gr00t_offline_source(
    dataset_dir: str,
    *,
    policy: Any,
    dataset_adapter: Any = None,
) -> WorkerSampler:
    """Draw from a replay buffer of the demonstrations and encode as GR00T fine-tuning does."""
    draw = _N17ReplayDraw(dataset_dir, policy, dataset_adapter=dataset_adapter)
    return WorkerSampler(draw, num_workers=min(8, max(0, (os.cpu_count() or 1) - 2)))


def _row_tasks(batch: dict[str, Any], override: str) -> list[str]:
    """One instruction per row: the override, else each demonstration's own task."""
    count = len(batch["reward"])
    if override:
        return [override] * count
    tasks = [str(info.get("task") or "") for info in batch["info"]]
    if len(tasks) != count or any(not task for task in tasks):
        raise KeyError("a demonstration has no task; set base_model.task_instruction or meta/tasks.jsonl")
    return tasks


class _N17ReplayDraw:
    """Offline N1.7 batches: the demonstration replay buffer, encoded with augmentation.

    Every step is kept, the last action repeated past the episode's end, so the
    terminal reward IQL learns from is in the buffer. The next observation is
    encoded the same way as the current one, so critic targets see the same
    augmentation distribution as their inputs; its action target is unused.
    """

    def __init__(self, dataset_dir: str, policy: Any, *, dataset_adapter: Any = None) -> None:
        from isaac_rl_posttraining.policies.gr00t.data import (
            read_task,
            register_embodiment,
        )
        from isaac_rl_posttraining.policies.gr00t.demonstrations import (
            Gr00tN17DemonstrationSource,
        )
        from isaac_rl_posttraining.training.demonstration_buffer import (
            load_demonstration_buffer,
        )

        processor = policy.processor
        tag = str(getattr(policy, "embodiment_tag", "new_embodiment"))
        register_embodiment(
            processor,
            dataset_dir,
            tag,
            int(policy.action_window),
            state_groups=getattr(dataset_adapter, "fallback_state_groups", None),
            action_groups=getattr(dataset_adapter, "fallback_action_groups", None),
        )
        layout = getattr(dataset_adapter, "modality_json", None)
        if layout is None:
            with open(os.path.join(dataset_dir, "meta", "modality.json")) as f:
                layout = json.load(f)
        config = processor.modality_configs[tag]
        cameras = tuple(config["video"].modality_keys)
        self.override = str(getattr(policy, "instruction", "") or "")
        self.source = Gr00tN17DemonstrationSource(
            processor,
            modality_layout=layout,
            embodiment_tag=tag,
            cameras=cameras,
            instruction=self.override or read_task(os.path.join(dataset_dir, "meta")),
            augment=True,
        )
        self.buffer = load_demonstration_buffer(
            dataset_dir,
            action_horizon=len(config["action"].delta_indices),
            cameras=cameras,
            pad_episode_end=True,
            assemble=getattr(dataset_adapter, "assemble", None),
        )
        self._sampler: tuple[int, Any] | None = None

    def _uniform(self) -> Any:
        # Built per process, after DataLoader workers are seeded, so workers draw different rows.
        from isaac_rl_posttraining.buffers.samplers import UniformSampler

        if self._sampler is None or self._sampler[0] != os.getpid():
            self._sampler = (os.getpid(), UniformSampler(self.buffer))
        return self._sampler[1]

    def __call__(self, batch_size: int) -> dict[str, Any]:
        batch = self._uniform().sample(batch_size)
        tasks = _row_tasks(batch, self.override)
        current = {"obs": batch["obs"], "action": batch["action"], "tasks": tasks}
        nxt = {"obs": batch["next_obs"], "action": batch["action"], "tasks": tasks}
        cur = self.source(current)
        nxt = self.source(nxt)
        return {
            "inputs_cur": cur,
            "inputs_next": nxt,
            "action": cur["action"],
            "reward": torch.as_tensor(batch["reward"], dtype=torch.float32),
            "done": torch.as_tensor(batch["done"], dtype=torch.float32),
        }
