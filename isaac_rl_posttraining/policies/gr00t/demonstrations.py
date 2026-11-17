# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T demonstrations in the model's own input format.

GR00T's action head does not read a LeRobot dataset directly: it wants
tokenized language, tiled pixel values, and state and action padded into the
fixed grid its embodiment-agnostic head emits, all normalized by the
statistics the checkpoint was trained with.

Reproducing that encoding by hand would be a second implementation of the
processor, and a silent normalization mismatch is the documented failure mode.

The two generations reach that encoding differently, which is why they are two
functions rather than one with a branch. N1.7 rebuilds the checkpoint's
*processor*, which carries the modality layout and the statistics together.
N1.5 has no processor: its encoding is a data config's modality transform --
the same one the policy loads with -- and the statistics have to be bound to
it separately, from the checkpoint rather than from the dataset.

In both, the window the demonstrations span is checked against the one the
policy executes, which is not the head's maximum. A GR00T N1.7 head emits 40
steps of 132 dimensions while a 28-DoF embodiment occupies 16 x 28 of it; the
rest is padding the head masks.

One thing neither builder can check: the Eagle processor splits a frame into
tiles of its own size without resizing, so a demonstration recorded at a
higher resolution than the simulator renders is tokenized into more tiles than
the policy ever sees. That follows from the dataset and the checkpoint, not
from anything configurable here -- record demonstrations at the resolution the
rollout uses.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaac_rl_posttraining.buffers.samplers import Sampler

__all__ = [
    "Gr00tN15DemonstrationSource",
    "Gr00tN17DemonstrationSource",
    "encode_training_steps",
    "open_gr00t_n15_demonstration_source",
    "open_gr00t_n17_demonstration_source",
]


def _build_processor(checkpoint: Path, backbone: str | None) -> Any:
    """Rebuild the checkpoint's processor, so normalization matches the model.

    Statistics, modality layout and embodiment ids come from the checkpoint,
    never from the dataset: a dataset's own statistics would renormalize the
    demonstrations into a distribution the model was not trained on.
    """
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    directory = (
        checkpoint / "processor" if (checkpoint / "processor").is_dir() else checkpoint
    )
    options = json.loads((directory / "processor_config.json").read_text())[
        "processor_kwargs"
    ]
    options["statistics"] = json.loads((directory / "statistics.json").read_text())
    options["embodiment_id_mapping"] = json.loads(
        (directory / "embodiment_id.json").read_text()
    )
    if backbone:
        options["model_name"] = backbone
        options.setdefault("transformers_loading_kwargs", {})["local_files_only"] = True
    return Gr00tN1d7Processor(**options)


def _dataset_instruction(root: Path) -> str:
    """The dataset's own task string, which the model was trained against."""
    path = root / "meta" / "tasks.jsonl"
    first = path.read_text(encoding="utf-8").splitlines()[0]
    return str(json.loads(first)["task"])


def _groups(layout: Mapping[str, Any], modality: str) -> list[tuple[str, int, int]]:
    """``(name, start, end)`` per modality group, in layout order."""
    entries = sorted(layout[modality].items(), key=lambda item: int(item[1]["start"]))
    return [(name, int(spec["start"]), int(spec["end"])) for name, spec in entries]


def encode_training_steps(
    processor: Any,
    collator: Any,
    images: Sequence[Mapping[str, np.ndarray]],
    states: Sequence[np.ndarray],
    action_chunks: Sequence[np.ndarray],
    tasks: Sequence[str],
    embodiment_tag: str,
    modality_json: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Encode samples as upstream fine-tuning does: one ``VLAStepData`` each, processor in training mode.

    Training mode applies the augmentation the checkpoint's processor carries
    (random crop, color jitter, state dropout); the processor is returned to
    eval mode afterwards, so inference keeps its own encoding.

    Args:
        processor: The GR00T N1.7 processor with the embodiment registered.
        collator: The model's ``Gr00tN1d7DataCollator``.
        images: One ``{view: [H, W, 3] uint8}`` mapping per sample.
        states: One flat raw state ``[S]`` per sample, sliced by ``modality_json``.
        action_chunks: One raw action chunk ``[T, A]`` per sample, at least the
            configured action window long.
        tasks: One instruction per sample.
        embodiment_tag: Registered embodiment tag.
        modality_json: The flat state/action layout of the arrays.

    Returns:
        Model inputs including the normalized ``action`` and its ``action_mask``.
    """
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType, VLAStepData

    config = processor.modality_configs[embodiment_tag]
    horizon = len(config["action"].delta_indices)
    views = list(config["video"].modality_keys)
    state_specs = {key: modality_json["state"][key] for key in config["state"].modality_keys}
    action_specs = {key: modality_json["action"][key] for key in config["action"].modality_keys}
    tag = EmbodimentTag(embodiment_tag)
    was_training = bool(processor.training)
    processor.train()
    try:
        features = []
        for frames, state, chunk, task in zip(images, states, action_chunks, tasks, strict=True):
            if len(chunk) < horizon:
                raise ValueError(f"action chunk has {len(chunk)} steps; the processor needs {horizon}")
            state = np.asarray(state, np.float32)
            chunk = np.asarray(chunk, np.float32)
            step = VLAStepData(
                images={view: [np.asarray(frames[view], dtype=np.uint8)] for view in views},
                states={key: state[None, int(s["start"]) : int(s["end"])] for key, s in state_specs.items()},
                actions={
                    key: chunk[:horizon, int(s["start"]) : int(s["end"])] for key, s in action_specs.items()
                },
                text=task,
                embodiment=tag,
            )
            features.append(processor([{"type": MessageType.EPISODE_STEP.value, "content": step}]))
    finally:
        if not was_training:
            processor.eval()
    return dict(collator(features)["inputs"])


class Gr00tN17DemonstrationSource:
    """Normalize a drawn batch with the checkpoint's own ``processor``.

    ``modality_layout`` is the dataset's ``meta/modality.json``. With
    ``augment``, each row is encoded as upstream fine-tuning encodes it (see
    :func:`encode_training_steps`); without it, through the processor's
    inference path, which applies no augmentation.
    """

    def __init__(
        self,
        processor: Any,
        *,
        modality_layout: Mapping[str, Any],
        embodiment_tag: str,
        cameras: Sequence[str],
        instruction: str,
        language_key: str = "annotation.human.task_description",
        augment: bool = False,
    ) -> None:
        self.processor = processor
        self.modality_layout = modality_layout
        self.state_groups = _groups(modality_layout, "state")
        self.action_groups = _groups(modality_layout, "action")
        self.embodiment_tag = embodiment_tag
        self.cameras = tuple(cameras)
        self.instruction = instruction
        self.language_key = language_key
        self.max_action_horizon = int(processor.max_action_horizon)
        self.max_action_dim = int(processor.max_action_dim)
        self.augment = bool(augment)
        self._collator: Any = None

    def _instructions(self, batch: Mapping[str, Any], count: int) -> list[str]:
        """Per-sample tasks when the batch carries them, else the one instruction."""
        tasks = batch.get("tasks")
        if tasks is None:
            return [self.instruction] * count
        tasks = [str(task) for task in tasks]
        if len(tasks) != count:
            raise ValueError(f"expected {count} task strings, got {len(tasks)}")
        return tasks

    def _encode_training(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        if self._collator is None:
            from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import (
                Gr00tN1d7DataCollator,
            )

            self._collator = Gr00tN1d7DataCollator(model_name=self.processor.model_name, model_type="qwen")
        observation = batch["obs"]
        states = np.asarray(observation["state"])
        frames = {camera: np.asarray(observation[camera]) for camera in self.cameras}
        return encode_training_steps(
            self.processor,
            self._collator,
            [{camera: frames[camera][index] for camera in self.cameras} for index in range(len(states))],
            states,
            np.asarray(batch["action"]),
            self._instructions(batch, len(states)),
            self.embodiment_tag,
            self.modality_layout,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        dataset_root: str | Path,
        *,
        embodiment_tag: str,
        instruction: str | None = None,
        backbone: str | None = None,
    ) -> Gr00tN17DemonstrationSource:
        """Rebuild the checkpoint's processor and read the dataset's layout.

        ``instruction`` defaults to the dataset's own task string, so the
        demonstrations carry the language the model was trained against.
        """
        processor = _build_processor(Path(checkpoint), backbone)
        layout = json.loads(
            (Path(dataset_root) / "meta" / "modality.json").read_text(encoding="utf-8")
        )
        if instruction is None:
            instruction = _dataset_instruction(Path(dataset_root))
        # Which views the model reads is the checkpoint's own answer.
        cameras = tuple(
            processor.modality_configs[embodiment_tag]["video"].modality_keys
        )
        return cls(
            processor,
            modality_layout=layout,
            embodiment_tag=embodiment_tag,
            cameras=cameras,
            instruction=instruction,
        )

    def __call__(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """``model_inputs`` for the drawn batch, normalized by the checkpoint."""
        if self.augment:
            return self._encode_training(batch)
        observation = batch["obs"]
        flat_state = np.asarray(observation["state"])
        flat_action = np.asarray(batch["action"])

        inputs: dict[str, Any] = {}
        for name, start, end in self.state_groups:
            inputs[f"state.{name}"] = flat_state[:, start:end]
        for camera in self.cameras:
            frames = np.asarray(observation[camera])
            inputs[f"video.{camera}"] = frames[:, None]
        inputs[self.language_key] = self._instructions(batch, flat_state.shape[0])

        from gr00t.data.embodiment_tags import EmbodimentTag

        tag = EmbodimentTag(self.embodiment_tag)
        processed = dict(self.processor.process_observation(inputs, tag))

        # Relative-action checkpoints reference the state a chunk started from,
        # and read it as a history, so the state carries a time axis of one.
        states = {
            name: flat_state[:, None, start:end]
            for name, start, end in self.state_groups
        }
        actions = {
            name: flat_action[..., start:end]
            for name, start, end in self.action_groups
        }
        normalized = []
        for index in range(flat_action.shape[0]):
            normalized.append(
                self.processor.state_action_processor.apply_action(
                    action={name: value[index] for name, value in actions.items()},
                    embodiment_tag=self.embodiment_tag,
                    state={name: value[index] for name, value in states.items()},
                )
            )
        stacked = {
            name: np.stack([sample[name] for sample in normalized])
            for name in normalized[0]
        }
        chunk = np.concatenate(
            [stacked[name] for name, _, _ in self.action_groups], axis=-1
        )

        # The head emits a fixed grid and reads action_mask to know which part
        # this embodiment fills, which is what the masked supervised loss
        # divides by.
        samples, horizon, width = chunk.shape
        grid = np.zeros(
            (samples, self.max_action_horizon, self.max_action_dim), dtype=np.float32
        )
        mask = np.zeros_like(grid)
        grid[:, :horizon, :width] = chunk
        mask[:, :horizon, :width] = 1.0
        processed["action"] = torch.as_tensor(grid, dtype=torch.float32)
        processed["action_mask"] = torch.as_tensor(mask, dtype=torch.float32)

        # process_observation is the inference path and drops the state's time
        # axis; the head is trained against a history of one.
        state = processed["state"]
        if state.ndim == 2:
            processed["state"] = state[:, None]
        processed["embodiment_id"] = processed["embodiment_id"].to(torch.int64)
        return processed


def open_gr00t_n17_demonstration_source(
    dataset_path: str,
    *,
    checkpoint: str,
    embodiment_tag: str,
    action_horizon: int,
    seed: int = 0,
    backbone_model_name_or_path: str | None = None,
) -> tuple[Sampler, Gr00tN17DemonstrationSource]:
    """Serve co-training batches from demonstrations held in a replay buffer.

    The buffer keeps them in env space and the checkpoint's own processor
    normalizes each drawn batch, so storage stays normalization-agnostic.

    Args:
        dataset_path: LeRobot dataset directory holding the demonstrations.
        checkpoint: The policy's checkpoint; its processor defines the
            encoding, so the anchor is normalized exactly as the model expects.
        embodiment_tag: Modality layout to encode under. It must be one the
            checkpoint's processor knows, and its camera and state groups must
            match the dataset's ``meta/modality.json``.
        action_horizon: The window the policy executes, checked against the one
            the checkpoint's demonstrations span.
        seed: Draw seed; a run is replayable from it.
        backbone_model_name_or_path: Local backbone directory, when the
            checkpoint's baked path is not present.

    Returns:
        The controller's sampler, and the encoding a rank applies to its rows.

    Raises:
        ImportError: If the ``gr00t`` package is not installed.
    """
    import numpy as np

    from isaac_rl_posttraining.buffers.samplers import UniformSampler
    from isaac_rl_posttraining.training.demonstration_buffer import (
        load_demonstration_buffer,
    )

    source = Gr00tN17DemonstrationSource.from_checkpoint(
        Path(checkpoint),
        dataset_path,
        embodiment_tag=embodiment_tag,
        backbone=backbone_model_name_or_path,
    )
    # The demonstrations span whatever window the checkpoint was trained on. If
    # that is not the window the policy executes, the supervised target teaches
    # a different action geometry than the model emits, and nothing downstream
    # would notice.
    chunk = len(
        source.processor.modality_configs[embodiment_tag]["action"].delta_indices
    )
    if chunk != action_horizon:
        raise ValueError(
            f"the checkpoint's demonstrations span {chunk} steps but the policy "
            f"executes {action_horizon}; co-training would teach a different "
            "action geometry than the model emits"
        )
    buffer = load_demonstration_buffer(
        dataset_path, action_horizon=action_horizon, cameras=source.cameras
    )
    return UniformSampler(buffer, rng=np.random.default_rng(seed)), source


# ----------------------------------------------------------------------
# N1.5: the data config's modality transform, over the same rows
# ----------------------------------------------------------------------
def _checkpoint_metadata(checkpoint: Path, embodiment_tag: str) -> Any:
    """The checkpoint's statistics for ``embodiment_tag``.

    Mirrors the native policy's metadata load, including the tag aliasing a
    fine-tuned checkpoint needs: the block is keyed by the tag the checkpoint
    was trained under, while the schema validates against the pinned tag whose
    action-head slot the weights occupy.
    """
    from gr00t.data.schema import DatasetMetadata

    from isaac_rl_posttraining.policies.gr00t.n15 import resolve_model_embodiment_tag

    path = checkpoint / "experiment_cfg" / "metadata.json"
    blocks = json.loads(path.read_text())
    block = blocks.get(embodiment_tag)
    if block is None:
        raise KeyError(
            f"the checkpoint has no statistics for embodiment {embodiment_tag!r} in "
            f"{path}; it knows {sorted(blocks)}"
        )
    return DatasetMetadata.model_validate(
        {**block, "embodiment_tag": resolve_model_embodiment_tag(embodiment_tag)}
    )


class Gr00tN15DemonstrationSource:
    """Normalize a drawn batch with the checkpoint's data-config ``transform``.

    N1.5 has no processor. The data config's modality transform -- the same one
    the policy loads with, carrying the checkpoint's statistics -- does the
    whole job: it normalizes, pads the target into the head's grid and produces
    the mask. So this class only puts drawn rows into the layout that transform
    reads, and batches them with its *own* Eagle processor rather than building
    a second one that could tokenize differently.

    ``cameras`` are unprefixed, as the buffer stores them.
    """

    def __init__(
        self,
        transform: Any,
        *,
        modality_layout: Mapping[str, Any],
        cameras: Sequence[str],
        instruction: str,
        language_key: str = "annotation.human.task_description",
    ) -> None:
        self.transform = transform
        self.state_groups = _groups(modality_layout, "state")
        self.action_groups = _groups(modality_layout, "action")
        self.cameras = tuple(cameras)
        self.instruction = instruction
        self.language_key = language_key
        self.transform.train()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        dataset_root: str | Path,
        *,
        embodiment_tag: str,
        data_config: str,
        instruction: str | None = None,
    ) -> Gr00tN15DemonstrationSource:
        """Build the policy's own transform and bind the checkpoint's statistics.

        The statistics come from the checkpoint rather than the dataset: a
        dataset's own would renormalize the anchor into a distribution the
        model was never trained on, and the loss would still look like a loss.
        """
        from isaac_rl_posttraining.policies.gr00t.n15 import resolve_data_config

        config = resolve_data_config(data_config)
        transform = config.transform()
        transform.set_metadata(_checkpoint_metadata(Path(checkpoint), embodiment_tag))
        layout = json.loads(
            (Path(dataset_root) / "meta" / "modality.json").read_text(encoding="utf-8")
        )
        if instruction is None:
            instruction = _dataset_instruction(Path(dataset_root))
        # Which views the model reads is the data config's answer.
        cameras = tuple(
            key.removeprefix("video.")
            for key in config.modality_config()["video"].modality_keys
        )
        return cls(
            transform,
            modality_layout=layout,
            cameras=cameras,
            instruction=instruction,
        )

    def _eagle_processor(self) -> Any:
        from gr00t.model.transforms import GR00TTransform

        stages = [s for s in self.transform.transforms if isinstance(s, GR00TTransform)]
        if len(stages) != 1:
            raise ValueError(
                f"expected exactly one GR00TTransform in the data config's transform, "
                f"found {len(stages)}; there is no Eagle processor to batch with"
            )
        return stages[0].eagle_processor

    def __call__(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """``model_inputs`` for the drawn batch, normalized by the checkpoint."""
        from gr00t.model.transforms import collate

        observation = batch["obs"]
        flat_state = np.asarray(observation["state"])
        flat_action = np.asarray(batch["action"])

        features = []
        for index in range(flat_state.shape[0]):
            # One window in the layout the dataset would have handed it.
            sample: dict[str, Any] = {}
            for name, start, end in self.state_groups:
                sample[f"state.{name}"] = flat_state[index, None, start:end]
            for name, start, end in self.action_groups:
                sample[f"action.{name}"] = flat_action[index, :, start:end]
            for camera in self.cameras:
                sample[f"video.{camera}"] = np.asarray(observation[camera])[index][None]
            sample[self.language_key] = [self.instruction]
            features.append(self.transform.apply(sample))
        return collate(features, self._eagle_processor())


def open_gr00t_n15_demonstration_source(
    dataset_path: str,
    *,
    checkpoint: str,
    embodiment_tag: str,
    data_config: str,
    action_horizon: int,
    seed: int = 0,
    **_ignored: Any,
) -> tuple[Sampler, Gr00tN15DemonstrationSource]:
    """Serve co-training batches from demonstrations held in a replay buffer.

    The N1.7 builder above with the other source.

    Args:
        dataset_path: LeRobot dataset directory holding the demonstrations.
        checkpoint: The policy's checkpoint; its statistics define the
            normalization, so the anchor matches what the model was trained on.
        embodiment_tag: Modality layout to encode under, and the statistics key.
        data_config: The data config the policy loads with. The demonstrations
            must be encoded by the same one, or the anchor's modality split and
            padding would not be the model's.
        action_horizon: The window the policy executes, checked against the one
            the data config's demonstrations span.
        seed: Draw seed; a run is replayable from it.

    Returns:
        The controller's sampler, and the encoding a rank applies to its rows.

    Raises:
        ImportError: If the ``gr00t`` package is not installed.
        ValueError: If the demonstrations span a different action window.
    """
    from isaac_rl_posttraining.buffers.samplers import UniformSampler
    from isaac_rl_posttraining.policies.gr00t.n15 import resolve_data_config
    from isaac_rl_posttraining.training.demonstration_buffer import (
        load_demonstration_buffer,
    )

    # The window the demonstrations span is the data config's, not the head's
    # maximum. A target of a different length teaches an action geometry the
    # model does not emit, and nothing downstream would notice.
    chunk = len(
        resolve_data_config(data_config).modality_config()["action"].delta_indices
    )
    if chunk != action_horizon:
        raise ValueError(
            f"the data config's demonstrations span {chunk} steps but the policy "
            f"executes {action_horizon}; co-training would teach a different "
            "action geometry than the model emits"
        )
    source = Gr00tN15DemonstrationSource.from_checkpoint(
        Path(checkpoint),
        dataset_path,
        embodiment_tag=embodiment_tag,
        data_config=data_config,
    )
    buffer = load_demonstration_buffer(
        dataset_path, action_horizon=action_horizon, cameras=source.cameras
    )
    return UniformSampler(buffer, rng=np.random.default_rng(seed)), source
