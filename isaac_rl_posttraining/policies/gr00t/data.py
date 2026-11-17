# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T data bridge: embodiment registration, input building, offline buffers.

Connects LeRobot-format datasets and harness observations to a GR00T
processor without depending on GR00T's training configuration stack:

* Register a dataset's embodiment -- modality layout plus normalization
  statistics -- into the processor (:func:`register_embodiment`), preferring
  the configuration a fine-tuned checkpoint baked in over anything rebuilt
  from dataset metadata.
* Convert a harness observation batch (``{"ego_view_image", "state"}``) into
  processed GR00T model inputs (:func:`build_inputs`).
* Serve offline RL batches from single-task or multi-task LeRobot datasets
  (:class:`Gr00tOfflineBuffer`, :class:`Gr00tMultiTaskOfflineBuffer`).

All imports of the external ``gr00t`` package are function-level, so this
module imports without the GR00T stack installed. Reading episode videos
additionally requires ``decord``.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from typing import Any

import numpy as np
import torch

from isaac_rl_posttraining.training.demonstrations import (
    read_reward_done as _read_reward_done,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_modality_configs",
    "convert_stats",
    "register_embodiment",
    "register_baked_embodiment",
    "has_baked_embodiment",
    "modality_json_from_processor",
    "build_inputs",
    "decode_action_chunks",
    "normalize_action_chunk",
    "read_task",
    "Gr00tOfflineBuffer",
    "Gr00tMultiTaskOfflineBuffer",
]

_STAT_KEYS = ["mean", "std", "min", "max", "q01", "q99"]
_STATE_KEY = "observation.state"
_MODALITY_JSON = "modality.json"


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_modality_configs(
    modality_json: dict,
    action_horizon: int,
    state_groups: list[str] | None = None,
    action_groups: list[str] | None = None,
) -> dict:
    """Convert LeRobot ``modality.json`` groups into GR00T modality configs.

    ``process_observation`` reads ``observation[f"{modality}.{key}"]`` for
    each configured key, so the modality keys are the group names (for
    state/action) and the view names (for video).

    Args:
        modality_json: Parsed ``meta/modality.json``.
        action_horizon: Length of the action ``delta_indices`` window.
        state_groups: When given, select only these state groups, in this
            order (intersected with what the file declares) -- used to pin a
            policy's trained subset and drop groups it never predicts.
        action_groups: Same selection for action groups.

    Returns:
        ``{modality: ModalityConfig}`` for state, action, video, and
        language.
    """
    from gr00t.data.types import ModalityConfig

    mj_state = modality_json.get("state", {})
    mj_action = modality_json.get("action", {})
    if state_groups is not None:
        state_keys = [k for k in state_groups if k in mj_state]
    else:
        # Only groups backed by observation.state; declared-but-unpopulated
        # end-effector pose groups from shared templates are excluded.
        state_keys = [k for k, v in mj_state.items() if v.get("original_key", _STATE_KEY) == _STATE_KEY]
    if action_groups is not None:
        action_keys = [k for k in action_groups if k in mj_action]
    else:
        action_keys = list(mj_action.keys())
    video_keys = list(modality_json.get("video", {}).keys())
    annotation_keys = list(modality_json.get("annotation", {}).keys())

    configs = {
        "state": ModalityConfig(delta_indices=[0], modality_keys=state_keys),
        "action": ModalityConfig(
            delta_indices=list(range(action_horizon)), modality_keys=action_keys
        ),
        "video": ModalityConfig(delta_indices=[0], modality_keys=video_keys),
    }
    # process_observation indexes the observation dict directly by the first
    # language modality key, so it must be the full "annotation.<group>" path.
    language_keys = [f"annotation.{k}" for k in annotation_keys] or [
        "annotation.human.task_description"
    ]
    configs["language"] = ModalityConfig(delta_indices=[0], modality_keys=language_keys)
    return configs


def convert_stats(
    stats_json: dict,
    modality_json: dict,
    embodiment_tag: str,
    state_groups: list[str] | None = None,
    action_groups: list[str] | None = None,
) -> dict:
    """Convert LeRobot per-feature statistics into GR00T's nested layout.

    Each group's ``[start:end]`` slice is cut out of its original key's stat
    vectors, producing ``{embodiment_tag: {modality: {group: {stat: [...]}}}}``.

    Args:
        stats_json: Parsed ``meta/stats.json``.
        modality_json: Parsed ``meta/modality.json``.
        embodiment_tag: Embodiment the statistics are registered under.
        state_groups: When given, emit stats for only these state groups (in
            order) so widths line up with the modality config.
        action_groups: Same selection for action groups.

    Returns:
        The nested statistics dict.
    """
    want = {"state": state_groups, "action": action_groups}
    out: dict[str, dict[str, dict[str, dict[str, list]]]] = {embodiment_tag: {}}
    for modality in ("state", "action"):
        out[embodiment_tag][modality] = {}
        groups = modality_json.get(modality, {})
        allow = want[modality]
        names = [g for g in allow if g in groups] if allow is not None else list(groups.keys())
        for group in names:
            spec = groups[group]
            original_key = spec.get("original_key", _STATE_KEY if modality == "state" else "action")
            if original_key not in stats_json:
                continue
            start, end = int(spec["start"]), int(spec["end"])
            feature_stats = stats_json[original_key]
            out[embodiment_tag][modality][group] = {
                stat: list(np.asarray(feature_stats[stat], dtype=np.float64).reshape(-1)[start:end])
                for stat in _STAT_KEYS
                if stat in feature_stats
            }
    return out


def _register_modality_config(processor: Any, embodiment_tag: str, config: dict) -> None:
    """Set a modality config on the processor and every relevant sub-processor.

    Some processors delegate state/action normalization to a sub-processor
    that keeps its own copy of ``modality_configs``; setting only the
    top-level mapping would leave the sub-processor missing the tag and make
    ``set_statistics`` fail. The config is therefore mirrored onto every
    attached sub-processor that exposes its own ``modality_configs``.
    """
    targets = [processor]
    for attr in (
        "state_action_processor",
        "video_processor",
        "language_processor",
        "state_processor",
        "action_processor",
    ):
        sub = getattr(processor, attr, None)
        if sub is not None and getattr(sub, "modality_configs", None) is not None:
            targets.append(sub)
    for target in targets:
        target.modality_configs[embodiment_tag] = config


def _baked_config(processor: Any, embodiment_tag: str) -> dict | None:
    """Return the modality config the checkpoint baked for a tag, if any.

    ``save_pretrained`` persists both the modality configs and the
    statistics, and ``from_pretrained`` restores them. For a fine-tuned
    checkpoint this baked config is the authoritative description of the
    action space the policy was trained on; rebuilding from dataset metadata
    instead can pull group layouts the policy never predicts and mismatch
    the statistics at decode time.
    """
    configs = getattr(processor, "modality_configs", None) or {}
    config = configs.get(embodiment_tag)
    if (
        isinstance(config, dict)
        and "action" in config
        and getattr(config["action"], "modality_keys", None)
    ):
        return config
    return None


def has_baked_embodiment(processor: Any, embodiment_tag: str) -> bool:
    """Whether the checkpoint baked a modality config for ``embodiment_tag``.

    True for fine-tuned checkpoints (the baked config is the authoritative
    action/state layout); False for a raw base model, which must register an
    embodiment from a dataset instead.
    """
    return _baked_config(processor, embodiment_tag) is not None


def _try_reuse_baked(processor: Any, embodiment_tag: str) -> bool:
    """Reuse the checkpoint's baked config for a tag verbatim, if present.

    The baked train-time statistics restored by ``from_pretrained`` are kept
    untouched; the only adjustment is mirroring the config onto the
    sub-processors so all copies agree. The baked action ``delta_indices``
    are deliberately not overridden: they encode the trained decode horizon,
    which is separate from the action head's flow horizon, and extending
    them would make ``decode_action`` emit untrained trailing steps.

    Returns:
        ``True`` when a baked config was found and re-registered.
    """
    baked = _baked_config(processor, embodiment_tag)
    if baked is None:
        return False
    _register_modality_config(processor, embodiment_tag, baked)
    return True


def _processor_statistics(processor: Any, embodiment_tag: str) -> dict | None:
    """Return the normalization statistics registered for a tag, if any.

    Checkpoint processors keep the statistics on the state/action
    sub-processor (mirrored on a top-level cache when present); either
    location is accepted.
    """
    for holder in (getattr(processor, "state_action_processor", None), processor):
        stats = getattr(holder, "statistics", None)
        if isinstance(stats, dict) and embodiment_tag in stats:
            return stats[embodiment_tag]
    return None


def _stats_width(group_stats: Any) -> int | None:
    """Width of a group's first available statistics vector, if any."""
    if isinstance(group_stats, dict):
        for stat_name in _STAT_KEYS:
            if stat_name in group_stats:
                return int(np.asarray(group_stats[stat_name]).reshape(-1).size)
    return None


def modality_json_from_processor(processor: Any, embodiment_tag: str) -> dict:
    """Derive the flat state/action layout from a checkpoint's baked config.

    The harness slices flat state/action vectors by a LeRobot-style
    ``modality.json`` layout (``{modality: {group: {start, end}}}``). A
    fine-tuned checkpoint that baked its embodiment (modality configs plus
    statistics restored by ``from_pretrained``) fully determines that
    layout: the group order comes from the baked modality keys and each
    group's width from its statistics vectors. This makes the checkpoint
    self-describing -- no dataset directory is needed to run inference.

    Args:
        processor: Processor with the embodiment baked.
        embodiment_tag: Tag to derive the layout for.

    Returns:
        ``{"state": {...}, "action": {...}}`` with contiguous
        ``{"start", "end"}`` spans in the trained group order.

    Raises:
        RuntimeError: If the tag has no baked modality config or statistics,
            or a group's width cannot be read from the statistics.
    """
    baked = _baked_config(processor, embodiment_tag)
    if baked is None:
        available = sorted(getattr(processor, "modality_configs", None) or {})
        raise RuntimeError(
            f"no baked modality config for embodiment tag {embodiment_tag!r}; "
            f"checkpoint provides {available}"
        )
    stats = _processor_statistics(processor, embodiment_tag)
    if stats is None:
        raise RuntimeError(
            f"no baked statistics for embodiment tag {embodiment_tag!r}; "
            "register a dataset instead (register_embodiment)"
        )

    layout: dict[str, dict[str, dict[str, int]]] = {}
    for modality in ("state", "action"):
        groups = stats.get(modality, {})
        offset = 0
        spans: dict[str, dict[str, int]] = {}
        for key in baked[modality].modality_keys:
            width = _stats_width(groups.get(key))
            if not width:
                raise RuntimeError(
                    f"cannot size {modality} group {key!r} for tag {embodiment_tag!r}: "
                    f"no statistics vector found (groups with stats: {sorted(groups)})"
                )
            spans[key] = {"start": offset, "end": offset + width}
            offset += width
        layout[modality] = spans
    return layout


def register_baked_embodiment(processor: Any, embodiment_tag: str) -> dict:
    """Register an embodiment purely from the checkpoint's baked config.

    The datasetless counterpart of :func:`register_embodiment` for
    checkpoints that ship their embodiment (modality configs plus
    statistics) baked in -- e.g. the released GR00T-N1.7 task checkpoints.
    The baked config is mirrored onto the sub-processors and the flat
    vector layout is derived from it.

    Args:
        processor: Processor restored via ``from_pretrained``.
        embodiment_tag: Tag to register.

    Returns:
        The derived modality layout (see
        :func:`modality_json_from_processor`).

    Raises:
        RuntimeError: If the checkpoint did not bake this tag.
    """
    modality_json = modality_json_from_processor(processor, embodiment_tag)
    if not _try_reuse_baked(processor, embodiment_tag):
        raise RuntimeError(
            f"embodiment tag {embodiment_tag!r} is not baked into the processor"
        )
    return modality_json


def register_embodiment(
    processor: Any,
    dataset_dir: str,
    embodiment_tag: str,
    action_horizon: int,
    state_groups: list[str] | None = None,
    action_groups: list[str] | None = None,
) -> None:
    """Register an embodiment's modality config and statistics on a processor.

    Preferred path: reuse the config and statistics the checkpoint baked
    under this tag (see :func:`_try_reuse_baked`). Fallback (a raw base
    model with no baked config): build the config and statistics from the
    dataset's ``meta/``, restricted to ``state_groups`` / ``action_groups``
    when given -- a policy that predicts only a subset passes the groups it
    was trained on so the dataset's extra full-body groups are dropped.

    Args:
        processor: The GR00T processor to register into.
        dataset_dir: LeRobot dataset root containing ``meta/``.
        embodiment_tag: Tag to register under.
        action_horizon: Action window length for the fallback config.
        state_groups: State groups to register, in order; ``None`` keeps
            every group the dataset declares.
        action_groups: Action groups to register, in order; ``None`` keeps
            every group the dataset declares.
    """
    if _try_reuse_baked(processor, embodiment_tag):
        return
    meta = os.path.join(dataset_dir, "meta")
    modality_json = _load_json(os.path.join(meta, _MODALITY_JSON))
    config = build_modality_configs(
        modality_json,
        action_horizon,
        state_groups=state_groups,
        action_groups=action_groups,
    )
    _register_modality_config(processor, embodiment_tag, config)
    stats = convert_stats(
        _load_json(os.path.join(meta, "stats.json")),
        modality_json,
        embodiment_tag,
        state_groups=state_groups,
        action_groups=action_groups,
    )
    processor.set_statistics(stats, override=True)


def build_inputs(
    processor: Any,
    obs_batch: dict[str, Any],
    embodiment_tag: str,
    modality_json: dict,
    task: str | list[str],
    device: str,
) -> Any:
    """Convert a harness observation batch into processed GR00T model inputs.

    ``process_observation`` consumes a whole batch at once: each video view
    as ``[B, 1, H, W, C]`` (it stacks views internally), each state group as
    ``[B, 1, d]``, and the language annotation as a list of ``B`` strings.
    The video, state, and language keys are sourced from the checkpoint's
    modality config -- exactly the keys native GR00T evaluation reads -- so
    the native-decode path here matches the reference evaluation.

    Camera routing: a per-view entry ``"video.<key>"`` in ``obs_batch``
    feeds that view directly (multi-camera embodiments such as the LIBERO
    checkpoints, trained on a scene plus a wrist camera). Views without a
    per-view entry fall back to the shared ``"ego_view_image"`` frame
    (single-camera environments feed the same frame to every view).

    Images are passed through at their native resolution: the processor's
    trained image transform performs the only resize. Feeding pre-shrunk
    frames is out-of-distribution for the model and degrades manipulation.

    Args:
        processor: The GR00T processor with the embodiment registered.
        obs_batch: ``{"state": [B, S]}`` plus either per-view
            ``"video.<key>": [B, H, W, C]`` entries, a shared
            ``"ego_view_image": [B, H, W, C]``, or both.
        embodiment_tag: Registered embodiment tag.
        modality_json: The modality layout used to slice state groups.
        task: One instruction broadcast over the batch, or a list of ``B``
            per-sample instructions.
        device: Unused placement hint kept for call-site symmetry.

    Returns:
        Processed model inputs ready for the backbone and action head.

    Raises:
        KeyError: If a configured video view has neither a per-view entry
            nor a shared frame to fall back to.
    """
    from gr00t.data.embodiment_tags import EmbodimentTag

    def _uint8(value: Any) -> np.ndarray:
        array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        return array.astype(np.uint8)

    embodiment = EmbodimentTag(embodiment_tag)
    state = obs_batch["state"]
    state = (
        state.detach().cpu().numpy() if torch.is_tensor(state) else np.asarray(state)
    ).astype(np.float32)
    batch_size = state.shape[0]
    shared_image = obs_batch.get("ego_view_image")

    modality_config = processor.modality_configs[embodiment_tag]
    video_keys = list(modality_config["video"].modality_keys)
    state_keys = list(modality_config["state"].modality_keys)
    language_key = modality_config["language"].modality_keys[0]

    annotations = list(task) if isinstance(task, (list, tuple)) else [task] * batch_size

    observation: dict[str, Any] = {language_key: annotations}
    for video_key in video_keys:
        frame = obs_batch.get(f"video.{video_key}", shared_image)
        if frame is None:
            raise KeyError(
                f"no frame for video view {video_key!r}: provide "
                f"'video.{video_key}' or a shared 'ego_view_image' entry "
                f"(configured views: {video_keys})"
            )
        observation[f"video.{video_key}"] = _uint8(frame)[:, None]  # [B, 1, H, W, C]
    state_specs = modality_json.get("state", {})
    for state_key in state_keys:
        if state_key in state_specs:
            spec = state_specs[state_key]
            observation[f"state.{state_key}"] = state[
                :, None, int(spec["start"]) : int(spec["end"])
            ]  # [B, 1, d]
    return processor.process_observation(observation, embodiment)


def decode_action_chunks(
    processor: Any,
    normalized: np.ndarray,
    embodiment_tag: str,
    modality_json: dict,
    states: np.ndarray,
) -> np.ndarray:
    """Decode a batch of normalized head outputs into the raw action space.

    Mirrors native GR00T evaluation exactly: ``processor.decode_action``
    un-normalizes each action group and converts relative groups to absolute
    targets using the current state as the per-group reference. Every
    operation is per-dimension (slice, un-normalize, add reference), so latent
    dimensions stay aligned with action dimensions.

    Args:
        processor: The GR00T processor with the embodiment registered.
        normalized: Head output of shape ``[B, H, D]`` (padded width).
        embodiment_tag: Registered embodiment tag.
        modality_json: The modality layout used to slice state groups.
        states: Raw state vectors ``[B, S]`` supplying the relative-action references.

    Returns:
        Raw action chunks of shape ``[B, H', A_raw]`` (``H'`` the checkpoint's
        action window).
    """
    from gr00t.data.embodiment_tags import EmbodimentTag

    tag = EmbodimentTag(embodiment_tag)
    modality_config = processor.modality_configs[embodiment_tag]
    action_keys = modality_config["action"].modality_keys
    raw_width = sum(int(spec["end"]) - int(spec["start"]) for spec in modality_json["action"].values())
    # State keyed by the processor's state modality keys, shaped [B, T=1, d] --
    # the layout decode_action expects for relative group references.
    state_specs = modality_json.get("state", {})
    batched_states = {}
    for state_key in modality_config["state"].modality_keys:
        group = state_key.split("state.", 1)[-1]
        if group in state_specs:
            spec = state_specs[group]
            batched_states[state_key] = states[:, None, int(spec["start"]) : int(spec["end"])].astype(np.float32)
    decoded = processor.decode_action(np.asarray(normalized)[:, :, :raw_width], tag, batched_states)
    # decode_action returns arrays keyed by the action modality keys, each of
    # shape [B, T, d]; concatenate in the trained group order.
    return np.concatenate([np.asarray(decoded[key]) for key in action_keys], axis=-1)


def normalize_action_chunk(
    processor: Any,
    action_chunk: np.ndarray,
    embodiment_tag: str,
    modality_json: dict,
    state: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a raw action chunk into the model's padded action space.

    Each sample's raw vector is sliced into the dataset's action groups,
    normalized per group by the processor, concatenated in the processor's
    trained group order, and zero-padded to the model's maximum action
    width with a 0/1 validity mask.

    Args:
        processor: The GR00T processor with the embodiment registered.
        action_chunk: Raw actions, shape ``[B, H, A_raw]``.
        embodiment_tag: Registered embodiment tag.
        modality_json: The modality layout used to slice action groups.
        state: Optional raw states ``[B, S]`` supplying the per-group
            reference for embodiments trained on relative actions.

    Returns:
        ``(action, action_mask)`` float tensors of shape
        ``[B, H, max_action_dim]``.
    """
    state_action_processor = processor.state_action_processor
    action_keys = processor.modality_configs[embodiment_tag]["action"].modality_keys
    groups = modality_json.get("action", {})
    state_groups = {
        g: s
        for g, s in modality_json.get("state", {}).items()
        if s.get("original_key", _STATE_KEY) == _STATE_KEY
    }
    max_action_dim = int(processor.max_action_dim)
    batch_size = action_chunk.shape[0]
    out_actions, out_masks = [], []
    for b in range(batch_size):
        # Only the groups the trained action config uses; any extra groups
        # the dataset declares are skipped.
        action_dict = {
            g: action_chunk[b][:, int(groups[g]["start"]) : int(groups[g]["end"])].astype(
                np.float32
            )
            for g in action_keys
            if g in groups
        }
        state_dict = None
        if state is not None:
            state_dict = {
                g: state[b][None, int(s["start"]) : int(s["end"])].astype(np.float32)
                for g, s in state_groups.items()
            }
        normalized = state_action_processor.apply_action(
            action=action_dict, embodiment_tag=embodiment_tag, state=state_dict
        )
        flat = np.concatenate([normalized[k] for k in action_keys], axis=-1)  # [H, A_raw]
        width = flat.shape[1]
        flat = np.concatenate(
            [flat, np.zeros((flat.shape[0], max_action_dim - width), np.float32)], axis=-1
        )
        mask = np.ones_like(flat)
        mask[:, width:] = 0.0
        out_actions.append(flat)
        out_masks.append(mask)
    return (
        torch.from_numpy(np.stack(out_actions)).float(),
        torch.from_numpy(np.stack(out_masks)).float(),
    )


def read_task(meta_dir: str) -> str:
    """Read the task instruction from a dataset's ``meta/tasks.jsonl``.

    Args:
        meta_dir: The dataset's ``meta`` directory.

    Returns:
        The first task description.
    """
    with open(os.path.join(meta_dir, "tasks.jsonl")) as f:
        return json.loads(f.readline())["task"]


def _passthrough_arrays(df: Any) -> tuple[np.ndarray, np.ndarray]:
    """Contiguous state/action straight from the standard LeRobot columns.

    The generic assembler used when no dataset adapter is configured: the
    ``observation.state`` and ``action`` columns are already the contiguous
    per-step vectors the buffer expects.

    Returns:
        ``(state, action)`` float32 arrays.
    """
    states = np.stack(df[_STATE_KEY].to_numpy()).astype(np.float32)
    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
    return states, actions


class Gr00tOfflineBuffer:
    """Offline buffer over one LeRobot dataset, yielding GR00T training batches.

    Reads per-episode parquet files (state, action, optional reward/done),
    decodes ego-view frames on demand with ``decord``, samples single-step
    transitions with an ``action_horizon``-long action chunk, and converts
    them to processed model inputs. Registers the dataset's embodiment
    (modality config plus statistics) into the processor on construction.

    The buffer is embodiment-agnostic: how a raw episode becomes contiguous
    state/action arrays, and the flat layout that slices them, come from an
    injected ``dataset_adapter`` (see
    :class:`isaac_rl_posttraining.envs.embodiments.DatasetAdapter`).

    Attributes:
        modality_json: The modality layout in effect -- the adapter's when
            it supplies one, else the dataset's own ``meta/modality.json``.
        task: The dataset's task instruction.
    """

    def __init__(
        self,
        dataset_dir: str,
        processor: Any,
        embodiment_tag: str,
        action_horizon: int = 40,
        image_size: int = 224,
        device: str = "cuda",
        max_episodes: int | None = None,
        video_key: str = "ego_view",
        task: str | None = None,
        video_reader_cache_size: int = 16,
        dataset_adapter: Any = None,
    ) -> None:
        """Build the buffer and register the embodiment.

        Args:
            dataset_dir: LeRobot dataset root.
            processor: GR00T processor to register into and normalize with.
            embodiment_tag: Embodiment tag for registration and processing.
            action_horizon: Action-chunk length attached to each transition.
            image_size: Side length frames are resized to when they do not
                already match.
            device: Placement hint forwarded to :func:`build_inputs`.
            max_episodes: Optional cap on the number of episodes loaded.
            video_key: Video view whose frames are decoded.
            task: Task instruction override; defaults to the dataset's.
            video_reader_cache_size: Maximum number of open video readers.
                Each reader holds substantial resident memory, so the cache
                is kept small and evicted readers are freed explicitly.
            dataset_adapter: Embodiment adapter (``assemble`` plus optional
                ``modality_json`` / fallback groups) mapping raw episodes to
                contiguous arrays; ``None`` uses the raw ``observation.state``
                / ``action`` columns and the dataset's own modality.
        """
        self.processor = processor
        self.tag = embodiment_tag
        self.H = action_horizon
        self.image_size = image_size
        self.device = device
        self.video_key = video_key
        self.adapter = dataset_adapter
        meta = os.path.join(dataset_dir, "meta")
        self.modality_json = _load_json(os.path.join(meta, _MODALITY_JSON))
        if dataset_adapter is not None and dataset_adapter.modality_json is not None:
            # The adapter's layout slices the arrays it assembles; the
            # dataset's modality.json describes columns it does not emit.
            self.modality_json = dataset_adapter.modality_json
        register_embodiment(
            processor,
            dataset_dir,
            embodiment_tag,
            action_horizon,
            state_groups=getattr(dataset_adapter, "fallback_state_groups", None),
            action_groups=getattr(dataset_adapter, "fallback_action_groups", None),
        )
        self.task = task or read_task(meta)
        self.episodes: list[dict] = self._load_episodes(dataset_dir, max_episodes)
        self.index = [
            (e, f) for e, episode in enumerate(self.episodes) for f in range(episode["length"] - 1)
        ]
        self._video_readers: dict[int, Any] = {}
        self._video_reader_cache_size = int(video_reader_cache_size)
        logger.info(
            "gr00t offline buffer: %d episodes, %d transitions",
            len(self.episodes),
            len(self.index),
        )

    def _load_episodes(self, dataset_dir: str, max_episodes: int | None) -> list[dict]:
        """Load episode arrays and video paths from a dataset directory.

        Each episode's contiguous state/action arrays come from the injected
        dataset adapter (or the raw ``observation.state`` / ``action``
        columns when none is configured); :attr:`modality_json`, fixed at
        construction, stays aligned with them.
        """
        import pandas as pd

        assemble = self.adapter.assemble if self.adapter is not None else _passthrough_arrays
        parquet_files = sorted(
            glob.glob(os.path.join(dataset_dir, "data", "chunk-*", "episode_*.parquet"))
        )
        if max_episodes:
            parquet_files = parquet_files[:max_episodes]
        episodes: list[dict] = []
        for parquet_file in parquet_files:
            episode_id = re.search(r"episode_(\d+)\.parquet", parquet_file).group(1)
            chunk = os.path.basename(os.path.dirname(parquet_file))
            df = pd.read_parquet(parquet_file)
            states, actions = assemble(df)
            reward, done = _read_reward_done(df, len(states))
            video = os.path.join(
                dataset_dir,
                "videos",
                chunk,
                f"observation.images.{self.video_key}",
                f"episode_{episode_id}.mp4",
            )
            episodes.append(
                {
                    "states": states,
                    "actions": actions,
                    "reward": reward,
                    "done": done,
                    "video": video,
                    "length": len(states),
                }
            )
        return episodes

    def _reader(self, episode_idx: int) -> Any:
        """Return a cached video reader for an episode, evicting oldest first."""
        if episode_idx not in self._video_readers:
            import decord

            while len(self._video_readers) >= self._video_reader_cache_size:
                evicted = self._video_readers.pop(next(iter(self._video_readers)))
                del evicted  # free the evicted reader's buffers immediately
            self._video_readers[episode_idx] = decord.VideoReader(
                self.episodes[episode_idx]["video"]
            )
        return self._video_readers[episode_idx]

    def _frame(self, episode_idx: int, frame_idx: int) -> np.ndarray:
        """Decode one frame, dropping alpha and resizing when needed."""
        frame = self._reader(episode_idx)[frame_idx].asnumpy()
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.shape[0] != self.image_size or frame.shape[1] != self.image_size:
            import cv2

            frame = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return frame.astype(np.uint8)

    def _episode_task(self, episode: dict) -> str:
        """Instruction for one episode (per-episode when present)."""
        return episode.get("task", self.task)

    def __len__(self) -> int:
        return len(self.index)

    def sample(self, batch_size: int) -> dict[str, Any]:
        """Sample a batch of transitions as GR00T training inputs.

        Returns:
            ``{"gr00t_cur", "gr00t_next", "action", "reward", "done"}``:
            processed inputs for the current and next observations, the
            normalized action chunk (also attached to ``gr00t_cur`` with its
            validity mask), and float32 reward/done tensors.
        """
        idxs = np.random.randint(0, len(self.index), size=batch_size)  # NOSONAR - draws from the globally seeded NumPy stream
        cur_imgs, nxt_imgs, cur_states, nxt_states = [], [], [], []
        actions, rewards, dones, tasks = [], [], [], []
        for k in idxs:
            episode_idx, frame_idx = self.index[k]
            episode = self.episodes[episode_idx]
            length = episode["length"]
            cur_imgs.append(self._frame(episode_idx, frame_idx))
            nxt_imgs.append(self._frame(episode_idx, frame_idx + 1))
            cur_states.append(episode["states"][frame_idx])
            nxt_states.append(episode["states"][frame_idx + 1])
            chunk_ids = [min(frame_idx + j, length - 1) for j in range(self.H)]
            actions.append(episode["actions"][chunk_ids])
            rewards.append(episode["reward"][frame_idx])
            dones.append(episode["done"][frame_idx])
            tasks.append(self._episode_task(episode))
        cur = build_inputs(
            self.processor,
            {"ego_view_image": np.stack(cur_imgs), "state": np.stack(cur_states)},
            self.tag,
            self.modality_json,
            tasks,
            self.device,
        )
        nxt = build_inputs(
            self.processor,
            {"ego_view_image": np.stack(nxt_imgs), "state": np.stack(nxt_states)},
            self.tag,
            self.modality_json,
            tasks,
            self.device,
        )
        norm_action, action_mask = normalize_action_chunk(
            self.processor, np.stack(actions), self.tag, self.modality_json, state=np.stack(cur_states)
        )
        cur["action"] = norm_action
        cur["action_mask"] = action_mask
        return {
            "gr00t_cur": cur,
            "gr00t_next": nxt,
            "action": norm_action,
            "reward": torch.tensor(rewards, dtype=torch.float32),
            "done": torch.tensor(dones, dtype=torch.float32),
        }


def _combined_stats(
    all_states: np.ndarray, all_actions: np.ndarray, modality_json: dict, embodiment_tag: str
) -> dict:
    """Build statistics from the union of several tasks' state/action arrays.

    One set of normalization statistics (mean/std/min/max/q01/q99 per
    dimension) covers the whole multi-task mixture.
    """

    def _stats(x: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "mean": x.mean(0),
            "std": x.std(0) + 1e-8,
            "min": x.min(0),
            "max": x.max(0),
            "q01": np.quantile(x, 0.01, axis=0),
            "q99": np.quantile(x, 0.99, axis=0),
        }

    stats_json = {
        _STATE_KEY: {k: v.tolist() for k, v in _stats(all_states).items()},
        "action": {k: v.tolist() for k, v in _stats(all_actions).items()},
    }
    return convert_stats(stats_json, modality_json, embodiment_tag)


class Gr00tMultiTaskOfflineBuffer(Gr00tOfflineBuffer):
    """Multi-task buffer over several LeRobot datasets, one embodiment.

    All tasks share one action layout; each transition keeps its own task
    instruction (the language conditions the policy on the task), and one
    combined set of normalization statistics -- the union over all tasks --
    is registered unless the checkpoint already baked a config for the tag.
    Yields the same batches as the single-task buffer.
    """

    def __init__(
        self,
        dataset_dirs: list[str],
        processor: Any,
        embodiment_tag: str,
        action_horizon: int = 40,
        image_size: int = 224,
        device: str = "cuda",
        max_episodes_per_task: int | None = None,
        video_key: str = "ego_view",
        video_reader_cache_size: int = 16,
        dataset_adapter: Any = None,
    ) -> None:
        """Build the buffer over ``dataset_dirs`` and register the embodiment.

        Args:
            dataset_dirs: One LeRobot dataset root per task.
            processor: GR00T processor to register into and normalize with.
            embodiment_tag: Embodiment tag for registration and processing.
            action_horizon: Action-chunk length attached to each transition.
            image_size: Side length frames are resized to when needed.
            device: Placement hint forwarded to :func:`build_inputs`.
            max_episodes_per_task: Optional per-task episode cap.
            video_key: Video view whose frames are decoded.
            video_reader_cache_size: Maximum number of open video readers.
            dataset_adapter: Embodiment adapter shared across tasks (see
                :class:`Gr00tOfflineBuffer`); ``None`` uses the raw columns
                and the first dataset's modality.
        """
        self.processor = processor
        self.tag = embodiment_tag
        self.H = action_horizon
        self.image_size = image_size
        self.device = device
        self.video_key = video_key
        self.adapter = dataset_adapter
        # The modality layout is shared across tasks: the adapter's when it
        # supplies one, else the first dataset's own modality.json.
        if dataset_adapter is not None and dataset_adapter.modality_json is not None:
            self.modality_json = dataset_adapter.modality_json
        else:
            self.modality_json = _load_json(os.path.join(dataset_dirs[0], "meta", _MODALITY_JSON))
        self.episodes: list[dict] = []
        all_states, all_actions = [], []
        for dataset_dir in dataset_dirs:
            task = read_task(os.path.join(dataset_dir, "meta"))
            episodes = self._load_episodes(dataset_dir, max_episodes_per_task)
            for episode in episodes:
                episode["task"] = task
                all_states.append(episode["states"])
                all_actions.append(episode["actions"])
            self.episodes.extend(episodes)
        # Prefer the checkpoint's baked config and statistics; only a raw
        # base model gets the rebuilt config plus combined union statistics.
        if not _try_reuse_baked(self.processor, embodiment_tag):
            _register_modality_config(
                self.processor,
                embodiment_tag,
                build_modality_configs(self.modality_json, action_horizon),
            )
            self.processor.set_statistics(
                _combined_stats(
                    np.concatenate(all_states),
                    np.concatenate(all_actions),
                    self.modality_json,
                    embodiment_tag,
                ),
                override=True,
            )
        self.task = self.episodes[0]["task"]  # representative default instruction
        self.index = [
            (e, f) for e, episode in enumerate(self.episodes) for f in range(episode["length"] - 1)
        ]
        self._video_readers = {}
        self._video_reader_cache_size = int(video_reader_cache_size)
        num_tasks = len({episode["task"] for episode in self.episodes})
        logger.info(
            "gr00t multi-task buffer: %d tasks, %d episodes, %d transitions",
            num_tasks,
            len(self.episodes),
            len(self.index),
        )



# Public alias: combined multi-task normalization statistics.
combined_stats = _combined_stats