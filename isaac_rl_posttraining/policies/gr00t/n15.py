# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.5 as a VLA policy: loading, observation mapping, and decoding.

:class:`Gr00tN15Policy` loads an open-source GR00T N1.5 checkpoint (Eagle
vision-language backbone plus flow-matching action head) through the pinned
``gr00t`` package's own policy class, so the checkpoint's baked normalization
statistics and modality transforms apply exactly as in native evaluation. It
exposes the inference surface of the N1.7 :class:`~.policy.Gr00tPolicy`
(``act``, ``act_with_noise``, ``infer_reference``, and the shape properties),
so :class:`~.wrapper.SteeredGr00tWrapper` steers either generation unchanged.

N1.5 checkpoints are self-describing: ``config.json`` declares the model
type, ``experiment_cfg/metadata.json`` carries the per-embodiment statistics,
and a data config (the modality keys and transform pipeline the checkpoint
was finetuned with) completes the observation contract. No dataset is needed.

All imports of the external ``gr00t`` package happen inside the functions
that need them, so this module imports without the GR00T stack installed.
"""

from __future__ import annotations

import contextlib
import json
import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger
import torch
import torch.nn as nn

from isaac_rl_posttraining.policies.flow_sde import masked_mean
from isaac_rl_posttraining.policies.gr00t.model import (
    LatentInjectionConfig,
    _randn_override,
    build_base_noise,
)
from isaac_rl_posttraining.policies.gr00t.vla import VLAPolicy, register_vla

if TYPE_CHECKING:
    from gr00t.data.dataset import ModalityConfig
    from gr00t.data.transform.base import ComposedModalityTransform
    from gr00t.experiment.data_config import BaseDataConfig


__all__ = [
    "N15_MODEL_TYPE",
    "Gr00tN15Model",
    "Gr00tN15Policy",
    "LoadedN15Checkpoint",
    "is_n15_checkpoint",
    "load_n15_checkpoint",
    "masked_max",
    "masked_mean",
    "read_model_type",
    "resolve_data_config",
    "resolve_model_embodiment_tag",
]

#: ``model_type`` a GR00T N1.5 checkpoint declares in its ``config.json``.
N15_MODEL_TYPE = "gr00t_n1_5"


# ----------------------------------------------------------------------
# Checkpoint inspection and loading
# ----------------------------------------------------------------------


def read_model_type(base_ckpt: str) -> str | None:
    """The ``model_type`` a checkpoint declares in its ``config.json``.

    Args:
        base_ckpt: Local checkpoint directory or Hugging Face model id (the
            config is fetched from the Hub cache for ids).

    Returns:
        The declared model type, or ``None`` when the config declares none.

    Raises:
        ValueError: If ``base_ckpt`` is empty.
    """
    if not base_ckpt:
        raise ValueError("base_ckpt is required (a checkpoint directory or Hugging Face model id)")
    if os.path.isdir(base_ckpt):
        config_path = os.path.join(base_ckpt, "config.json")
    else:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(base_ckpt, "config.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    model_type = config.get("model_type")
    return None if model_type is None else str(model_type)


def is_n15_checkpoint(base_ckpt: str) -> bool:
    """Whether ``base_ckpt`` declares itself a GR00T N1.5 checkpoint."""
    return read_model_type(base_ckpt) == N15_MODEL_TYPE


def resolve_data_config(
    spec: str | BaseDataConfig | type[BaseDataConfig],
) -> BaseDataConfig:
    """Resolve a data-config spec into a ``gr00t`` data config instance.

    Args:
        spec: A ``DATA_CONFIG_MAP`` key or a ``"module:Class"`` path (both
            resolved with the pinned package's ``load_data_config``
            semantics), a data config class, or an already-built instance.

    Returns:
        The data config instance.

    Raises:
        TypeError: If ``spec`` is none of the accepted forms.
    """
    from gr00t.experiment.data_config import BaseDataConfig, load_data_config

    if isinstance(spec, str):
        return load_data_config(spec)
    if isinstance(spec, type) and issubclass(spec, BaseDataConfig):
        return spec()
    if isinstance(spec, BaseDataConfig):
        return spec
    raise TypeError(
        "data_config must be a DATA_CONFIG_MAP key, a 'module:Class' string, a "
        f"BaseDataConfig subclass, or an instance; got {type(spec).__name__}"
    )


@dataclass(frozen=True)
class LoadedN15Checkpoint:
    """A loaded N1.5 checkpoint: the frozen model plus its observation contract.

    Attributes:
        model: The ``GR00T_N1_5`` module (frozen, bf16, eval mode, on device).
        transform: The data config's modality transform bound to the
            checkpoint's statistics; ``apply`` builds model inputs and
            ``unapply`` decodes normalized actions.
        modality_config: The data config's modality configs (video, state,
            action, language), each with ``modality_keys`` and ``delta_indices``.
        action_widths: Raw width of each action key, in action-key order.
        native_policy: The pinned ``Gr00tPolicy`` the bundle was built from,
            kept only for the decode-parity diagnostic.
    """

    model: nn.Module
    transform: ComposedModalityTransform
    modality_config: dict[str, ModalityConfig]
    action_widths: dict[str, int]
    native_policy: Any = None


#: Embodiment tags of fine-tuned checkpoints that the pinned ``gr00t`` release does not
#: name, mapped to the pinned tag occupying the same action-head/state-encoder slot.
#: RLinf's GR00T fork trains its IsaacLab and LIBERO Franka checkpoints under tags it
#: assigns projector index 31 -- the index the public release reserves for
#: ``new_embodiment`` -- so the weights load unchanged; only the statistics key in
#: ``experiment_cfg/metadata.json`` differs (see :class:`_MetadataAliasedGr00tPolicy`).
EMBODIMENT_TAG_ALIASES: dict[str, str] = {
    "isaaclab_franka": "new_embodiment",
    "libero_franka": "new_embodiment",
}


def resolve_model_embodiment_tag(embodiment_tag: str) -> str:
    """The pinned ``gr00t`` embodiment tag that loads ``embodiment_tag``'s weights.

    Tags the pinned release knows map to themselves; the fine-tune tags in
    :data:`EMBODIMENT_TAG_ALIASES` map to the pinned tag sharing their slot.

    Raises:
        ValueError: For a tag the pinned release neither knows nor aliases.
    """
    from gr00t.data.embodiment_tags import EmbodimentTag

    known = {tag.value for tag in EmbodimentTag}
    if embodiment_tag in known:
        return embodiment_tag
    try:
        return EMBODIMENT_TAG_ALIASES[embodiment_tag]
    except KeyError as exc:
        raise ValueError(
            f"embodiment tag {embodiment_tag!r} is unknown to the pinned gr00t release and has no "
            f"alias; known tags {sorted(known)}, aliases {sorted(EMBODIMENT_TAG_ALIASES)}"
        ) from exc


def _make_native_policy(base_ckpt: str, embodiment_tag: str, modality_config: Any, transform: Any, device: str) -> Any:
    """Construct the pinned ``Gr00tPolicy``, aliasing the metadata key when needed."""
    from gr00t.model.policy import Gr00tPolicy as NativeGr00tPolicy

    model_tag = resolve_model_embodiment_tag(embodiment_tag)
    if model_tag == embodiment_tag:
        return NativeGr00tPolicy(
            model_path=base_ckpt,
            embodiment_tag=embodiment_tag,
            modality_config=modality_config,
            modality_transform=transform,
            device=device,
        )

    class _MetadataAliasedGr00tPolicy(NativeGr00tPolicy):
        """``Gr00tPolicy`` whose statistics come from ``metadata.json[<fine-tune tag>]``.

        The model itself is built under the aliased pinned tag (same projector
        slot); the statistics block is validated with that tag substituted so the
        pinned schema accepts it.
        """

        # Native hook.
        def _load_metadata(self, exp_cfg_dir: Path) -> None:  # noqa: D401
            from gr00t.data.schema import DatasetMetadata

            metadata_path = Path(exp_cfg_dir) / "metadata.json"
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadatas = json.load(f)
            metadata_dict = metadatas.get(embodiment_tag)
            if metadata_dict is None:
                raise ValueError(
                    f"No metadata found for embodiment tag {embodiment_tag!r} in {metadata_path}"
                )
            metadata = DatasetMetadata.model_validate(
                {**metadata_dict, "embodiment_tag": self.embodiment_tag.value}
            )
            self._modality_transform.set_metadata(metadata)
            self.metadata = metadata

    logger.info(
        f"embodiment tag {embodiment_tag!r} loads through the pinned tag {model_tag!r} "
        f"(same action-head slot); statistics read from metadata.json[{embodiment_tag!r}]"
    )
    return _MetadataAliasedGr00tPolicy(
        model_path=base_ckpt,
        embodiment_tag=model_tag,
        modality_config=modality_config,
        modality_transform=transform,
        device=device,
    )


def load_n15_checkpoint(
    base_ckpt: str,
    data_config: str | BaseDataConfig | type[BaseDataConfig],
    embodiment_tag: str,
    device: str,
) -> LoadedN15Checkpoint:
    """Load an N1.5 checkpoint through the pinned ``gr00t`` policy class.

    The pinned :class:`gr00t.model.policy.Gr00tPolicy` runs the native
    loading sequence -- ``GR00T_N1_5.from_pretrained`` in bf16, the action
    head rebuilt for the data config's action window when it differs, and
    the modality transform bound to the embodiment's statistics from
    ``experiment_cfg/metadata.json`` -- so decoding here matches native
    evaluation. Only the model, the transform, and the layout are kept.

    Args:
        base_ckpt: Checkpoint directory or Hugging Face model id.
        data_config: Data config spec; see :func:`resolve_data_config`.
        embodiment_tag: Embodiment whose statistics the checkpoint bakes; a
            fine-tune tag from :data:`EMBODIMENT_TAG_ALIASES` is accepted.
        device: Device the model is moved to.

    Returns:
        The loaded checkpoint bundle.
    """
    config = resolve_data_config(data_config)
    modality_config = config.modality_config()
    native = _make_native_policy(base_ckpt, embodiment_tag, modality_config, config.transform(), device)
    model = native.model
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    action_widths: dict[str, int] = {}
    for key in modality_config["action"].modality_keys:
        # Keys read "action.<part>"; the metadata indexes parts without the prefix.
        part = key.split(".", 1)[1]
        action_widths[key] = int(native.metadata.modalities.action[part].shape[0])
    logger.info(
        f"loaded GR00T N1.5 checkpoint {base_ckpt} (embodiment {embodiment_tag!r}, action widths {action_widths})"
    )
    return LoadedN15Checkpoint(
        model=model,
        transform=native.modality_transform,
        modality_config=modality_config,
        action_widths=action_widths,
        native_policy=native,
    )


# ----------------------------------------------------------------------
# Feature pooling
# ----------------------------------------------------------------------


def masked_max(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Max of ``tokens`` ``[B, T, D]`` over the tokens ``mask`` ``[B, T]`` marks valid.

    Returns a float32 tensor of shape ``[B, D]``.
    """
    invalid = ~mask.to(torch.bool)[..., None]
    return tokens.float().masked_fill(invalid, float("-inf")).amax(dim=1)


# ----------------------------------------------------------------------
# Model wrapper
# ----------------------------------------------------------------------


class Gr00tN15Model(nn.Module):
    """Thin wrapper over a loaded GR00T N1.5 model, shaped like :class:`~.model.Gr00tModel`.

    Exposes native and latent-steered decoding plus pooled backbone features.
    The wrapped model is expected to provide ``prepare_input``, ``backbone``
    (returning ``backbone_features`` and ``backbone_attention_mask``),
    ``action_head`` (with ``config`` and ``num_inference_timesteps``), and
    ``get_action``.

    Attributes:
        base_model: The wrapped GR00T N1.5 model.
        latent_injection: Knobs for :meth:`get_action_with_noise`.
        action_horizon: Flow horizon ``H`` read from the model config.
        action_dim: Padded flow action width ``D`` read from the model config.
        feat_dim: Backbone feature width (the action head's cross-attention
            input width), used to size feature consumers.
    """

    def __init__(
        self, base_model: nn.Module, latent_injection: LatentInjectionConfig | None = None
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.latent_injection = (
            latent_injection if latent_injection is not None else LatentInjectionConfig()
        )
        self.action_horizon = int(base_model.config.action_horizon)
        self.action_dim = int(base_model.config.action_dim)
        self.feat_dim = int(base_model.action_head.config.backbone_embedding_dim)

    @property
    def denoising_steps(self) -> int:
        """Number of Euler steps the flow head integrates per decode."""
        return int(self.base_model.action_head.num_inference_timesteps)

    @denoising_steps.setter
    def denoising_steps(self, value: int) -> None:
        self.base_model.action_head.num_inference_timesteps = int(value)

    def backbone_tokens(self, inputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the vision-language backbone only.

        Args:
            inputs: Already-processed GR00T model inputs.

        Returns:
            ``(backbone_features, attention_mask)`` of shapes ``[B, T, D]``
            and ``[B, T]``; the mask marks valid (non-padding) tokens.
        """
        backbone_inputs, _ = self.base_model.prepare_input(inputs)
        output = self.base_model.backbone(backbone_inputs)
        return output["backbone_features"], output["backbone_attention_mask"]

    def extract_features(self, inputs: dict[str, Any]) -> torch.Tensor:
        """Pooled backbone features for critic heads.

        Args:
            inputs: Already-processed GR00T model inputs.

        Returns:
            Float32 tensor of shape ``[B, feat_dim]``: the mean over valid
            backbone tokens.
        """
        tokens, mask = self.backbone_tokens(inputs)
        return masked_mean(tokens, mask)

    @torch.no_grad()
    def get_action(self, inputs: dict[str, Any]) -> torch.Tensor:
        """Sample a normalized action chunk natively (the un-steered base policy).

        Returns:
            Predicted normalized actions of shape ``[B, H, D]``.
        """
        return self.base_model.get_action(inputs)["action_pred"]

    @torch.no_grad()
    def get_action_with_noise(self, inputs: dict[str, Any], latent: torch.Tensor) -> torch.Tensor:
        """Decode an action chunk with an external latent as the flow noise.

        The latent is expanded to the full ``[B, H, D]`` initial noise by
        :func:`~.model.build_base_noise` under :attr:`latent_injection`,
        then substituted for the head's internal ``torch.randn`` draw.

        Args:
            inputs: Already-processed GR00T model inputs (batch size ``B``
                must match the latent's).
            latent: Steering latent; see :func:`~.model.build_base_noise`
                for accepted shapes.

        Returns:
            Predicted normalized actions of shape ``[B, H, D]``.
        """
        param = next(self.base_model.parameters())
        lat = torch.as_tensor(latent, device=param.device, dtype=param.dtype)
        base = build_base_noise(lat, self.action_horizon, self.action_dim, self.latent_injection)
        with _randn_override(base, (base.shape[0], self.action_horizon, self.action_dim)):
            return self.base_model.get_action(inputs)["action_pred"]


# ----------------------------------------------------------------------
# Policy
# ----------------------------------------------------------------------


@register_vla("gr00t_n1_5")
class Gr00tN15Policy(VLAPolicy):
    """GR00T N1.5 policy with native and latent-steered decoding.

    The checkpoint loads frozen with its baked statistics bound to the data
    config's transform, so inference needs no dataset registration. The
    policy decodes one observation at a time; batched callers slice per
    sample (see :class:`~.base_model.Gr00tBaseModelPolicy`).

    Attributes:
        model: The :class:`Gr00tN15Model` wrapper around the base model.
        transform: The modality transform (observation to inputs, action
            decode) bound to the checkpoint's statistics.
        modality_config: The data config's modality configs.
        action_widths: Raw width of each action key, in action-key order.
        task: Default task instruction for observations without a ``"prompt"``.
    """

    obs_format = "gr00t"
    needs_chunked_data = True

    def __init__(
        self,
        base_ckpt: str,
        data_config: str | BaseDataConfig | type[BaseDataConfig],
        embodiment_tag: str = "new_embodiment",
        device: str = "cuda",
        denoising_steps: int | None = None,
        latent_injection: LatentInjectionConfig | None = None,
    ) -> None:
        """Load the checkpoint and bind the data config's transform.

        Args:
            base_ckpt: Checkpoint directory or Hugging Face model id.
            data_config: The data config the checkpoint was finetuned with;
                see :func:`resolve_data_config` for accepted forms.
            embodiment_tag: Embodiment whose baked statistics apply.
            device: Device the model lives on.
            denoising_steps: Optional override of the flow head's number of
                Euler integration steps; ``None`` keeps the checkpoint's.
            latent_injection: Steering-noise knobs forwarded to
                :class:`Gr00tN15Model`.

        Raises:
            ValueError: If ``denoising_steps`` is not a positive integer.
        """
        super().__init__()
        if denoising_steps is not None and int(denoising_steps) < 1:
            raise ValueError(f"denoising_steps must be a positive integer; got {denoising_steps!r}")
        self.device = device
        self.embodiment_tag = embodiment_tag
        self.task: str | None = None
        self.compute_dtype = torch.bfloat16

        loaded = load_n15_checkpoint(
            base_ckpt, data_config, embodiment_tag=embodiment_tag, device=device
        )
        self._loaded = loaded
        self.model = Gr00tN15Model(loaded.model, latent_injection=latent_injection)
        if denoising_steps is not None:
            self.model.denoising_steps = int(denoising_steps)
        self.transform = loaded.transform
        self.modality_config = loaded.modality_config
        self.action_widths = dict(loaded.action_widths)
        self.video_keys = list(loaded.modality_config["video"].modality_keys)
        self.state_keys = list(loaded.modality_config["state"].modality_keys)
        self.action_keys = list(loaded.modality_config["action"].modality_keys)
        self.language_key = str(loaded.modality_config["language"].modality_keys[0])

    @property
    def action_horizon(self) -> int:
        """Flow horizon ``H`` of the wrapped model."""
        return self.model.action_horizon

    @property
    def decoded_action_horizon(self) -> int:
        """Action steps one decode emits: the data config's action window."""
        return len(self.modality_config["action"].delta_indices)

    @property
    def raw_action_dim(self) -> int:
        """Width of one decoded action step in the raw environment space."""
        return sum(self.action_widths.values())

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_baked_embodiment(self, task: str | None = None) -> None:
        """Record the default instruction; the statistics are bound at load.

        An N1.5 checkpoint's ``experiment_cfg/metadata.json`` statistics are
        already set on the transform by :func:`load_n15_checkpoint`, so the
        datasetless path has nothing left to register but the default task
        language for observations that carry no ``"prompt"`` entry.
        """
        if task is not None:
            self.task = task

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self, obs: dict[str, Any], deterministic: bool = False, actor_state: Any = None
    ) -> tuple[torch.Tensor, Any]:
        """Decode a raw action chunk for one observation, natively.

        Args:
            obs: Observation dict; see :meth:`build_model_inputs`.
            deterministic: Ignored; flow sampling is inherently stochastic.
            actor_state: Passed through unchanged.

        Returns:
            ``(action, actor_state)`` with ``action`` of shape
            ``[H_decoded, A_raw]`` on :attr:`device`.
        """
        inputs = self.build_model_inputs(obs)
        with self._autocast():
            normalized = self.model.get_action(inputs)  # [1, H, D]
        action = self.decode_action(normalized)  # [H_decoded, A_raw]
        self._log_decode_diagnostics(obs, normalized, action)
        return torch.as_tensor(action, device=self.device).float(), actor_state

    def _log_decode_diagnostics(self, obs: dict[str, Any], normalized: torch.Tensor, action: np.ndarray) -> None:
        """Log the first few decodes (inputs, normalized and raw outputs) for pipeline checks."""
        count = getattr(self, "_diag_count", 0)
        if count >= 3:
            return
        self._diag_count = count + 1
        frames = {
            key: (tuple(np.asarray(obs[key]).shape), str(np.asarray(obs[key]).dtype), float(np.asarray(obs[key]).mean()))
            for key in self.video_keys
        }
        states = {key: np.round(np.asarray(obs[key], dtype=np.float32).reshape(-1), 3).tolist() for key in self.state_keys}
        norm = normalized.float().cpu()
        logger.info(
            f"[n15-diag {self._diag_count}] prompt={self._obs_task(obs)!r} frames={frames} states={states} "
            f"normalized action min/max={norm.min().item():.3f}/{norm.max().item():.3f} "
            f"raw action shape={action.shape} min/max={action.min():.3f}/{action.max():.3f} "
            f"first row={np.round(action[0], 3).tolist()}"
        )
        native = getattr(self._loaded, "native_policy", None) if hasattr(self, "_loaded") else None
        if native is not None:
            try:
                single = {key: np.asarray(obs[key]).reshape(-1, *np.asarray(obs[key]).shape[-3:]) for key in self.video_keys}
                single.update({key: np.asarray(obs[key], dtype=np.float32).reshape(1, -1) for key in self.state_keys})
                single[self.language_key] = [self._obs_task(obs)]
                ref = native.get_action(single)
                ref_row = np.concatenate([np.asarray(ref[key])[0] for key in self.action_keys], axis=-1)
                logger.info(
                    f"[n15-diag {self._diag_count}] native Gr00tPolicy first row={np.round(ref_row, 3).tolist()} "
                    f"max|ours-native|={float(np.abs(action[0] - ref_row).max()):.3f} (stochastic flow: expect small but nonzero)"
                )
            except Exception as exc:
                logger.warning(f"[n15-diag] native parity check failed: {type(exc).__name__}: {exc}")

    @torch.no_grad()
    def act_with_noise(self, obs: dict[str, Any], latent: torch.Tensor) -> np.ndarray:
        """Decode a raw action chunk steered by an external latent.

        Args:
            obs: Observation dict; see :meth:`build_model_inputs`.
            latent: Steering latent for a batch of one; see
                :func:`~.model.build_base_noise` for accepted shapes.

        Returns:
            Raw action chunk of shape ``[H_decoded, A_raw]``.
        """
        inputs = self.build_model_inputs(obs)
        with self._autocast():
            normalized = self.model.get_action_with_noise(inputs, latent)  # [1, H, D]
        return self.decode_action(normalized)

    @torch.no_grad()
    def act_batch(self, obs_list: list[dict[str, Any]], latents: np.ndarray | None = None) -> np.ndarray:
        """Decode raw action chunks for ``B`` observations in one model forward.

        The batched counterpart of :meth:`act` / :meth:`act_with_noise` used by
        the vectorized collector (``SteeredGr00tWrapper.infer_batch``): the
        observations are stacked into one ``[B]`` batch of model inputs, the
        flow is sampled natively or steered by per-sample latents
        (``[B, latent_dim]`` or ``[B, K, latent_dim]``; see
        :func:`~.model.build_base_noise`), and each sample's chunk is decoded
        with the checkpoint's statistics. Batches must be homogeneous: the same
        modality keys and one shared instruction (the single-task vector-env
        case; per-sample instructions would need padded language batches).

        Returns:
            ``[B, H_decoded, A_raw]`` raw action chunks.

        Raises:
            ValueError: If the observations carry different instructions, or a
                latent batch does not match ``B``.
        """
        inputs = self.build_model_inputs_batch(obs_list)
        with self._autocast():
            if latents is None:
                normalized = self.model.get_action(inputs)  # [B, H, D]
            else:
                lat = torch.as_tensor(np.asarray(latents, dtype=np.float32), device=self.device)
                if lat.ndim == 1:
                    lat = lat[None]
                if lat.shape[0] != len(obs_list):
                    raise ValueError(
                        f"latents carry {lat.shape[0]} samples but the batch has {len(obs_list)} observations"
                    )
                normalized = self.model.get_action_with_noise(inputs, lat)  # [B, H, D]
        return self.decode_action_batch(normalized)

    def build_model_inputs_batch(self, obs_list: list[dict[str, Any]]) -> dict[str, Any]:
        """Stack ``B`` observations (each as in :meth:`build_model_inputs`) into one batch.

        Frames become ``[B, T=1, H, W, C]``, state parts ``[B, T=1, d]``, and
        the shared instruction a ``[B]`` string array, the layout GR00T's own
        multi-env evaluation feeds its transform.

        Raises:
            ValueError: If ``obs_list`` is empty or the instructions differ.
            KeyError: If an observation lacks a modality key.
        """
        if not obs_list:
            raise ValueError("build_model_inputs_batch needs at least one observation")
        tasks = {self._obs_task(obs) for obs in obs_list}
        if len(tasks) != 1:
            raise ValueError(
                "batched decode needs one shared instruction across the batch; got "
                f"{len(tasks)} distinct instructions: {sorted(tasks)}"
            )
        for index, obs in enumerate(obs_list):
            missing = [key for key in self.video_keys + self.state_keys if key not in obs]
            if missing:
                raise KeyError(f"observation {index} lacks the checkpoint's modality keys {missing}")
        observation: dict[str, Any] = {}
        for key in self.video_keys:
            observation[key] = np.concatenate([self._frame(obs[key], key) for obs in obs_list], axis=0)
        for key in self.state_keys:
            observation[key] = np.concatenate([self._state(obs[key]) for obs in obs_list], axis=0)
        observation[self.language_key] = np.asarray([tasks.pop()] * len(obs_list))
        return self.transform.apply(observation)

    def decode_action_batch(self, normalized: torch.Tensor) -> np.ndarray:
        """Decode ``[B, H, D]`` normalized head outputs into ``[B, H_decoded, A_raw]``."""
        decoded = self.transform.unapply({"action": normalized.float().cpu()})
        return np.concatenate([np.asarray(decoded[key]) for key in self.action_keys], axis=-1)

    @torch.no_grad()
    def infer_reference(
        self, obs: dict[str, Any], image_transform_mode: str = "eval"
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Decode the native reference action plus pooled backbone features.

        Args:
            obs: Observation dict; see :meth:`build_model_inputs`.
            image_transform_mode: ``"train"`` or ``"eval"``. The transform
                runs in eval mode either way (no augmentation), so the frozen
                reference decodes identically; the mode is validated to honor
                the reference-inference contract, not applied.

        Returns:
            ``(features, reference)`` with mean- and max-pooled backbone
            tokens of shape ``[1, 2 * feat_dim]`` (float32, on the model's
            device) and the raw reference chunk ``[1, H_decoded, A_raw]``.
        """
        if image_transform_mode not in ("train", "eval"):
            raise ValueError("image_transform_mode must be 'train' or 'eval'")
        inputs = self.build_model_inputs(obs)
        with self._autocast():
            tokens, mask = self.model.backbone_tokens(inputs)  # [1, T, D], [1, T]
            normalized = self.model.get_action(inputs)  # [1, H, D]
        pooled = torch.cat([masked_mean(tokens, mask), masked_max(tokens, mask)], dim=1)
        reference = self.decode_action(normalized)[None]  # [1, H_decoded, A_raw]
        return pooled, reference

    def build_model_inputs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Convert one observation into processed model inputs (batch of one).

        The observation carries one entry per configured modality key: each
        camera view ``"video.<view>"`` as a uint8 frame ``[H, W, C]`` or
        ``[1, H, W, C]``, each state part ``"state.<part>"`` as a float
        vector (a leading singleton axis is accepted), and optionally a
        ``"prompt"`` string overriding the registered default instruction.
        Entries under other keys are ignored. The transform receives the
        batched layout native evaluation uses -- frames ``[B=1, T=1, H, W, C]``,
        states ``[B=1, T=1, d]``, and the instruction as a one-element string
        array under the data config's language key.

        Raises:
            KeyError: If a configured video or state key is missing.
            TypeError: If a frame is not uint8.
            ValueError: If a frame is not a single ``[H, W, C]`` image, or no
                instruction is available.
        """
        missing = [key for key in self.video_keys + self.state_keys if key not in obs]
        if missing:
            raise KeyError(
                f"observation lacks the checkpoint's modality keys {missing} "
                f"(video views {self.video_keys}, state parts {self.state_keys})"
            )
        observation: dict[str, Any] = {}
        for key in self.video_keys:
            observation[key] = self._frame(obs[key], key)
        for key in self.state_keys:
            observation[key] = self._state(obs[key])
        observation[self.language_key] = np.asarray([self._obs_task(obs)])
        return self.transform.apply(observation)

    def decode_action(self, normalized: torch.Tensor) -> np.ndarray:
        """Decode normalized head outputs into the raw action space.

        Mirrors native GR00T N1.5 evaluation: the transform's ``unapply``
        splits the padded flow width into the action keys, un-normalizes
        each key with the checkpoint's statistics, and drops the padding.

        Args:
            normalized: Head output of shape ``[1, H, D]`` (padded width).

        Returns:
            Raw action chunk of shape ``[H_decoded, A_raw]`` in action-key order.
        """
        decoded = self.transform.unapply({"action": normalized.float().cpu()})
        return np.concatenate([np.asarray(decoded[key])[0] for key in self.action_keys], axis=-1)

    def _autocast(self) -> AbstractContextManager[Any]:
        """bf16 autocast on CUDA, as in native N1.5 inference; a no-op elsewhere."""
        if torch.device(self.device).type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _obs_task(self, obs: dict[str, Any]) -> str:
        """The instruction for one observation: its prompt, else the default."""
        prompt = obs.get("prompt")
        if isinstance(prompt, str):
            return prompt
        if (
            isinstance(prompt, (list, tuple))
            and len(prompt) > 0
            and all(isinstance(entry, str) for entry in prompt)
        ):
            return prompt[0]  # batch-of-one inference: one instruction
        if self.task is None:
            raise ValueError(
                "no task instruction: put a 'prompt' string in the observation or "
                "register a default with register_baked_embodiment(task=...)"
            )
        return self.task

    @staticmethod
    def _frame(value: Any, key: str) -> np.ndarray:
        """Shape one camera frame as ``[B=1, T=1, H, W, C]`` uint8."""
        frame = np.asarray(value.detach().cpu() if torch.is_tensor(value) else value)
        if frame.dtype != np.uint8:
            raise TypeError(f"frame {key!r} must be uint8, got {frame.dtype}")
        if frame.ndim == 3:
            frame = frame[None]
        if frame.ndim != 4 or frame.shape[0] != 1:
            raise ValueError(
                f"frame {key!r} must be one [H, W, C] image (optionally [1, H, W, C]); "
                f"got shape {frame.shape}"
            )
        return frame[None]

    @staticmethod
    def _state(value: Any) -> np.ndarray:
        """Shape one state part as ``[B=1, T=1, d]`` float32."""
        state = np.asarray(
            value.detach().cpu() if torch.is_tensor(value) else value, dtype=np.float32
        )
        return state.reshape(1, 1, -1)
