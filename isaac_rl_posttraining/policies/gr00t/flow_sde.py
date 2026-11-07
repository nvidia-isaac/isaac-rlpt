# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.5 as a trainable flow-SDE actor-critic.

The N1.5 action head (state and action encoders, a DiT conditioned on the
Eagle vision-language tokens, an action decoder) is reached through its
submodules exactly as its native ``get_action`` does, so the stochastic step
and the learner's re-evaluation see the same velocity as native decoding.
Only the action head and the value head train; the Eagle backbone stays frozen
(RLinf's ``tune_llm=False, tune_visual=False``) and its raw outputs can travel
with a decision as cached features.

The external ``gr00t`` package is imported only inside
:func:`load_flow_sde_actor_critic`; the adapter works on any object with the
N1.5 model's surface (see the unit tests). The shared sampling math lives in
:mod:`isaac_rl_posttraining.policies.flow_sde`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch import nn

from isaac_rl_posttraining.policies.feature_cache import unpack_features
from isaac_rl_posttraining.policies.flow_sde import (
    FlowConditioning,
    FlowSDEActorCritic,
    disable_dropout,
    flow_sde_step_moments,
    gaussian_log_prob,
    masked_mean,
)
from isaac_rl_posttraining.policies.gr00t.n15 import load_n15_checkpoint
from isaac_rl_posttraining.policies.trainable import (
    STOCHASTIC_STEP_KEY,
    FlowSDEConfig,
    FlowSDESample,
)

__all__ = [
    "BF16_PIXEL_KEY",
    "PIXEL_KEY",
    "STOCHASTIC_STEP_KEY",
    "FlowSDEConfig",
    "FlowSDESample",
    "Gr00tFlowSDEActorCritic",
    "flow_sde_step_moments",
    "gaussian_log_prob",
    "load_flow_sde_actor_critic",
    "pack_model_inputs",
    "unpack_model_inputs",
]

#: Eagle pixel tensor key as produced by the GR00T transform.
PIXEL_KEY = "eagle_pixel_values"
#: Transport key of the pixel tensor stored as raw bfloat16 bits (int16), see :func:`pack_model_inputs`.
BF16_PIXEL_KEY = "eagle_pixel_values_bf16"
_IMAGE_SIZES_KEY = "eagle_image_sizes"


@dataclass(frozen=True)
class _N15Conditioning:
    """The head's view of one batch: processed backbone tokens, their mask, and the encoded state."""

    vl_embs: torch.Tensor
    mask: torch.Tensor
    state_features: torch.Tensor
    embodiment_id: torch.Tensor


def _sample_flow_time(head: Any, samples: int, device: torch.device) -> torch.Tensor:
    """The head's own time schedule, drawn in float32.

    ``head.sample_time`` draws from a Beta whose concentrations were built when
    the checkpoint loaded, so on a bf16 checkpoint they are bf16 -- and Beta
    samples through ``dirichlet``, which has no bfloat16 kernel. Nothing else
    reaches this: the flow-SDE path derives its timestep from the Euler step
    index and never samples a time. So the schedule is the head's, taken from
    its own distribution and ``noise_s``; only the sampling dtype differs.
    """
    distribution = torch.distributions.Beta(
        head.beta_dist.concentration1.float(), head.beta_dist.concentration0.float()
    )
    sample = distribution.sample([samples]).to(device)
    return (head.config.noise_s - sample) / head.config.noise_s


class Gr00tFlowSDEActorCritic(FlowSDEActorCritic):
    """GR00T N1.5 with flow-SDE sampling and a value head, for PPO.

    The value head is the one added to a flow-matching VLA by Chen et al.,
    "πRL: Online RL Fine-tuning for Flow-based
    Vision-Language-Action Models", arXiv:2510.25889.

    Args:
        model: A GR00T N1.5 model (``backbone``, ``action_head``, ``prepare_input``).
        transform: The checkpoint's modality transform (``apply`` / ``unapply``).
        video_keys: Camera keys of the transform, in order.
        state_keys: State keys of the transform, in order.
        action_keys: Action keys of the transform, in order.
        language_key: The transform's instruction key.
        action_window: Executed action steps per chunk (the data config's window).
        valid_action_dim: Real action width inside the padded flow width.
        config: Flow-SDE settings.

    With a frozen backbone, :meth:`act` exposes the raw Eagle outputs (features
    and attention mask, taken *before* the action head's own normalization and
    self-attention, which stay trainable) and :meth:`forward` accepts them under
    ``cached_features=`` so training skips the backbone and needs none of the
    ``eagle_*`` inputs.
    """

    frozen_submodules = ("model.backbone",)
    cached_feature_names = ("backbone_features", "backbone_attention_mask")
    BACKBONE_INPUT_PREFIX = "eagle_"

    def __init__(
        self,
        model: nn.Module,
        *,
        transform: Any,
        video_keys: Sequence[str],
        state_keys: Sequence[str],
        action_keys: Sequence[str],
        language_key: str,
        action_window: int,
        valid_action_dim: int,
        config: FlowSDEConfig,
    ) -> None:
        head = model.action_head
        parameter = next(head.parameters())
        super().__init__(
            video_keys=video_keys,
            state_keys=state_keys,
            action_window=action_window,
            valid_action_dim=valid_action_dim,
            chunk_shape=(int(head.action_horizon), int(head.action_dim)),
            value_input_dim=int(head.config.backbone_embedding_dim) + int(head.config.input_embedding_dim),
            config=config,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        self.model = model
        self.transform = transform
        self.action_keys = list(action_keys)
        self.language_key = str(language_key)
        head.num_inference_timesteps = config.denoising_steps

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.action_head.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.model.action_head.parameters()).device

    @property
    def backbone_trainable(self) -> bool:
        return any(parameter.requires_grad for parameter in self.model.backbone.parameters())

    # ------------------------------------------------------------------
    # Inputs and outputs
    # ------------------------------------------------------------------
    def build_inputs(
        self,
        frames: Mapping[str, np.ndarray],
        states: Mapping[str, np.ndarray],
        instruction: str,
    ) -> dict[str, torch.Tensor]:
        """Run the checkpoint's transform on a batch of ``N`` observations.

        Args:
            frames: Per camera key, uint8 frames ``[N, H, W, 3]``.
            states: Per state key, float ``[N, d]`` parts.
            instruction: The shared task instruction.

        Returns:
            Processed model inputs (CPU tensors) in the transform's batched layout.
        """
        missing = [key for key in self.video_keys if key not in frames]
        missing += [key for key in self.state_keys if key not in states]
        if missing:
            raise KeyError(f"observations lack the checkpoint's modality keys {missing}")
        batch = int(np.asarray(frames[self.video_keys[0]]).shape[0])
        observation: dict[str, Any] = {}
        for key in self.video_keys:
            frame = np.asarray(frames[key])
            if frame.dtype != np.uint8 or frame.ndim != 4 or frame.shape[0] != batch:
                raise ValueError(f"frame {key!r} must be uint8 [N, H, W, 3], got {frame.dtype} {frame.shape}")
            observation[key] = frame[:, None]
        for key in self.state_keys:
            state = np.asarray(states[key], dtype=np.float32)
            if state.ndim != 2 or state.shape[0] != batch:
                raise ValueError(f"state {key!r} must be float [N, d], got {state.shape}")
            observation[key] = state[:, None, :]
        observation[self.language_key] = np.asarray([instruction] * batch)
        return dict(self.transform.apply(observation))

    def decode(
        self,
        normalized: torch.Tensor,
        *,
        inputs: Mapping[str, torch.Tensor],
        states: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """Decode ``[B, H, D]`` normalized chunks into ``[B, window, A_raw]`` raw actions (absolute: the state is not needed)."""
        del inputs, states
        decoded = self.transform.unapply({"action": normalized.detach().float().cpu()})
        return np.concatenate([np.asarray(decoded[key]) for key in self.action_keys], axis=-1)

    def learner_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Processed inputs the training forward still needs once the backbone outputs are cached."""
        return {key: value for key, value in inputs.items() if not key.startswith(self.BACKBONE_INPUT_PREFIX)}

    def pack_inputs(self, inputs: Mapping[str, torch.Tensor], batch_size: int) -> list[dict[str, np.ndarray]]:
        return pack_model_inputs(inputs, batch_size)

    def unpack_inputs(self, batch: Mapping[str, Any], device: torch.device | str) -> dict[str, torch.Tensor]:
        return unpack_model_inputs(batch, device)

    # ------------------------------------------------------------------
    # The head's pieces
    # ------------------------------------------------------------------
    def _backbone(self, inputs: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], Any]:
        """Run the backbone: its raw outputs (the cacheable part) and the head's inputs."""
        backbone_inputs, action_inputs = self.model.prepare_input(dict(inputs))
        with torch.set_grad_enabled(self.backbone_trainable and torch.is_grad_enabled()):
            outputs = self.model.backbone(backbone_inputs)
        return {name: outputs[name] for name in self.cached_feature_names}, action_inputs

    scores_demonstrations = True

    def supervised_loss(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """Score demonstrations with the head's own flow-matching loss, per sample.

        The head's training forward already computes this, but reduces it to one
        scalar over the whole batch, and :meth:`CoTrainer.reduce` needs ``[B]``
        so a slice is weighted by the demonstrations it actually scored. The
        recipe below is that forward -- fresh noise and time per call, the
        head's own encoders, DiT and decoder -- stopping before its reduction.

        The masked mean is over the elements this embodiment occupies: the head
        emits a padded ``[B, horizon, max_action_dim]`` grid, so averaging the
        whole grid would divide by the padding and scale the co-training weight
        down by that ratio.
        """
        head = self.model.action_head
        raw, action_inputs = self._backbone(batch)
        vl_embs = head.process_backbone_output(dict(raw))["backbone_features"]
        embodiment_id = action_inputs.embodiment_id
        state_features = head.state_encoder(action_inputs.state, embodiment_id)

        actions = action_inputs.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        time = _sample_flow_time(head, actions.shape[0], actions.device).to(actions.dtype)
        noisy = (1 - time[:, None, None]) * noise + time[:, None, None] * actions
        velocity = actions - noise

        buckets = (time * head.num_timestep_buckets).long()
        action_features = head.action_encoder(noisy, buckets, embodiment_id)
        if head.config.add_pos_embed:
            positions = torch.arange(action_features.shape[1], dtype=torch.long, device=actions.device)
            action_features = action_features + head.position_embedding(positions)[None]
        future_tokens = head.future_tokens.weight[None].expand(vl_embs.shape[0], -1, -1)
        tokens = torch.cat((state_features, future_tokens, action_features), dim=1)

        hidden = head.model(
            hidden_states=tokens,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=raw["backbone_attention_mask"],
            timestep=buckets,
        )
        predicted = head.action_decoder(hidden, embodiment_id)[:, -actions.shape[1] :]

        mask = action_inputs.action_mask.to(torch.float32)
        error = torch.nn.functional.mse_loss(predicted.float(), velocity.float(), reduction="none") * mask
        samples = error.shape[0]
        return error.reshape(samples, -1).sum(dim=1) / mask.reshape(samples, -1).sum(dim=1).clamp_min(1.0)

    def condition(
        self, inputs: Mapping[str, torch.Tensor], cached_features: Mapping[str, Any] | None = None
    ) -> FlowConditioning:
        if cached_features is None:
            raw, action_inputs = self._backbone(inputs)
        else:
            cached = unpack_features(cached_features, self.device)
            missing = [name for name in self.cached_feature_names if name not in cached]
            if missing:
                raise KeyError(f"cached features lack {missing}; expected {list(self.cached_feature_names)}")
            _, action_inputs = self.model.prepare_input(dict(inputs))
            raw = {name: cached[name] for name in self.cached_feature_names}
            raw["backbone_features"] = raw["backbone_features"].to(self.dtype)
        head = self.model.action_head
        # The head's own normalization and self-attention over the tokens stay trainable.
        processed = head.process_backbone_output(dict(raw))
        vl_embs = processed["backbone_features"]
        embodiment_id = action_inputs.embodiment_id
        state_features = head.state_encoder(action_inputs.state, embodiment_id)
        return FlowConditioning(
            context=_N15Conditioning(vl_embs, raw["backbone_attention_mask"], state_features, embodiment_id),
            cached=raw,
            batch_size=int(vl_embs.shape[0]),
        )

    def velocity(self, conditioning: FlowConditioning, x_t: torch.Tensor, step_index: torch.Tensor) -> torch.Tensor:
        head = self.model.action_head
        context: _N15Conditioning = conditioning.context
        buckets = (step_index.to(x_t.device).float() / float(self.denoising_steps) * head.num_timestep_buckets).long()
        action_features = head.action_encoder(x_t, buckets, context.embodiment_id)
        if head.config.add_pos_embed:
            positions = torch.arange(action_features.shape[1], dtype=torch.long, device=x_t.device)
            action_features = action_features + head.position_embedding(positions)[None]
        future_tokens = head.future_tokens.weight[None].expand(context.vl_embs.shape[0], -1, -1)
        tokens = torch.cat((context.state_features, future_tokens, action_features), dim=1)
        hidden = head.model(hidden_states=tokens, encoder_hidden_states=context.vl_embs, timestep=buckets)
        return head.action_decoder(hidden, context.embodiment_id)[:, -head.action_horizon :]

    def value_features(self, conditioning: FlowConditioning) -> torch.Tensor:
        context: _N15Conditioning = conditioning.context
        pooled = masked_mean(context.vl_embs, context.mask).to(context.state_features.dtype)
        return torch.cat((pooled, context.state_features.reshape(context.state_features.shape[0], -1)), dim=1)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def spec_options(cls, base_model: Any) -> dict[str, Any]:
        """N1.5 loads through its data config and the embodiment tag whose statistics it bakes."""
        for name in ("data_config", "embodiment_tag"):
            if getattr(base_model, name) is None:
                raise ValueError(f"base_model.{name} is required by the GR00T N1.5 flow-SDE adapter")
        return {"data_config": str(base_model.data_config), "embodiment_tag": str(base_model.embodiment_tag)}

    @classmethod
    def load(
        cls,
        checkpoint: str,
        *,
        device: str,
        flow_sde: FlowSDEConfig,
        value_head_seed: int = 0,
        data_config: str,
        embodiment_tag: str,
    ) -> Gr00tFlowSDEActorCritic:
        return load_flow_sde_actor_critic(
            checkpoint, data_config, embodiment_tag, config=flow_sde, device=device, value_head_seed=value_head_seed
        )


# ----------------------------------------------------------------------
# Transport of processed inputs
# ----------------------------------------------------------------------


def pack_model_inputs(inputs: Mapping[str, torch.Tensor], batch_size: int) -> list[dict[str, np.ndarray]]:
    """Split processed model inputs into per-sample NumPy records.

    Pixel values are stored as the raw bits of their bfloat16 cast (``int16``
    under :data:`BF16_PIXEL_KEY`), which is exactly what the model consumed at
    decision time and half the size of float32. Per-image tensors
    (``[B * n_images, ...]``) regain a per-sample image axis.
    """
    records: list[dict[str, np.ndarray]] = [{} for _ in range(batch_size)]
    for key, tensor in inputs.items():
        value = tensor.detach().cpu()
        if key == PIXEL_KEY:
            key = BF16_PIXEL_KEY
            value = value.to(torch.bfloat16).view(torch.int16)
        if value.ndim >= 1 and value.shape[0] != batch_size and value.shape[0] % batch_size == 0:
            value = value.reshape(batch_size, value.shape[0] // batch_size, *value.shape[1:])
        if value.ndim == 0 or value.shape[0] != batch_size:
            raise ValueError(f"input {key!r} with shape {tuple(value.shape)} cannot be split into {batch_size} samples")
        array = value.numpy()
        for index in range(batch_size):
            # np.array keeps 0-d entries (embodiment id) 0-d; ascontiguousarray would promote them.
            records[index][key] = np.array(array[index], copy=True)
    return records


def unpack_model_inputs(batch: Mapping[str, torch.Tensor], device: torch.device | str) -> dict[str, torch.Tensor]:
    """Invert :func:`pack_model_inputs` for a stacked ``[B, ...]`` batch.

    Entries that were never packed (no raw-bit pixel key, per-image sizes
    already flat) pass through unchanged, so already-processed inputs are safe.
    """
    inputs: dict[str, torch.Tensor] = {}
    for key, tensor in batch.items():
        value = torch.as_tensor(tensor).to(device)
        if key == BF16_PIXEL_KEY:
            key = PIXEL_KEY
            value = value.contiguous().view(torch.bfloat16)
            value = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
        elif key == _IMAGE_SIZES_KEY and value.ndim == 3:
            value = value.reshape(value.shape[0] * value.shape[1], value.shape[2])
        inputs[key] = value
    return inputs


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def load_flow_sde_actor_critic(
    checkpoint: str,
    data_config: Any,
    embodiment_tag: str,
    *,
    config: FlowSDEConfig,
    device: str | None = None,
    value_head_seed: int = 0,
) -> Gr00tFlowSDEActorCritic:
    """Load an N1.5 checkpoint as a trainable flow-SDE actor-critic.

    The model loads through the pinned native policy class (baked statistics,
    bf16, see :func:`~.n15.load_n15_checkpoint`); then the action head is made
    trainable, the backbone stays frozen, dropout is disabled and a freshly
    initialized value head (seeded, so every rank and worker starts identical)
    is attached. Without ``device`` the model lands on the current CUDA device
    (the one a learner rank pinned), or on the CPU when CUDA is unavailable.
    """
    if device is None:
        device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    loaded = load_n15_checkpoint(checkpoint, data_config, embodiment_tag=embodiment_tag, device=device)
    model = loaded.model
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in model.action_head.parameters():
        parameter.requires_grad_(True)
    dropped = 0 if config.keep_dropout else disable_dropout(model)
    modality = loaded.modality_config
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(value_head_seed)
        actor_critic = Gr00tFlowSDEActorCritic(
            model,
            transform=loaded.transform,
            video_keys=list(modality["video"].modality_keys),
            state_keys=list(modality["state"].modality_keys),
            action_keys=list(modality["action"].modality_keys),
            language_key=str(modality["language"].modality_keys[0]),
            action_window=len(modality["action"].delta_indices),
            valid_action_dim=sum(loaded.action_widths.values()),
            config=config,
        )
    trainable = sum(p.numel() for p in actor_critic.parameters() if p.requires_grad)
    total = sum(p.numel() for p in actor_critic.parameters())
    logger.info(
        f"flow-SDE actor-critic: {trainable / 1e6:.1f}M trainable of {total / 1e6:.1f}M parameters "
        f"(frozen backbone), {dropped} dropout layers disabled, K={config.denoising_steps}, "
        f"noise_level={config.noise_level}"
    )
    return actor_critic
