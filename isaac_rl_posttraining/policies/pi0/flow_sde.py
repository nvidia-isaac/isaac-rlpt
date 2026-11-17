# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""pi0 / pi0.5 (openpi, PyTorch weights) as a trainable flow-SDE actor-critic.

Experimental -- written against openpi's PyTorch model
(``openpi.models_pytorch.pi0_pytorch.PI0Pytorch``: ``embed_prefix``,
``paligemma_with_expert.forward``, ``denoise_step``) and exercised on a
stand-in with that surface; it has not run with real openpi weights yet.

pi0 integrates its flow from noise at ``t = 1`` to actions at ``t = 0`` with
the velocity ``u_t = noise - actions`` (``x_{t+dt} = x_t + dt u_t``,
``dt = -1/K``). The shared flow-SDE math runs from noise at 0 to data at 1, so
the adapter maps grid step ``k`` to pi0 time ``1 - k/K`` and negates the
predicted velocity; the chains it stores are the states pi0's own Euler decode
visits. The behavior-cloning loss does not use that grid: it draws openpi's
continuous time, ``Beta(1.5, 1) * 0.999 + 0.001``, and hands ``velocity`` the
shared-time complement ``1 - t``.

The PaliGemma prefix (images and language) is frozen and encoded once per
decode into its key-value cache, which the action expert attends to at every
step; the action expert and the projections train. The cache is far too large
to ship with a decision, so decisions carry the model inputs themselves (no
cached features) and the learner recomputes the prefix.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch import nn

from isaac_rl_posttraining.policies.flow_sde import (
    FlowConditioning,
    FlowSDEActorCritic,
    critic_token_sequence,
    disable_dropout,
    masked_mean,
)
from isaac_rl_posttraining.policies.trainable import FlowSDEConfig

__all__ = [
    "IMAGE_SLOTS",
    "Pi0FlowSDEActorCritic",
    "action_input_steps",
    "attention_2d_masks",
    "encode_action_chunk",
    "sample_openpi_time",
]

#: Image parsers copy actions and need cameras, so only these change the action space.
_ACTION_SPACE_TRANSFORMS = frozenset({"DeltaActions", "SubsampleActions"})

#: pi0's camera slots, in the order the model embeds them (openpi's ``IMAGE_KEYS``).
IMAGE_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
#: The instruction key of openpi's input transforms.
PROMPT_KEY = "prompt"
VIDEO_PREFIX = "video."
STATE_PREFIX = "state."


def action_input_steps(data_inputs: Sequence[Any], normalize: Any, pad: Any) -> list[Any]:
    """Delta (and subsample), then ``Normalize``, then pad to the model action dim."""
    steps = [step for step in data_inputs if type(step).__name__ in _ACTION_SPACE_TRANSFORMS]
    steps.append(normalize)
    steps.append(pad)
    return steps


def encode_action_chunk(
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None,
    states: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Raw ``[B, S]`` states and ``[B, H, A]`` chunks through the action pipeline."""
    states = np.asarray(states, np.float32)
    actions = np.asarray(actions, np.float32)
    if transform is None:
        return actions.copy()
    encoded = []
    for index in range(actions.shape[0]):
        # DeltaActions writes into the array it is given.
        out = transform({"state": states[index].copy(), "actions": actions[index].copy()})
        encoded.append(np.asarray(out["actions"], np.float32))
    return np.stack(encoded)


def sample_openpi_time(batch: int, device: torch.device) -> torch.Tensor:
    """openpi's training time, shape ``[B]`` float32: ``Beta(1.5, 1) * 0.999 + 0.001``."""
    alpha = torch.tensor(1.5, dtype=torch.float32, device=device)
    beta = torch.tensor(1.0, dtype=torch.float32, device=device)
    return torch.distributions.Beta(alpha, beta).sample((batch,)) * 0.999 + 0.001


def attention_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """openpi's block-causal attention: token ``i`` sees token ``j`` when ``j``'s block is not after ``i``'s, and ``j`` is not padding."""
    cumsum = torch.cumsum(att_masks.to(torch.long), dim=1)
    attend = cumsum[:, None, :] <= cumsum[:, :, None]
    padding = pad_masks[:, None, :] & pad_masks[:, :, None]
    return attend & padding


@dataclass(frozen=True)
class _Pi0Conditioning:
    """One decode's frozen prefix: its key-value cache, padding mask, last hidden states, and the state."""

    prefix_cache: Any
    prefix_pad_mask: torch.Tensor
    prefix_hidden: torch.Tensor
    state: torch.Tensor


class Pi0FlowSDEActorCritic(FlowSDEActorCritic):
    """pi0 / pi0.5 with flow-SDE sampling and a value head, for PPO.

    The value head is the one added to a flow-matching VLA by Chen et al.,
    "πRL: Online RL Fine-tuning for Flow-based
    Vision-Language-Action Models", arXiv:2510.25889, which reports on these
    two base_models.

    Args:
        model: openpi's PyTorch pi0 model.
        input_transform: openpi's input pipeline for one sample (repack,
            data and model transforms, normalization, tokenization); it maps
            the openpi observation dict to ``image`` / ``image_mask`` slot
            dicts, ``state``, ``tokenized_prompt`` and ``tokenized_prompt_mask``.
        output_transform: openpi's output pipeline for one sample
            (un-normalization, relative-to-absolute, the embodiment's slicing).
        observation_keys: Observation key (``video.<cam>`` or ``state.<part>``)
            to the openpi key the input transform expects.
        action_window: Executed action steps per chunk.
        valid_action_dim: Real action width inside pi0's padded flow width.
        prefix_width: Hidden width of the PaliGemma prefix (the value head pools it).
        config: Flow-SDE settings.
    """

    frozen_submodules = ("model.paligemma_with_expert.paligemma",)
    cached_feature_names = ()

    def __init__(
        self,
        model: nn.Module,
        *,
        input_transform: Callable[[dict[str, Any]], dict[str, Any]],
        output_transform: Callable[[dict[str, Any]], dict[str, Any]],
        action_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        observation_keys: Mapping[str, str],
        action_window: int,
        valid_action_dim: int,
        prefix_width: int,
        config: FlowSDEConfig,
    ) -> None:
        video_keys = []
        state_keys = []
        for key in observation_keys:
            if key.startswith(VIDEO_PREFIX):
                video_keys.append(key)
            elif key.startswith(STATE_PREFIX):
                state_keys.append(key)
            else:
                raise ValueError(
                    f"observation key {key!r} must start with {VIDEO_PREFIX!r} (a camera) or {STATE_PREFIX!r} (a state part)"
                )
        if not video_keys or not state_keys:
            raise ValueError("observation_keys must map at least one camera and one state part")
        parameter = next(model.action_out_proj.parameters())
        super().__init__(
            video_keys=video_keys,
            state_keys=state_keys,
            action_window=action_window,
            valid_action_dim=valid_action_dim,
            chunk_shape=(int(model.config.action_horizon), int(model.config.action_dim)),
            value_input_dim=int(prefix_width) + int(model.config.action_dim),
            config=config,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        self.model = model
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.action_transform = action_transform
        self.observation_keys = dict(observation_keys)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.action_out_proj.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.model.action_out_proj.parameters()).device

    @property
    def prefix_trainable(self) -> bool:
        return any(parameter.requires_grad for parameter in self.model.paligemma_with_expert.paligemma.parameters())

    # ------------------------------------------------------------------
    # Inputs and outputs
    # ------------------------------------------------------------------
    def build_inputs(
        self,
        frames: Mapping[str, np.ndarray],
        states: Mapping[str, np.ndarray],
        instruction: str | Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """Run openpi's input transforms sample by sample and stack the results.

        Images come out of the transforms as ``[H, W, 3]`` floats and are moved
        to the channels-first layout the model embeds. ``instruction`` is one
        string for the batch, or one string per sample.
        """
        missing = [key for key in self.video_keys if key not in frames]
        missing += [key for key in self.state_keys if key not in states]
        if missing:
            raise KeyError(f"observations lack the mapped keys {missing}")
        batch = int(np.asarray(frames[self.video_keys[0]]).shape[0])
        if isinstance(instruction, str):
            prompts = [instruction] * batch
        else:
            prompts = [str(text) for text in instruction]
            if len(prompts) != batch:
                raise ValueError(f"expected {batch} instructions, got {len(prompts)}")
        columns: dict[str, list[np.ndarray]] = {}
        for index in range(batch):
            sample: dict[str, Any] = {PROMPT_KEY: prompts[index]}
            for key in self.video_keys:
                sample[self.observation_keys[key]] = np.asarray(frames[key][index])
            for key in self.state_keys:
                sample[self.observation_keys[key]] = np.asarray(states[key][index], dtype=np.float32)
            transformed = self.input_transform(sample)
            for slot in IMAGE_SLOTS:
                image = np.asarray(transformed["image"][slot])
                columns.setdefault(f"image/{slot}", []).append(np.moveaxis(image, -1, 0))
                columns.setdefault(f"image_mask/{slot}", []).append(np.asarray(transformed["image_mask"][slot], dtype=bool))
            for name in ("state", "tokenized_prompt", "tokenized_prompt_mask"):
                columns.setdefault(name, []).append(np.asarray(transformed[name]))
        return {key: torch.as_tensor(np.stack(values)) for key, values in columns.items()}

    def encode_actions(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Raw chunks to the flow target: delta, the checkpoint's norm stats, then pad."""
        return encode_action_chunk(self.action_transform, states, actions)

    def decode(
        self,
        normalized: torch.Tensor,
        *,
        inputs: Mapping[str, torch.Tensor],
        states: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """Run openpi's output transforms sample by sample: ``[B, H, D]`` normalized chunks to ``[B, window, A_raw]``."""
        del states
        chunks = normalized.detach().float().cpu().numpy()
        model_states = inputs["state"].detach().float().cpu().numpy()
        decoded = []
        for index in range(chunks.shape[0]):
            # openpi's policy hands the transformed state along so relative action groups find their reference.
            outputs = self.output_transform({"state": model_states[index], "actions": chunks[index]})
            decoded.append(np.asarray(outputs["actions"], dtype=np.float32)[: self.action_window])
        return np.stack(decoded)

    # ------------------------------------------------------------------
    # The model's pieces
    # ------------------------------------------------------------------
    def condition(
        self, inputs: Mapping[str, torch.Tensor], cached_features: Mapping[str, Any] | None = None
    ) -> FlowConditioning:
        if cached_features is not None:
            raise ValueError("pi0 has no cacheable features; its prefix cache is recomputed by the learner")
        images = [inputs[f"image/{slot}"].to(device=self.device, dtype=self.dtype) for slot in IMAGE_SLOTS]
        image_masks = [inputs[f"image_mask/{slot}"].to(self.device).bool() for slot in IMAGE_SLOTS]
        tokens = inputs["tokenized_prompt"].to(self.device)
        token_masks = inputs["tokenized_prompt_mask"].to(self.device).bool()
        state = inputs["state"].to(device=self.device, dtype=self.dtype)
        with torch.set_grad_enabled(self.prefix_trainable and torch.is_grad_enabled()):
            prefix_embs, prefix_pad, prefix_att = self.model.embed_prefix(images, image_masks, tokens, token_masks)
            prefix_pad = prefix_pad.bool()
            position_ids = torch.cumsum(prefix_pad.to(torch.long), dim=1) - 1
            (prefix_hidden, _), prefix_cache = self.model.paligemma_with_expert.forward(
                attention_mask=attention_2d_masks(prefix_pad, prefix_att),
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
        return FlowConditioning(
            context=_Pi0Conditioning(prefix_cache, prefix_pad, prefix_hidden, state),
            cached={},
            batch_size=int(state.shape[0]),
        )

    def velocity(self, conditioning: FlowConditioning, x_t: torch.Tensor, step_index: torch.Tensor) -> torch.Tensor:
        context: _Pi0Conditioning = conditioning.context
        # An integer is grid step k (shared time k/K). A float is already that shared time.
        # pi0's clock is 1 - t, and its velocity points from actions to noise.
        if step_index.is_floating_point():
            shared_time = step_index.to(device=x_t.device, dtype=torch.float32).reshape(-1)
        else:
            shared_time = step_index.to(device=x_t.device, dtype=torch.float32) / float(self.denoising_steps)
        pi0_time = 1.0 - shared_time
        u_t = self.model.denoise_step(
            context.state, context.prefix_cache, context.prefix_pad_mask, x_t, pi0_time.to(dtype=x_t.dtype)
        )
        return -u_t

    def _flow_match_time(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        pi0_time = sample_openpi_time(batch, self.device)
        shared = 1.0 - pi0_time
        return shared.to(dtype=self.dtype).reshape(batch, 1, 1), shared

    def value_features(self, conditioning: FlowConditioning) -> torch.Tensor:
        context: _Pi0Conditioning = conditioning.context
        pooled = masked_mean(context.prefix_hidden, context.prefix_pad_mask).to(context.state.dtype)
        return torch.cat((pooled, context.state), dim=1)

    def _critic_token_sequence(
        self, conditioning: FlowConditioning
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        context: _Pi0Conditioning = conditioning.context
        state = context.state
        proj = getattr(self.model, "state_proj", None)
        if isinstance(proj, nn.Module):
            state = proj(state)
        return critic_token_sequence(context.prefix_hidden, context.prefix_pad_mask, state)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def spec_options(cls, base_model: Any) -> dict[str, Any]:
        """pi0 loads through its openpi training config; the observation mapping names the cameras and state parts."""
        for name in ("openpi_config", "observation_keys", "action_horizon", "action_dim"):
            if getattr(base_model, name) is None:
                raise ValueError(f"base_model.{name} is required by the pi0 flow-SDE adapter")
        return {
            "openpi_config": str(base_model.openpi_config),
            "observation_keys": {str(key): str(value) for key, value in dict(base_model.observation_keys).items()},
            "action_window": int(base_model.action_horizon),
            "action_dim": int(base_model.action_dim),
        }

    @classmethod
    def load(
        cls,
        checkpoint: str,
        *,
        device: str,
        flow_sde: FlowSDEConfig,
        value_head_seed: int = 0,
        openpi_config: str,
        observation_keys: Mapping[str, str],
        action_window: int,
        action_dim: int,
    ) -> Pi0FlowSDEActorCritic:
        """Load an openpi PyTorch checkpoint as a trainable flow-SDE actor-critic.

        Follows ``openpi.policies.policy_config.create_trained_policy``: the
        training config names the model and the data pipeline, the checkpoint
        supplies the weights and the normalization statistics. The PaliGemma
        prefix is frozen, the action expert and projections train, dropout is
        disabled and a freshly initialized value head (seeded) is attached.
        """
        from openpi import transforms as openpi_transforms
        from openpi.training import checkpoints as openpi_checkpoints
        from openpi.training import config as openpi_configs

        train_config = openpi_configs.get_config(openpi_config)
        model = train_config.model.load_pytorch(train_config, checkpoint)
        model.to(device)
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        norm_stats = openpi_checkpoints.load_norm_stats(f"{checkpoint}/assets", data_config.asset_id)
        normalize = openpi_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
        pad = next(
            (
                step
                for step in data_config.model_transforms.inputs
                if type(step).__name__ == "PadStatesAndActions"
            ),
            None,
        )
        if pad is None:
            pad = openpi_transforms.PadStatesAndActions(int(model.config.action_dim))
        input_transform = openpi_transforms.compose(
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                normalize,
                *data_config.model_transforms.inputs,
            ]
        )
        action_transform = openpi_transforms.compose(
            action_input_steps(data_config.data_transforms.inputs, normalize, pad)
        )
        output_transform = openpi_transforms.compose(
            [
                *data_config.model_transforms.outputs,
                openpi_transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
                *data_config.repack_transforms.outputs,
            ]
        )
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        for parameter in model.paligemma_with_expert.paligemma.parameters():
            parameter.requires_grad_(False)
        dropped = 0 if flow_sde.keep_dropout else disable_dropout(model)
        prefix_width = int(model.paligemma_with_expert.paligemma.config.text_config.hidden_size)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(value_head_seed)
            actor_critic = cls(
                model,
                input_transform=input_transform,
                output_transform=output_transform,
                action_transform=action_transform,
                observation_keys=observation_keys,
                action_window=action_window,
                valid_action_dim=action_dim,
                prefix_width=prefix_width,
                config=flow_sde,
            )
        trainable = sum(p.numel() for p in actor_critic.parameters() if p.requires_grad)
        total = sum(p.numel() for p in actor_critic.parameters())
        logger.info(
            f"flow-SDE actor-critic (pi0): {trainable / 1e6:.1f}M trainable of {total / 1e6:.1f}M parameters "
            f"(frozen PaliGemma prefix), {dropped} dropout layers disabled, K={flow_sde.denoising_steps}, "
            f"noise_level={flow_sde.noise_level}"
        )
        return actor_critic
