# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.7 as a trainable flow-SDE actor-critic.

N1.7 pairs a Cosmos-Reason2 (Qwen3-VL) vision-language backbone with a
flow-matching action head of the same family as N1.5's: category-specific
state and action encoders, a DiT conditioned on the backbone tokens, and an
action decoder, integrated in ``num_inference_timesteps`` Euler steps. The
adapter reaches those submodules exactly as the head's own
``get_action_with_features`` does (no future tokens, state history flattened
into one token), so the stochastic step and the learner's re-evaluation see the
same velocity as native decoding.

The backbone stays frozen and its outputs are cached: Qwen3-VL consumes
flattened image patches whose layout has no per-sample split, so a decision
cannot ship the raw backbone inputs. It ships the backbone's ``backbone_features``
and attention mask (plus the image-token mask when the head uses the alternate
vision-language DiT) together with the head's own inputs, the state and the
embodiment id; ``requires_cached_features`` tells the launcher so.

Loading goes through the repository's N1.7 policy (``policies/gr00t/policy.py``)
so the checkpoint, its processor and its baked embodiment resolve the same way
as for steering. The external ``gr00t`` package is imported only there.
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
    critic_token_sequence,
    disable_dropout,
    masked_mean,
)
from isaac_rl_posttraining.policies.gr00t import data as gr00t_data
from isaac_rl_posttraining.policies.trainable import FlowSDEConfig

__all__ = ["Gr00tN17FlowSDEActorCritic", "ensure_n17_embodiment"]

#: The processor's video views appear in observations under this prefix (``video.<view>``).
VIDEO_PREFIX = "video."
#: The modality layout's state groups appear in observations under this prefix (``state.<group>``).
STATE_PREFIX = "state."


@dataclass(frozen=True)
class _N17Conditioning:
    """The head's view of one batch: processed backbone tokens, their masks, and the encoded state."""

    vl_embs: torch.Tensor
    mask: torch.Tensor
    image_mask: torch.Tensor | None
    state_features: torch.Tensor
    embodiment_id: torch.Tensor


class Gr00tN17FlowSDEActorCritic(FlowSDEActorCritic):
    """GR00T N1.7 with flow-SDE sampling and a value head, for PPO.

    The value head is the one added to a flow-matching VLA by Chen et al.,
    "πRL: Online RL Fine-tuning for Flow-based
    Vision-Language-Action Models", arXiv:2510.25889.

    Args:
        model: A GR00T N1.7 model (``backbone``, ``action_head``, ``prepare_input``).
        processor: The checkpoint's processor with the embodiment registered.
        embodiment_tag: The registered embodiment tag.
        modality_json: The flat state/action layout of that embodiment.
        video_keys: Observation keys of the processor's video views, in order.
        action_keys: Action groups of the processor's modality config, in order.
        action_window: Executed action steps per chunk (at most the head's horizon).
        config: Flow-SDE settings.
    """

    frozen_submodules = ("model.backbone",)
    cached_feature_names = ("backbone_features", "backbone_attention_mask")
    requires_cached_features = True
    #: The processed inputs the action head itself consumes.
    HEAD_INPUT_KEYS = ("state", "embodiment_id")

    def __init__(
        self,
        model: nn.Module,
        *,
        processor: Any,
        embodiment_tag: str,
        modality_json: Mapping[str, Any],
        video_keys: Sequence[str],
        action_keys: Sequence[str],
        action_window: int,
        config: FlowSDEConfig,
    ) -> None:
        head = model.action_head
        parameter = next(head.parameters())
        groups = sorted(modality_json["state"].items(), key=lambda item: int(item[1]["start"]))
        self.state_slices = tuple(
            (f"{STATE_PREFIX}{name}", int(spec["start"]), int(spec["end"])) for name, spec in groups
        )
        raw_action_width = sum(int(spec["end"]) - int(spec["start"]) for spec in modality_json["action"].values())
        super().__init__(
            video_keys=video_keys,
            state_keys=[key for key, _, _ in self.state_slices],
            action_window=action_window,
            valid_action_dim=raw_action_width,
            chunk_shape=(int(head.action_horizon), int(head.action_dim)),
            value_input_dim=int(head.config.backbone_embedding_dim) + int(head.config.input_embedding_dim),
            config=config,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        self.model = model
        self.processor = processor
        self.embodiment_tag = str(embodiment_tag)
        self.modality_json = dict(modality_json)
        self.action_keys = list(action_keys)
        head.num_inference_timesteps = config.denoising_steps
        # The alternate DiT attends to text and image tokens separately, so it needs the image-token mask too.
        self.alternate_dit = bool(head.config.use_alternate_vl_dit)
        if self.alternate_dit:
            self.cached_feature_names = (*self.cached_feature_names, "image_mask")

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
    def _flat_state(self, states: Mapping[str, np.ndarray]) -> np.ndarray:
        """Concatenate the observation's state groups into the flat state the processor slices."""
        parts = []
        for key, start, end in self.state_slices:
            part = np.asarray(states[key], dtype=np.float32)
            if part.ndim != 2 or part.shape[1] != end - start:
                raise ValueError(f"state {key!r} must be float [N, {end - start}], got {part.shape}")
            parts.append(part)
        return np.concatenate(parts, axis=-1)

    def build_inputs(
        self,
        frames: Mapping[str, np.ndarray],
        states: Mapping[str, np.ndarray],
        instruction: str,
    ) -> dict[str, torch.Tensor]:
        """Run the checkpoint's processor on a batch of ``N`` observations (frames ``[N, H, W, 3]`` per view, states ``[N, d]`` per group)."""
        missing = [key for key in self.video_keys if key not in frames]
        missing += [key for key in self.state_keys if key not in states]
        if missing:
            raise KeyError(f"observations lack the checkpoint's modality keys {missing}")
        observation: dict[str, Any] = {"state": self._flat_state(states)}
        for key in self.video_keys:
            frame = np.asarray(frames[key])
            if frame.dtype != np.uint8 or frame.ndim != 4:
                raise ValueError(f"frame {key!r} must be uint8 [N, H, W, 3], got {frame.dtype} {frame.shape}")
            observation[key] = frame
        inputs = gr00t_data.build_inputs(
            self.processor, observation, self.embodiment_tag, self.modality_json, instruction, str(self.device)
        )
        return dict(inputs)

    def decode(
        self,
        normalized: torch.Tensor,
        *,
        inputs: Mapping[str, torch.Tensor],
        states: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """Decode ``[B, H, D]`` normalized chunks into ``[B, window, A_raw]`` raw actions; relative groups reference the raw state."""
        del inputs
        decoded = gr00t_data.decode_action_chunks(
            self.processor,
            normalized.detach().float().cpu().numpy(),
            self.embodiment_tag,
            self.modality_json,
            self._flat_state(states),
        )
        return decoded[:, : self.action_window]

    def learner_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Only the head's own inputs travel; the backbone's outputs go as cached features."""
        return {key: inputs[key] for key in self.HEAD_INPUT_KEYS}

    # ------------------------------------------------------------------
    # The head's pieces
    # ------------------------------------------------------------------
    scores_demonstrations = True

    def supervised_loss(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """Score demonstrations with the model's own action loss, masked.

        The head emits ``[B, max_horizon, max_action_dim]`` and masks the part
        this embodiment occupies -- on a 28-DoF G1 that is 16x28 of a 40x132
        grid. Averaging the whole grid would divide by the padding and scale
        the co-training weight down by the ratio (11.8x, measured), so the
        reduction is a masked mean, which is also what the model's own scalar
        loss reports.
        """
        outputs = self.model(dict(batch))
        action_loss = outputs["action_loss"].float()
        mask = outputs["action_mask"].to(action_loss.dtype)
        samples = action_loss.shape[0]
        flat_loss = action_loss.reshape(samples, -1)
        flat_mask = mask.reshape(samples, -1)
        return (flat_loss * flat_mask).sum(dim=1) / flat_mask.sum(dim=1).clamp_min(1.0)

    def condition(
        self, inputs: Mapping[str, torch.Tensor], cached_features: Mapping[str, Any] | None = None
    ) -> FlowConditioning:
        head = self.model.action_head
        if cached_features is None:
            backbone_inputs, _ = self.model.prepare_input(dict(inputs))
            with torch.set_grad_enabled(self.backbone_trainable and torch.is_grad_enabled()):
                outputs = self.model.backbone(backbone_inputs)
            raw = {name: outputs[name] for name in self.cached_feature_names}
        else:
            cached = unpack_features(cached_features, self.device)
            missing = [name for name in self.cached_feature_names if name not in cached]
            if missing:
                raise KeyError(f"cached features lack {missing}; expected {list(self.cached_feature_names)}")
            raw = {name: cached[name] for name in self.cached_feature_names}
            raw["backbone_features"] = raw["backbone_features"].to(self.dtype)
        # The head's own normalization and self-attention over the tokens stay trainable.
        processed = head.process_backbone_output(dict(raw))
        vl_embs = processed["backbone_features"]
        state = inputs["state"].to(device=self.device, dtype=self.dtype)
        embodiment_id = inputs["embodiment_id"].to(self.device)
        # The head folds the state history into one token: [B, T, S] -> [B, 1, T * S].
        state_features = head.state_encoder(state.reshape(state.shape[0], 1, -1), embodiment_id)
        return FlowConditioning(
            context=_N17Conditioning(
                vl_embs=vl_embs,
                mask=raw["backbone_attention_mask"],
                image_mask=raw["image_mask"] if self.alternate_dit else None,
                state_features=state_features,
                embodiment_id=embodiment_id,
            ),
            cached=raw,
            batch_size=int(vl_embs.shape[0]),
        )

    def velocity(self, conditioning: FlowConditioning, x_t: torch.Tensor, step_index: torch.Tensor) -> torch.Tensor:
        head = self.model.action_head
        context: _N17Conditioning = conditioning.context
        buckets = (step_index.to(x_t.device).float() / float(self.denoising_steps) * head.num_timestep_buckets).long()
        action_features = head.action_encoder(x_t, buckets, context.embodiment_id)
        if head.config.add_pos_embed:
            positions = torch.arange(action_features.shape[1], dtype=torch.long, device=x_t.device)
            action_features = action_features + head.position_embedding(positions)[None]
        tokens = torch.cat((context.state_features, action_features), dim=1)
        if self.alternate_dit:
            hidden = head.model(
                hidden_states=tokens,
                encoder_hidden_states=context.vl_embs,
                timestep=buckets,
                image_mask=context.image_mask,
                backbone_attention_mask=context.mask,
            )
        else:
            hidden = head.model(hidden_states=tokens, encoder_hidden_states=context.vl_embs, timestep=buckets)
        return head.action_decoder(hidden, context.embodiment_id)[:, -head.action_horizon :]

    def value_features(self, conditioning: FlowConditioning) -> torch.Tensor:
        context: _N17Conditioning = conditioning.context
        pooled = masked_mean(context.vl_embs, context.mask).to(context.state_features.dtype)
        return torch.cat((pooled, context.state_features.reshape(context.state_features.shape[0], -1)), dim=1)

    def _critic_token_sequence(
        self, conditioning: FlowConditioning
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        context: _N17Conditioning = conditioning.context
        return critic_token_sequence(context.vl_embs, context.mask, context.state_features)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def spec_options(cls, base_model: Any) -> dict[str, Any]:
        """N1.7 loads a baked embodiment, or registers one from ``dataset_dir``."""
        for name in ("embodiment_tag", "action_horizon"):
            if getattr(base_model, name) is None:
                raise ValueError(f"base_model.{name} is required by the GR00T N1.7 flow-SDE adapter")
        options = {
            "embodiment_tag": str(base_model.embodiment_tag),
            "action_window": int(base_model.action_horizon),
            "backbone_model_name_or_path": base_model.backbone_model_name_or_path,
        }
        dataset_dir = _optional_str(getattr(base_model, "dataset_dir", None))
        if dataset_dir:
            options["dataset_dir"] = dataset_dir
            options["dataset_adapter"] = _optional_str(getattr(base_model, "dataset_adapter", None))
        return options

    @classmethod
    def load(
        cls,
        checkpoint: str,
        *,
        device: str,
        flow_sde: FlowSDEConfig,
        value_head_seed: int = 0,
        embodiment_tag: str,
        action_window: int,
        backbone_model_name_or_path: str | None = None,
        dataset_dir: str | None = None,
        dataset_adapter: Any = None,
    ) -> Gr00tN17FlowSDEActorCritic:
        """Load an N1.7 checkpoint as a trainable flow-SDE actor-critic.

        Fine-tuned G1 checkpoints bake ``new_embodiment`` into the processor.
        The released pretrained checkpoint ``nvidia/GR00T-N1.7-3B`` does not:
        IQL then registers the G1 layout and statistics from
        ``dataset_dir``. The action head is made trainable, the backbone
        stays frozen, dropout is disabled and a freshly initialized value
        head (seeded, so every rank and worker starts identical) is attached.
        """
        from isaac_rl_posttraining.policies.gr00t.policy import Gr00tPolicy

        policy = Gr00tPolicy(
            checkpoint,
            backbone_model_name_or_path=backbone_model_name_or_path,
            embodiment_tag=embodiment_tag,
            device=device,
        )
        ensure_n17_embodiment(
            policy,
            embodiment_tag=embodiment_tag,
            action_window=action_window,
            dataset_dir=dataset_dir,
            dataset_adapter=dataset_adapter,
        )
        model = policy.model.base_model
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in model.action_head.parameters():
            parameter.requires_grad_(True)
        dropped = 0 if flow_sde.keep_dropout else disable_dropout(model)
        modality = policy.processor.modality_configs[embodiment_tag]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(value_head_seed)
            actor_critic = cls(
                model,
                processor=policy.processor,
                embodiment_tag=embodiment_tag,
                modality_json=policy.modality_json,
                video_keys=[f"{VIDEO_PREFIX}{view}" for view in modality["video"].modality_keys],
                action_keys=list(modality["action"].modality_keys),
                action_window=action_window,
                config=flow_sde,
            )
        trainable = sum(p.numel() for p in actor_critic.parameters() if p.requires_grad)
        total = sum(p.numel() for p in actor_critic.parameters())
        logger.info(
            f"flow-SDE actor-critic (N1.7): {trainable / 1e6:.1f}M trainable of {total / 1e6:.1f}M parameters "
            f"(frozen backbone), {dropped} dropout layers disabled, K={flow_sde.denoising_steps}, "
            f"noise_level={flow_sde.noise_level}"
        )
        return actor_critic


def _optional_str(value: Any) -> str | None:
    """Hydra ``null`` and empty strings both mean 'not set'."""
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "null", "None"}:
        return None
    return text


def ensure_n17_embodiment(
    policy: Any,
    *,
    embodiment_tag: str,
    action_window: int,
    dataset_dir: str | None,
    dataset_adapter: Any,
) -> None:
    """Register the embodiment the N1.7 adapter infers with.

    Fine-tuned G1 checkpoints bake ``new_embodiment``. The released
    pretrained ``nvidia/GR00T-N1.7-3B`` does not, so IQL registers the G1
    layout and statistics from the demonstration dataset instead.
    """
    if gr00t_data.has_baked_embodiment(policy.processor, embodiment_tag):
        policy.register_baked_embodiment()
        return
    if not dataset_dir:
        raise ValueError(
            f"N1.7 checkpoint has no baked embodiment {embodiment_tag!r}; "
            "set base_model.dataset_dir so the layout can be registered from demonstrations"
        )
    if isinstance(dataset_adapter, str):
        raise TypeError("dataset_adapter must be resolved by the launcher (resolve_adapter_option)")
    decode_horizon = getattr(dataset_adapter, "decode_horizon", None) or int(action_window)
    gr00t_data.register_embodiment(
        policy.processor,
        dataset_dir,
        embodiment_tag,
        decode_horizon,
        state_groups=getattr(dataset_adapter, "fallback_state_groups", None),
        action_groups=getattr(dataset_adapter, "fallback_action_groups", None),
    )
    policy.modality_json = gr00t_data.modality_json_from_processor(
        policy.processor, embodiment_tag
    )
