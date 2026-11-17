# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.7 as a VLA policy: loading and action decoding.

:class:`Gr00tPolicy` loads the open-source GR00T N1.7 model (vision-language
backbone plus flow-matching action head) and wraps it in
:class:`~.model.Gr00tModel`. Steering algorithms keep the whole model frozen
and use only the inference paths (:meth:`Gr00tPolicy.act` and
:meth:`Gr00tPolicy.act_with_noise`).

All imports of the external ``gr00t``, ``transformers``, and ``peft``
packages happen inside the methods that need them, so this module imports
without the GR00T stack installed.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaac_rl_posttraining.policies.gr00t import data as gr00t_data
from isaac_rl_posttraining.policies.gr00t.finetune import FinetuneSpec
from isaac_rl_posttraining.policies.gr00t.model import Gr00tModel, LatentInjectionConfig
from isaac_rl_posttraining.policies.gr00t.vla import VLAPolicy, register_vla
from isaac_rl_posttraining.utils.torch_utils import require_cuda_kernels

logger = logging.getLogger(__name__)

__all__ = ["Gr00tPolicy", "make_offline_buffer", "make_multitask_buffer"]

_VIDEO_KEY_PREFIX = "video."

_DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@register_vla("gr00t")
class Gr00tPolicy(VLAPolicy):
    """GR00T N1.7 policy with latent-steered decoding.

    The base model loads frozen; :meth:`configure_optimizers` can attach
    LoRA adapters to the backbone for finetuning phases. Dataset modality
    and normalization statistics must be registered before inference by
    building an offline buffer (:meth:`make_offline_buffer` or
    :meth:`make_multitask_buffer`).

    Attributes:
        model: The :class:`~.model.Gr00tModel` wrapper around the base model.
        processor: The checkpoint's processor (observation and action
            transforms), or ``None`` when it could not be loaded.
        modality_json: Modality layout registered by the offline buffer.
        task: Default task instruction registered by the offline buffer.
    """

    obs_format = "gr00t"
    needs_chunked_data = True

    def __init__(
        self,
        base_ckpt: str,
        backbone_model_name_or_path: str | None = None,
        embodiment_tag: str = "new_embodiment",
        device: str = "cuda",
        action_horizon: int | None = None,
        latent_injection: LatentInjectionConfig | None = None,
    ) -> None:
        """Load the base model and the processor.

        Args:
            base_ckpt: Local checkpoint directory (or Hugging Face model id)
                of the GR00T N1.7 model.
            backbone_model_name_or_path: Optional local directory or model id
                overriding the backbone named by the checkpoint.
            embodiment_tag: Embodiment the policy runs as; datasets register
                their modality and statistics under this tag.
            device: Device the model lives on.
            action_horizon: Optional override of the checkpoint's flow
                horizon. The horizon only sets the flow sequence length, so
                shortening it is shape-safe; the padded action width is
                fixed by the checkpoint.
            latent_injection: Steering-noise knobs forwarded to
                :class:`~.model.Gr00tModel`.
        """
        super().__init__()
        require_cuda_kernels(device)
        self.device = device
        self.embodiment_tag = embodiment_tag
        self._phase = "offline"
        self.modality_json: dict | None = None  # set when a dataset registers
        self.task: str | list[str] | None = None

        # Importing the model module registers the GR00T N1.7 config and
        # model classes with transformers' Auto* factories, so
        # AutoModel.from_pretrained can build the class from the checkpoint.
        import gr00t.model.gr00t_n1d7.gr00t_n1d7  # noqa: F401
        import transformers
        from transformers import AutoConfig, AutoModel

        # bf16 throughout: the vision-language backbone uses attention
        # kernels that require half precision.
        model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
        resolved_backbone = backbone_model_name_or_path
        self._backbone_alias: tempfile.TemporaryDirectory[str] | None = None
        if resolved_backbone is not None:
            model_config = AutoConfig.from_pretrained(base_ckpt, trust_remote_code=True)
            original_backbone = str(model_config.model_name)
            alias_relative = Path(original_backbone)
            safe_alias = not alias_relative.is_absolute() and ".." not in alias_relative.parts
            if os.path.isdir(resolved_backbone) and original_backbone not in resolved_backbone and safe_alias:
                self._backbone_alias = tempfile.TemporaryDirectory(prefix="rlpt-gr00t-backbone-")
                alias = Path(self._backbone_alias.name, alias_relative)
                alias.parent.mkdir(parents=True, exist_ok=True)
                alias.symlink_to(Path(resolved_backbone).resolve(), target_is_directory=True)
                resolved_backbone = str(alias)
            model_config.model_name = resolved_backbone
            model_kwargs.update(
                config=model_config,
                transformers_loading_kwargs={
                    "trust_remote_code": True,
                    "local_files_only": os.path.isdir(resolved_backbone),
                },
            )
        base_model = AutoModel.from_pretrained(base_ckpt, **model_kwargs)
        if action_horizon is not None:
            logger.info(
                "overriding action_horizon %s -> %s",
                base_model.config.action_horizon,
                action_horizon,
            )
            base_model.config.action_horizon = int(action_horizon)
            base_model.action_head.config.action_horizon = int(action_horizon)
            base_model.action_head.action_horizon = int(action_horizon)
        # The base loads frozen; adapters attached later re-enable gradients
        # on just the adapter parameters.
        for param in base_model.parameters():
            param.requires_grad_(False)
        self.model = Gr00tModel(base_model, latent_injection=latent_injection)
        self.compute_dtype = torch.bfloat16
        self.model.base_model.to(device=device, dtype=self.compute_dtype)

        # Processor: registering the checkpoint's processor class is
        # best-effort (older checkpoints may not ship one).
        try:
            import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7  # noqa: F401
        except Exception as exc:
            logger.warning("GR00T processor class registration skipped (%s)", exc)
        self.processor = None
        try:
            processor_kwargs: dict[str, Any] = {"trust_remote_code": True}
            if resolved_backbone is not None:
                processor_kwargs.update(
                    model_name=resolved_backbone,
                    transformers_loading_kwargs={
                        "trust_remote_code": True,
                        "local_files_only": os.path.isdir(resolved_backbone),
                    },
                )
            self.processor = transformers.AutoProcessor.from_pretrained(
                self._resolve_processor_dir(base_ckpt), **processor_kwargs
            )
            self.processor.eval()
        except Exception as exc:
            logger.warning(
                "processor not loaded (%s: %s)", type(exc).__name__, str(exc)[:140]
            )

    def _online_modality_json(self, buffer_modality_json: dict) -> dict:
        """The layout of the flat state/action the environment exchanges online.

        The offline buffer slices the arrays *it* assembles by its own layout —
        with the G1 WBC adapter that is the dataset's full-body state, legs
        included (``right_arm`` at ``[29:36]``). The environment, however,
        sends the policy's state in the baked group order (the decoupled
        upper-body groups, ``left_arm`` at ``[0:7]``); slicing that with the
        buffer's spans mis-assigns every joint group and runs off the end. A
        checkpoint that bakes its embodiment therefore defines the online
        layout — the same one the datasetless path registers — and only a raw
        base model (nothing baked) falls back to the buffer's.
        """
        if gr00t_data.has_baked_embodiment(self.processor, self.embodiment_tag):
            return gr00t_data.modality_json_from_processor(self.processor, self.embodiment_tag)
        return buffer_modality_json

    @staticmethod
    def _resolve_processor_dir(base_ckpt: str) -> str:
        """Where the checkpoint keeps its processor files.

        GR00T-N1.7 fine-tuning saves the processor under a ``processor/``
        subdirectory while ``AutoProcessor`` expects it at the model root;
        mirror upstream ``Gr00tPolicy`` and fall back to the subdirectory when
        the root carries no ``processor_config.json``. A model id or a root
        that ships its own processor resolves to itself.
        """
        subdir = os.path.join(base_ckpt, "processor")
        if os.path.isdir(subdir) and not os.path.exists(
            os.path.join(base_ckpt, "processor_config.json")
        ):
            return subdir
        return base_ckpt

    @property
    def action_horizon(self) -> int:
        """Flow horizon ``H`` of the wrapped model."""
        return self.model.action_horizon

    @property
    def decoded_action_horizon(self) -> int:
        """Action steps ``decode_action`` emits: the registered action window.

        A checkpoint's baked modality config fixes this window (16 for the
        released N1.7 processors) independently of the flow head's
        noise-window length (``model.action_horizon``).
        """
        # Precondition failures raise RuntimeError: inside an nn.Module
        # property an AttributeError would be masked by Module.__getattr__
        # and surface as "no attribute decoded_action_horizon".
        if self.processor is None:
            raise RuntimeError(
                "decoded_action_horizon requires a loaded processor; "
                "register a dataset (make_offline_buffer/make_multitask_buffer) first"
            )
        try:
            mc = self.processor.modality_configs[self.embodiment_tag]
        except (KeyError, AttributeError) as exc:
            raise RuntimeError(
                f"no modality config registered for embodiment tag "
                f"{self.embodiment_tag!r}: {exc}"
            ) from exc
        return len(mc["action"].delta_indices)

    @property
    def raw_action_dim(self) -> int:
        """Width of one decoded action step in the raw environment space.

        Computed from the registered modality layout as the total width of
        the trained action groups -- the width :meth:`act` and
        :meth:`act_with_noise` return.

        Raises:
            RuntimeError: If no dataset has been registered yet.
        """
        if self.modality_json is None or self.processor is None:
            raise RuntimeError(
                "raw_action_dim requires registered dataset modality; build an "
                "offline buffer (make_offline_buffer / make_multitask_buffer) first."
            )
        action_keys = self.processor.modality_configs[self.embodiment_tag]["action"].modality_keys
        groups = self.modality_json.get("action", {})
        return sum(
            int(groups[key]["end"]) - int(groups[key]["start"])
            for key in action_keys
            if key in groups
        )

    # ------------------------------------------------------------------
    # Fine-tuning structure
    # ------------------------------------------------------------------

    def apply_finetune_spec(self, finetune_spec: FinetuneSpec | None = None) -> None:
        """Attach LoRA adapters per the spec (no optimizers are built here).

        Called before loading a fine-tuned checkpoint so the adapter
        parameters exist in the state-dict keys; the optimizer over the
        resulting trainable set is built by the caller's training loop.

        Args:
            finetune_spec: Differential-finetuning spec. The first rule
                carrying a LoRA config determines the adapter attached to
                the backbone.
        """
        lora_rules = finetune_spec.lora_rules() if finetune_spec is not None else []
        if lora_rules:
            from peft import LoraConfig, get_peft_model

            rule = lora_rules[0].lora
            backbone = self.model.base_model.backbone
            self.model.base_model.backbone = get_peft_model(
                backbone,
                LoraConfig(
                    r=rule.get("r", 16),
                    lora_alpha=rule.get("alpha", 32),
                    lora_dropout=rule.get("dropout", 0.0),
                    target_modules=rule.get("target_modules", _DEFAULT_LORA_TARGET_MODULES),
                    bias="none",
                ),
            )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self, obs: dict[str, Any], deterministic: bool = False, actor_state: Any = None
    ) -> tuple[torch.Tensor, Any]:
        """Decode a raw action chunk for one observation, natively.

        Builds model inputs for a batch of one, samples the action head, and
        decodes per dimension into the environment's raw action space
        (un-normalization plus relative-to-absolute conversion against the
        current state).

        Args:
            obs: ``{"ego_view_image" | "ego_view": [H, W, C] or [1, H, W, C],
                "state": [S]}``.
            deterministic: Ignored; flow sampling is inherently stochastic.
            actor_state: Passed through unchanged.

        Returns:
            ``(action, actor_state)`` with ``action`` of shape
            ``[H, A_raw]`` on :attr:`device`.
        """
        state = self._obs_state(obs)
        inputs = self._build_model_inputs(obs, state)
        result = self.model.get_action(inputs)
        normalized = result["action_pred"].float().cpu().numpy()[0]  # [H, D]
        action = self._unapply(normalized, state)  # [H, A_raw]
        return torch.as_tensor(action, device=self.device).float(), actor_state

    @torch.no_grad()
    def act_with_noise(self, obs: dict[str, Any], latent: torch.Tensor) -> np.ndarray:
        """Decode a raw action chunk steered by an external latent.

        Same observation-to-action path as :meth:`act`, but the flow initial
        noise is built from ``latent`` (see
        :meth:`~.model.Gr00tModel.get_action_with_noise`).

        Args:
            obs: Observation dict as in :meth:`act`.
            latent: Steering latent for a batch of one; see
                :func:`~.model.build_base_noise` for accepted shapes.

        Returns:
            Raw action chunk of shape ``[H, A_raw]``.
        """
        state = self._obs_state(obs)
        inputs = self._build_model_inputs(obs, state)
        action_pred = self.model.get_action_with_noise(inputs, latent)  # [1, H, D]
        normalized = action_pred.float().cpu().numpy()[0]
        return self._unapply(normalized, state)

    @torch.no_grad()
    def infer_reference(
        self, obs: dict[str, Any], image_transform_mode: str = "eval"
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Decode the native reference action plus pooled backbone features.

        The frozen policy's own un-steered action chunk (what it would execute
        without steering) together with mean/max-pooled backbone tokens, both
        from one preprocessing pass. Feeds the reference-conditioned and
        direct-action-edit learners, which condition on the base action and
        the scene features rather than steering the flow noise.

        Args:
            obs: Observation dict as in :meth:`act`.
            image_transform_mode: ``"train"`` or ``"eval"``. GR00T's processor
                applies its single trained image transform either way, so the
                frozen reference decodes identically; the mode is validated to
                honor the reference-inference contract, not applied.

        Returns:
            ``(features, reference)`` with pooled features of shape
            ``[1, 2 * feat_dim]`` on :attr:`device` and the raw reference
            action chunk of shape ``[1, H, A_raw]``.
        """
        if image_transform_mode not in ("train", "eval"):
            raise ValueError("image_transform_mode must be 'train' or 'eval'")
        state = self._obs_state(obs)
        inputs = self._build_model_inputs(obs, state)
        backbone_features, _ = self.model.extract_features(inputs)  # [1, T, D]
        pooled = torch.cat(
            [backbone_features.mean(dim=1), backbone_features.amax(dim=1)], dim=1
        ).float()  # [1, 2 * feat_dim]: mean+max pool; f32 since the backbone runs in bf16 (numpy has no bf16)
        normalized = self.model.get_action(inputs)["action_pred"].float().cpu().numpy()[0]  # [H, D]
        reference = self._unapply(normalized, state)[None]  # [1, H, A_raw]
        return pooled, reference

    @torch.no_grad()
    def act_batch(
        self, obs_list: list[dict[str, Any]], latents: np.ndarray | None = None
    ) -> np.ndarray:
        """Decode raw action chunks for a batch of observations in ONE forward.

        The batched counterpart of :meth:`act` / :meth:`act_with_noise`: the
        per-sample path decoded one env at a time, so N steering decodes ran
        sequentially; this stacks the N observations into one ``[B]``-batch of
        model inputs, runs a single backbone + action-head forward (native, or
        steered by per-env latents via :meth:`~.model.Gr00tModel.get_action_with_noise`),
        and decodes each sample's chunk against its own state. The backbone
        processes each sequence independently, so the batched decode has the
        same steering semantics as the per-sample path and is equivalent in
        distribution; it is bit-identical only when the RNG state is aligned
        (the un-steered padding noise is drawn fresh per call, so ``N``
        separate draws differ from one ``[N]`` draw), e.g. row 0 under a shared
        seed. Only homogeneous batches are supported — one shared instruction
        and identical camera keys across envs, the single-task vector-env case —
        anything else fails loud rather than batching unvalidated padding.

        Args:
            obs_list: ``B`` single-env observation dicts (each as in :meth:`act`).
            latents: ``[B, latent_dim]`` / ``[B, K, latent_dim]`` per-env
                steering latents, or ``None`` for native flow noise.

        Returns:
            ``[B, H, A_raw]`` raw action chunks.

        Raises:
            ValueError: If the envs carry different instructions or camera keys.
        """
        states = [self._obs_state(obs) for obs in obs_list]
        inputs = self._build_model_inputs_batch(obs_list, states)
        if latents is None:
            action_pred = self.model.get_action(inputs)["action_pred"]  # [B, H, D]
        else:
            lat = torch.as_tensor(np.asarray(latents, dtype=np.float32), device=self.device)
            action_pred = self.model.get_action_with_noise(inputs, lat)  # [B, H, D]
        normalized = action_pred.float().cpu().numpy()  # [B, H, D]
        return np.stack([self._unapply(normalized[b], states[b]) for b in range(len(obs_list))])

    @torch.no_grad()
    def prepare_initial_noise(self, latent: np.ndarray) -> np.ndarray:
        """Expand one steering latent while advancing the caller's RNG stream."""
        prepared = self.model.prepare_initial_noise(
            torch.as_tensor(np.asarray(latent, dtype=np.float32), device=self.device)
        )
        return prepared.float().cpu().numpy()

    @torch.no_grad()
    def sample_native_initial_noise(self) -> np.ndarray:
        """Draw one native ``[1, H, D]`` flow-noise sample on the current RNG stream."""
        return self.model.sample_native_initial_noise(1).float().cpu().numpy()

    @torch.no_grad()
    def act_batch_with_initial_noise(
        self,
        obs_list: list[dict[str, Any]],
        initial_noise: np.ndarray,
    ) -> np.ndarray:
        """Decode independently sampled initial noise in one model forward."""
        states = [self._obs_state(obs) for obs in obs_list]
        inputs = self._build_model_inputs_batch(obs_list, states)
        action_pred = self.model.get_action_with_initial_noise(
            inputs,
            torch.as_tensor(
                np.asarray(initial_noise, dtype=np.float32),
                device=self.device,
            ),
        )
        normalized = action_pred.float().cpu().numpy()
        return np.stack(
            [
                self._unapply(normalized[index], states[index])
                for index in range(len(obs_list))
            ]
        )

    def _build_model_inputs_batch(
        self, obs_list: list[dict[str, Any]], states: list[np.ndarray]
    ) -> Any:
        """Batched counterpart of :meth:`_build_model_inputs`: stack ``B`` samples.

        Mirrors the single-sample builder but stacks the state, the shared ego
        frame, and any per-view ``video.<key>`` entries along a leading ``[B]``
        axis. The batch must be homogeneous: every env carries the same camera
        keys and the same instruction, which is then broadcast over the batch
        exactly as GR00T's own multi-env evaluation does (per-sample
        instructions would pad to the batch max — an unvalidated path, so it is
        refused rather than silently batched). Camera routing and
        native-resolution handling are unchanged.
        """
        first = obs_list[0]
        video_keys = sorted(key for key in first if key.startswith(_VIDEO_KEY_PREFIX))
        has_ego = "ego_view_image" in first or "ego_view" in first
        for index, obs in enumerate(obs_list[1:], start=1):
            obs_video_keys = sorted(
                key for key in obs if key.startswith(_VIDEO_KEY_PREFIX)
            )
            obs_has_ego = "ego_view_image" in obs or "ego_view" in obs
            if obs_video_keys != video_keys or obs_has_ego != has_ego:
                raise ValueError(
                    "batched decode needs every env to carry the same camera keys; env 0 has "
                    f"video={video_keys} ego={has_ego} but env {index} has "
                    f"video={obs_video_keys} ego={obs_has_ego}"
                )
        tasks = {self._obs_task(obs) for obs in obs_list}
        if len(tasks) != 1:
            raise ValueError(
                "batched decode needs one shared instruction across the batch (the "
                f"single-task case); got {len(tasks)} distinct instructions: {sorted(map(str, tasks))}"
            )

        obs_batch: dict[str, Any] = {
            "state": np.stack([np.asarray(s, dtype=np.float32) for s in states])
        }
        for key in video_keys:
            obs_batch[key] = np.stack([self._frame_batch(obs[key])[0] for obs in obs_list])
        if has_ego:
            obs_batch["ego_view_image"] = np.stack([self._obs_image(obs) for obs in obs_list])
        return gr00t_data.build_inputs(
            self.processor,
            obs_batch,
            self.embodiment_tag,
            self.modality_json,
            tasks.pop(),
            self.device,
        )

    def _build_model_inputs(self, obs: dict[str, Any], state: np.ndarray) -> Any:
        """Convert one observation into processed model inputs (batch of 1).

        Per-view camera entries (``"video.<key>"``) pass through to the
        matching modality views; a shared ``ego_view_image``/``ego_view``
        frame (when present) covers any view without its own entry. A
        ``"prompt"`` string riding in the observation (the convention of the
        in-process environment providers) overrides the registered default
        task instruction, so multi-task suites condition on the right
        language without re-registering.
        """
        obs_batch: dict[str, Any] = {"state": state[None]}
        for key, value in obs.items():
            if key.startswith(_VIDEO_KEY_PREFIX):
                obs_batch[key] = self._frame_batch(value)
        if "ego_view_image" in obs or "ego_view" in obs:
            obs_batch["ego_view_image"] = self._obs_image(obs)[None]
        return gr00t_data.build_inputs(
            self.processor,
            obs_batch,
            self.embodiment_tag,
            self.modality_json,
            self._obs_task(obs),
            self.device,
        )

    def _obs_task(self, obs: dict[str, Any]) -> str | list[str] | None:
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
        return self.task

    @staticmethod
    def _frame_batch(value: Any) -> np.ndarray:
        """Shape one camera frame as a batch of one (``[1, H, W, C]``)."""
        frame = np.asarray(value.detach().cpu() if torch.is_tensor(value) else value)
        if frame.ndim == 4 and frame.shape[0] == 1:
            return frame
        return frame[None]

    def _unapply(self, normalized: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Decode one normalized head output ``[H, D]`` into the raw action space ``[H, A_raw]``.

        The current state supplies the reference of relative action groups
        (see :func:`~.data.decode_action_chunks`).
        """
        return gr00t_data.decode_action_chunks(
            self.processor, normalized[None], self.embodiment_tag, self.modality_json, state[None]
        )[0]

    @staticmethod
    def _obs_image(obs: dict[str, Any]) -> np.ndarray:
        """Extract the ego-view image, dropping a singleton batch axis."""
        image = obs["ego_view_image"] if "ego_view_image" in obs else obs["ego_view"]
        image = np.asarray(image.detach().cpu() if torch.is_tensor(image) else image)
        # Vectorized environments yield [1, H, W, C]; the batch axis is
        # re-added in _build_model_inputs, so drop it here to avoid feeding
        # a 5-D stack to the processor.
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        return image

    @staticmethod
    def _obs_state(obs: dict[str, Any]) -> np.ndarray:
        """Extract the state vector as a flat float32 array."""
        state = obs["state"]
        return np.asarray(
            state.detach().cpu() if torch.is_tensor(state) else state, dtype=np.float32
        ).reshape(-1)

    # ------------------------------------------------------------------
    # Dataset registration
    # ------------------------------------------------------------------

    def register_baked_embodiment(self, task: str | None = None) -> None:
        """Register the embodiment from the checkpoint's baked processor config.

        The datasetless registration path: released task checkpoints bake
        their embodiment (modality configs plus normalization statistics)
        into the processor, making the checkpoint self-describing. The flat
        state/action layout is derived from that baked config, so inference
        needs no dataset directory at all.

        Args:
            task: Optional default instruction for observations that carry
                no ``"prompt"`` entry. In-process environment providers put
                the task language in the observation, which takes precedence
                per call.

        Raises:
            RuntimeError: If the checkpoint's processor was not loaded, or
                the checkpoint did not bake :attr:`embodiment_tag`.
        """
        if self.processor is None:
            raise RuntimeError(
                "GR00T processor not loaded; a baked embodiment cannot be registered."
            )
        self.modality_json = gr00t_data.register_baked_embodiment(
            self.processor, self.embodiment_tag
        )
        if task is not None:
            self.task = task
        logger.info(
            "registered baked embodiment %r (state %d-dim, action %d-dim)",
            self.embodiment_tag,
            max((s["end"] for s in self.modality_json["state"].values()), default=0),
            max((s["end"] for s in self.modality_json["action"].values()), default=0),
        )

    def make_offline_buffer(
        self,
        dataset_dir: str,
        max_episodes: int | None = None,
        image_size: int = 224,
        dataset_adapter: Any = None,
    ) -> "gr00t_data.Gr00tOfflineBuffer":
        """Build a single-task offline buffer and register its embodiment.

        Registers the dataset's modality and normalization statistics into
        the processor and shares the layout and task instruction with the
        inference paths.

        Args:
            dataset_dir: LeRobot dataset root.
            max_episodes: Optional cap on the number of episodes loaded.
            image_size: Frame side length for the buffer's decoded images.
            dataset_adapter: Embodiment adapter mapping raw episodes to
                contiguous arrays and their modality; ``None`` uses the raw
                columns and the dataset's own modality.

        Returns:
            The constructed buffer.

        Raises:
            RuntimeError: If the checkpoint's processor was not loaded.
        """
        if self.processor is None:
            raise RuntimeError("GR00T processor not loaded; cannot build an offline buffer.")
        buffer = gr00t_data.Gr00tOfflineBuffer(
            dataset_dir,
            self.processor,
            self.embodiment_tag,
            action_horizon=self.model.action_horizon,
            image_size=image_size,
            device=self.device,
            max_episodes=max_episodes,
            dataset_adapter=dataset_adapter,
        )
        self.modality_json = self._online_modality_json(buffer.modality_json)
        self.task = buffer.task
        return buffer

    def make_multitask_buffer(
        self,
        dataset_dirs: list[str],
        max_episodes_per_task: int | None = None,
        image_size: int = 224,
        dataset_adapter: Any = None,
    ) -> "gr00t_data.Gr00tMultiTaskOfflineBuffer":
        """Build a multi-task offline buffer and register combined statistics.

        Args:
            dataset_dirs: One LeRobot dataset root per task.
            max_episodes_per_task: Optional per-task episode cap.
            image_size: Frame side length for the buffer's decoded images.
            dataset_adapter: Embodiment adapter shared across tasks; ``None``
                uses the raw columns and the first dataset's modality.

        Returns:
            The constructed buffer.

        Raises:
            RuntimeError: If the checkpoint's processor was not loaded.
        """
        if self.processor is None:
            raise RuntimeError("GR00T processor not loaded; cannot build an offline buffer.")
        buffer = gr00t_data.Gr00tMultiTaskOfflineBuffer(
            list(dataset_dirs),
            self.processor,
            self.embodiment_tag,
            action_horizon=self.model.action_horizon,
            image_size=image_size,
            device=self.device,
            max_episodes_per_task=max_episodes_per_task,
            dataset_adapter=dataset_adapter,
        )
        self.modality_json = self._online_modality_json(buffer.modality_json)
        self.task = buffer.task
        return buffer

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, save_dir: str) -> None:
        """Save the base model as a plain state dict.

        Serialization goes to a node-local temporary file first and is then
        copied into ``save_dir``: writing multi-gigabyte checkpoints
        directly onto networked filesystems can fail mid-write, while a
        local serialize-then-copy is reliable. Call after
        :meth:`configure_optimizers` so any adapter parameters exist in the
        state-dict keys.

        Args:
            save_dir: Destination directory (created if needed).
        """
        os.makedirs(save_dir, exist_ok=True)

        def _save(obj: Any, name: str) -> None:
            tmp = os.path.join(tempfile.gettempdir(), f"_save_{os.getpid()}_{name}")
            torch.save(obj, tmp)
            shutil.copyfile(tmp, os.path.join(save_dir, name))
            os.remove(tmp)

        _save(self.model.base_model.state_dict(), "base_state.pt")

    def load(self, load_dir: str) -> None:
        """Load the base model state from a directory.

        Accepts checkpoints written by :meth:`save` (``base_state.pt``) as
        well as ``save_pretrained``-style directories (safetensors or
        ``pytorch_model.bin``). Apply the finetuning spec first
        (:meth:`apply_finetune_spec`) when the checkpoint contains adapter
        keys, so the state-dict keys match.

        Args:
            load_dir: Checkpoint directory.
        """
        base_path = os.path.join(load_dir, "base_state.pt")
        if os.path.exists(base_path):
            state = torch.load(base_path, map_location="cpu", weights_only=True)
        else:
            safetensor_files = sorted(glob.glob(os.path.join(load_dir, "*.safetensors")))
            if safetensor_files:
                from safetensors.torch import load_file

                # Sharded checkpoints spread the state dict over several
                # files; merge every shard (loading only the first silently
                # restored a fraction of the model).
                state = {}
                for shard in safetensor_files:
                    state.update(load_file(shard))
            else:
                state = torch.load(
                    os.path.join(load_dir, "pytorch_model.bin"),
                    map_location="cpu",
                    weights_only=True,
                )
        missing, unexpected = self.model.base_model.load_state_dict(state, strict=False)
        logger.info(
            "loaded base model (%d tensors; missing %d, unexpected %d)",
            len(state),
            len(missing),
            len(unexpected),
        )
        if missing:
            logger.warning(
                "base-model load left %d parameters untouched (first: %s); this is only "
                "safe when those weights were already correct (e.g. the checkpoint was "
                "also the from_pretrained source)",
                len(missing),
                missing[0],
            )


def make_offline_buffer(
    policy: Gr00tPolicy,
    dataset_dir: str,
    max_episodes: int | None = None,
    image_size: int = 224,
    dataset_adapter: Any = None,
) -> "gr00t_data.Gr00tOfflineBuffer":
    """Module-level convenience for :meth:`Gr00tPolicy.make_offline_buffer`."""
    return policy.make_offline_buffer(
        dataset_dir,
        max_episodes=max_episodes,
        image_size=image_size,
        dataset_adapter=dataset_adapter,
    )


def make_multitask_buffer(
    policy: Gr00tPolicy,
    dataset_dirs: list[str],
    max_episodes_per_task: int | None = None,
    image_size: int = 224,
    dataset_adapter: Any = None,
) -> "gr00t_data.Gr00tMultiTaskOfflineBuffer":
    """Module-level convenience for :meth:`Gr00tPolicy.make_multitask_buffer`."""
    return policy.make_multitask_buffer(
        dataset_dirs,
        max_episodes_per_task=max_episodes_per_task,
        image_size=image_size,
        dataset_adapter=dataset_adapter,
    )
