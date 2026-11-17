# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frozen GR00T policy as seen by latent-steering RL algorithms.

:class:`SteeredGr00tWrapper` loads a GR00T policy (base checkpoint plus an
optional finetuned checkpoint), registers dataset normalization statistics,
freezes every parameter, and exposes the narrow surface steering algorithms
consume: ``infer(observations, noise)`` returning a decoded action chunk,
plus the ``action_horizon`` / ``action_dim`` / ``latent_dim`` shape
properties. DSRL is the first such algorithm; the wrapper itself is
algorithm-agnostic -- it only decodes latents.

Steering runs SAC (or any latent-space learner) in the frozen flow-matching
policy's initial-noise space: the learner outputs a latent per step, and the
wrapper decodes it into raw environment actions through the policy's action
head. The policy is never trained.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence

import numpy as np
import torch

from isaac_rl_posttraining.data.spec import ArraySpec, ObservationSpec
from isaac_rl_posttraining.policies.gr00t.data import read_task
from isaac_rl_posttraining.policies.gr00t.finetune import FinetuneSpec, GroupSpec
from isaac_rl_posttraining.policies.gr00t.model import LatentInjectionConfig
from isaac_rl_posttraining.policies.gr00t.vla import make_vla_policy

# Registers the "gr00t" (N1.7) and "gr00t_n1_5" VLA policies with the registry.
import isaac_rl_posttraining.policies.gr00t.policy  # noqa: F401
from isaac_rl_posttraining.policies.gr00t.n15 import is_n15_checkpoint

logger = logging.getLogger(__name__)

__all__ = ["SteeredGr00tWrapper", "Gr00tDSRLWrapper"]

_DEFAULT_LORA = {"r": 16, "alpha": 32}


class SteeredGr00tWrapper:
    """Frozen GR00T policy with latent-steered and native decoding.

    For the steering learners, critic and actor observation features come
    from a separate embedding (for example a frozen vision encoder applied by
    an environment wrapper), keeping them decoupled from the GR00T backbone,
    so :meth:`get_features` is intentionally not implemented. The reference-
    conditioned and direct-action-edit learners instead take the pooled
    backbone features carried by :meth:`infer_reference`.

    Attributes:
        policy: The underlying :class:`~.policy.Gr00tPolicy` (N1.7) or
            :class:`~.n15.Gr00tN15Policy` (N1.5), selected by the
            checkpoint's declared model type.
        noise_dim: Steerable latent width per step, when configured.
        action_bound: Symmetric bound of the learner's latent action space
            (stored for consumers that size action spaces off the wrapper).
    """

    def __init__(
        self,
        base_ckpt: str,
        backbone_model_name_or_path: str | None = None,
        bc_ckpt: str | None = None,
        dataset_dir: str | None = None,
        embodiment_tag: str = "new_embodiment",
        device: str = "cuda",
        noise_dim: int | None = None,
        action_bound: float = 2.0,
        multitask_dirs: Sequence[str] | None = None,
        task_dir: str | None = None,
        action_horizon: int | None = None,
        latent_injection: LatentInjectionConfig | None = None,
        task_instruction: str | None = None,
        dataset_adapter: Any = None,
        data_config: str | None = None,
        denoising_steps: int | None = None,
    ) -> None:
        """Load, register statistics, freeze.

        A checkpoint declaring ``model_type: gr00t_n1_5`` in its
        ``config.json`` loads through the N1.5 path (:mod:`.n15`); it is
        self-describing (data config plus baked statistics), so the dataset
        and finetuned-checkpoint fields below do not apply to it. Every other
        checkpoint loads through the N1.7 path exactly as before.

        Exactly one normalization-statistics source is used, in this order:

        * Multi-task: ``multitask_dirs`` registers combined statistics over
          all task datasets (matching how a multi-task policy was trained),
          and ``task_dir`` selects the instruction of the task being
          steered.
        * Single-task: ``dataset_dir`` registers that dataset's modality,
          statistics, and task instruction.
        * Baked (both unset): the checkpoint's own baked processor config
          registers ``embodiment_tag`` -- released task checkpoints (e.g.
          GR00T-N1.7-LIBERO) ship modality configs and statistics in the
          checkpoint, so no dataset is needed. The task language comes from
          each observation's ``"prompt"`` entry (or ``task_instruction``).

        Args:
            base_ckpt: GR00T base checkpoint directory (or model id).
            backbone_model_name_or_path: Optional local directory or model id
                overriding the backbone named by the checkpoint.
            bc_ckpt: Optional finetuned checkpoint loaded on top of the
                base (the LoRA adapter structure is applied first so keys
                match).
            dataset_dir: Single-task dataset root.
            embodiment_tag: Embodiment tag the checkpoint was trained under.
            device: Device the policy runs on.
            noise_dim: Steerable latent width per step; defaults to the
                decoded action width.
            action_bound: Symmetric bound of the learner's latent action
                space.
            multitask_dirs: Dataset roots of every task in a multi-task
                mixture; mutually exclusive with the single-task path.
            task_dir: Dataset root of the task being steered (multi-task
                only); sets the language instruction.
            action_horizon: Optional flow-horizon override; must match the
                checkpoint's training horizon, otherwise the decoded chunk
                is mis-sized.
            latent_injection: Steering-noise knobs (clipping, scaling,
                tiling, base-noise freshness).
            task_instruction: Default instruction when the observations
                carry no ``"prompt"`` entry (baked path only; the dataset
                paths read the instruction from the dataset).
            dataset_adapter: Embodiment adapter injected by the env-aware
                caller, mapping a dataset's raw episodes to the contiguous
                arrays and modality the buffer registers (the env layer owns
                this; the wrapper only forwards it). ``None`` uses the
                buffer's generic raw-column path.
            data_config: N1.5 only (required there): the data config the
                checkpoint was finetuned with -- a ``gr00t``
                ``DATA_CONFIG_MAP`` key or a ``"module:Class"`` path.
            denoising_steps: N1.5 only: override of the flow head's number
                of denoising steps; ``None`` keeps the checkpoint's value.

        Raises:
            TypeError: If ``multitask_dirs`` is a bare string.
            ValueError: If a ``multitask_dirs`` entry does not exist, or a
                field is set that the checkpoint's generation does not
                support (see above).
            RuntimeError: If no dataset is configured and the checkpoint
                bakes no config for ``embodiment_tag``.
        """
        self.device = device
        self.noise_dim = noise_dim
        self.action_bound = float(action_bound)
        if is_n15_checkpoint(base_ckpt):
            unsupported = [
                name
                for name, value in (
                    ("bc_ckpt", bc_ckpt),
                    ("dataset_dir", dataset_dir),
                    ("multitask_dirs", multitask_dirs),
                    ("action_horizon", action_horizon),
                )
                if value
            ]
            if unsupported:
                raise ValueError(
                    "GR00T N1.5 checkpoints are self-describing (data config plus baked "
                    f"statistics); these fields do not apply to them: {unsupported}"
                )
            if data_config is None:
                raise ValueError(
                    "GR00T N1.5 checkpoints require data_config (the data config the "
                    "checkpoint was finetuned with, e.g. a 'module:Class' path)"
                )
            self.policy = make_vla_policy(
                "gr00t_n1_5",
                base_ckpt=base_ckpt,
                data_config=data_config,
                device=device,
                embodiment_tag=embodiment_tag,
                denoising_steps=denoising_steps,
                latent_injection=latent_injection,
            )
        else:
            if data_config is not None or denoising_steps is not None:
                raise ValueError(
                    "data_config and denoising_steps apply to GR00T N1.5 checkpoints only; "
                    f"{base_ckpt!r} is not one"
                )
            self.policy = make_vla_policy(
                "gr00t",
                base_ckpt=base_ckpt,
                backbone_model_name_or_path=backbone_model_name_or_path,
                device=device,
                embodiment_tag=embodiment_tag,
                action_horizon=action_horizon,
                latent_injection=latent_injection,
            )
        if bc_ckpt:
            # Apply the adapter structure so the finetuned checkpoint's keys
            # match, then load it. Without a finetuned checkpoint the base
            # model is used verbatim -- no adapters are attached.
            self.policy.apply_finetune_spec(
                FinetuneSpec(
                    default_lr=1e-4,
                    groups=[GroupSpec(pattern="*backbone*", trainable=True, lora=_DEFAULT_LORA)],
                )
            )
            self.policy.load(bc_ckpt)

        self._register_statistics(
            multitask_dirs=multitask_dirs,
            task_dir=task_dir,
            dataset_dir=dataset_dir,
            dataset_adapter=dataset_adapter,
            task_instruction=task_instruction,
        )

        # Steering never trains the policy; only the latent-space learner learns.
        self.policy.model.eval()
        for param in self.policy.model.parameters():
            param.requires_grad_(False)
        logger.info("steered GR00T wrapper ready (frozen); action_horizon=%d", self.action_horizon)

    def _register_statistics(
        self,
        *,
        multitask_dirs: Sequence[str] | None,
        task_dir: str | None,
        dataset_dir: str | None,
        dataset_adapter: Any,
        task_instruction: str | None,
    ) -> None:
        """Register normalization statistics from exactly one source (see ``__init__``)."""
        if isinstance(multitask_dirs, str):
            raise TypeError(
                "multitask_dirs must be a sequence of dataset directories, not a string"
            )
        if multitask_dirs:
            dirs = [str(d) for d in multitask_dirs]
            missing = [d for d in dirs if not os.path.isdir(d)]
            if missing:
                raise ValueError(f"multitask_dirs entries do not exist: {missing}")
            # combined statistics, shared modality
            self.policy.make_multitask_buffer(dirs, dataset_adapter=dataset_adapter)
            if task_dir:
                self.policy.task = read_task(os.path.join(task_dir, "meta"))
            logger.info(
                "multi-task statistics over %d datasets; task=%r", len(dirs), self.policy.task
            )
        elif dataset_dir:
            # One episode suffices: only the registration side effect is needed.
            self.policy.make_offline_buffer(
                dataset_dir, max_episodes=1, dataset_adapter=dataset_adapter
            )
        else:
            # Datasetless: the checkpoint's baked embodiment is authoritative.
            self.policy.register_baked_embodiment(task=task_instruction)

    @property
    def action_horizon(self) -> int:
        """Environment steps decoded per :meth:`infer` call.

        Reads the processor's registered action window (what
        ``decode_action`` actually emits), which a checkpoint's baked
        modality config controls — not the flow head's noise-window length
        (see :attr:`latent_horizon`).
        """
        return int(self.policy.decoded_action_horizon)

    @property
    def latent_horizon(self) -> int:
        """Length of the flow head's initial-noise window (steering window)."""
        return int(self.policy.model.action_horizon)

    @property
    def action_dim(self) -> int:
        """Width of one decoded action step in the raw environment space."""
        return int(self.policy.raw_action_dim)

    @property
    def latent_dim(self) -> int:
        """Steerable latent width per step."""
        return int(self.noise_dim) if self.noise_dim else self.action_dim

    def input_spec(self) -> ObservationSpec | None:
        """The state width the registered modality layout slices, as the launcher's load-time check.

        The layout's state groups tile the proprioception vector from index 0,
        so the observation must be exactly that wide: a wider vector (legs
        included) would be sliced as if its leading entries were the arms.
        Unknown until a dataset or a baked config registers the layout, and
        unknown for a base model that carries no flat layout at all (N1.5 keeps
        modality keys and delta indices, not start/end offsets).
        """
        layout = getattr(self.policy, "modality_json", None)
        if layout is None or not layout.get("state"):
            return None
        width = max(int(group["end"]) for group in layout["state"].values())
        return ObservationSpec({"state": ArraySpec((width,), "float32")})

    def infer(
        self, observations: dict[str, Any], noise: np.ndarray | None
    ) -> dict[str, np.ndarray]:
        """Decode one observation into a raw action chunk.

        Args:
            observations: Single-sample observation dict
                (``{"ego_view_image" | "ego_view", "state"}``).
            noise: Steering latent, or ``None`` for the native base policy
                (internal Gaussian flow noise, no steering -- the base
                success-rate baseline decodes exactly like native
                evaluation). Accepted shapes: ``[latent_dim]`` or
                ``[1, latent_dim]`` (broadcast across the horizon), or
                ``[1, K, latent_dim]`` (per-step latents).

        Returns:
            ``{"actions": [1, H, A_raw]}``.

        Raises:
            ValueError: If ``noise`` carries more than one sample or has an
                unsupported rank. (A ``[K, latent_dim]`` per-step latent
                must be passed as ``[1, K, latent_dim]``; a 2-D array is
                read as ``[batch, latent_dim]``.)
        """
        if noise is None:
            action, _ = self.policy.act(observations)
            action = (
                action.detach().cpu().numpy() if hasattr(action, "detach") else np.asarray(action)
            )
            return {"actions": action[None]}  # [1, H, A_raw]
        latent = np.asarray(noise, dtype=np.float32)
        if latent.ndim == 1:
            latent = latent[None]  # [latent_dim] -> [1, latent_dim]
        if latent.ndim not in (2, 3) or latent.shape[0] != 1:
            raise ValueError(
                "noise must be [latent_dim], [1, latent_dim], or [1, K, latent_dim] "
                f"(single sample); got shape {np.shape(noise)}"
            )
        action = self.policy.act_with_noise(
            observations, torch.as_tensor(latent, device=self.device)
        )
        return {"actions": np.asarray(action)[None]}  # [1, H, A_raw]

    #: GR00T decodes a whole batch in one forward (see :meth:`infer_batch`), so
    #: the vectorized collector steers all envs with a single call instead of
    #: looping. Base models that can only decode one sample at a time (e.g. pi0,
    #: whose transforms are single-example) omit this flag and are looped.
    supports_batched_infer: bool = True

    @property
    def supports_seeded_batch_decode(self) -> bool:
        """Whether the policy decodes per-session initial noise in one batch (GR00T N1.7, not N1.5)."""
        return all(
            callable(getattr(self.policy, name, None))
            for name in (
                "prepare_initial_noise",
                "sample_native_initial_noise",
                "act_batch_with_initial_noise",
            )
        )

    def prepare_seeded_batch_noise(self, noise: np.ndarray) -> np.ndarray:
        """Sample the full flow noise for one fixed-seed evaluation session."""
        latent = np.asarray(noise, dtype=np.float32)
        if latent.ndim == 1:
            latent = latent[None]
        if latent.ndim not in (2, 3) or latent.shape[0] != 1:
            raise ValueError(
                "seeded batch noise must describe one sample; "
                f"got shape {np.shape(noise)}"
            )
        return self.policy.prepare_initial_noise(latent)

    def prepare_seeded_native_noise(self) -> np.ndarray:
        """Sample one native flow-noise tensor for an isolated evaluation RNG."""
        return self.policy.sample_native_initial_noise()

    def infer_prepared_batch(
        self,
        observations: dict[str, Any] | list[dict[str, Any]],
        initial_noise: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Decode per-session initial noise without drawing shared batch RNG."""
        obs_list = self._as_obs_list(observations)
        prepared = np.asarray(initial_noise, dtype=np.float32)
        if prepared.ndim != 3 or prepared.shape[0] != len(obs_list):
            raise ValueError(
                "prepared initial noise must be [B, H, D] with "
                f"B == {len(obs_list)}; got shape {prepared.shape}"
            )
        actions = self.policy.act_batch_with_initial_noise(obs_list, prepared)
        return {"actions": np.asarray(actions)}

    def infer_batch(
        self, observations: dict[str, Any] | list[dict[str, Any]], noise: np.ndarray | None
    ) -> dict[str, np.ndarray]:
        """Decode ``B`` observations into raw action chunks in one model forward.

        The batched counterpart of :meth:`infer`. ``observations`` is either a
        batched dict — each array entry carrying a leading ``[B]`` axis, the
        shape the vectorized collector holds — or a list of ``B`` single-env
        dicts. ``noise`` is ``[B, latent_dim]`` / ``[B, K, latent_dim]`` per-env
        latents, or ``None`` for native. Returns ``{"actions": [B, H, A_raw]}``.

        Raises:
            ValueError: If ``noise``'s leading axis does not match ``B``.
        """
        obs_list = self._as_obs_list(observations)
        latent: np.ndarray | None = None
        if noise is not None:
            latent = np.asarray(noise, dtype=np.float32)
            if latent.ndim not in (2, 3) or latent.shape[0] != len(obs_list):
                raise ValueError(
                    "batched noise must be [B, latent_dim] or [B, K, latent_dim] with "
                    f"B == {len(obs_list)}; got shape {np.shape(noise)}"
                )
        actions = self.policy.act_batch(obs_list, latent)  # [B, H, A_raw]
        return {"actions": np.asarray(actions)}

    @staticmethod
    def _as_obs_list(
        observations: dict[str, Any] | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Split a batched observation dict into ``B`` single-env dicts.

        A list passes through. For a dict, the batch size is read from the
        leading axis of ``state`` (else a 4-D image); entries with that leading
        axis are indexed per env, and everything else (e.g. a shared prompt
        string) is passed through unchanged.
        """
        if isinstance(observations, (list, tuple)):
            return list(observations)

        batch = SteeredGr00tWrapper._observation_batch_size(observations)
        if batch is None:
            return [observations]

        obs_list: list[dict[str, Any]] = []
        for b in range(batch):
            obs_b: dict[str, Any] = {}
            for key, value in observations.items():
                if (
                    (isinstance(value, np.ndarray) or torch.is_tensor(value))
                    and value.ndim >= 1
                    and value.shape[0] == batch
                ):
                    obs_b[key] = value[b]
                elif isinstance(value, (list, tuple)) and len(value) == batch:
                    obs_b[key] = value[b]
                else:
                    obs_b[key] = value
            obs_list.append(obs_b)
        return obs_list

    @staticmethod
    def _observation_batch_size(observations: dict[str, Any]) -> int | None:
        """Leading batch axis of ``state`` (else a 4-D image); ``None`` if unbatched."""
        state = observations.get("state")
        if state is not None:
            arr = state.detach().cpu().numpy() if torch.is_tensor(state) else np.asarray(state)
            if arr.ndim >= 2:
                return int(arr.shape[0])
        for key in ("ego_view_image", "ego_view"):
            image = observations.get(key)
            if image is not None and hasattr(image, "ndim") and image.ndim == 4:
                return int(image.shape[0])
        return None

    def get_features(self, observations: dict[str, Any]) -> torch.Tensor:
        """Not implemented: the learner's features come from a separate encoder.

        Raises:
            NotImplementedError: Always; configure the learner to use its
                own observation embedding rather than GR00T backbone
                features, keeping the critic decoupled from the policy.
        """
        raise NotImplementedError(
            "SteeredGr00tWrapper has no backbone-feature path; use a separate observation "
            "embedding for the learner so the critic stays decoupled from the GR00T backbone."
        )

    def infer_reference(
        self, observations: dict[str, Any], *, image_transform_mode: str = "eval"
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Native reference action plus pooled backbone features.

        The frozen policy's un-steered action chunk and mean/max-pooled
        backbone tokens, for the reference-conditioned and direct-action-edit
        learners. Unlike :meth:`get_features` (deliberately absent), these
        learners consume the pooled features carried alongside the reference,
        so the steering critic elsewhere stays decoupled from the backbone.

        Args:
            observations: Single-sample observation dict as in :meth:`infer`.
            image_transform_mode: ``"train"`` or ``"eval"``; the frozen policy
                decodes through its native image transform either way.

        Returns:
            ``(features, reference)`` -- features ``[1, 2 * feat_dim]`` on the
            policy device, reference the raw action chunk ``[1, H, A_raw]``.
        """
        return self.policy.infer_reference(
            observations, image_transform_mode=image_transform_mode
        )

    def __call__(
        self, observations: dict[str, Any], noise: np.ndarray | None
    ) -> dict[str, np.ndarray]:
        """Alias for :meth:`infer`."""
        return self.infer(observations, noise)


#: Established import name for the wrapper (DSRL was the first consumer).
Gr00tDSRLWrapper = SteeredGr00tWrapper
