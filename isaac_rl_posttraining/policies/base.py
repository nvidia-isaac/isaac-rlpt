# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base model interface for RL post-training.

This module defines :class:`BaseModelPolicy`, the contract between pretrained
robot foundation policies (GR00T, pi0, ...) and the RL algorithms in
:mod:`isaac_rl_posttraining.algorithms`. Latent-steering methods such as DSRL treat the
pretrained policy as a black box that maps observations plus an optional
injected latent to a chunk of raw environment actions; this interface is that
boundary.

:class:`MockBaseModelPolicy`, a deterministic base model with a position-coded
decode, is provided in this module explicitly for contract tests.
"""

from __future__ import annotations

import abc
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np

__all__ = ["DEFAULT_LATENT_CLIP", "BaseModelPolicy", "MockBaseModelPolicy"]

#: Default symmetric bound applied to injected latents at injection time.
#: See contract note 3 on :class:`BaseModelPolicy`.
DEFAULT_LATENT_CLIP = 3.0
#: Backbone feature width the mock reports; its pooled features are twice as wide.
MOCK_FEATURE_DIM = 8


class BaseModelPolicy(abc.ABC):
    """Abstract interface between a pretrained policy and RL algorithms.

    A subclass loads and runs one kind of pretrained base model, chosen by
    the ``base_model`` config group (in Python, by
    :class:`~isaac_rl_posttraining.algorithms.BaseModel`). It exposes exactly
    one inference hook, :meth:`infer`, plus the three shape properties RL code
    needs to size its networks and buffers:

    * :attr:`action_horizon` — ``H``, environment steps produced per call.
    * :attr:`action_dim` — ``A_raw``, the width of one decoded action step.
    * :attr:`latent_dim` — steerable latent dimensions per action step.

    Contract
    --------
    Implementations must satisfy all of the following; the tests written
    against :class:`MockBaseModelPolicy` document the expected behavior.

    1. **Native mode.** ``infer(observations, noise=None)`` samples the
       base model's own initial noise and must be distributionally identical to
       the un-steered base policy. Algorithms rely on this as the
       "no intervention" baseline.
    2. **Per-dimension decode.** After the model head, each raw action
       dimension must be produced from the corresponding head output
       dimension alone: slicing, un-normalization, and relative-to-absolute
       integration are all per-dimension operations. No cross-dimension
       mixing (projections, whitening, ...) may happen after the head, so
       latent dimensions stay aligned with action dimensions.
    3. **Latent clipping.** Whether injected latents are clipped is up to the
       base model. The default — implemented by :class:`MockBaseModelPolicy` and
       recommended for new base models — is to clip every injected entry to
       ``[-DEFAULT_LATENT_CLIP, DEFAULT_LATENT_CLIP]`` (``±3``) at injection
       time. Base models that deviate must say so in their docstring.
    4. **Fresh padding noise.** Noise entries not covered by the caller's
       ``noise`` argument — steps beyond ``K``, and any internal noise
       dimensions beyond :attr:`latent_dim` — are drawn fresh from
       ``N(0, 1)`` on every call, never zeroed and never cached, so partial
       steering degrades gracefully to the native distribution.
    5. **Steering capability.** :attr:`supports_steering` reports whether the
       base model can accept an external latent at all. It defaults to ``True``;
       base models that cannot inject a latent (for example remote-inference
       base models whose serving protocol exposes no noise channel) override it
       to return ``False``, and their :meth:`infer` must then raise
       ``NotImplementedError`` with a clear message whenever ``noise`` is not
       ``None``. Native mode (note 1) is required of every base model.
    6. **Serving.** :attr:`servable` declares whether several rollout workers
       may share one instance through a policy server
       (``placement.policy.mode=served`` or ``placement.parts.base.mode=served``).
       A base model that keeps one episode's state without a session id (its
       ``start_episode()`` takes none) sets it to ``False``; the dry run then
       refuses a served placement.
    """

    #: Whether rollout workers may share one instance through a policy server (contract note 6).
    servable: ClassVar[bool] = True

    @property
    @abc.abstractmethod
    def action_horizon(self) -> int:
        """``H``: number of environment steps produced per :meth:`infer` call."""

    @property
    @abc.abstractmethod
    def action_dim(self) -> int:
        """``A_raw``: width of one decoded action step in the raw action space."""

    @property
    @abc.abstractmethod
    def latent_dim(self) -> int:
        """Number of steerable latent dimensions per action step."""

    @property
    def latent_horizon(self) -> int:
        """``K_max``: maximum per-step steering latents accepted by :meth:`infer`.

        For most base models this equals :attr:`action_horizon`. Base models whose
        generative process denoises a longer window than they decode (for
        example a flow-matching head with a 40-step noise window that emits a
        16-step action chunk) report the noise-window length here, so steering
        code can size latents against the window while execution code sizes
        itself against :attr:`action_horizon`.
        """
        return self.action_horizon

    @property
    def supports_steering(self) -> bool:
        """Whether :meth:`infer` accepts an externally injected latent.

        ``True`` (the default) means the full contract applies: ``noise`` may
        be ``None``, ``[B, latent_dim]``, or ``[B, K, latent_dim]``. Base models
        that cannot accept an external latent — the serving path samples its
        own initial noise and exposes no injection channel — return ``False``;
        for those base models :meth:`infer` must raise ``NotImplementedError``
        with a clear message whenever ``noise`` is not ``None``, and only
        native mode (``noise=None``) is available. Latent-steering algorithms
        check this flag before sizing their latent actors.
        """
        return True

    @property
    def supports_finetune(self) -> bool:
        """Whether this base model's own weights can be updated by an algorithm.

        ``False`` (the default) marks a *frozen* base: the algorithm improves it
        without touching its weights (e.g. DSRL steers it through :meth:`infer`).
        A *trainable* base model returns ``True`` and exposes an ``update(batch)``
        surface (see :class:`~isaac_rl_posttraining.policies.gr00t.vla.VLAPolicy`),
        letting a direct-finetuning algorithm optimize it in place. This is the
        mutability counterpart of :attr:`supports_steering`; a finetuning
        algorithm checks it before wiring the base model as its trainable actor.
        """
        return False

    def input_spec(self) -> Any:
        """The observation entries this base model reads, with the widths its checkpoint fixes.

        Returns an :class:`~isaac_rl_posttraining.data.spec.ObservationSpec`
        once the loaded checkpoint knows them (GR00T's state layout fixes the
        proprioception width it slices), or ``None`` when the base model cannot
        state them; the launcher compares the spec with the environment's
        observation before any rollout so a wrongly laid-out state vector
        fails at load rather than at the first decode.
        """
        return None

    @classmethod
    def from_dsrl_config(cls, cfg: Any, device: Any, dataset_adapter: Any = None) -> "BaseModelPolicy":
        """Build this base model for a DSRL run from the composed config.

        The uniform construction hook :func:`build_base_model
        <isaac_rl_posttraining.runners.off_policy_launch.build_base_model>` calls after resolving
        ``base_model.name`` through the registry, so adding a base model needs no
        change to the training script. Implementations read their fields from
        the ``base_model`` config group (and ``dsrl`` for latent geometry) and
        return the steering-capable object DSRL consumes — usually
        ``cls(...)``; GR00T returns its dedicated steering wrapper.

        Args:
            cfg: The composed (OmegaConf) run config.
            device: Torch device the policy runs on.

        Raises:
            NotImplementedError: If the base model cannot be built for a DSRL run
                from config (the default).
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement from_dsrl_config; it cannot be "
            "constructed for a DSRL run through build_base_model."
        )

    @abc.abstractmethod
    def infer(self, observations: dict, noise: np.ndarray | None) -> dict:
        """Run one inference call, optionally steering the initial latent.

        Args:
            observations: Base-model-specific observation dict (camera images,
                proprioception, a language instruction, ...) with a leading
                batch dimension ``B`` on every array entry.
            noise: Latent to inject in place of the base model's own initial
                noise. One of:

                * ``None`` — native mode. The base model samples its own
                  initial noise; the result must match the un-steered base
                  policy (contract note 1).
                * shape ``[B, latent_dim]`` — one latent per batch element,
                  broadcast across all ``H`` action steps.
                * shape ``[B, K, latent_dim]`` with ``K <= H`` — per-step
                  latents for the first ``K`` steps; the remaining ``H - K``
                  steps receive fresh ``N(0, 1)`` noise (contract note 4).

        Returns:
            A dict with key ``"actions"`` holding a ``np.ndarray`` of shape
            ``[B, H, A_raw]``, decoded per-dimension into the environment's
            raw action space (contract note 2).

        Raises:
            ValueError: If ``noise`` has an unsupported shape.
            NotImplementedError: If ``noise`` is not ``None`` and the base model
                does not support steering (:attr:`supports_steering` is
                ``False``; contract note 5).
        """


class MockBaseModelPolicy(BaseModelPolicy):
    """Deterministic stand-in base model for contract tests.

    The mock models its head noise as a ``[B, H, action_dim]`` tensor whose
    first :attr:`latent_dim` entries per step are steerable, and decodes it
    with a position-coded, strictly per-dimension rule::

        actions[b, h, a] = u[b, h, a] + 10 * h + 1000 * a

    Injected latents are clipped to ``±DEFAULT_LATENT_CLIP`` first, so every
    steered entry stays well below the code spacing and tests can recover
    both the position and the injected value from any output entry:
    ``a = round(v / 1000)``, ``h = round((v - 1000 * a) / 10)``, and the
    residual is the (clipped) injected latent. This makes broadcast
    semantics, per-step placement, and the absence of cross-dimension mixing
    directly observable. The coding assumes ``action_horizon <= 100``.

    Entries not covered by the ``noise`` argument are drawn from an
    instance-owned generator seeded at construction: fresh on every call, as
    the contract requires, yet reproducible per instance. With
    ``latent_dim == action_dim`` (the default) and a full
    ``[B, H, latent_dim]`` noise argument, the output is a pure function of
    the inputs.

    The batch size is taken from the first array-valued observation entry,
    falling back to ``noise`` and then to ``1``, so tests may pass an empty
    observation dict.
    """

    def __init__(
        self,
        action_horizon: int = 8,
        action_dim: int = 6,
        latent_dim: int | None = None,
        seed: int = 0,
    ) -> None:
        self._action_horizon = int(action_horizon)
        self._action_dim = int(action_dim)
        self._latent_dim = self._action_dim if latent_dim is None else int(latent_dim)
        if not 0 < self._latent_dim <= self._action_dim:
            raise ValueError("latent_dim must be in (0, action_dim] for the mock base_model")
        self._rng = np.random.default_rng(seed)
        # GR00T's shape surface for pooled backbone features (``policy.model.feat_dim``).
        self.policy = SimpleNamespace(model=SimpleNamespace(feat_dim=MOCK_FEATURE_DIM))
        # Position code, shape [H, A_raw]; broadcasts over the batch.
        self._position_code = (
            10.0 * np.arange(self._action_horizon, dtype=np.float64)[:, None]
            + 1000.0 * np.arange(self._action_dim, dtype=np.float64)[None, :]
        )

    @classmethod
    def from_dsrl_config(cls, cfg: Any, device: Any, dataset_adapter: Any = None) -> "MockBaseModelPolicy":
        """Construct the mock base model from the ``base_model`` config group."""
        from omegaconf import OmegaConf

        return cls(
            action_horizon=int(OmegaConf.select(cfg, "base_model.action_horizon", default=40) or 40),
            action_dim=int(OmegaConf.select(cfg, "base_model.action_dim", default=35) or 35),
        )

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def infer(self, observations: dict, noise: np.ndarray | None = None) -> dict:
        horizon, latent_dim = self._action_horizon, self._latent_dim
        batch_size = self._batch_size(observations, noise)
        # Contract note 4: everything starts as fresh N(0, 1) ...
        u = self._rng.standard_normal((batch_size, horizon, self._action_dim))
        if noise is not None:
            noise_arr = np.asarray(noise, dtype=np.float64)
            broadcast_all_steps = noise_arr.ndim == 2
            if broadcast_all_steps:
                noise_arr = noise_arr[:, None, :]
            if (
                noise_arr.ndim != 3
                or noise_arr.shape[0] != batch_size
                or noise_arr.shape[1] > horizon
                or noise_arr.shape[2] != latent_dim
            ):
                raise ValueError(
                    f"noise must have shape [B, {latent_dim}] or [B, K, {latent_dim}] "
                    f"with B == {batch_size} and K <= {horizon}; got {np.shape(noise)}"
                )
            # Contract note 3: clip injected latents at injection time.
            steered = np.clip(noise_arr, -DEFAULT_LATENT_CLIP, DEFAULT_LATENT_CLIP)
            if broadcast_all_steps:
                steered = np.broadcast_to(steered, (batch_size, horizon, latent_dim))
            # ... and only the steered block is overwritten.
            u[:, : steered.shape[1], :latent_dim] = steered
        # Contract note 2: per-dimension decode (element-wise shift only).
        actions = (u + self._position_code).astype(np.float32)
        return {"actions": actions}

    def infer_reference(self, observations: dict, *, image_transform_mode: str = "eval") -> tuple[Any, np.ndarray]:
        """The reference-action surface of SAC-edit and reference-conditioned RFS.

        The pooled features are zeros of width ``2 * policy.model.feat_dim`` (the
        mock has no backbone); the reference is the mock's own decode.
        """
        import torch

        del image_transform_mode
        features = torch.zeros(1, 2 * self.policy.model.feat_dim)
        return features, self.infer(observations, None)["actions"]

    @staticmethod
    def _batch_size(observations: dict, noise: np.ndarray | None) -> int:
        for value in observations.values():
            if isinstance(value, np.ndarray) and value.ndim > 0:
                return int(value.shape[0])
        if noise is not None:
            return int(np.shape(noise)[0])
        return 1
