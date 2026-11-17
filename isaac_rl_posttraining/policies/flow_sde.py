# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flow-SDE sampling, chunk log-densities and a value head over any flow-matching action head.

A flow-matching action head integrates a learned velocity field from Gaussian
noise to an action chunk in ``K`` Euler steps, so the executed action has no
tractable density. Following RLinf's GR00T recipe, one of the ``K`` steps is
replaced by a stochastic (SDE) step whose transition is a closed-form
Gaussian, the other steps stay deterministic, and the chunk log-probability
is that step's Gaussian log-density summed over the executed action window.
The learner recomputes it from the stored denoising chain under the updated
head, which is all PPO's likelihood ratio needs. A value head reads pooled
features of the frozen part together with the head's own state embedding.

:class:`FlowSDEActorCritic` holds that math once. A base model adapter subclasses
it and supplies its model's pieces: how observations become model inputs and
how those are encoded once per decode (:meth:`~FlowSDEActorCritic.condition`),
the head's velocity at a chain state (:meth:`~FlowSDEActorCritic.velocity`),
the features the value head pools (:meth:`~FlowSDEActorCritic.value_features`),
and how normalized chunks decode. The flow runs on the uniform grid
``t_k = k / K`` from noise at ``t = 0`` to data at ``t = 1``; a base model whose
flow is parameterized the other way maps its time inside ``velocity`` (see the
pi0 adapter).
"""

from __future__ import annotations

import abc
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from isaac_rl_posttraining.policies.feature_cache import pack_features, unpack_features
from isaac_rl_posttraining.policies.trainable import (
    STOCHASTIC_STEP_KEY,
    FlowSDEConfig,
    FlowSDESample,
)

__all__ = [
    "FlowConditioning",
    "FlowSDEActorCritic",
    "disable_dropout",
    "flow_sde_step_moments",
    "gaussian_log_prob",
    "masked_mean",
    "split_infer_observations",
    "value_head_mlp",
]


def flow_sde_step_moments(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    step_index: torch.Tensor,
    *,
    num_steps: int,
    noise_level: float,
    stochastic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard deviation of the next chain state ``x_{k+1}``.

    The flow runs from noise at time 0 to data at time 1 on the uniform grid
    ``t_k = k / K``. The velocity gives the noise and data estimates
    ``x0 = x_t - v t`` and ``x1 = x_t + v (1 - t)``; the deterministic Euler step
    is their interpolation at ``t + dt`` (``x_t + v dt``) with zero noise. The
    stochastic step (RLinf's ``flow_sde``) shrinks the noise-estimate weight by
    ``sigma_k^2 dt / (2 (1 - t))`` with ``sigma_k = noise_level sqrt((1 - t_k) / t_k)``
    (``t_0`` replaced by ``t_1``) and adds Gaussian noise with standard deviation
    ``sqrt(dt) sigma_k``.

    Args:
        x_t: Current chain state ``[B, H, D]``.
        velocity: Predicted velocity at ``x_t`` ``[B, H, D]``.
        step_index: Step index per sample ``[B]`` (integer tensor).
        num_steps: Number of Euler steps ``K``.
        noise_level: SDE noise scale.
        stochastic: Whether this is the noised step.

    Returns:
        ``(mean, std)`` shaped like ``x_t``; ``std`` is zero when deterministic.
    """
    grid = torch.linspace(0.0, 1.0, num_steps + 1, device=x_t.device, dtype=torch.float32)
    index = step_index.to(device=x_t.device, dtype=torch.long)
    t = grid[index][:, None, None]
    dt = (grid[index + 1] - grid[index])[:, None, None]
    x_t32 = x_t.float()
    velocity32 = velocity.float()
    x0_pred = x_t32 - velocity32 * t
    x1_pred = x_t32 + velocity32 * (1.0 - t)
    x1_weight = t + dt
    if stochastic:
        safe_grid = torch.where(grid == 0, grid[1], grid)
        sigmas = noise_level * torch.sqrt((1.0 - grid) / safe_grid)[:-1]
        sigma = sigmas[index][:, None, None]
        x0_weight = 1.0 - x1_weight - sigma.square() * dt / (2.0 * (1.0 - t))
        std = torch.sqrt(dt) * sigma
    else:
        x0_weight = 1.0 - x1_weight
        std = torch.zeros_like(t)
    mean = x0_pred * x0_weight + x1_pred * x1_weight
    return mean.to(x_t.dtype), std.expand_as(mean).to(x_t.dtype)


def gaussian_log_prob(sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Elementwise Gaussian log-density, evaluated in float32 (``std > 0``)."""
    std32 = std.float()
    z = (sample.float() - mean.float()) / std32
    return -torch.log(std32) - 0.5 * math.log(2.0 * math.pi) - 0.5 * z.square()


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``tokens`` ``[B, T, D]`` over the tokens ``mask`` ``[B, T]`` marks valid, as float32 ``[B, D]``."""
    weights = mask.to(torch.float32)[..., None]
    return (tokens.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def critic_token_sequence(
    tokens: torch.Tensor, valid: torch.Tensor, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Observation tokens plus one state token, and a padding mask (True = ignore).

    ``valid`` is ``[B, T]``; nonzero entries are real tokens. The state token is
    always real. Both tensors are detached so a critic cannot train the backbone
    or the action head that produced them.
    """
    tokens = tokens.detach()
    state = state.detach()
    if state.ndim == 2:
        state = state.unsqueeze(1)
    if valid.shape[:2] != tokens.shape[:2]:
        raise ValueError(
            f"token mask {tuple(valid.shape)} does not match tokens {tuple(tokens.shape[:2])}"
        )
    width = int(tokens.shape[-1])
    state_width = int(state.shape[-1])
    if state_width < width:
        # shortcut: zero-pad a narrower state token up to the token width (N1.7 1536 vs 2048, pi0 expert vs PaliGemma); a critic-owned projection if that subspace underfits.
        state = torch.nn.functional.pad(state, (0, width - state_width))
    elif state_width != width:
        raise ValueError(f"state embedding width {state_width} exceeds token width {width}")
    sequence = torch.cat((tokens, state.to(dtype=tokens.dtype)), dim=1)
    valid_bool = valid if valid.dtype == torch.bool else valid != 0
    padding = ~valid_bool
    state_padding = torch.zeros(valid_bool.shape[0], 1, dtype=torch.bool, device=valid_bool.device)
    return sequence, torch.cat((padding, state_padding), dim=1)


def disable_dropout(module: nn.Module) -> int:
    """Swap dropout layers for identities so behavior and learner passes agree; returns how many."""
    replaced = 0
    for name, child in list(module.named_children()):  # NOSONAR - the loop replaces children
        if isinstance(child, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            setattr(module, name, nn.Identity())
            replaced += 1
        else:
            replaced += disable_dropout(child)
    return replaced


def value_head_mlp(input_dim: int, hidden_sizes: Sequence[int]) -> nn.Sequential:
    """RLinf's value MLP: ReLU hidden layers, Kaiming init, small-normal last layer."""
    layers: list[nn.Module] = []
    width = input_dim
    for hidden in hidden_sizes:
        linear = nn.Linear(width, int(hidden))
        nn.init.kaiming_normal_(linear.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(linear.bias)
        layers.extend((linear, nn.ReLU()))
        width = int(hidden)
    last = nn.Linear(width, 1)
    nn.init.normal_(last.weight, mean=0.0, std=0.02)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


@dataclass
class FlowConditioning:
    """What one decode conditions on, computed once per batch.

    Attributes:
        context: The base model's own encodings (tokens, masks, state features,
            key-value caches, ...) that :meth:`FlowSDEActorCritic.velocity`
            and :meth:`FlowSDEActorCritic.value_features` read.
        cached: Raw outputs of the frozen part, one entry per name in
            :attr:`FlowSDEActorCritic.cached_feature_names`; shipped with the
            decision when features are cached.
        batch_size: Number of samples in the batch.
    """

    context: Any
    cached: dict[str, torch.Tensor]
    batch_size: int


class FlowSDEActorCritic(nn.Module, abc.ABC):
    """A flow-matching policy with flow-SDE sampling and a value head, for PPO.

    The offline methods use it too: IQL calls :meth:`feature_tokens` and
    :meth:`actor_bc_loss`, SFT only :meth:`actor_bc_loss`, and their
    evaluation decodes with :meth:`infer`.

    Implements :class:`~isaac_rl_posttraining.policies.trainable.TrainablePolicy`
    for any base model that supplies the abstract pieces below. ``act`` runs the
    Euler chain with one stochastic step and returns the chain, its
    log-probability and the value; ``forward(observations, actions)`` is the
    PPO learner contract and re-evaluates the stochastic step of stored chains
    under the current weights, returning the log-probability, a zero entropy
    surrogate (the squashed flow has no closed-form entropy and RLinf trains
    without an entropy bonus) and the value, all ``[B]`` float32.

    Flow-SDE and the critic head it is trained against are the recipe of
    Chen et al., "πRL: Online RL Fine-tuning for Flow-based
    Vision-Language-Action Models", arXiv:2510.25889, whose RLinf
    implementation :func:`value_head_mlp` follows down to the initialization.

    Args:
        video_keys: Camera keys of the observation, in the model's order.
        state_keys: State keys of the observation, in the model's order.
        action_window: Executed action steps per chunk.
        valid_action_dim: Real action width inside the padded flow width.
        chunk_shape: ``(H, D)`` of the head's flow: full horizon and padded width.
        value_input_dim: Width of :meth:`value_features`.
        config: Flow-SDE settings.
        device: Device the head's parameters live on (the value head joins it).
        dtype: Dtype of the head's parameters (the value head joins it).
    """

    #: Dotted paths (relative to this module) of the parts that never train.
    frozen_submodules: tuple[str, ...] = ()
    #: Outputs of the frozen part a decision can ship instead of the inputs they came from.
    cached_feature_names: tuple[str, ...] = ()
    #: The raw model inputs have no per-sample layout to ship, so decisions must travel as cached features.
    requires_cached_features: bool = False

    def __init__(
        self,
        *,
        video_keys: Sequence[str],
        state_keys: Sequence[str],
        action_window: int,
        valid_action_dim: int,
        chunk_shape: tuple[int, int],
        value_input_dim: int,
        config: FlowSDEConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        horizon, width = (int(size) for size in chunk_shape)
        if not 1 <= int(action_window) <= horizon:
            raise ValueError(f"action_window must lie in [1, {horizon}], got {action_window}")
        if not 1 <= int(valid_action_dim) <= width:
            raise ValueError(f"valid_action_dim must lie in [1, {width}], got {valid_action_dim}")
        self.video_keys = list(video_keys)
        self.state_keys = list(state_keys)
        self.action_window = int(action_window)
        self.valid_action_dim = int(valid_action_dim)
        self.chunk_shape = (horizon, width)
        self.config = config
        self.value_head = value_head_mlp(int(value_input_dim), config.value_hidden_sizes)
        self.value_head.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # The base model's pieces
    # ------------------------------------------------------------------
    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        """Device of the head's parameters."""

    @property
    @abc.abstractmethod
    def dtype(self) -> torch.dtype:
        """Dtype of the head's parameters (the chains are kept in it)."""

    @abc.abstractmethod
    def build_inputs(
        self,
        frames: Mapping[str, np.ndarray],
        states: Mapping[str, np.ndarray],
        instruction: str,
    ) -> dict[str, torch.Tensor]:
        """Model inputs for a batch of ``N`` observations (uint8 frames ``[N, H, W, 3]`` per camera key, float states ``[N, d]`` per state key)."""

    @abc.abstractmethod
    def decode(
        self,
        normalized: torch.Tensor,
        *,
        inputs: Mapping[str, torch.Tensor],
        states: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """Decode ``[B, H, D]`` normalized chunks into ``[B, window, A_raw]`` raw actions.

        ``inputs`` are the model inputs the chunk was decided from and
        ``states`` the raw state parts behind them, for base models whose action
        groups are relative to the current state.
        """

    @abc.abstractmethod
    def condition(
        self, inputs: Mapping[str, torch.Tensor], cached_features: Mapping[str, Any] | None = None
    ) -> FlowConditioning:
        """Encode one batch: run the frozen part (or take its cached outputs) and the head's input encoders.

        With ``cached_features`` (stacked records from
        :func:`~isaac_rl_posttraining.policies.feature_cache.pack_features`)
        the frozen part is not run and ``inputs`` hold only what
        :meth:`learner_inputs` kept.
        """

    @abc.abstractmethod
    def velocity(self, conditioning: FlowConditioning, x_t: torch.Tensor, step_index: torch.Tensor) -> torch.Tensor:
        """The head's velocity ``[B, H, D]`` at chain state ``x_t`` and grid step ``step_index`` (``[B]``, noise at step 0)."""

    @abc.abstractmethod
    def value_features(self, conditioning: FlowConditioning) -> torch.Tensor:
        """Features ``[B, value_input_dim]`` the value head reads, in the head's dtype."""

    @classmethod
    @abc.abstractmethod
    def spec_options(cls, base_model: Any) -> dict[str, Any]:
        """The base model's own ``load`` keyword arguments, extracted and validated from the ``base_model`` config section."""

    @classmethod
    @abc.abstractmethod
    def load(
        cls,
        checkpoint: str,
        *,
        device: str,
        flow_sde: FlowSDEConfig,
        value_head_seed: int = 0,
        **options: Any,
    ) -> FlowSDEActorCritic:
        """Load ``checkpoint`` as a trainable actor-critic on ``device``."""

    # ------------------------------------------------------------------
    # Shared behavior
    # ------------------------------------------------------------------
    @property
    def denoising_steps(self) -> int:
        return int(self.config.denoising_steps)

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def learner_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Inputs the training forward still needs once the frozen part's outputs are cached (all of them, by default)."""
        return dict(inputs)

    def pack_inputs(self, inputs: Mapping[str, torch.Tensor], batch_size: int) -> list[dict[str, np.ndarray]]:
        """Split batch-leading model inputs into per-sample NumPy records (bfloat16 travels as raw bits)."""
        return pack_features(inputs, batch_size)

    def unpack_inputs(self, batch: Mapping[str, Any], device: torch.device | str) -> dict[str, torch.Tensor]:
        """Invert :meth:`pack_inputs` for a stacked ``[B, ...]`` batch."""
        return unpack_features(batch, device)

    def value(self, conditioning: FlowConditioning) -> torch.Tensor:
        """Value estimate ``[B]`` (float32)."""
        return self.value_head(self.value_features(conditioning))[:, 0].float()

    def _chunk_log_prob(self, sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        density = gaussian_log_prob(sample, mean, std)
        return density[:, : self.action_window, : self.valid_action_dim].sum(dim=(1, 2))

    @torch.no_grad()
    def act(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        stochastic_step: int | None,
        noise: torch.Tensor | None = None,
    ) -> FlowSDESample:
        """Decode one batch, noising ``stochastic_step`` (``None`` = deterministic).

        Args:
            inputs: Model inputs (see :meth:`build_inputs`).
            stochastic_step: Index in ``range(K)`` of the SDE step, or ``None``
                for the plain Euler decode used at evaluation time.
            noise: Optional initial noise ``[B, H, D]`` (tests); drawn otherwise.
        """
        steps = self.denoising_steps
        if stochastic_step is not None and not 0 <= int(stochastic_step) < steps:
            raise ValueError(f"stochastic_step must lie in range({steps}), got {stochastic_step}")
        conditioning = self.condition(inputs)
        batch = conditioning.batch_size
        horizon, width = self.chunk_shape
        if noise is None:
            x_t = torch.randn(batch, horizon, width, dtype=self.dtype, device=self.device)
        else:
            x_t = noise.to(device=self.device, dtype=self.dtype)
        chain = [x_t]
        log_prob = torch.zeros(batch, dtype=torch.float32, device=self.device)
        for step in range(steps):
            index = torch.full((batch,), step, dtype=torch.long, device=self.device)
            velocity = self.velocity(conditioning, x_t, index)
            stochastic = stochastic_step is not None and step == int(stochastic_step)
            mean, std = flow_sde_step_moments(
                x_t, velocity, index, num_steps=steps, noise_level=self.config.noise_level, stochastic=stochastic
            )
            if stochastic:
                x_t = mean + torch.randn_like(mean) * std
                log_prob = self._chunk_log_prob(x_t, mean, std)
            else:
                x_t = mean
            chain.append(x_t)
        value = self.value(conditioning)
        return FlowSDESample(
            actions=x_t,
            chains=torch.stack(chain, dim=1),
            log_prob=log_prob,
            value=value,
            stochastic_step=None if stochastic_step is None else int(stochastic_step),
            cached_features={name: tensor.detach() for name, tensor in conditioning.cached.items()},
        )

    def infer(self, observations: Mapping[str, Any]) -> np.ndarray:
        """Deterministic Euler decode of one observation batch to raw actions ``[B, H, A]``."""
        inputs, states = self._infer_model_inputs(observations)
        sample = self.act(inputs, stochastic_step=None)
        return np.asarray(self.decode(sample.actions, inputs=inputs, states=states), dtype=np.float32)

    def feature_tokens(self, inputs: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``(tokens [B, T, D], key_padding_mask [B, T] or None)``.

        True in the mask marks padding. A context that is already ``[B, T, D]``
        is returned unchanged. Otherwise the pooled value features are one token.
        Computed under ``no_grad``: critic gradients stay out of the backbone and
        the action head. The PPO value head still reads :meth:`value_features`.
        """
        with torch.no_grad():
            conditioning = self.condition(inputs)
            sequenced = self._critic_token_sequence(conditioning)
            if sequenced is not None:
                return sequenced
            features = self.value_features(conditioning)
            if features.ndim == 2:
                return features.unsqueeze(1), None
            if features.ndim == 3:
                return features, None
            raise ValueError(
                f"{type(self).__name__}.value_features must return [B, D] or [B, T, D], got {tuple(features.shape)}"
            )

    def _critic_token_sequence(
        self, conditioning: FlowConditioning
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        """Token sequence for the IQL critic, or ``None`` to fall back to the pooled features."""
        context = conditioning.context
        if torch.is_tensor(context) and context.ndim == 3:
            return context, None
        return None

    def actor_bc_loss(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        """Per-sample flow-matching (or family-supervised) action loss, shape ``[B]``."""
        if self.scores_demonstrations:
            return self.supervised_loss(inputs)
        if "action" not in inputs:
            raise KeyError("actor_bc_loss requires inputs['action']")
        action = inputs["action"]
        cond_inputs = {key: value for key, value in inputs.items() if key != "action"}
        conditioning = self.condition(cond_inputs)
        batch = conditioning.batch_size
        horizon, width = self.chunk_shape
        action_t = action.to(device=self.device, dtype=self.dtype)
        if action_t.ndim != 3:
            raise ValueError(f"action must have shape [B, H, A], got {tuple(action_t.shape)}")
        padded = torch.zeros(batch, horizon, width, device=self.device, dtype=self.dtype)
        steps = min(horizon, action_t.shape[1])
        dims = min(width, action_t.shape[2])
        padded[:, :steps, :dims] = action_t[:, :steps, :dims]
        noise = torch.randn_like(padded)
        # On the decode grid t_k = k / K, so the stated time is the time x_t is built at.
        time, time_arg = self._flow_match_time(batch)
        x_t = (1.0 - time) * noise + time * padded
        velocity = self.velocity(conditioning, x_t, time_arg)
        target = padded - noise
        error = (velocity - target)[:, : self.action_window, : self.valid_action_dim]
        return error.float().pow(2).mean(dim=(1, 2))

    def _flow_match_time(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared flow time ``[B, 1, 1]`` and the tensor ``velocity`` is called with.

        The default is one uniform grid step ``k`` per sample, ``t = k / K``.
        """
        step = torch.randint(self.denoising_steps, (batch,), device=self.device)
        time = (step.to(dtype=self.dtype) / self.denoising_steps).reshape(batch, 1, 1)
        return time, step

    def _infer_model_inputs(
        self, observations: Mapping[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
        """Turn an ``infer`` observation dict into model inputs and raw state parts.

        Accepts the model's own keys, G1's ``ego_view_image``, or the harness
        keys ``observation_keys`` maps (LIBERO's ``observation/image``).
        """
        renamed = _rename_observation_keys(observations, getattr(self, "observation_keys", None))
        parts = split_infer_observations(
            renamed,
            video_keys=self.video_keys,
            state_keys=self.state_keys,
            state_slices=getattr(self, "state_slices", None),
        )
        if parts is not None:
            frames, states = parts
            return self.build_inputs(frames, states, _infer_instruction(observations)), states
        inputs = {
            key: value if torch.is_tensor(value) else torch.as_tensor(value)
            for key, value in observations.items()
            if key not in ("instruction", "language", "prompt", "task") and not isinstance(value, str)
        }
        batch = _batch_leading_size(inputs)
        states = {
            key: np.asarray(observations[key]) if key in observations else np.zeros((batch, 1), dtype=np.float32)
            for key in self.state_keys
        }
        return inputs, states

    #: Whether this family can score demonstration actions. The
    #: ``supervised_batch`` keyword lives on this base class, so a signature
    #: alone would advertise the capability for every subclass; a family opts
    #: in by setting this True alongside :meth:`supervised_loss`.
    scores_demonstrations: bool = False

    def supervised_loss(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """Per-sample supervised loss over demonstrations, shaped ``[B]``.

        Implemented by a policy family that can score demonstration actions in
        its own parameterization. The result must already account for padding:
        an embodiment-agnostic head emits a fixed grid and masks the part an
        embodiment uses, so averaging the whole grid divides by the padding and
        silently scales the co-training weight.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot score demonstrations; it has no supervised loss"
        )

    def forward(  # NOSONAR - the optional 4th output is the documented co-training contract
        self,
        observations: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
        cached_features: Mapping[str, Any] | None = None,
        supervised_batch: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Re-evaluate stored chains: ``(log_prob, entropy, value)``, each ``[B]`` float32.

        ``cached_features`` (stacked records from :func:`pack_features`) replace
        the frozen part's pass; ``observations`` then hold only the inputs
        :meth:`learner_inputs` kept.

        With ``supervised_batch`` the co-training anchor is scored in the same
        call and returned as a fourth output, so its gradients reach the shared
        modules through one forward rather than two. It must travel through
        this call: FSDP all-gathers parameters around the module's forward, and
        a separate method would see the rank's shard instead.
        """
        if STOCHASTIC_STEP_KEY not in observations:
            raise KeyError(f"PPO observations must carry {STOCHASTIC_STEP_KEY!r}")
        step_index = observations[STOCHASTIC_STEP_KEY].reshape(-1).long()
        inputs = self.unpack_inputs(
            {key: value for key, value in observations.items() if key != STOCHASTIC_STEP_KEY},
            self.device,
        )
        conditioning = self.condition(inputs, cached_features)
        steps = self.denoising_steps
        chains = actions.to(device=self.device, dtype=self.dtype)
        if chains.ndim != 4 or chains.shape[1] != steps + 1:
            raise ValueError(f"chains must have shape [B, {steps + 1}, H, D], got {tuple(chains.shape)}")
        rows = torch.arange(chains.shape[0], device=chains.device)
        x_prev = chains[rows, step_index]
        x_next = chains[rows, step_index + 1]
        velocity = self.velocity(conditioning, x_prev, step_index)
        mean, std = flow_sde_step_moments(
            x_prev, velocity, step_index, num_steps=steps, noise_level=self.config.noise_level, stochastic=True
        )
        log_prob = self._chunk_log_prob(x_next, mean, std)
        value = self.value(conditioning)
        entropy = torch.zeros_like(log_prob)
        if supervised_batch is not None:
            return log_prob, entropy, value, self.supervised_loss(supervised_batch)
        return log_prob, entropy, value


def _rename_observation_keys(
    observations: Mapping[str, Any],
    observation_keys: Mapping[str, str] | None,
) -> Mapping[str, Any]:
    """Copy harness keys (``observation/image``) onto the mapped model keys."""
    if not observation_keys:
        return observations
    renamed = dict(observations)
    for model_key, external_key in observation_keys.items():
        if model_key not in renamed and external_key in renamed:
            renamed[model_key] = renamed[external_key]
    return renamed


def split_infer_observations(
    observations: Mapping[str, Any],
    *,
    video_keys: Sequence[str],
    state_keys: Sequence[str],
    state_slices: Sequence[tuple[str, int, int]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    """Split an ``infer`` dict into camera frames and state groups, or ``None``.

    ``None`` means the dict already looks like processed model inputs (no
    cameras), so :meth:`FlowSDEActorCritic._infer_model_inputs` should pass
    it through. Harness observations (``ego_view_image`` plus a flat
    ``state``) are rewritten onto ``video_keys`` / ``state_keys``.
    """
    video_keys = list(video_keys)
    state_keys = list(state_keys)
    if video_keys and all(key in observations for key in video_keys):
        frames = {key: _batched_image(observations[key]) for key in video_keys}
        if state_keys and all(key in observations for key in state_keys):
            states = {key: _batched_vector(observations[key]) for key in state_keys}
        elif "state" in observations:
            states = _split_flat_state(observations["state"], state_keys, state_slices)
        else:
            raise KeyError(
                f"infer observations have cameras {video_keys} but neither "
                f"grouped state keys {state_keys} nor a flat 'state'"
            )
        return frames, states
    shared = observations.get("ego_view_image", observations.get("ego_view"))
    if shared is None or not video_keys:
        return None
    frame = _batched_image(shared)
    frames = {key: frame for key in video_keys}
    if "state" not in observations:
        raise KeyError("infer observations need 'state' when cameras come from ego_view_image")
    states = _split_flat_state(observations["state"], state_keys, state_slices)
    return frames, states


def _batched_image(value: Any) -> np.ndarray:
    """A uint8 image batch ``[N, H, W, 3]`` from a single frame or a batch."""
    frame = np.asarray(value)
    if frame.ndim == 3:
        frame = frame[None]
    if frame.ndim != 4:
        raise ValueError(f"image must be [H, W, 3] or [N, H, W, 3], got {frame.shape}")
    return frame


def _batched_vector(value: Any) -> np.ndarray:
    """A float batch ``[N, D]`` from a vector or a batch of vectors."""
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim == 1:
        vector = vector[None]
    if vector.ndim != 2:
        raise ValueError(f"state must be [D] or [N, D], got {vector.shape}")
    return vector


def _split_flat_state(
    value: Any,
    state_keys: Sequence[str],
    state_slices: Sequence[tuple[str, int, int]] | None,
) -> dict[str, np.ndarray]:
    """Cut a flat proprio vector into the model's named state groups."""
    flat = _batched_vector(value)
    if state_slices:
        return {key: flat[:, start:end] for key, start, end in state_slices}
    if len(state_keys) == 1:
        return {state_keys[0]: flat}
    raise KeyError(
        f"a flat 'state' of width {flat.shape[1]} cannot be split into {list(state_keys)} "
        "without state_slices"
    )


def _infer_instruction(observations: Mapping[str, Any]) -> str:
    """A language instruction carried as a string, list, or array; empty when absent."""
    for key in ("instruction", "language", "prompt", "task"):
        if key not in observations:
            continue
        value = observations[key]
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)) and value:
            return str(value[0])
        if isinstance(value, np.ndarray) and value.size:
            return str(value.reshape(-1)[0])
    return ""


def _batch_leading_size(inputs: Mapping[str, Any]) -> int:
    """Batch size of an observation dict, ``1`` when nothing is batched."""
    for value in inputs.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
        if isinstance(value, np.ndarray) and value.ndim > 0:
            return int(value.shape[0])
    return 1
