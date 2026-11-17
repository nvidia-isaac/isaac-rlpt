# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PPO update logic hosted by the provider-neutral learner-group contract."""

from __future__ import annotations

import contextlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from isaac_rl_posttraining.algorithms.ppo.config import PPOConfig
from isaac_rl_posttraining.policies.feature_cache import CACHED_FEATURES_KEY
from isaac_rl_posttraining.runtimes.learner import (
    LearnerBatch,
    LearnerContext,
    LearnerModuleManager,
)
from isaac_rl_posttraining.training.demonstration_buffer import DEMONSTRATIONS_KEY

__all__ = [
    "BuildModel",
    "PPOActorCritic",
    "PPOLearner",
]


@runtime_checkable
class PPOActorCritic(Protocol):
    """Minimum model surface needed by the PPO optimizer.

    ``__call__`` returns the log-probability, entropy, and value of every sample,
    each shaped ``[batch]``. A model that implements
    :class:`~isaac_rl_posttraining.policies.feature_cache.FeatureCaching` also
    accepts ``cached_features=`` (the activations its rollout side recorded) so
    frozen sub-modules are not recomputed during training.
    """

    def parameters(self, recurse: bool = True): ...

    def __call__(
        self,
        observations: Any,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


BuildModel = Callable[[], nn.Module]


def _tensor_tree(
    value: np.ndarray | Mapping[str, np.ndarray],
    device: torch.device,
) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(value, Mapping):
        return {
            name: _tensor_from_array(array, device) for name, array in value.items()
        }
    return _tensor_from_array(value, device)


def _tensor_from_array(value: np.ndarray, device: torch.device) -> torch.Tensor:
    array = np.asarray(value)
    if not array.flags.writeable:
        array = array.copy()
    return torch.as_tensor(array, device=device)


def _index_tree(value: Any, indices: torch.Tensor) -> Any:
    if isinstance(value, Mapping):
        return {name: tensor[indices] for name, tensor in value.items()}
    return value[indices]


def _leading_size(value: Any) -> int:
    if isinstance(value, Mapping):
        sizes = {_leading_size(item) for item in value.values()}
        if len(sizes) != 1:
            raise ValueError("PPO observation fields must share a leading dimension")
        return sizes.pop()
    if value.ndim < 1:
        raise ValueError("PPO batch tensors must carry a leading dimension")
    return int(value.shape[0])


def _require_finite(value: Any, *, name: str) -> None:
    if isinstance(value, Mapping):
        for field_name, tensor in value.items():
            _require_finite(tensor, name=f"{name}.{field_name}")
        return
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"PPO batch field {name!r} must be finite")


def _distributed_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _global_mean(value: torch.Tensor) -> torch.Tensor:
    totals = torch.stack((value.detach().sum(), value.new_tensor(value.numel())))
    totals = _distributed_sum(totals)
    return totals[0] / totals[1].clamp_min(1.0)


def _normalize_advantages(value: torch.Tensor) -> torch.Tensor:
    totals = torch.stack(
        (
            value.detach().sum(),
            value.detach().square().sum(),
            value.new_tensor(value.numel()),
        )
    )
    totals = _distributed_sum(totals)
    mean = totals[0] / totals[2].clamp_min(1.0)
    variance = totals[1] / totals[2].clamp_min(1.0) - mean.square()
    return (value - mean) / torch.sqrt(variance.clamp_min(0.0) + 1e-8)


def _huber(error: torch.Tensor, delta: float) -> torch.Tensor:
    absolute = error.abs()
    return torch.where(
        absolute < delta, 0.5 * error.square(), delta * (absolute - 0.5 * delta)
    )


def _cast_learner_fields(tensors: dict[str, Any], samples: int) -> None:
    """Flatten the per-sample scalars and cast actions and float observations to float32."""
    for name in ("old_log_probs", "advantages", "returns", "old_values"):
        tensor = tensors[name]
        assert isinstance(tensor, torch.Tensor)
        if tensor.numel() != samples:
            raise ValueError(
                f"PPO batch field {name!r} must have one scalar per sample"
            )
        tensors[name] = tensor.float().reshape(samples)
    actions = tensors["actions"]
    assert isinstance(actions, torch.Tensor)
    tensors["actions"] = actions.float()
    observations = tensors["observations"]
    if isinstance(observations, Mapping):
        tensors["observations"] = {
            name: value.float() if value.is_floating_point() else value
            for name, value in observations.items()
        }
    elif observations.is_floating_point():
        tensors["observations"] = observations.float()


class PPOLearner:
    """Algorithm-owned clipped PPO worker suitable for local or FSDP modules."""

    _REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {
            "observations",
            "actions",
            "old_log_probs",
            "advantages",
            "returns",
            "old_values",
        }
    )
    _OPTIONAL: ClassVar[frozenset[str]] = frozenset(
        {CACHED_FEATURES_KEY, DEMONSTRATIONS_KEY}
    )

    def __init__(
        self,
        context: LearnerContext,
        modules: LearnerModuleManager,
        build_model: BuildModel,
        config: PPOConfig,
        frozen_submodules: Sequence[str] = (),
        cotrainer: Any = None,
    ) -> None:
        self.context = context
        self.config = config
        self.device = torch.device(context.device)
        prepare_kwargs: dict[str, Any] = {"name": "actor_critic", "publish": True}
        if frozen_submodules:
            prepare_kwargs["frozen_submodules"] = tuple(frozen_submodules)
        model = modules.prepare(build_model(), **prepare_kwargs)
        if not isinstance(model, nn.Module):
            raise TypeError("PPO build_model must return torch.nn.Module")
        self.model = model
        self.optimizer = self._build_optimizer()
        modules.register_optimizer(
            "actor_critic",
            self.optimizer,
            module=self.model,
        )
        # Demonstration co-training is optional; when enabled its supervised
        # term rides the same backward as the RL loss (see _slice_loss). Verify
        # the model can score demonstrations now, rather than discovering
        # mid-run that the anchor was silently absent.
        self.cotrainer = cotrainer
        if cotrainer is not None:
            cotrainer.check_supported(self.model)
        self.updates = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        named = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        if not named:
            raise ValueError("PPO model has no trainable parameters")
        if self.config.optimizer == "adam":
            optimizer_class = torch.optim.Adam
        else:
            optimizer_class = torch.optim.AdamW
        return optimizer_class(
            self._parameter_groups(named),
            lr=self.config.learning_rate,
            betas=tuple(self.config.adam_betas),
            eps=self.config.optimizer_eps,
            weight_decay=self.config.weight_decay,
        )

    def _parameter_groups(
        self, named: list[tuple[str, torch.nn.Parameter]]
    ) -> list[dict[str, Any]]:
        """One group, or policy/value groups when the value head has its own rate."""
        if self.config.value_learning_rate is None:
            groups: list[dict[str, Any]] = [
                {"params": [parameter for _, parameter in named]}
            ]
        else:
            pattern = self.config.value_parameter_pattern
            value = [parameter for name, parameter in named if pattern in name]
            policy = [parameter for name, parameter in named if pattern not in name]
            if not value or not policy:
                raise ValueError(
                    "value_learning_rate needs both value parameters (names containing "
                    f"{pattern!r}) and policy parameters; found {len(value)} and {len(policy)}"
                )
            groups = [
                {"params": policy, "lr": self.config.learning_rate},
                {"params": value, "lr": self.config.value_learning_rate},
            ]
        return groups

    def _validate_batch(self, batch: LearnerBatch) -> dict[str, Any]:
        missing = sorted(self._REQUIRED.difference(batch.tensors))
        unexpected = sorted(
            set(batch.tensors).difference(self._REQUIRED | self._OPTIONAL)
        )
        if missing or unexpected:
            raise ValueError(
                f"PPO batch fields mismatch; missing={missing}, unexpected={unexpected}"
            )
        # The demonstrations ride along rather than being part of the RL
        # batch: they are drawn independently, so they neither have one row per
        # sample nor belong on the device before the rank encodes them.
        tensors = {
            name: _tensor_tree(value, self.device)
            for name, value in batch.tensors.items()
            if name != DEMONSTRATIONS_KEY
        }
        sizes = {name: _leading_size(value) for name, value in tensors.items()}
        if set(sizes.values()) != {batch.samples}:
            raise ValueError(
                f"PPO batch leading dimensions {sizes} do not match samples={batch.samples}"
            )
        for name, value in tensors.items():
            if name != CACHED_FEATURES_KEY:
                _require_finite(value, name=name)
        if batch.samples % self.config.minibatch_size:
            raise ValueError(
                "rank-local PPO samples must be divisible by minibatch_size: "
                f"{batch.samples} vs {self.config.minibatch_size}"
            )
        self._require_equal_rank_samples(batch.samples)
        _cast_learner_fields(tensors, batch.samples)
        if DEMONSTRATIONS_KEY in batch.tensors:
            tensors[DEMONSTRATIONS_KEY] = batch.tensors[DEMONSTRATIONS_KEY]
        return tensors

    def _require_equal_rank_samples(self, samples: int) -> None:
        sample_count = torch.tensor([samples], device=self.device, dtype=torch.int64)
        if dist.is_available() and dist.is_initialized():
            gathered = [
                torch.zeros_like(sample_count) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered, sample_count)
            counts = tuple(int(value.item()) for value in gathered)
            if len(set(counts)) != 1:
                raise ValueError(
                    f"PPO learner ranks require equal sample counts: {counts}"
                )

    def _no_sync_unless(self, last_slice: bool):
        """Skip FSDP gradient reduction on all but the last accumulation slice."""
        no_sync = getattr(self.model, "no_sync", None)
        if last_slice or not callable(no_sync):
            return contextlib.nullcontext()
        return no_sync()

    def _slice_loss(
        self,
        tensors: dict[str, Any],
        advantages: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        observations = _index_tree(tensors["observations"], indices)
        actions = tensors["actions"][indices]
        old_log_prob = tensors["old_log_probs"][indices]
        returns = tensors["returns"][indices]
        old_values = tensors["old_values"][indices]
        slice_advantages = advantages[indices]

        # A supervised co-training batch travels INTO the model call, never as a
        # separate method on the module. Under FSDP the parameter all-gather is
        # installed on the module's forward, so a direct method call would run
        # against this rank's flat shard -- silently wrong, or a size mismatch.
        # It also lets a model fuse both batches into one backbone pass.
        extra: dict[str, Any] = {}
        if CACHED_FEATURES_KEY in tensors:
            extra["cached_features"] = _index_tree(
                tensors[CACHED_FEATURES_KEY], indices
            )
        cotraining = self.cotrainer is not None and self.cotrainer.enabled
        if cotraining:
            extra["supervised_batch"] = self.cotrainer.next_batch(
                tensors[DEMONSTRATIONS_KEY], self._cotrain_batch_index, self.device
            )
            self._cotrain_batch_index += 1

        outputs = self.model(observations, actions, **extra)
        if cotraining:
            new_log_prob, entropy, values, per_sample_bc = outputs
        else:
            new_log_prob, entropy, values = outputs
            per_sample_bc = None
        new_log_prob = new_log_prob.float().reshape(-1)
        entropy = entropy.float().reshape(-1)
        values = values.float().reshape(-1)
        log_ratio = (new_log_prob - old_log_prob).clamp(-20.0, 20.0)
        ratio = log_ratio.exp()
        unclipped = ratio * slice_advantages
        clipped = (
            ratio.clamp(1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
            * slice_advantages
        )
        policy_loss = -torch.minimum(unclipped, clipped).mean()

        if self.config.value_loss == "huber":
            value_error = _huber(returns - values, self.config.huber_delta)
        else:
            value_error = 0.5 * (values - returns).square()
        if self.config.value_clip is None:
            value_loss = value_error.mean()
        else:
            clipped_values = old_values + (values - old_values).clamp(
                -self.config.value_clip,
                self.config.value_clip,
            )
            if self.config.value_loss == "huber":
                clipped_error = _huber(
                    returns - clipped_values, self.config.huber_delta
                )
            else:
                clipped_error = 0.5 * (clipped_values - returns).square()
            value_loss = torch.maximum(value_error, clipped_error).mean()
        entropy_mean = entropy.mean()
        loss = (
            policy_loss
            + self.config.value_loss_coef * value_loss
            - self.config.entropy_coef * entropy_mean
        )
        # The supervised anchor joins THIS loss rather than taking its own
        # backward: each slice runs inside one gradient-sync window, so a second
        # backward would both double the collective traffic and let the two terms
        # see different parameter versions.
        bc_loss = None
        if per_sample_bc is not None:
            bc_loss = self.cotrainer.reduce(per_sample_bc)
            loss = loss + self.cotrainer.beta * bc_loss
        with torch.no_grad():
            stats = {
                "policy_loss": policy_loss.detach(),
                "value_loss": value_loss.detach(),
                "entropy": entropy_mean.detach(),
                "approx_kl": ((ratio - 1.0) - log_ratio).mean(),
                "clip_fraction": (
                    ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean()
                ),
                "value_mean": values.mean(),
                "explained_variance": 1.0
                - (returns - values).var(unbiased=False)
                / returns.var(unbiased=False).clamp_min(1e-8),
            }
            if bc_loss is not None:
                stats["bc_loss"] = bc_loss.detach()
        return loss, stats

    def _check_finite_loss(self, loss: torch.Tensor) -> None:
        finite = torch.tensor(
            float(bool(torch.isfinite(loss).item())), device=self.device
        )
        finite = _distributed_sum(finite)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if int(finite.item()) != world_size:
            raise FloatingPointError("PPO produced a non-finite loss")

    def _clip_gradients(self) -> torch.Tensor:
        if self.config.max_grad_norm is None:
            return torch.zeros((), device=self.device)
        distributed_clip = getattr(self.model, "clip_grad_norm_", None)
        if callable(distributed_clip):
            return distributed_clip(self.config.max_grad_norm)
        return torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )

    def train_step(self, batch: LearnerBatch) -> dict[str, float]:
        tensors = self._validate_batch(batch)
        advantages = tensors["advantages"]
        assert isinstance(advantages, torch.Tensor)
        if self.config.normalize_advantages:
            advantages = _normalize_advantages(advantages)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed + self.updates)
        totals: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "value_mean": 0.0,
            "explained_variance": 0.0,
            "grad_norm": 0.0,
        }
        micro = self.config.micro_batch_size or self.config.minibatch_size
        self._cotrain_batch_index = 0
        minibatches = 0
        epochs_completed = 0
        early_stopped = False
        early_stop_kl = 0.0
        # The first slice is evaluated before any update: its ratio measures how
        # exactly the learner reproduces the behavior log-probabilities.
        first_slice_stats: dict[str, float] = {}
        started = time.perf_counter()
        for _epoch in range(self.config.epochs):
            permutation = torch.randperm(batch.samples, generator=generator)
            epoch_kl_sum = torch.zeros((), device=self.device)
            epoch_samples = torch.zeros((), device=self.device)
            for start in range(0, batch.samples, self.config.minibatch_size):
                minibatch_indices = permutation[
                    start : start + self.config.minibatch_size
                ].to(self.device)
                minibatch_stats = self._accumulate_minibatch(
                    tensors, advantages, minibatch_indices, micro, first_slice_stats
                )
                approx_kl = minibatch_stats["approx_kl"]
                stop_kl = self._kl_above_target(approx_kl)
                if stop_kl is not None:
                    early_stopped = True
                    early_stop_kl = stop_kl
                    break
                self._optimizer_step(minibatch_stats, totals)
                minibatches += 1
                epoch_kl_sum += approx_kl.detach() * len(minibatch_indices)
                epoch_samples += len(minibatch_indices)
            if early_stopped:
                break
            epochs_completed += 1
            if self._kl_above_target(epoch_kl_sum / epoch_samples) is not None:
                break

        self.updates += 1
        elapsed = time.perf_counter() - started
        divisor = max(minibatches, 1)
        metrics = {name: value / divisor for name, value in totals.items()}
        metrics.update(first_slice_stats)
        metrics.update(
            {
                "epochs_completed": float(epochs_completed),
                "minibatches": float(minibatches),
                "early_stopped": float(early_stopped),
                "early_stop_kl": early_stop_kl,
                "optimizer_seconds": elapsed,
                "learner_rank_samples_per_second": batch.samples / elapsed,
            }
        )
        metrics.update(self._peak_memory_metrics())
        if torch.cuda.is_available():
            # Return the update's activations to the GPU: simulators that render on it would starve.
            torch.cuda.empty_cache()
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("PPO produced a non-finite metric")
        return metrics

    def _accumulate_minibatch(
        self,
        tensors: dict[str, Any],
        advantages: torch.Tensor,
        minibatch_indices: torch.Tensor,
        micro: int,
        first_slice_stats: dict[str, float],
    ) -> dict[str, torch.Tensor]:
        """Backpropagate one minibatch in micro slices; return its sample-weighted stats.

        Fills ``first_slice_stats`` from the very first slice of the update.
        """
        slices = minibatch_indices.split(micro)
        self.optimizer.zero_grad(set_to_none=True)
        minibatch_stats: dict[str, torch.Tensor] = {}
        for slice_index, indices in enumerate(slices):
            with self._no_sync_unless(slice_index == len(slices) - 1):
                loss, stats = self._slice_loss(tensors, advantages, indices)
                self._check_finite_loss(loss)
                (loss / len(slices)).backward()
            if not first_slice_stats:
                first_slice_stats.update(
                    {
                        "initial_approx_kl": float(stats["approx_kl"]),
                        "initial_clip_fraction": float(stats["clip_fraction"]),
                    }
                )
            weight = len(indices) / len(minibatch_indices)
            for name, value in stats.items():
                minibatch_stats[name] = minibatch_stats.get(name, 0.0) + value * weight
        return minibatch_stats

    def _optimizer_step(
        self, minibatch_stats: dict[str, torch.Tensor], totals: dict[str, float]
    ) -> None:
        """Clip, step, and add the minibatch statistics into ``totals``."""
        grad_norm = self._clip_gradients()
        self.optimizer.step()

        minibatch_stats["grad_norm"] = grad_norm.detach()
        for name, value in minibatch_stats.items():
            # `.get` so an optional statistic (e.g. bc_loss, reported
            # only while co-training) does not need a pre-seeded total.
            totals[name] = totals.get(name, 0.0) + float(value)

    def _kl_above_target(self, approx_kl: torch.Tensor) -> float | None:
        """Global mean of ``approx_kl`` when it exceeds ``target_kl``, else ``None``."""
        if self.config.target_kl is None:
            return None
        global_kl = float(_global_mean(approx_kl))
        if global_kl > self.config.target_kl:
            return global_kl
        return None

    def _peak_memory_metrics(self) -> dict[str, float]:
        if self.device.type != "cuda":
            return {}
        return {
            "peak_allocated_bytes": float(torch.cuda.max_memory_allocated(self.device)),
            "peak_reserved_bytes": float(torch.cuda.max_memory_reserved(self.device)),
        }

    def checkpoint_state(self) -> dict[str, np.ndarray]:
        payload = json.dumps({"updates": self.updates}, sort_keys=True).encode("utf-8")
        return {"ppo_state": np.frombuffer(payload, dtype=np.uint8).copy()}

    def load_checkpoint_state(self, state: Mapping[str, np.ndarray]) -> None:
        if set(state) != {"ppo_state"}:
            raise ValueError("PPO checkpoint must contain only ppo_state")
        payload = np.asarray(state["ppo_state"], dtype=np.uint8).tobytes()
        decoded = json.loads(payload.decode("utf-8"))
        updates = decoded.get("updates")
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError("PPO checkpoint update count must be non-negative")
        self.updates = updates

    def close(self) -> None:
        # The learner holds no resources beyond its modules.
        pass
