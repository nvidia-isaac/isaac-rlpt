# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shard learner modules with PyTorch FSDP and checkpoint them; runtime-agnostic."""

from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel,
    MixedPrecision,
    StateDictType,
)

from isaac_rl_posttraining.runtimes.base import PolicyState
from isaac_rl_posttraining.runtimes.fsdp.checkpoint import (
    LEARNER_CHECKPOINT_SCHEMA_VERSION,
    validate_learner_checkpoint_manifest,
)
from isaac_rl_posttraining.runtimes.learner import (
    average_gradients_across_ranks,
    check_only_frozen_missing,
    frozen_parameter_prefixes,
    without_prefixes,
)
from isaac_rl_posttraining.runtimes.policy_snapshot import policy_state_from_tensors

__all__ = ["FSDPModuleManager"]


class FSDPModuleManager:
    """Shard a learner's modules with PyTorch FSDP and save or restore them as one checkpoint.

    A module prepared with ``shard=False`` stays whole on every rank instead:
    rank 0's weights are broadcast once and gradients are averaged after each
    backward (plain data parallelism), so code may call any of its methods.

    ``device`` is the rank's device: the current CUDA device by default, or
    ``cpu`` for ranks on a gloo process group (tests and CPU-only runs).
    """

    def __init__(
        self, mixed_precision: str, device: torch.device | None = None
    ) -> None:
        self._mixed_precision = mixed_precision
        self._device = (
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else torch.device(device)
        )
        self._on_cpu = self._device.type == "cpu"
        if self._on_cpu and mixed_precision != "none":
            raise ValueError("learner mixed precision needs CUDA ranks")
        self._modules: dict[str, torch.nn.Module] = {}
        self._replicated: set[str] = set()
        self._frozen_prefixes: dict[str, tuple[str, ...]] = {}
        self._optimizers: dict[str, tuple[torch.nn.Module, torch.optim.Optimizer]] = {}
        self._published_module: torch.nn.Module | None = None
        self._published_name: str | None = None

    def _precision_policy(self) -> MixedPrecision | None:
        if self._mixed_precision == "none":
            return None
        if self._mixed_precision == "bf16":
            dtype = torch.bfloat16
        else:
            raise ValueError(
                f"unsupported learner mixed precision {self._mixed_precision!r}"
            )
        return MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        )

    def training_context(self):
        if self._on_cpu:
            return contextlib.nullcontext()
        if self._mixed_precision == "none":
            return torch.autocast(device_type="cuda", enabled=False)
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    def prepare(
        self,
        module: torch.nn.Module,
        *,
        name: str,
        publish: bool = False,
        frozen_submodules: Sequence[str] = (),
        shard: bool = True,
    ) -> torch.nn.Module:
        if not isinstance(module, torch.nn.Module):
            raise TypeError("learner module manager requires torch.nn.Module")
        if not name or name in self._modules:
            raise ValueError(
                f"learner module name must be unique and non-empty: {name!r}"
            )
        # Frozen submodules stay whole on every rank (no all-gather per
        # forward) and are absent from published states and checkpoints.
        prefixes = frozen_parameter_prefixes(module, frozen_submodules)
        if shard:
            ignored = [module.get_submodule(path) for path in frozen_submodules]
            if self._on_cpu:
                # FSDP syncs module states only on accelerators; broadcast rank 0's weights here.
                _broadcast_states(module)
            wrapped: torch.nn.Module = FullyShardedDataParallel(
                module,
                device_id=self._device if self._on_cpu else self._device.index,
                mixed_precision=self._precision_policy(),
                sync_module_states=not self._on_cpu,
                use_orig_params=True,
                ignored_modules=ignored,
            )
        else:
            wrapped = _replicate(module, self._device)
            self._replicated.add(name)
        self._modules[name] = wrapped
        self._frozen_prefixes[name] = prefixes
        if publish:
            if self._published_module is not None:
                raise ValueError("exactly one learner module may publish policy state")
            self._published_module = wrapped
            self._published_name = name
        return wrapped

    def register_optimizer(
        self,
        name: str,
        optimizer: torch.optim.Optimizer,
        *,
        module: torch.nn.Module,
    ) -> None:
        if not name or name in self._optimizers:
            raise ValueError(
                f"learner optimizer name must be unique and non-empty: {name!r}"
            )
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("learner module manager requires torch.optim.Optimizer")
        prepared = next(
            (candidate for candidate in self._modules.values() if candidate is module),
            None,
        )
        if prepared is None:
            raise ValueError("optimizer module was not prepared by this manager")
        self._optimizers[name] = (prepared, optimizer)

    def _replicated_modules(self) -> list[torch.nn.Module]:
        return [self._modules[name] for name in self._replicated]

    def validate(self) -> None:
        if not self._modules:
            raise ValueError("learner spec must prepare at least one module")
        if self._published_module is None:
            raise ValueError(
                "learner spec must mark exactly one module with publish=True"
            )
        if not self._optimizers:
            raise ValueError("learner spec must register at least one optimizer")

    def _full_state_dict(self, name: str, module: torch.nn.Module) -> dict[str, Any]:
        """The module's whole state dict on rank 0 (empty or partial elsewhere)."""
        if name in self._replicated:
            return {
                key: value.detach().cpu() for key, value in module.state_dict().items()
            }
        # Offloading a CPU module's gathered state to the CPU crashes torch 2.9's FSDP.
        config = FullStateDictConfig(offload_to_cpu=not self._on_cpu, rank0_only=True)
        with FullyShardedDataParallel.state_dict_type(
            module,
            StateDictType.FULL_STATE_DICT,
            config,
        ):
            return module.state_dict()

    def policy_state(self, version: int) -> PolicyState | None:
        if self._published_module is None or self._published_name is None:
            raise RuntimeError("learner spec did not mark a module with publish=True")
        state_dict = self._full_state_dict(self._published_name, self._published_module)
        if dist.get_rank() != 0:
            return None
        assert self._published_name is not None
        return policy_state_from_tensors(
            state_dict,
            version=version,
            exclude_prefixes=self._frozen_prefixes[self._published_name],
        )

    @staticmethod
    def _raise_collective_errors(
        context: str,
        local_error: str | None,
    ) -> None:
        errors: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(errors, local_error)
        failures = tuple(error for error in errors if error is not None)
        if failures:
            raise RuntimeError(f"{context}: " + "; ".join(failures))

    @staticmethod
    def _normalized_algorithm_state(
        state: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        normalized: dict[str, np.ndarray] = {}
        for name, value in state.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "learner checkpoint state names must be non-empty strings"
                )
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise TypeError(f"learner checkpoint state {name!r} uses object dtype")
            normalized[name] = np.ascontiguousarray(array).copy()
        return normalized

    def save_checkpoint(
        self,
        directory: str,
        algorithm_state: Mapping[str, np.ndarray],
        learner_step: int,
    ) -> None:
        path = Path(directory)
        if not path.is_absolute():
            raise ValueError(
                "distributed learner checkpoint directory must be absolute"
            )
        rank = dist.get_rank()
        normalized_algorithm_state = None
        state_error = None
        try:
            normalized_algorithm_state = self._normalized_algorithm_state(
                algorithm_state
            )
        # Propagate validation to every rank.
        except Exception as exc:  # noqa: BLE001
            state_error = f"rank {rank}: {type(exc).__name__}: {exc}"
        self._raise_collective_errors(
            "distributed learner checkpoint state validation failed",
            state_error,
        )
        if normalized_algorithm_state is None:
            raise RuntimeError("distributed learner checkpoint validation lost state")
        checkpoint_ids = [uuid.uuid4().hex if rank == 0 else None]
        dist.broadcast_object_list(checkpoint_ids, src=0)
        checkpoint_id = str(checkpoint_ids[0])
        directory_error = None
        try:
            if rank == 0:
                path.mkdir(parents=True, exist_ok=True)
        # Propagate to every rank.
        except Exception as exc:  # noqa: BLE001
            directory_error = f"rank {rank}: {type(exc).__name__}: {exc}"
        self._raise_collective_errors(
            "distributed learner checkpoint directory creation failed",
            directory_error,
        )

        models: dict[str, dict[str, torch.Tensor]] = {}
        optimizers: dict[str, dict[str, Any]] = {}
        config = FullStateDictConfig(offload_to_cpu=not self._on_cpu, rank0_only=True)
        optimizer_config = FullOptimStateDictConfig(
            offload_to_cpu=not self._on_cpu,
            rank0_only=True,
        )
        for name, module in self._modules.items():
            state = self._full_state_dict(name, module)
            # Frozen submodules are copies of the base checkpoint; the
            # learner rebuilds them from it on load.
            models[name] = without_prefixes(state, self._frozen_prefixes[name])
        for name, (module, optimizer) in self._optimizers.items():
            if module in self._replicated_modules():
                # Every rank holds the same full state; rank 0 writes it.
                optimizers[name] = optimizer.state_dict() if rank == 0 else {}
                continue
            with FullyShardedDataParallel.state_dict_type(
                module,
                StateDictType.FULL_STATE_DICT,
                config,
                optimizer_config,
            ):
                optimizers[name] = FullyShardedDataParallel.optim_state_dict(
                    module,
                    optimizer,
                )
        optimizer_materialization_error = None
        materialized_optimizers = tuple(
            name for name, state in optimizers.items() if state
        )
        if rank != 0 and materialized_optimizers:
            optimizer_materialization_error = (
                f"rank {rank}: full optimizer state unexpectedly materialized for "
                f"{materialized_optimizers}"
            )
        self._raise_collective_errors(
            "distributed learner checkpoint optimizer sharding check failed",
            optimizer_materialization_error,
        )
        payload_error = None
        try:
            if rank == 0:
                payload = {
                    "checkpoint_id": checkpoint_id,
                    "models": models,
                    "optimizers": optimizers,
                }
                temporary = path / "learner.pt.tmp"
                torch.save(payload, temporary)
                temporary.replace(path / "learner.pt")

            local_payload = {
                "algorithm": normalized_algorithm_state,
                "checkpoint_id": checkpoint_id,
                "learner_step": learner_step,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": None if self._on_cpu else torch.cuda.get_rng_state(),
            }
            temporary = path / f"rank_{rank}.pt.tmp"
            torch.save(local_payload, temporary)
            temporary.replace(path / f"rank_{rank}.pt")
        # Propagate to every rank.
        except Exception as exc:  # noqa: BLE001
            payload_error = f"rank {rank}: {type(exc).__name__}: {exc}"
        self._raise_collective_errors(
            "distributed learner checkpoint payload write failed",
            payload_error,
        )

        manifest_error = None
        try:
            if rank == 0:
                manifest = {
                    "schema_version": LEARNER_CHECKPOINT_SCHEMA_VERSION,
                    "checkpoint_id": checkpoint_id,
                    "world_size": dist.get_world_size(),
                    "module_names": sorted(self._modules),
                    "optimizer_names": sorted(self._optimizers),
                    "learner_step": learner_step,
                }
                manifest_temporary = path / "manifest.json.tmp"
                with manifest_temporary.open("w", encoding="utf-8") as stream:
                    json.dump(manifest, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                manifest_temporary.replace(path / "manifest.json")
        # Propagate to every rank.
        except Exception as exc:  # noqa: BLE001
            manifest_error = f"rank {rank}: {type(exc).__name__}: {exc}"
        self._raise_collective_errors(
            "distributed learner checkpoint manifest write failed",
            manifest_error,
        )

    def load_checkpoint(
        self,
        directory: str,
    ) -> tuple[dict[str, np.ndarray], int]:
        path = Path(directory)
        if not path.is_absolute():
            raise ValueError(
                "distributed learner checkpoint directory must be absolute"
            )
        learner_path = path / "learner.pt"
        rank_path = path / f"rank_{dist.get_rank()}.pt"
        manifest_path = path / "manifest.json"
        payload = None
        local_payload = None
        checkpoint_id = ""
        learner_step = -1
        preflight_error = None
        try:
            if (
                not manifest_path.is_file()
                or not learner_path.is_file()
                or not rank_path.is_file()
            ):
                raise FileNotFoundError(
                    f"incomplete distributed learner checkpoint at {path}"
                )
            with manifest_path.open(encoding="utf-8") as stream:
                manifest = json.load(stream)
            checkpoint_id, learner_step = validate_learner_checkpoint_manifest(
                manifest,
                world_size=dist.get_world_size(),
                module_names=tuple(self._modules),
                optimizer_names=tuple(self._optimizers),
            )
            payload = torch.load(learner_path, map_location="cpu", weights_only=True)
            # The rank file carries the algorithm's NumPy state, which needs unpickling.
            local_payload = torch.load(  # NOSONAR - own checkpoint with NumPy state
                rank_path, map_location="cpu", weights_only=False
            )
            if payload["checkpoint_id"] != checkpoint_id:
                raise ValueError(
                    "distributed learner checkpoint files are mixed or stale"
                )
            if local_payload["checkpoint_id"] != checkpoint_id:
                raise ValueError("distributed learner rank file is mixed or stale")
            if local_payload["learner_step"] != learner_step:
                raise ValueError(
                    "distributed learner rank step disagrees with manifest"
                )
        # Propagate preflight to every rank.
        except Exception as exc:  # noqa: BLE001
            preflight_error = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
        self._raise_collective_errors(
            "distributed learner checkpoint preflight failed",
            preflight_error,
        )
        if payload is None or local_payload is None:
            raise RuntimeError("distributed learner checkpoint preflight lost state")
        config = FullStateDictConfig(offload_to_cpu=not self._on_cpu, rank0_only=False)
        for name, module in self._modules.items():
            prefixes = self._frozen_prefixes[name]
            if name in self._replicated:
                result = module.load_state_dict(
                    payload["models"][name], strict=not prefixes
                )
            else:
                with FullyShardedDataParallel.state_dict_type(
                    module,
                    StateDictType.FULL_STATE_DICT,
                    config,
                ):
                    result = module.load_state_dict(
                        payload["models"][name], strict=not prefixes
                    )
            if prefixes:
                check_only_frozen_missing(
                    result.missing_keys, result.unexpected_keys, prefixes
                )
        for name, (module, optimizer) in self._optimizers.items():
            if module in self._replicated_modules():
                optimizer.load_state_dict(payload["optimizers"][name])
                continue
            optimizer_state = FullyShardedDataParallel.optim_state_dict_to_load(
                module,
                optimizer,
                payload["optimizers"][name],
            )
            optimizer.load_state_dict(optimizer_state)

        rng_error = None
        try:
            torch.set_rng_state(local_payload["torch_rng"])
            if local_payload["cuda_rng"] is not None:
                torch.cuda.set_rng_state(local_payload["cuda_rng"])
        # Propagate to every rank.
        except Exception as exc:  # noqa: BLE001
            rng_error = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
        self._raise_collective_errors(
            "distributed learner checkpoint RNG restore failed",
            rng_error,
        )
        return (
            self._normalized_algorithm_state(local_payload["algorithm"]),
            learner_step,
        )


def _broadcast_states(module: torch.nn.Module) -> None:
    """Overwrite this rank's parameters and buffers with rank 0's."""
    with torch.no_grad():
        for tensor in module.state_dict().values():
            dist.broadcast(tensor, src=0)


def _replicate(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Place a whole copy of ``module`` on this rank: rank 0's weights, gradients averaged across ranks."""
    module.to(device)
    _broadcast_states(module)
    average_gradients_across_ranks(tuple(module.parameters()))
    return module
