# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RLInf-hosted policy servers: an RLInf worker around the shared policy-server replica."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from isaac_rl_posttraining.runtimes.base import PolicyContext
from isaac_rl_posttraining.runtimes.rlinf.cluster import rlinf_worker_class
from isaac_rl_posttraining.runtimes.serving.batcher import Batcher
from isaac_rl_posttraining.runtimes.serving.replica import PolicyReplica

__all__ = ["PolicyServerWorker", "make_policy_worker"]


def make_policy_worker(worker_base: type) -> Any:
    """Build the policy-server worker class on RLInf's worker base."""
    return rlinf_worker_class("RLPTPolicyServer", PolicyServerWorker, worker_base)


class PolicyServerWorker(PolicyReplica):
    """Policy servers hosted by RLInf; ``make_policy_worker`` mixes this into RLInf's worker."""

    # Provided by RLInf's worker.
    _rank: int
    _world_size: int

    def __init__(
        self,
        policy_spec: Callable[[PolicyContext], Any],
        device: str,
        batcher: Batcher,
        collect_metrics: bool,
        name: str = "policy",
    ) -> None:
        super(PolicyReplica, self).__init__()  # RLInf's worker, which sets _rank
        super().__init__(
            PolicyContext(
                policy_index=int(self._rank),
                num_policies=int(self._world_size),
                device="cuda:0" if device == "gpu" else "cpu",
                placement="policy_server",
            )
        )
        self.add_policy(name, policy_spec, batcher, "builtin", collect_metrics)
