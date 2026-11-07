# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-rank FSDP check: the supervised batch must enter through the model call.

Run under torchrun; prints PASS/FAIL lines consumed by tests/unit/test_cotrain.py.
"""

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel


class Fused(nn.Module):
    """Mirrors the contract: supervised batch in, per-sample loss as 4th output."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(64, 64, bias=False)

    def forward(  # NOSONAR - mirrors the co-training contract (4th output)
        self, observations, actions, *, supervised_batch=None
    ):
        logp = self.lin(observations).sum(-1)
        if supervised_batch is None:
            return logp, logp, logp
        bc = (
            (self.lin(supervised_batch["obs"]) - supervised_batch["target"])
            .square()
            .mean(-1)
        )
        return logp, logp, logp, bc

    def actor_bc_loss(self, batch):  # the wrong route, kept to prove it fails
        return (self.lin(batch["obs"]) - batch["target"]).square().mean(-1)


def main():
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    model = FullyShardedDataParallel(
        Fused().cuda(), device_id=rank, use_orig_params=True
    )
    dev = f"cuda:{rank}"
    obs = torch.randn(4, 64, device=dev)
    demo = {
        "obs": torch.randn(4, 64, device=dev),
        "target": torch.zeros(4, 64, device=dev),
    }

    # the contract route
    try:
        _, _, _, bc = model(obs, obs, supervised_batch=demo)
        bc.mean().backward()
        grads = [p.grad is not None for p in model.parameters()]
        ok = bc.shape == (4,) and any(grads)
        if rank == 0:
            print(
                f"CALL_ROUTE {'PASS' if ok else 'FAIL'} bc_shape={tuple(bc.shape)} grads={any(grads)}"
            )
    # Any failure is the result.
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            print(f"CALL_ROUTE FAIL {type(exc).__name__}: {exc}")

    # the route my first patch used -- must NOT work under FSDP
    try:
        model.actor_bc_loss(demo)
        if rank == 0:
            print("METHOD_ROUTE UNEXPECTEDLY_WORKED")
    # Any failure is the result.
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            print(f"METHOD_ROUTE RAISES_AS_EXPECTED {type(exc).__name__}")

    dist.destroy_process_group()


main()
