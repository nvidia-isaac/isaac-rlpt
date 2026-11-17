# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch FSDP learner pieces any multi-process runtime can host.

:class:`~isaac_rl_posttraining.runtimes.fsdp.modules.FSDPModuleManager` shards
an algorithm's modules and writes the manifest-versioned checkpoint;
:class:`~isaac_rl_posttraining.runtimes.fsdp.worker.FSDPLearnerWorker` runs one
rank. Importing the submodules needs ``torch.distributed``; this package
module itself imports nothing heavy.
"""

from isaac_rl_posttraining.runtimes.fsdp.checkpoint import (
    LEARNER_CHECKPOINT_SCHEMA_VERSION,
    validate_learner_checkpoint_manifest,
)

__all__ = ["LEARNER_CHECKPOINT_SCHEMA_VERSION", "validate_learner_checkpoint_manifest"]
