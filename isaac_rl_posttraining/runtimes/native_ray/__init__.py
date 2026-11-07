# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The native runtime's multi-process mode: rollout workers, policy servers and learner ranks as Ray actors.

:class:`~isaac_rl_posttraining.runtimes.native.NativeRuntime` opens these
when it is given a placement plan; nothing here is used by a single-process
run. Importing the package needs no Ray; opening a group does (the ``ray``
extra).

- :mod:`.cluster`: connect to Ray, reserve each role's GPUs per node, wait on actors.
- :mod:`.rollout`: :class:`~.rollout.RayRolloutGroup`, one actor per rollout worker.
- :mod:`.serving`: :class:`~.serving.RayPolicyService`, policy-server replicas.
- :mod:`.learner`: :class:`~.learner.RayLearnerGroup`, learner ranks in one process group.
"""
