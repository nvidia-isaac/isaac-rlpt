# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment adapters and wrappers.

Layout:

* :mod:`~isaac_rl_posttraining.envs.transport` -- the transport seam between
  the learner and remote environment endpoints (URIs, wire schema, native
  TCP implementation).
* :mod:`~isaac_rl_posttraining.envs.zmq_transport` -- the ZeroMQ transport
  (``ipc://`` and ``zmq+tcp://`` endpoints, msgpack(+numpy) frames; needs
  the optional ``[zmq]`` extra only when actually constructed).
* :mod:`~isaac_rl_posttraining.envs.providers` -- the by-name registry of
  environment providers (one per simulator integration; see
  ``simulators/README.md``).
* :mod:`~simulators.g1_arena.env` -- :class:`ArenaEnv`, the
  Gymnasium client for the Isaac Lab Arena environment server (provider
  ``"g1_arena"``).
* :mod:`~simulators.libero.env` -- :class:`LiberoEnv`, the
  in-process wrapper for the LIBERO benchmark suites (provider
  ``"libero"``; the LIBERO stack itself is optional and imported lazily).
* :mod:`~simulators.libero.policies.gr00t` --
  :class:`LiberoGr00tAdapter`, the LIBERO observation/action conventions
  of the released GR00T-N1.7 LIBERO checkpoints.
* :mod:`~isaac_rl_posttraining.envs.steering_wrappers` -- observation glue
  and space definitions for latent-steering learners.
* :mod:`~isaac_rl_posttraining.envs.action_wrappers` -- open-loop execution
  of policy action chunks.
* :mod:`~isaac_rl_posttraining.envs.obs_wrappers`,
  :mod:`~isaac_rl_posttraining.envs.language_wrappers`,
  :mod:`~isaac_rl_posttraining.envs.dino_wrapper` -- dict-observation
  adapters, fixed-instruction embeddings, and DINOv2 image features.

Torch-dependent wrapper modules are not imported here so that this package
stays importable in lightweight processes (e.g. the simulator-side env
server, which only needs :mod:`transport`).
"""

from isaac_rl_posttraining.envs.action_wrappers import (
    ActionChunkingWrapper,
    VectorActionChunkingWrapper,
)
from isaac_rl_posttraining.envs.providers import (
    available_providers,
    get_provider,
    register_provider,
)
from isaac_rl_posttraining.envs.steering_wrappers import (
    DummyDSRLEnv,
    LatentSpaceEnv,
    ResidualFlowSpaceEnv,
    SimpleDSRLWrapper,
    SteeringObsWrapper,
)
from isaac_rl_posttraining.envs.transport import (
    PROTOCOL_VERSION,
    EnvTransport,
    NativeTcpServer,
    NativeTcpTransport,
    TransportError,
    transport_from_uri,
)
from isaac_rl_posttraining.envs.vector_backends import (
    VECTOR_BACKENDS,
    BatchedEnvAdapter,
    SubprocVectorEnv,
    build_vector_env,
)
from isaac_rl_posttraining.envs.zmq_transport import (
    ZmqServer,
    ZmqTransport,
)

__all__ = [
    "ActionChunkingWrapper",
    "BatchedEnvAdapter",
    "DummyDSRLEnv",
    "EnvTransport",
    "LatentSpaceEnv",
    "NativeTcpServer",
    "NativeTcpTransport",
    "PROTOCOL_VERSION",
    "ResidualFlowSpaceEnv",
    "SimpleDSRLWrapper",
    "SteeringObsWrapper",
    "SubprocVectorEnv",
    "TransportError",
    "VECTOR_BACKENDS",
    "VectorActionChunkingWrapper",
    "ZmqServer",
    "ZmqTransport",
    "available_providers",
    "build_vector_env",
    "get_provider",
    "register_provider",
    "transport_from_uri",
]
