# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured (dataclass) schema for the training entry point.

:class:`TrainConfig` mirrors ``configs/train.yaml``;
``scripts/train.py`` converts the composed Hydra config into these
dataclasses (via :func:`isaac_rl_posttraining.utils.config_utils.convert_hydra_to_dataclass`)
to validate fields and resolve interpolations. Keys present in the YAML but
not declared here are refused by the conversion — keep the two in sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from isaac_rl_posttraining.algorithms.dsrl import DSRLSteeringConfig
from isaac_rl_posttraining.algorithms.rfs import RFSModulationConfig
from isaac_rl_posttraining.configs.base_model_schema import BaseModelConfig
from isaac_rl_posttraining.configs.runtime_schema import (
    EnvsPlacementConfig,
    LearnerPlacementConfig,
    NativeRuntimeSettings,
    PlacementConfig,
    PolicyPlacementConfig,
    RLInfRuntimeSettings,
    RuntimeConfig,
)

__all__ = [
    "EnvironmentConfig",
    "TrainingConfig",
    "LoggingConfig",
    "EvaluationConfig",
    "ModelConfig",
    "LearnerConfig",
    "NativeRuntimeSettings",
    "PPOTrainingConfig",
    "RLInfRuntimeSettings",
    "RuntimeConfig",
    "PlacementConfig",
    "ScheduleConfig",
    "EnvsPlacementConfig",
    "PolicyPlacementConfig",
    "LearnerPlacementConfig",
    "BaseModelConfig",
    "MLPConfig",
    "RNNConfig",
    "TransformerConfig",
    "PolicyConfig",
    "ValueFunctionConfig",
    "ReplayBufferConfig",
    "DSRLSteeringConfig",
    "RFSModulationConfig",
    "DirectActionConfig",
    "TrainConfig",
]


@dataclass
class EnvironmentConfig:
    """Environment selection and reward shaping (g1_arena/gr00t reference defaults)."""

    env_name: str = field(default="arena/drill_lift", metadata={"help": "Environment name (e.g. 'arena/drill_lift')"})
    use_full_state: bool = field(
        default=False,
        metadata={"help": "Use the raw state observation only (True) instead of DINO image embeddings (False)"},
    )
    sparse_success_reward: bool = field(
        default=False,
        metadata={
            "help": "Replace the env reward with a terminal-outcome reward "
            "(values from success_reward/failure_reward/step_reward)"
        },
    )
    success_reward: float = field(default=1.0, metadata={"help": "Reward on a success terminal (sparse scheme)"})
    failure_reward: float = field(
        default=0.0,
        metadata={
            "help": "Reward on a failure terminal, incl. timeouts. -1.0 with success_reward=0 makes the "
            "common outcome carry signal, so Q calibrates faster"
        },
    )
    step_reward: float = field(default=0.0, metadata={"help": "Reward on non-terminal steps (sparse scheme)"})
    max_episode_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Client-side episode step cap; train.yaml defaults it to the task's cap from "
            "simulator.max_episode_steps (a number, or a per-task map keyed by the env_name task)"
        },
    )
    image_keys: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "Camera entries the learner embeds with DINO; null follows the simulator entry's "
            "mapped image keys"
        },
    )
    critic_image_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "When set, attach 'critic_image' (first image key downsized to this square "
            "size, uint8) for pixel-input actor/critic encoders"
        },
    )
    use_init_states: bool = field(
        default=True, metadata={"help": "Draw initial scenes from the benchmark init-state files"}
    )
    fixed_init_state_id: Optional[int] = field(
        default=None,
        metadata={"help": "Pin every reset to this benchmark init-state index (None: sample)"},
    )


@dataclass
class TrainingConfig:
    """Training-loop settings."""

    num_envs: int = field(
        default=1, metadata={"help": "Environments the rollout workers host (the simulator entry places them)"}
    )
    num_rollouts: int = field(
        default=100_000, metadata={"help": "Environment steps to collect/train for (0 = eval-only)"}
    )
    chunk_size: Optional[int] = field(default=None, metadata={"help": "Actor chunk length (None = non-chunked MLP)"})
    use_rnn: bool = field(default=False, metadata={"help": "Use the RNN actor/critic instead of the transformer"})
    seed: int = field(default=0, metadata={"help": "Random seed"})
    load_dir: Optional[str] = field(
        default=None, metadata={"help": "Checkpoint directory to warm-start the SAC actor/critic from"}
    )
    save_interval: int = field(default=10_000, metadata={"help": "Checkpoint interval in env steps"})
    checkpoint_retention: str = field(
        default="all",
        metadata={"help": "'all' keeps every checkpoint; 'best_last_final' prunes to the best-eval, last, and final"},
    )
    rollout_count_by: str = field(
        default="step", metadata={"help": "Count the rollout budget in env steps ('step') or episodes ('episode')"}
    )
    train_after_episode: bool = field(default=False, metadata={"help": "Train only after episode completion"})
    max_updates_per_burst: Optional[int] = field(
        default=None,
        metadata={
            "help": "Cap on SAC updates scheduled per training phase (each update is "
            "num_updates_per_train_step gradient steps). None keeps one update per collected "
            "decision at any num_envs; a cap bounds the learner's time per phase when many envs "
            "collect faster than the learner can consume at that ratio. With a cap set, the "
            "train_after_episode cadence also counts the warm-up data before learning_starts, so "
            "its first burst fires at learning_starts (bounded by the cap) instead of at the next "
            "episode end."
        },
    )
    policy_decode_batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Maximum frozen-policy decode batch during training. Evaluation is always "
            "uncapped so it decodes all active evaluation environments together."
        },
    )


@dataclass
class LoggingConfig:
    """Experiment logging settings."""

    wandb_project: str = field(default="isaac_rl_posttraining", metadata={"help": "W&B project name"})
    wandb_entity: Optional[str] = field(default=None, metadata={"help": "W&B entity name"})
    wandb_offline: bool = field(default=False, metadata={"help": "Run W&B in offline mode"})
    wandb_log_dir_base: Optional[str] = field(
        default=None,
        metadata={"help": "W&B files go to <this>/<wandb_name> (null: WANDB_DIR when set, else logs/wandb)"},
    )
    wandb_name: str = field(default="train_rl", metadata={"help": "W&B run name"})
    wandb_prefix: Optional[str] = field(
        default="offline", metadata={"help": "Prefix for W&B metric names (e.g. 'offline', 'online')"}
    )
    log_level: str = field(default="INFO", metadata={"help": "Console log level"})


@dataclass
class EvaluationConfig:
    """Evaluation settings."""

    eval_freq: int = field(default=10_000, metadata={"help": "Evaluation frequency in env steps"})
    eval_num_episodes: int = field(default=25, metadata={"help": "Episodes per evaluation"})
    eval_num_envs: int = field(
        default=1,
        metadata={
            "help": "Environments the evaluation env hosts; >1 evaluates that many episodes in "
            "parallel behind the batched backend. eval_num_episodes must be an exact multiple "
            "of this value; batched evaluation skips video recording"
        },
    )
    eval_record_video: bool = field(default=True, metadata={"help": "Record evaluation videos"})
    eval_on_first_step: bool = field(default=False, metadata={"help": "Evaluate once before any training"})
    eval_at_episode_boundary: bool = field(
        default=False,
        metadata={"help": "Only evaluate at episode boundaries (required when train and eval share the env)"},
    )
    eval_base: bool = field(
        default=False,
        metadata={"help": "Evaluate the NATIVE base policy (no steering latents) through the same pipeline"},
    )
    action_selection: str = field(
        default="sample",
        metadata={
            "help": "How the actor picks evaluation actions: 'sample' (the faithful evaluation of a "
            "stochastic steering policy) or 'mean' (its mean action, which underestimates success)"
        },
    )
    eval_deterministic: bool = field(
        default=False,
        metadata={"help": "Deprecated alias of eval.action_selection (true = mean)"},
    )
    eval_max_steps: int = field(
        default=1000,
        metadata={"help": "Hard per-episode step cap so a non-terminating env cannot hang evaluation"},
    )
    reset_seed: Optional[int] = field(
        default=None,
        metadata={
            "help": "Seed evaluation resets per environment and episode so evaluations replay "
            "the same initial scenes (null keeps them unseeded)"
        },
    )
    video_fps: int = field(default=20, metadata={"help": "Evaluation video framerate"})


@dataclass
class ModelConfig:
    """Feature-extractor model settings."""

    dinov2_model: Optional[str] = field(
        default="facebook/dinov2-base", metadata={"help": "DINOv2 model id for image embeddings (None disables)"}
    )
    sentence_model: Optional[str] = field(
        default=None, metadata={"help": "SentenceTransformer model id for text embeddings (None disables)"}
    )
    image_encoder_type: Optional[str] = field(
        default=None, metadata={"help": "Trainable image encoder: 'small', 'impala', or None (MLP featurizers)"}
    )
    impala_nn_scale: int = field(default=1, metadata={"help": "IMPALA channel-width multiplier"})
    impala_num_blocks_per_stack: int = field(default=2, metadata={"help": "IMPALA residual blocks per stack"})
    impala_use_smaller: bool = field(default=False, metadata={"help": "Use the reduced-depth IMPALA variant"})
    impala_output_dim: Optional[int] = field(default=None, metadata={"help": "IMPALA embedding projection width"})
    impala_bounded_head: bool = field(
        default=False, metadata={"help": "LayerNorm+tanh bound on the IMPALA embedding"}
    )
    small_bottleneck_dim: int = field(
        default=50, metadata={"help": "Bottleneck width of the 'small' conv encoder"}
    )
    featurizer_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Per-key MLP featurizer width for non-image keys (None: 384)"},
    )


@dataclass
class LearnerConfig:
    """The one-rank learner's device (``learner``); the rank count is ``placement.learner.ranks``."""

    device: str = field(
        default="auto",
        metadata={"help": "Device of a one-rank on-policy learner in the training process: auto, cpu, or gpu"},
    )

    def __post_init__(self) -> None:
        if self.device not in ("auto", "cpu", "gpu"):
            raise ValueError("learner.device must be auto, cpu, or gpu")


@dataclass
class ScheduleConfig:
    """When collection, updates and weight publishes happen (``schedule``)."""

    kind: str = field(
        default="sync",
        metadata={
            "help": "sync: collect a round, train on it, publish; overlapped: collect the next round "
            "while the learner trains (policy lag 1); async: rollout workers collect continuously"
        },
    )
    max_policy_lag: Optional[int] = field(
        default=None,
        metadata={
            "help": "Drop training data whose policy lag (weight publishes since it was collected) "
            "exceeds this (null: the schedule's own)"
        },
    )
    publish_every: Optional[int] = field(
        default=None,
        metadata={"help": "Async: publish every N learner updates (null: once per training burst)"},
    )


@dataclass
class MLPConfig:
    """MLP architecture."""

    hidden_dims: tuple = field(default=(256, 256), metadata={"help": "Hidden layer widths"})
    activation: str = field(default="relu", metadata={"help": "Activation function"})
    use_layer_norm: bool = field(default=False, metadata={"help": "LayerNorm after each layer"})
    dropout_rate: float = field(default=0.0, metadata={"help": "Dropout probability (0 disables)"})


@dataclass
class RNNConfig:
    """RNN architecture."""

    rnn_type: str = field(default="LSTM", metadata={"help": "RNN type (LSTM, GRU, RNN)"})
    rnn_hidden_size: int = field(default=256, metadata={"help": "Recurrent hidden width"})
    rnn_num_layers: int = field(default=1, metadata={"help": "Stacked recurrent layers"})
    rnn_dropout: float = field(default=0.0, metadata={"help": "Inter-layer recurrent dropout"})
    rnn_bidirectional: bool = field(default=False, metadata={"help": "Bidirectional RNN"})
    feature_hidden_dims: Optional[list] = field(default=None, metadata={"help": "MLP widths before the RNN"})
    output_hidden_dims: Optional[list] = field(default=None, metadata={"help": "MLP widths after the RNN"})
    activation: str = field(default="relu", metadata={"help": "Activation for the MLP parts"})
    use_layer_norm: bool = field(default=False, metadata={"help": "LayerNorm inside the MLP parts"})
    dropout_rate: float = field(default=0.0, metadata={"help": "Dropout inside the MLP parts"})


@dataclass
class TransformerConfig:
    """Transformer architecture."""

    d_model: int = field(default=128, metadata={"help": "Token embedding width"})
    nhead: int = field(default=8, metadata={"help": "Attention heads"})
    num_encoder_layers: int = field(default=1, metadata={"help": "Encoder depth"})
    transformer_dropout: float = field(default=0.1, metadata={"help": "Dropout inside the transformer"})
    transformer_activation: str = field(default="gelu", metadata={"help": "Transformer activation (relu, gelu)"})
    feature_hidden_dims: Optional[list] = field(
        default=None, metadata={"help": "MLP widths before the transformer"}
    )
    output_hidden_dims: Optional[list] = field(default=None, metadata={"help": "MLP widths after the transformer"})
    pooling_strategy: str = field(
        default="first", metadata={"help": "Token pooling (mean, first, attention, weighted_mean)"}
    )
    use_layer_norm: bool = field(default=True, metadata={"help": "LayerNorm inside the MLP parts"})
    activation: str = field(default="relu", metadata={"help": "Activation for the MLP parts"})
    dropout_rate: float = field(default=0.0, metadata={"help": "Dropout inside the MLP parts"})


@dataclass
class PolicyConfig:
    """Actor-network architecture group (``policy@policy``)."""

    mlp: MLPConfig = field(
        default_factory=lambda: MLPConfig(hidden_dims=(512, 512), activation="relu", use_layer_norm=False)
    )
    rnn: RNNConfig = field(default_factory=lambda: RNNConfig(feature_hidden_dims=[512, 512]))
    transformer: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(
            d_model=128,
            feature_hidden_dims=[768, 512, 256],
            transformer_dropout=0.1,
            transformer_activation="gelu",
            use_layer_norm=False,
        )
    )

    use_tanh_output: bool = field(default=True, metadata={"help": "Squash continuous actions through tanh"})
    deterministic: bool = field(default=False, metadata={"help": "Deterministic (non-distributional) policy"})
    log_std_init: float = field(default=0.0, metadata={"help": "Initial log-std"})
    log_std_min: float = field(default=-20.0, metadata={"help": "Log-std lower clamp"})
    log_std_max: float = field(default=2.0, metadata={"help": "Log-std upper clamp"})
    official_policy_init: bool = field(
        default=False,
        metadata={"help": "Initialize an MLP actor at the frozen-policy noise prior"},
    )

    def __post_init__(self) -> None:
        """Promote dict sub-configs to their dataclasses."""
        if isinstance(self.mlp, dict):
            self.mlp = MLPConfig(**self.mlp)
        if isinstance(self.rnn, dict):
            self.rnn = RNNConfig(**self.rnn)
        if isinstance(self.transformer, dict):
            self.transformer = TransformerConfig(**self.transformer)


@dataclass
class ValueFunctionConfig:
    """Critic-network architecture group (``value_function@value_function``)."""

    mlp: MLPConfig = field(
        default_factory=lambda: MLPConfig(hidden_dims=(768, 512, 256), activation="relu", use_layer_norm=True)
    )
    rnn: RNNConfig = field(default_factory=lambda: RNNConfig(feature_hidden_dims=[768, 256], activation="relu"))
    transformer: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(
            d_model=128,
            feature_hidden_dims=[768, 512, 256],
            transformer_dropout=0.1,
            transformer_activation="gelu",
            use_layer_norm=True,
            pooling_strategy="first",
        )
    )

    def __post_init__(self) -> None:
        """Promote dict sub-configs to their dataclasses."""
        if isinstance(self.mlp, dict):
            self.mlp = MLPConfig(**self.mlp)
        if isinstance(self.rnn, dict):
            self.rnn = RNNConfig(**self.rnn)
        if isinstance(self.transformer, dict):
            self.transformer = TransformerConfig(**self.transformer)


@dataclass
class ReplayBufferConfig:
    """Replay settings: one episode-organized buffer and how the learner's batches are drawn from it."""

    capacity: int = field(
        default=1_000_000,
        metadata={"help": "Transitions kept; once full, the oldest complete episodes are evicted whole"},
    )
    reward_aggregation: str = field(
        default="last",
        metadata={
            "help": "How a decision's per-step rewards become its one training reward: "
            "'last' (the outcome step) or 'sum'"
        },
    )
    success_fraction: Optional[float] = field(
        default=None,
        metadata={
            "help": "Share of every batch drawn from successful episodes, the rest from the other "
            "finished ones (null = uniform over every stored transition)"
        },
    )
    demonstrations: Optional[str] = field(
        default=None,
        metadata={"help": "Saved replay buffer directory of demonstrations mixed into every batch"},
    )
    demonstration_fraction: Optional[float] = field(
        default=None, metadata={"help": "Share of every batch drawn from the demonstrations"}
    )
    save_buffer_on_exit: bool = field(default=False, metadata={"help": "Save the buffer on exit or kill"})
    save_buffer_every: int = field(default=0, metadata={"help": "Save the buffer every N env steps (0 disables)"})
    save_buffer_dir: str = field(default="replay_buffers", metadata={"help": "Buffer save subdirectory"})


@dataclass
class PersistConfig:
    """Save every completed training episode as it is collected (off unless ``path`` is set)."""

    path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory the episodes stream to as part-NNNNN saved replay buffers; a relative path is "
            "under the run directory (null = off)"
        },
    )
    every_episodes: int = field(default=100, metadata={"help": "Completed episodes per part"})
    keep_frames: bool = field(
        default=False, metadata={"help": "Keep camera frames (uint8 images), which dominate the size on disk"}
    )


@dataclass
class DataConfig:
    """What the run keeps of its collected data besides training on it."""

    persist: PersistConfig = field(default_factory=PersistConfig)

    def __post_init__(self) -> None:
        if isinstance(self.persist, dict):
            self.persist = PersistConfig(**self.persist)


@dataclass
class DirectActionConfig:
    """Direct-action and frozen-reference geometry for SAC-edit.

    These learners do not steer the flow noise: the frozen policy decodes a
    reference action and the actor emits a bounded edit executed in the raw
    action space. This section configures the reference-action environment
    wrapper and the direct-action actor head.
    """

    action_horizon: Optional[int] = field(
        default=None, metadata={"help": "Steps per decoded chunk (null = the frozen policy's decoded horizon)"}
    )
    action_dim: Optional[int] = field(
        default=None, metadata={"help": "Learner-controlled action width (null = the frozen policy's action width)"}
    )
    action_bound: float = field(default=1.0, metadata={"help": "Symmetric bound of the normalized action chunk"})
    action_indices: Optional[list[int]] = field(
        default=None,
        metadata={"help": "Environment-action indices the learner controls (null = the first action_dim)"},
    )
    environment_action_defaults: Optional[list[float]] = field(
        default=None,
        metadata={"help": "Full environment-action vector for the uncontrolled dimensions"},
    )
    include_sim_state: bool = field(
        default=False, metadata={"help": "Keep the simulator state key in the learner observation"}
    )
    state_key: str = field(default="state", metadata={"help": "Observation key holding the simulator state"})
    reference_action_transform: str = field(
        default="identity",
        metadata={"help": "'identity' or 'absolute_to_relative' mapping of the frozen reference chunk"},
    )
    relative_action_scale: float = field(
        default=0.025, metadata={"help": "Scale for the absolute-to-relative reference transform"}
    )
    reference_image_transform_mode: str = field(
        default="eval", metadata={"help": "Frozen-policy image preprocessing mode for collection ('train'/'eval')"}
    )
    simnorm_dim: int = field(default=64, metadata={"help": "SimNorm group width in the actor head"})
    simnorm_groups: int = field(default=8, metadata={"help": "SimNorm group count in the actor head"})
    actor_edit_scale: Optional[float] = field(
        default=None, metadata={"help": "Bounded reference-edit scale (null disables the edit actor)"}
    )
    actor_reference_centered_full_support: bool = field(
        default=False, metadata={"help": "Full-support reference-centered actor (mutually exclusive with actor_edit_scale)"}
    )


@dataclass
class PPOTrainingConfig:
    """On-policy schedule and PPO update (the ``ppo`` section).

    Composed by the on-policy algorithms only: ``configs/algorithm/ppo.yaml``
    (GR00T flow-SDE fine-tuning) and ``configs/algorithm/dsrl_ppo.yaml``
    (PPO over the steering actor) each carry their recipe. Batch sizes are
    global: the rank-local PPO minibatch is ``global_batch_size / placement.learner.ranks``.
    """

    iterations: int = 100
    eval_only: bool = False
    resume_from: Optional[str] = None
    episodes_per_environment: int = 1
    eval_episodes_per_environment: int = 1
    eval_every: int = 10
    checkpoint_every: int = 10
    checkpoint_retention: str = "best_last_final"
    learning_rate: float = 3e-4
    value_learning_rate: Optional[float] = None
    optimizer: str = "adam"
    adam_betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    optimizer_eps: float = 1e-5
    weight_decay: float = 0.0
    clip_ratio: float = 0.2
    value_clip: Optional[float] = 0.2
    value_loss: str = "mse"
    huber_delta: float = 1.0
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0
    update_epochs: int = 4
    global_batch_size: int = 64
    micro_batch_size: Optional[int] = None
    max_grad_norm: Optional[float] = 0.5
    target_kl: Optional[float] = None
    gamma: float = 0.99
    gae_lambda: float = 0.95
    discount_per: str = "simulator_step"
    bootstrap_at_time_limit: bool = True
    eval_seed: int = 100_000
    # dsrl_ppo only (the simulator attachments build the evaluation workers).
    isolate_evaluation_workers: bool = True
    # Deterministic torch kernels in evaluation policy processes; deterministic_evaluation is
    # its deprecated alias.
    deterministic_kernels: bool = True
    deterministic_evaluation: bool = True
    sample_evaluation_actions: bool = True
    # GR00T flow-SDE fine-tuning only.
    noise_level: Optional[float] = None
    cache_features: bool = False
    # null = the base model adapter's own frozen parts (policies/trainable.py); a list overrides them.
    frozen_submodules: Optional[list[str]] = None
    # Demonstration co-training (isaac_rl_posttraining/training/cotrain.py):
    # L = L_PPO + cotrain_beta * L_BC over the ORIGINAL demonstrations, which
    # anchors the policy to the distribution the real robot was taught on.
    # Requires a base model whose forward accepts a supervised_batch; 0 disables it.
    cotrain_beta: float = 0.0
    cotrain_batch_size: int = 32
    # The demonstrations themselves: a LeRobot dataset directory, required
    # whenever cotrain_beta > 0. The executed action window and the draw order
    # come from the run, not from here.
    cotrain_dataset_path: Optional[str] = None


@dataclass
class TrainConfig:
    """Root config of ``scripts/train.py`` (mirrors ``train.yaml``)."""

    # Populated by the Hydra defaults tree; kept loose on purpose: the group
    # carries one algorithm's own schema (name, family, hyperparameters).
    algorithm: Any = field(default_factory=dict)

    # Simulator-integration metadata (config group "simulator"; see
    # simulators/README.md): provider name, task enumeration, episode caps.
    simulator: Any = field(default_factory=dict)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)

    placement: PlacementConfig = field(default_factory=PlacementConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)

    training: TrainingConfig = field(default_factory=TrainingConfig)
    buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvaluationConfig = field(default_factory=EvaluationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    base_model: BaseModelConfig = field(default_factory=BaseModelConfig)
    dsrl: DSRLSteeringConfig = field(default_factory=DSRLSteeringConfig)
    rfs: RFSModulationConfig = field(default_factory=RFSModulationConfig)
    direct_action: DirectActionConfig = field(default_factory=DirectActionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    value_function: ValueFunctionConfig = field(default_factory=ValueFunctionConfig)
    # Composed by the on-policy algorithms only.
    ppo: Optional[PPOTrainingConfig] = None

    def __post_init__(self) -> None:
        """Promote dict sub-configs to their dataclasses."""
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.runtime, dict):
            self.runtime = RuntimeConfig(**self.runtime)
        if isinstance(self.placement, dict):
            self.placement = PlacementConfig(**self.placement)
        if isinstance(self.schedule, dict):
            self.schedule = ScheduleConfig(**self.schedule)
        if isinstance(self.learner, dict):
            self.learner = LearnerConfig(**self.learner)
        if isinstance(self.buffer, dict):
            self.buffer = ReplayBufferConfig(**self.buffer)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.logging, dict):
            self.logging = LoggingConfig(**self.logging)
        if isinstance(self.eval, dict):
            self.eval = EvaluationConfig(**self.eval)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.env, dict):
            self.env = EnvironmentConfig(**self.env)
        if isinstance(self.base_model, dict):
            self.base_model = BaseModelConfig(**self.base_model)
        if isinstance(self.dsrl, dict):
            self.dsrl = DSRLSteeringConfig(**self.dsrl)
        if isinstance(self.rfs, dict):
            self.rfs = RFSModulationConfig(**self.rfs)
        if isinstance(self.ppo, dict):
            self.ppo = PPOTrainingConfig(**self.ppo)
        if isinstance(self.direct_action, dict):
            self.direct_action = DirectActionConfig(**self.direct_action)
        if isinstance(self.policy, dict):
            self.policy = PolicyConfig(**self.policy)
        if isinstance(self.value_function, dict):
            self.value_function = ValueFunctionConfig(**self.value_function)
