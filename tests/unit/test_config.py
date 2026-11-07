# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical Hydra configuration: the defaults tree composes and its load-bearing values stay pinned.

The repo ships one training config, ``isaac_rl_posttraining/configs/train.yaml``
(``algorithm=<name>`` selects the method), plus the config groups. These tests
compose the full defaults tree through Hydra and pin the values the launchers
and the rest of the suite rely on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "isaac_rl_posttraining" / "configs"
CANONICAL_NAME = "train"


@pytest.fixture()
def canonical_cfg():
    """Compose the canonical config through the full Hydra defaults tree."""
    path = CONFIG_DIR / f"{CANONICAL_NAME}.yaml"
    if not path.is_file():
        pytest.fail(f"canonical config not found at {path}", pytrace=False)
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        yield compose(config_name=CANONICAL_NAME)


def test_canonical_config_composes(canonical_cfg):
    """The full defaults tree (algorithm/policy/value_function groups) resolves."""
    required_sections = {
        "env",
        "training",
        "buffer",
        "logging",
        "eval",
        "model",
        "dsrl",
        "learner",
        "algorithm",  # algorithm: dsrl_sac
        "policy",  # policy@policy: transformer
        "value_function",  # value_function@value_function: transformer
        "base_model",  # base_model: gr00t
        "runtime",  # rollout execution provider
    }
    missing = required_sections - set(canonical_cfg.keys())
    assert not missing, f"canonical config is missing sections: {sorted(missing)}"


def test_canonical_recipe_pins(canonical_cfg):
    """The load-bearing defaults of the validated G1 Factory + GR00T DSRL recipe."""
    # Latent geometry: a 40-step chunk of 35-dim per-step latents; the first 8 decoded actions execute.
    assert int(canonical_cfg.training.chunk_size) == 40
    assert int(canonical_cfg.dsrl.noise_dim) == 35
    assert int(canonical_cfg.dsrl.action_exec_len) == 8
    assert float(canonical_cfg.dsrl.noise_action_bound) == pytest.approx(
        1.0
    )  # tight recipe (peak-validated)
    # Groups: the validated GR00T base model, the in-process runtime, a critic that pools the CLS token.
    assert str(canonical_cfg.base_model.name) == "gr00t"
    assert str(canonical_cfg.runtime.name) == "native"
    assert canonical_cfg.runtime.num_rollout_workers is None
    assert str(canonical_cfg.value_function.transformer.pooling_strategy) == "first"
    # Evaluation is scheduled, and the client-side episode cap is set (the Arena server enforces no horizon).
    for key in ("eval_freq", "eval_num_episodes", "eval_on_first_step"):
        assert key in canonical_cfg.eval, f"eval.{key} missing from canonical config"
    assert (
        int(canonical_cfg.eval.eval_freq) > 0
        and int(canonical_cfg.eval.eval_num_episodes) > 0
    )
    # The task's cap from the simulator group (drill_lift is LMDrillLift).
    assert int(canonical_cfg.env.max_episode_steps) == 450


def test_only_one_canonical_config():
    """One top-level config; algorithms are a group and task variants are CLI overrides, not files."""
    if not CONFIG_DIR.is_dir():
        pytest.fail(f"config dir missing at {CONFIG_DIR}", pytrace=False)
    top_level = sorted(
        p.name for ext in ("*.yaml", "*.yml") for p in CONFIG_DIR.glob(ext)
    )
    assert top_level == [f"{CANONICAL_NAME}.yaml"], (
        f"expected the {CANONICAL_NAME} root config only, found: {top_level}"
    )


def test_no_legacy_strings_in_configs():
    """No legacy codename and no dead package roots anywhere in shipped configs."""
    if not CONFIG_DIR.is_dir():
        pytest.fail(f"config dir missing at {CONFIG_DIR}", pytrace=False)
    # Joined at runtime so this test file never matches itself in repo scans.
    needles = ("robo" + "meter", "rl_policy" + "_learning")
    yaml_files = [p for ext in ("*.yaml", "*.yml") for p in CONFIG_DIR.rglob(ext)]
    assert yaml_files, f"no yaml files found under {CONFIG_DIR}"
    offenders = {}
    for path in yaml_files:
        text = path.read_text(encoding="utf-8").lower()
        hits = [n for n in needles if n in text]
        if hits:
            offenders[str(path.relative_to(REPO_ROOT))] = hits
    assert not offenders, f"legacy strings in configs: {offenders}"


def test_canonical_sac_hyperparameters(canonical_cfg):
    """DSRL SAC group values (arXiv:2603.02115 A5) that the SAC-core tests rely on."""
    alg = canonical_cfg.algorithm
    assert int(alg.learning_starts) == 1200
    assert int(alg.batch_size) == 64
    assert int(alg.num_critics) == 4
    assert int(alg.n_critics_to_sample) == 2
    assert int(alg.num_critic_updates_per_actor_update) == 4
    assert int(alg.num_updates_per_train_step) == 10
    assert int(alg.target_update_interval) == 1
    assert float(alg.gamma) == pytest.approx(0.995)
    assert float(alg.tau) == pytest.approx(0.005)
    assert str(alg.ent_coef) == "auto"
    assert float(alg.target_entropy) == pytest.approx(0.0)
    assert str(alg.critic_reduction) == "min"


# ---------------------------------------------------------------------------
# The base model config group (frozen-policy selection).
# ---------------------------------------------------------------------------

#: Every shipped base model group option (configs/base_model/<name>.yaml).
BASE_MODEL_GROUP_OPTIONS = sorted(
    path.stem for path in (CONFIG_DIR / "base_model").glob("*.yaml")
)


@pytest.mark.parametrize("name", BASE_MODEL_GROUP_OPTIONS)
def test_base_model_group_composes_all_options(name):
    """Every base model group option composes and carries a matching name."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=CANONICAL_NAME, overrides=[f"base_model={name}"])
    assert str(cfg.base_model.name) == name


def test_base_model_group_is_the_only_checkpoint_surface():
    """base_model.* carries the checkpoint selection; no dsrl.gr00t_* aliases."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name=CANONICAL_NAME,
            overrides=["base_model.ckpt=/checkpoints/ckpt_sentinel"],
        )
    assert str(cfg.base_model.ckpt) == "/checkpoints/ckpt_sentinel"
    assert cfg.base_model.backbone_model_name_or_path is None
    assert cfg.base_model.server_url is None
    for legacy_key in (
        "vla_type",
        "gr00t_base_ckpt",
        "gr00t_dataset_dir",
        "gr00t_embodiment_tag",
    ):
        assert legacy_key not in cfg.dsrl


@pytest.mark.parametrize(
    "runtime,pins",
    [
        (
            "rlinf",
            {
                "runtime.rlinf.policy_serving": "auto",
                "runtime.rlinf.num_policy_servers": None,
                "runtime.rlinf.policy_batch_size": 32,
                "runtime.rlinf.learner_first_accelerator": 0,
                "runtime.rlinf.first_accelerator": 1,
                "runtime.rlinf.learner_mixed_precision": "none",
            },
        ),
    ],
)
def test_runtime_group_options_compose_their_defaults(runtime, pins):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=CANONICAL_NAME, overrides=[f"runtime={runtime}"])
    assert str(cfg.runtime.name) == runtime
    for key, expected in pins.items():
        assert OmegaConf.select(cfg, key) == expected, key


def test_base_model_dreamzero_local_defaults_pinned():
    """In-process DreamZero group: the 5B/LIBERO runtime defaults."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name=CANONICAL_NAME, overrides=["base_model=dreamzero_local"]
        )
    assert str(cfg.base_model.embodiment_tag) == "libero_sim"
    assert str(cfg.base_model.tokenizer_path) == "google/umt5-xxl"
    assert bool(cfg.base_model.torch_compile) is True
    assert bool(cfg.base_model.text_encoder_cpu_offload) is False
    assert int(cfg.base_model.predict_micro_batch_size) == 1
    assert cfg.base_model.ckpt is None  # required at launch, never defaulted
    assert cfg.base_model.server_url is None  # the served path is dreamzero_server
    assert cfg.base_model.num_dit_steps is None


def test_base_model_dreamzero_server_geometry_pinned():
    """DreamZero served group: served-checkpoint geometry and a ws:// URL."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name=CANONICAL_NAME, overrides=["base_model=dreamzero_server"]
        )
    assert int(cfg.base_model.action_horizon) == 24
    assert int(cfg.base_model.action_dim) == 8
    assert str(cfg.base_model.server_url).startswith("ws://")
    # Steering geometry stays untouched: selecting a native-only base_model
    # must not silently alter the canonical latent pins.
    assert int(cfg.dsrl.noise_dim) == 35
    assert int(cfg.training.chunk_size) == 40


def test_base_model_mock_matches_canonical_latent_geometry():
    """The mock group mirrors the canonical 40x35 chunk for pipe-cleaning."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=CANONICAL_NAME, overrides=["base_model=mock"])
    assert int(cfg.base_model.action_horizon) == 40
    assert int(cfg.base_model.action_dim) == 35


def test_checkpoint_retention_survives_dataclass_round_trip():
    """``training.checkpoint_retention`` must reach the Trainer wiring.

    Regression: the field once lived on ``ReplayBufferConfig`` while
    the entry point read ``cfg.training.checkpoint_retention``, so the
    dataclass round-trip in the Hydra entrypoint silently dropped the
    override and retention stayed inert.
    """
    from dataclasses import asdict

    from isaac_rl_posttraining.configs.schema import TrainConfig, TrainingConfig
    from isaac_rl_posttraining.utils.config_utils import convert_hydra_to_dataclass

    assert "checkpoint_retention" in {
        f.name for f in __import__("dataclasses").fields(TrainingConfig)
    }

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name=CANONICAL_NAME,
            overrides=["training.checkpoint_retention=best_last_final"],
        )
    cfg_dc = convert_hydra_to_dataclass(cfg, TrainConfig)
    round_tripped = OmegaConf.create(asdict(cfg_dc))
    assert (
        OmegaConf.select(round_tripped.training, "checkpoint_retention", default="all")
        == "best_last_final"
    )


def test_critic_target_reduction_default_and_min_option():
    """TD-target reduction: default is pooled features + 'mean'; 'min' composes."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=CANONICAL_NAME)
    # Pooled features + mean target reduction is the recipe whose retained
    # checkpoints re-scored above base at N=100; 'min' remains an ablation knob.
    assert str(cfg.algorithm.critic_target_reduction) == "mean"
    assert bool(cfg.algorithm.pooled_critic_features) is True
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name=CANONICAL_NAME,
            overrides=["algorithm.critic_target_reduction=min"],
        )
    assert str(cfg.algorithm.critic_target_reduction) == "min"


def test_dataclass_defaults_match_canonical_yaml():
    """Schema dataclass defaults agree with the canonical YAML values.

    The entry point round-trips the composed config through the dataclasses;
    a dataclass default that contradicts the YAML is a latent bug the moment
    a config path skips composition.
    """
    from isaac_rl_posttraining.algorithms.dsrl import DSRLSteeringConfig
    from isaac_rl_posttraining.configs.schema import PolicyConfig

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=CANONICAL_NAME)

    dsrl_defaults = DSRLSteeringConfig()
    assert dsrl_defaults.noise_dim == int(cfg.dsrl.noise_dim)
    assert dsrl_defaults.noise_action_bound == pytest.approx(
        float(cfg.dsrl.noise_action_bound)
    )
    assert dsrl_defaults.action_exec_len == int(cfg.dsrl.action_exec_len)
    assert PolicyConfig().log_std_init == pytest.approx(float(cfg.policy.log_std_init))
    assert PolicyConfig().official_policy_init == bool(cfg.policy.official_policy_init)


def test_pixel_recipe_fields_survive_dataclass_round_trip():
    """The pixel-recipe knobs must reach their consumers through the round-trip.

    Regression guard for the schema-drop class of bug: ``env.use_init_states``
    once had no ``EnvironmentConfig`` field, so launcher overrides of it were
    silently discarded by the Hydra entrypoint's dataclass rebuild.
    """
    from dataclasses import asdict

    from isaac_rl_posttraining.configs.schema import TrainConfig
    from isaac_rl_posttraining.utils.config_utils import convert_hydra_to_dataclass

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name=CANONICAL_NAME,
            overrides=[
                "env.use_init_states=false",
                "env.fixed_init_state_id=3",
                "model.image_encoder_type=small",
                "model.small_bottleneck_dim=64",
                "model.featurizer_dim=64",
                "model.impala_bounded_head=true",
            ],
        )
    round_tripped = OmegaConf.create(
        asdict(convert_hydra_to_dataclass(cfg, TrainConfig))
    )
    assert round_tripped.env.use_init_states is False
    assert int(round_tripped.env.fixed_init_state_id) == 3
    assert str(round_tripped.model.image_encoder_type) == "small"
    assert int(round_tripped.model.small_bottleneck_dim) == 64
    assert int(round_tripped.model.featurizer_dim) == 64
    assert round_tripped.model.impala_bounded_head is True


# ---------------------------------------------------------------------------
# Removed and mistyped replay keys are refused, not dropped
# ---------------------------------------------------------------------------


def _convert(*overrides: str):
    from isaac_rl_posttraining.configs.schema import TrainConfig
    from isaac_rl_posttraining.utils.config_utils import convert_hydra_to_dataclass

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=CANONICAL_NAME, overrides=list(overrides))
    return convert_hydra_to_dataclass(cfg, TrainConfig)


def test_removed_buffer_split_override_is_refused_with_its_replacement():
    from isaac_rl_posttraining.utils.config_utils import RemovedConfigKeyError

    with pytest.raises(
        RemovedConfigKeyError,
        match=r"buffer\.split.*buffer\.success_fraction.*buffer\.demonstrations",
    ):
        _convert("+buffer.split=outcome")
    with pytest.raises(
        RemovedConfigKeyError, match=r"buffer\.save_buffer_images: removed"
    ):
        _convert("+buffer.save_buffer_images=false")


def test_save_buffer_settings_are_declared_replay_settings():
    converted = _convert()
    assert (
        converted.buffer.save_buffer_on_exit is False
        and converted.buffer.save_buffer_every == 0
    )
    assert (
        converted.buffer.reward_aggregation == "last"
        and converted.buffer.success_fraction is None
    )
    assert (
        _convert("buffer.save_buffer_on_exit=true").buffer.save_buffer_on_exit is True
    )
    assert _convert("++buffer.save_buffer_every=5000").buffer.save_buffer_every == 5000


def test_renamed_serialize_flag_is_refused_with_its_new_name():
    from isaac_rl_posttraining.utils.config_utils import RemovedConfigKeyError

    with pytest.raises(
        RemovedConfigKeyError, match=r"runtime\.rlinf\.serialize_specs_by_value"
    ):
        _convert("+runtime.rlinf.serialize_factory_by_value=true")


def test_unknown_buffer_key_is_refused_naming_the_real_settings():
    from isaac_rl_posttraining.utils.config_utils import RemovedConfigKeyError

    with pytest.raises(
        RemovedConfigKeyError,
        match=r"buffer\.succes_fraction: not a setting.*success_fraction",
    ):
        _convert("+buffer.succes_fraction=0.5")


def test_unknown_key_in_any_schema_group_is_refused():
    """A mistyped training or evaluation key fails at launch instead of running the default schedule."""
    from isaac_rl_posttraining.utils.config_utils import RemovedConfigKeyError

    with pytest.raises(
        RemovedConfigKeyError,
        match=r"training\.num_rolouts: not a setting of TrainingConfig.*num_rollouts",
    ):
        _convert("+training.num_rolouts=5")
    with pytest.raises(
        RemovedConfigKeyError,
        match=r"eval\.eval_num_episode: not a setting.*eval_num_episodes",
    ):
        _convert("+eval.eval_num_episode=3")


def test_yaml_form_of_a_removed_buffer_key_is_refused_too():
    from isaac_rl_posttraining.configs.schema import TrainConfig
    from isaac_rl_posttraining.utils.config_utils import (
        RemovedConfigKeyError,
        convert_hydra_to_dataclass,
    )

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=CANONICAL_NAME)
    plain = OmegaConf.to_container(cfg, resolve=True)
    plain["buffer"]["split"] = "outcome"
    plain["buffer"]["fractions"] = {"success": 0.5, "failure": 0.5}
    with pytest.raises(
        RemovedConfigKeyError, match=r"buffer\.fractions.*buffer\.split"
    ):
        convert_hydra_to_dataclass(plain, TrainConfig)


def test_the_old_backend_group_is_refused_naming_its_base_model_key():
    from isaac_rl_posttraining.utils.config_utils import (
        RemovedConfigKeyError,
        refuse_removed_overrides,
    )

    for override, message in (
        ("backend=gr00t", r"backend: renamed to base_model$"),
        (
            "backend.base_ckpt=/ckpt",
            r"backend\.base_ckpt: renamed to base_model\.ckpt$",
        ),
        (
            "+backend.dataset_dir=/data",
            r"backend\.dataset_dir: renamed to base_model\.dataset_dir$",
        ),
    ):
        with pytest.raises(RemovedConfigKeyError, match=message):
            refuse_removed_overrides([override])
    refuse_removed_overrides(["base_model=gr00t", "base_model.ckpt=/ckpt"])


def test_yaml_form_of_the_old_backend_group_is_refused_too():
    from isaac_rl_posttraining.configs.schema import TrainConfig
    from isaac_rl_posttraining.utils.config_utils import (
        RemovedConfigKeyError,
        convert_hydra_to_dataclass,
    )

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=CANONICAL_NAME)
    plain = OmegaConf.to_container(cfg, resolve=True)
    plain["backend"] = plain.pop("base_model")
    with pytest.raises(RemovedConfigKeyError, match=r"backend: renamed to base_model"):
        convert_hydra_to_dataclass(plain, TrainConfig)
