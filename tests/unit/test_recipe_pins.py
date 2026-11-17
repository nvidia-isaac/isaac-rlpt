# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the DSRL G1 Factory recipe so hyperparameters cannot drift by accident.

These values produced the README results (simulators/g1_arena/README.md,
"Reference DSRL results"). Changing any of them is a deliberate, reviewed
decision: update the pin in the same commit as the config change and say why.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "isaac_rl_posttraining" / "configs"


def _resolved():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(
            config_name="train", overrides=["base_model=mock", "simulator=g1_arena"]
        )


# The learning recipe: everything that changes what the agent learns.
LEARNING_RECIPE = {
    "algorithm.name": "dsrl_sac",  # registry key names the paradigm (zoo rename); same SAC recipe
    "dsrl.noise_action_bound": 1.0,
    "dsrl.action_exec_len": 8,
    "training.chunk_size": 40,
    "algorithm.learning_starts": 1200,
    "algorithm.ent_coef": "auto",
    "algorithm.ent_coef_parameterization": "log",
    "algorithm.target_entropy": 0,
    "algorithm.target_update_interval": 1,
    "algorithm.gamma": 0.995,
    "algorithm.tau": 0.005,
    "algorithm.batch_size": 64,
    "algorithm.num_critics": 4,
    "algorithm.n_critics_to_sample": 2,
    "algorithm.critic_target_reduction": "mean",
    "algorithm.pooled_critic_features": True,
    "algorithm.critic_reduction": "min",
    "algorithm.num_critic_updates_per_actor_update": 4,
    "algorithm.num_updates_per_train_step": 10,
    "algorithm.temperature_loss_type": "log_alpha",
    "algorithm.bootstrap_at_truncation": False,
    "algorithm.train_critic_with_entropy": False,
    "algorithm.train_actor_with_entropy": True,
    "algorithm.train_actor_with_kl_divergence": False,
    "algorithm.train_actor_with_prior_kl": False,
    "algorithm.latent_prior_kl_coef": 0.0,
    "algorithm.actor_optimizer_lr": 5e-5,
    "algorithm.critic_optimizer_lr": 1e-4,
    "algorithm.ent_coef_lr": 1e-4,
    "algorithm.actor_optimizer_betas": [0.9, 0.999],
    "algorithm.critic_optimizer_betas": [0.95, 0.999],
}

# The budget and evaluation protocol of the validated recipe: 40k env steps
# (runs collapse beyond ~45k), 50-episode evaluations every 5k steps, and a
# checkpoint at every evaluated step so retention can keep the best one.
BUDGET_AND_EVALUATION = {
    "training.num_rollouts": 40000,
    "training.save_interval": 5000,
    "eval.eval_freq": 5000,
    "eval.eval_num_episodes": 50,
}

# Knobs that must stay at their identity value unless deliberately changed
# (present on some branches only, so pinned to "absent or identity").
IDENTITY_IF_PRESENT = {
    "algorithm.ent_coef_init": 1.0,
    "algorithm.ent_coef_min": 0.0,
    "training.async_learner": False,
}


def _get(cfg, dotted, default=None):
    node = cfg
    for part in dotted.split("."):
        if node is None or part not in node:
            return default
        node = node[part]
    return node


def _plain(value):
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(value) if hasattr(value, "_get_node") else value
    # Any non-container value is returned as is.
    except Exception:  # noqa: BLE001
        return value


def test_learning_recipe_is_pinned():
    cfg = _resolved()
    drift = {}
    for key, expected in LEARNING_RECIPE.items():
        actual = _plain(_get(cfg, key, "<missing>"))
        if isinstance(expected, float):
            ok = (
                isinstance(actual, (int, float))
                and abs(float(actual) - expected) < 1e-12
            )
        else:
            ok = actual == expected
        if not ok:
            drift[key] = (expected, actual)
    assert not drift, f"recipe drift (expected, actual): {drift}"


def test_budget_and_evaluation_protocol_are_pinned():
    cfg = _resolved()
    actual = {key: _get(cfg, key, "<missing>") for key in BUDGET_AND_EVALUATION}
    assert actual == BUDGET_AND_EVALUATION


def test_identity_knobs_unchanged_if_present():
    cfg = _resolved()
    drift = {}
    for key, identity in IDENTITY_IF_PRESENT.items():
        actual = _get(cfg, key, None)
        if actual is not None and actual != identity:
            drift[key] = (identity, actual)
    assert not drift, f"identity knobs changed (expected, actual): {drift}"
