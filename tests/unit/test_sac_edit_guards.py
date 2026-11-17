# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAC-edit safety: non-finite actor inputs fail loudly, and the defaults that let it train on
G1 Factory (fp32, bounded edit, early learning start) are pinned; see simulators/g1_arena/README.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from isaac_rl_posttraining.algorithms.sac_edit.actor import (
    ReferenceCenteredSquashedGaussianDistribution,
    ReferenceEditDistribution,
)
from isaac_rl_posttraining.algorithms.sac_edit.reward_scaler import (
    AdaptiveRewardScaler,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "isaac_rl_posttraining" / "configs"

DISTRIBUTIONS = {
    "edit": lambda: ReferenceEditDistribution(action_dim=3, edit_scale=0.02),
    "reference_centered": lambda: ReferenceCenteredSquashedGaussianDistribution(
        action_dim=3
    ),
}


def _inputs():
    return {
        "mean_actions": torch.zeros(2, 3),
        "log_std": torch.zeros(2, 3),
        "reference_action": torch.full((2, 3), 0.1),
    }


@pytest.mark.parametrize("kind", sorted(DISTRIBUTIONS))
@pytest.mark.parametrize("bad", ["mean_actions", "log_std", "reference_action"])
def test_non_finite_inputs_raise_and_name_the_tensor(kind, bad):
    inputs = _inputs()
    inputs[bad][0, 1] = float("nan")  # one isfinite() check covers nan and inf alike
    with pytest.raises(FloatingPointError, match=bad):
        DISTRIBUTIONS[kind]().proba_distribution(
            inputs["mean_actions"],
            inputs["log_std"],
            reference_action=inputs["reference_action"],
        )


@pytest.mark.parametrize("kind", sorted(DISTRIBUTIONS))
def test_finite_inputs_parameterize_and_sample(kind):
    inputs = _inputs()
    dist = DISTRIBUTIONS[kind]().proba_distribution(
        inputs["mean_actions"],
        inputs["log_std"],
        reference_action=inputs["reference_action"],
    )
    action = dist.sample()
    assert action.shape == (2, 3)
    assert torch.isfinite(action).all()


def test_bounded_edit_stays_within_edit_scale():
    inputs = _inputs()
    dist = ReferenceEditDistribution(action_dim=3, edit_scale=0.02).proba_distribution(
        inputs["mean_actions"],
        inputs["log_std"] + 3.0,
        reference_action=inputs["reference_action"],
    )
    for _ in range(20):
        assert ((dist.sample() - inputs["reference_action"]).abs() <= 0.02 + 1e-6).all()


def _resolved(algorithm: str):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(
            config_name="train",
            overrides=[
                "base_model=mock",
                "simulator=g1_arena",
                f"algorithm={algorithm}",
            ],
        )


# The defaults that made SAC-edit learn on G1 Factory (README, "Algorithm zoo on G1 Factory").
FAST_SAC_EDIT_SAFE_DEFAULTS = {
    "algorithm.learning_starts": 4000,
    "algorithm.use_amp": False,
    "algorithm.clip_grad_norm": 1.0,
    "direct_action.actor_edit_scale": 0.02,
}

# The RFS recipe behind the README zoo table; the variants are explicit overrides.
RFS_RECIPE = {
    "rfs.residual_action_bound": 0.01,
    "rfs.proposal_conditioned_residual": False,
    "rfs.reference_action_conditioning": False,
    "dsrl.noise_action_bound": 1.0,
    "algorithm.learning_starts": 1200,
    "algorithm.actor_optimizer_lr": 5e-5,
    "algorithm.critic_optimizer_lr": 1e-4,
    "algorithm.batch_size": 64,
    "algorithm.gamma": 0.995,
    # the two critic settings where RFS deliberately differs from dsrl_sac.yaml (mean / true)
    "algorithm.critic_target_reduction": "min",
    "algorithm.pooled_critic_features": False,
}


def _select(cfg, dotted: str):
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


@pytest.mark.parametrize(
    "algorithm,pins", [("sac_edit", FAST_SAC_EDIT_SAFE_DEFAULTS), ("rfs", RFS_RECIPE)]
)
def test_zoo_defaults_are_pinned(algorithm, pins):
    cfg = _resolved(algorithm)
    drift = {k: (_select(cfg, k), v) for k, v in pins.items() if _select(cfg, k) != v}
    assert not drift, f"{algorithm} defaults drifted (actual, pinned): {drift}"


def _log_prob_of_unit_gaussian_edits(
    edit_scale: float, dims: int = 3, n: int = 4096
) -> torch.Tensor:
    torch.manual_seed(0)
    dist = ReferenceEditDistribution(
        action_dim=dims, edit_scale=edit_scale
    ).proba_distribution(
        torch.zeros(n, dims),
        torch.zeros(n, dims),
        reference_action=torch.zeros(n, dims),
    )
    return dist.log_prob(dist.sample(), dist.gaussian_actions)


def test_edit_density_is_measured_in_normalized_edit_units():
    """The log-density must not carry the -D*log(edit_scale) constant: with it, a 16x35 chunk at
    edit_scale 0.02 sits ~2200 nats above any reachable entropy target and alpha runs away."""
    tight = _log_prob_of_unit_gaussian_edits(0.02)
    wide = _log_prob_of_unit_gaussian_edits(0.5)
    assert torch.allclose(tight, wide), "log-density depends on edit_scale"
    entropy_per_dim = float(-tight.mean()) / 3
    assert -2.0 < entropy_per_dim < 2.0, (
        entropy_per_dim
    )  # a target of 0 nats is reachable


def test_edit_entropy_can_exceed_zero():
    """Some edit width has positive entropy, so target_entropy=0 lies inside the reachable range
    (a saturated squashed Gaussian has negative entropy, a unit-width one is close to uniform)."""
    torch.manual_seed(0)
    dims, n = 3, 4096
    entropies = []
    for log_std in (-1.0, -0.5, 0.0, 0.5):
        dist = ReferenceEditDistribution(
            action_dim=dims, edit_scale=0.02
        ).proba_distribution(
            torch.zeros(n, dims),
            torch.full((n, dims), log_std),
            reference_action=torch.zeros(n, dims),
        )
        entropies.append(
            float(-dist.log_prob(dist.sample(), dist.gaussian_actions).mean())
        )
    assert max(entropies) > 0.0, entropies


def test_reward_scaling_keeps_one_return_per_episode():
    """Decisions of two environments arrive interleaved; the scaler's statistics match the
    same episodes fed one after the other."""
    episode_a = ([-1.0, -2.0, -3.0], [False, False, True])
    episode_b = ([-0.5, -0.25], [False, True])

    def scaler():
        return AdaptiveRewardScaler(gamma=0.9, support_g_max=5.0)

    sequential = scaler()
    for episode_id, (rewards, dones) in enumerate((episode_a, episode_b)):
        sequential.update(
            rewards, dones, [False] * len(rewards), [episode_id] * len(rewards)
        )
    interleaved = scaler()
    interleaved.update([-1.0, -0.5], [False, False], [False, False], [0, 1])
    interleaved.update([-0.25, -2.0], [True, False], [False, False], [1, 0])
    interleaved.update([-3.0], [True], [False], [0])

    # The same returns in another order: the running moments agree up to rounding.
    for name, value in sequential.state_dict().items():
        assert torch.allclose(interleaved.state_dict()[name], value), name
    expected_returns = [-1.0, -2.9, -5.61, -0.5, -0.7]
    assert sequential.return_count.item() == len(expected_returns)
    assert sequential.return_mean.item() == pytest.approx(sum(expected_returns) / 5)


def test_reward_scaler_loads_a_checkpoint_from_before_per_episode_returns():
    """Checkpoints that kept one shared running return still load; their statistics carry over."""
    saved = AdaptiveRewardScaler(gamma=0.9, support_g_max=5.0)
    saved.update([1.0, 2.0], [False, True], [False, False], ["a", "a"])
    parent = torch.nn.Module()
    parent.reward_scaler = AdaptiveRewardScaler(gamma=0.9, support_g_max=5.0)
    old_state = {
        f"reward_scaler.{key}": value for key, value in saved.state_dict().items()
    }
    old_state["reward_scaler.discounted_return"] = torch.tensor(
        0.5, dtype=torch.float64
    )

    parent.load_state_dict(old_state, strict=True)

    assert torch.equal(parent.reward_scaler.return_max, saved.return_max)
    assert torch.equal(parent.reward_scaler.return_count, saved.return_count)
