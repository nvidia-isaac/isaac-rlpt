# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""zoo_recipes.sh resolves each validated condition to the documented overrides (dry run)."""

from __future__ import annotations

import os

import pytest

from example_workflows.conftest import TRAIN_LAUNCHER, dryrun_text, run_launcher_dryrun

ZOO_LAUNCHER = os.path.join(os.path.dirname(TRAIN_LAUNCHER), "zoo_recipes.sh")

PROTOCOL = [
    "launch.override=training.num_rollouts=40000",
    "launch.override=eval.eval_num_episodes=50",
    "launch.override=eval.eval_freq=5000",
    "launch.override=training.save_interval=5000",
    "launch.override=training.checkpoint_retention=best_last_final",
    "launch.override=eval.action_selection=sample",
    "launch.override=base_model.dataset_dir=null",
    "launch.override=base_model.task_instruction='Move the power drill from one table to another'",
    "launch.override=training.seed=1",
]
RFS = ["launch.override=algorithm=rfs", "launch.override=training.chunk_size=40"]
MODES = {
    "rfs": RFS,
    "rfs_pc": RFS
    + [
        "launch.override=rfs.proposal_conditioned_residual=true",
        "launch.override=rfs.reference_action_conditioning=true",
    ],
    "rfs_nb2": RFS + ["launch.override=dsrl.noise_action_bound=2.0"],
    "sac_edit": ["launch.override=algorithm=sac_edit"],
}


@pytest.mark.parametrize("mode", sorted(MODES))
def test_zoo_recipe_resolves_documented_overrides(launcher_sandbox, mode):
    result = run_launcher_dryrun(ZOO_LAUNCHER, [mode, "DrillPnP", 1], launcher_sandbox)
    assert result.returncode == 0, dryrun_text(result)
    lines = set(dryrun_text(result).splitlines())
    missing = [o for o in PROTOCOL + MODES[mode] if o not in lines]
    assert not missing, f"{mode}: missing overrides {missing}\n{dryrun_text(result)}"
    assert "plan.simulator=g1_arena" in lines


def test_zoo_recipe_variants_do_not_leak_across_modes(launcher_sandbox):
    """The plain `rfs` mode carries none of the variant knobs."""
    lines = set(
        dryrun_text(
            run_launcher_dryrun(ZOO_LAUNCHER, ["rfs", "DrillPnP"], launcher_sandbox)
        ).splitlines()
    )
    overrides = [line for line in lines if line.startswith("launch.override=")]
    assert not any(
        "launch.override=rfs." in line or "noise_action_bound" in line
        for line in overrides
    ), overrides
    assert not any("online_alg_name" in line for line in lines), (
        "the algorithm group is the only selector"
    )
    assert "launch.override=training.seed=0" in lines


def test_zoo_recipe_rejects_unknown_mode(launcher_sandbox):
    result = run_launcher_dryrun(ZOO_LAUNCHER, ["dsrl", "DrillPnP"], launcher_sandbox)
    assert result.returncode != 0
    assert "MODE must be" in dryrun_text(result)


def test_zoo_recipe_forwards_extra_overrides_after_the_recipe(launcher_sandbox):
    """Overrides after the seed reach the learner and come last, so they win over the recipe."""
    result = run_launcher_dryrun(
        ZOO_LAUNCHER,
        ["rfs_pc", "DrillPnP", 0, "training.save_interval=2500", "eval.eval_freq=2500"],
        launcher_sandbox,
    )
    assert result.returncode == 0, dryrun_text(result)
    lines = [
        line
        for line in dryrun_text(result).splitlines()
        if line.startswith("launch.override=")
    ]
    assert "launch.override=training.save_interval=2500" in lines
    assert lines.index("launch.override=eval.eval_freq=2500") > lines.index(
        "launch.override=eval.eval_freq=5000"
    )
