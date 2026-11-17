# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Placement follows the simulator entry: what each registered simulator may run, and what a plan says."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from example_workflows.dgx_example import layouts
from isaac_rl_posttraining.envs.providers import (
    available_simulator_entries,
    get_simulator_entry,
)

WORKFLOW = Path(__file__).resolve().parent.parent
REPO_ROOT = WORKFLOW.parent.parent


def test_a_served_simulator_runs_both_layouts_and_an_in_process_one_only_single() -> (
    None
):
    arena = layouts.capabilities("arena")
    assert arena.entry == "g1_arena" and arena.served
    assert arena.shapes == ("single", "split")
    assert arena.server_url_variables == ("ARENA_SERVER_URL", "ARENA_EVAL_SERVER_URL")
    assert arena.batched_collection
    for segment in ("isaaclab", "libero"):
        caps = layouts.capabilities(segment)
        assert not caps.served and caps.shapes == ("single",), segment
    # Isaac Lab clones its scenes in one simulation; LIBERO runs one simulator per worker.
    assert layouts.capabilities("isaaclab").batched_collection
    assert not layouts.capabilities("libero").batched_collection


def test_the_capability_table_covers_every_registered_entry() -> None:
    rows = layouts.all_capabilities()
    listed = {segment for caps in rows for segment in caps.segments}
    assert listed == set(available_simulator_entries())
    assert [caps.entry for caps in rows] == sorted(
        {get_simulator_entry(segment).name for segment in available_simulator_entries()}
    )
    table = layouts.capabilities_table()
    for caps in rows:
        assert f"`{caps.entry}`" in table


def test_readme_capability_table_is_current() -> None:
    """Checked in a fresh interpreter: other test modules register entries of their own in this process."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "example_workflows.dgx_example.layouts",
            "capabilities",
            "--check",
            str(WORKFLOW / "README.md"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"{result.stdout}{result.stderr}\n"
        "run python -m example_workflows.dgx_example.layouts capabilities --write README.md"
    )


def test_the_same_card_is_single_and_two_cards_are_split() -> None:
    single = layouts.plan("arena", 64, "GPU-a", "GPU-a")
    assert single.shape == "single" and single.train_server_url is None
    split = layouts.plan("arena", 64, "GPU-a", "GPU-b", port=7310)
    assert split.shape == "split" and split.train_server_url == "tcp://127.0.0.1:7310"
    assert split.values()["DGX_TRAIN_SERVER_URL_VARIABLE"] == "ARENA_SERVER_URL"


def test_batched_collection_and_its_override_follow_the_entry() -> None:
    assert layouts.plan("arena", 64, "0", "0").hydra_overrides == (
        "training.num_envs=64",
        "training.rollout_count_by=episode",
    )
    assert layouts.plan("arena", 1, "0", "0").hydra_overrides == (
        "training.num_envs=1",
    )
    unbatched = layouts.plan("libero", 8, "0", "0")
    assert not unbatched.batched
    assert unbatched.hydra_overrides == ("training.num_envs=8",)
    assert "DGX_TRAIN_SERVER_URL_VARIABLE" not in unbatched.values()


def test_several_servers_take_consecutive_ports_with_one_batched_worker_each() -> None:
    two = layouts.plan("arena", 64, "GPU-a", "GPU-b", port=7300, num_servers=2)
    assert two.num_servers == 2 and two.batched
    assert two.train_server_url == "tcp://127.0.0.1:7300,tcp://127.0.0.1:7301"
    assert two.values()["DGX_NUM_SERVERS"] == "2"
    assert layouts.plan("arena", 64, "0", "0").values()["DGX_NUM_SERVERS"] == "1"
    # The entry decides: Arena splits the environments equally, an in-process simulator has one simulation.
    with pytest.raises(ValueError, match="do not divide"):
        layouts.plan("arena", 9, "0", "1", num_servers=2)
    with pytest.raises(ValueError, match="one simulator"):
        layouts.plan("isaaclab", 8, "0", "0", num_servers=2)
    with pytest.raises(ValueError, match="num_servers"):
        layouts.plan("arena", 8, "0", "0", num_servers=0)


def test_a_split_needs_a_simulator_in_its_own_process() -> None:
    with pytest.raises(ValueError, match="same GPU for the single layout"):
        layouts.plan("isaaclab", 8, "0", "1")
    with pytest.raises(ValueError, match="same GPU for the single layout"):
        layouts.plan("libero", 2, "0", "1")


def test_plan_rejects_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="num_envs"):
        layouts.plan("arena", 0, "0", "0")
    with pytest.raises(ValueError, match="name a card"):
        layouts.plan("arena", 8, "", "0")
    with pytest.raises(ValueError, match="port"):
        layouts.plan("arena", 8, "0", "0", port=0)
    with pytest.raises(KeyError, match="no simulator entry"):
        layouts.plan("nope", 8, "0", "0")


def test_shell_lines_evaluate_in_bash() -> None:
    plan = layouts.plan("arena", 128, "GPU-sim", "GPU-learner")
    script = (
        "\n".join(plan.shell_lines())
        + '\necho "$DGX_SHAPE|$DGX_NUM_ENVS|$DGX_HYDRA_OVERRIDES|$DGX_TRAIN_SERVER_URL"'
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == (
        "split|128|training.num_envs=128 training.rollout_count_by=episode|tcp://127.0.0.1:7300"
    )


def test_cli_prints_a_plan_and_reports_a_refusal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        layouts.main(
            [
                "plan",
                "--simulator",
                "arena",
                "--num-envs",
                "8",
                "--simulator-gpu",
                "1",
                "--learner-gpu",
                "1",
            ]
        )
        == 0
    )
    assert "shape: single" in capsys.readouterr().out
    assert (
        layouts.main(
            [
                "plan",
                "--simulator",
                "libero",
                "--num-envs",
                "8",
                "--simulator-gpu",
                "0",
                "--learner-gpu",
                "1",
            ]
        )
        == 2
    )
    assert "cannot place this run" in capsys.readouterr().err
    assert layouts.main(["capabilities"]) == 0
    assert "| Simulator (`env.env_name` segment) |" in capsys.readouterr().out
