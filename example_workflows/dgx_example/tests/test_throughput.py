# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The throughput results: well-formed, consistent with the registry, rendered into the README, drawn."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from example_workflows.dgx_example import throughput
from example_workflows.dgx_example.readme_blocks import block_matches
from isaac_rl_posttraining.envs.providers import available_simulator_entries

WORKFLOW = Path(__file__).resolve().parent.parent
THROUGHPUT = WORKFLOW / "results" / "throughput.json"
DGX_STATION = WORKFLOW / "results" / "throughput_dgx_station.json"
SCHEDULES = WORKFLOW / "results" / "throughput_schedules.json"


def test_throughput_json_is_well_formed_and_rendered() -> None:
    results = throughput.load_throughput(THROUGHPUT)
    assert results.rows and results.layouts and results.gpus
    assert block_matches(
        WORKFLOW / "README.md", throughput.THROUGHPUT_BLOCK, throughput.render(results)
    ), (
        "run python -m example_workflows.dgx_example.throughput results/throughput.json --write README.md"
    )


def test_dgx_station_results_are_well_formed_and_rendered() -> None:
    """The DGX Station's layouts: their own file and block."""
    results = throughput.load_throughput(DGX_STATION)
    assert results.split_layouts, "the DGX Station set carries the split layout"
    assert block_matches(
        WORKFLOW / "README.md", "throughput_dgx_station", throughput.render(results)
    ), (
        "run python -m example_workflows.dgx_example.throughput results/throughput_dgx_station.json "
        "--write README.md --block throughput_dgx_station"
    )
    # A split's speed-up is over the single cards of its own runtime; the RLinf splits, with no
    # RLinf single-card layout in this set, get no speed-up block and are no column of the native one.
    speedup = throughput.render_speedup_table(results)
    assert "RLinf runtime" not in speedup, speedup
    assert speedup.count("over each single card") == 1, speedup
    servers = throughput.render_servers_table(results)
    assert servers.count("| GB300 DGX Station (both GPUs), RLinf runtime | ") == 2, (
        servers
    )


def test_schedule_results_are_well_formed_and_rendered() -> None:
    """The split at N = 256 on both schedules and both runtimes: every overlapped layout has its sync
    counterpart, every row the whole-iteration rate, and the README block is what the file renders."""
    results = throughput.load_throughput(SCHEDULES)
    assert block_matches(
        WORKFLOW / "README.md", "throughput_schedules", throughput.render(results)
    ), (
        "run python -m example_workflows.dgx_example.throughput results/throughput_schedules.json "
        "--write README.md --block throughput_schedules"
    )
    assert len(results.rows) == 5 and results.env_counts == [256]
    assert len(results.schedule_layouts) == 3
    for layout in results.schedule_layouts:
        if layout.simulator_servers == 1:
            sync = results.sync_counterpart(layout)
            assert (
                sync is not None and sync.rollout_runtime == layout.rollout_runtime
            ), layout
        else:
            # The two-server overlapped run is measured against the one-server overlapped run.
            assert results.one_server_counterpart(layout) is not None, layout
    for row in results.rows:
        assert row.iteration_rate is not None and row.collect_s is not None, row
        assert row.train_s is not None and row.publish_s is not None, row
    # The burst is sized to the round, which is what gives the schedule something to hide.
    assert results.recipe["max_updates_per_burst"] == 310
    table = throughput.render_schedule_table(results)
    assert table.count("GB300 DGX Station (both GPUs)") == 2, table
    # The runtime comparison keeps to the sync pair: a round started ahead has no clean collection rate.
    runtime_table = throughput.render_rollout_runtime_table(results)
    assert runtime_table.count("| GB300 DGX Station (both GPUs) |") == 1, runtime_table
    # No single-card layout here, so no speed-up block and no figure of it.
    assert throughput.render_speedup_table(results) == ""


def test_every_rlinf_row_has_its_native_counterpart() -> None:
    """An RLinf-runtime row's baseline is the native row on the same cards and N, from another run."""
    results = throughput.load_throughput(THROUGHPUT)
    for row in results.rows:
        layout = results.layout(row.layout)
        if layout.rollout_runtime == "native":
            continue
        native = results.native_counterpart(layout)
        assert native is not None, row
        baseline = results.row(native.key, row.num_envs)
        assert baseline is not None and baseline.rate is not None, row
        assert row.wandb_run != baseline.wandb_run, row


def test_rows_name_registered_simulators_and_carry_their_provenance() -> None:
    results = throughput.load_throughput(THROUGHPUT)
    segments = set(available_simulator_entries())
    for row in results.rows:
        assert row.simulator in segments, row
        if row.status == "finished":
            assert row.rate is not None and row.wandb_run, row
            assert row.env_step_s is not None and row.decode_s is not None, row
        else:
            assert row.rate is None and row.wandb_run is None, row


def test_speedup_table_divides_the_split_by_each_single_card() -> None:
    results = throughput.load_throughput(DGX_STATION)
    split = results.split_layouts[0]
    single = next(layout for layout in results.layouts if layout.shape == "single")
    num_envs = results.series(split.key)[-1].num_envs
    numerator = results.row(split.key, num_envs)
    denominator = results.row(single.key, num_envs)
    assert numerator is not None and numerator.rate is not None
    assert denominator is not None and denominator.rate is not None
    expected = f"{numerator.rate.mean / denominator.rate.mean:.2f}×"
    table = throughput.render_speedup_table(results)
    row = next(
        line for line in table.splitlines() if line.startswith(f"| {num_envs} |")
    )
    assert expected in row


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda data: data["rows"][0].pop("env_steps_per_s"), "finished row needs"),
        (
            lambda data: data["rows"][0].__setitem__("layout", "nowhere"),
            "unknown layout",
        ),
        (lambda data: data["rows"][0].__setitem__("status", "crashed"), "status"),
        (lambda data: data["rows"].append(dict(data["rows"][0])), "duplicate row"),
        (lambda data: data["recipe"].pop("task"), "recipe lacks"),
        (
            lambda data: data["layouts"]["gb300_only"].__setitem__("shape", "split"),
            "two cards",
        ),
        (
            lambda data: data["layouts"]["gb300_only"].pop("rollout_runtime"),
            "rollout_runtime",
        ),
        (
            lambda data: data["layouts"]["gb300_only"].__setitem__(
                "rollout_runtime", "ray"
            ),
            "rollout_runtime must be one of",
        ),
        (lambda data: data["layouts"]["gb300_only"].pop("schedule"), "schedule"),
        (
            lambda data: data["layouts"]["gb300_only"].__setitem__(
                "schedule", "lookahead"
            ),
            "schedule must be one of",
        ),
        (
            lambda data: data["rows"][0].__setitem__(
                "iteration_env_steps_per_s", {"mean": 0, "std": 0, "cycles": 3}
            ),
            "mean must be positive",
        ),
        (
            lambda data: data["layouts"]["gb300_only"].pop("simulator_servers"),
            "simulator_servers",
        ),
        (
            lambda data: data["layouts"]["gb300_only"].__setitem__(
                "simulator_servers", 0
            ),
            "simulator_servers",
        ),
    ],
)
def test_loader_names_the_offending_field(tmp_path: Path, mutate, message: str) -> None:
    data = json.loads(THROUGHPUT.read_text(encoding="utf-8"))
    mutate(data)
    bad = tmp_path / "throughput.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        throughput.load_throughput(bad)


def _with_rlinf_layout(tmp_path: Path) -> Path:
    """The single-card file plus the L40 layout measured again with the RLinf runtime placing the rollout worker."""
    data = json.loads(THROUGHPUT.read_text(encoding="utf-8"))
    data["layouts"]["l40_only_rlinf"] = {
        **data["layouts"]["l40_only"],
        "label": "L40 only, RLinf runtime",
        "rollout_runtime": "rlinf",
    }
    for row in [row for row in data["rows"] if row["layout"] == "l40_only"][:2]:
        rlinf_row = dict(row)
        rlinf_row["layout"] = "l40_only_rlinf"
        rlinf_row["wandb_run"] = f"rlinf{row['num_envs']}"
        rlinf_row["env_steps_per_s"] = {
            **row["env_steps_per_s"],
            "mean": row["env_steps_per_s"]["mean"] * 0.95,
        }
        data["rows"].append(rlinf_row)
    path = tmp_path / "throughput.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _with_overlapped_layout(tmp_path: Path) -> Path:
    """The single-card file plus the L40 layout measured again on the overlapped schedule, both with
    the whole-iteration rate and walls the trainer logs."""
    data = json.loads(THROUGHPUT.read_text(encoding="utf-8"))
    data["layouts"]["l40_only_overlapped"] = {
        **data["layouts"]["l40_only"],
        "label": "L40 only, overlapped schedule",
        "schedule": "overlapped",
    }
    for row in [row for row in data["rows"] if row["layout"] == "l40_only"][:2]:
        rate = row["env_steps_per_s"]["mean"]
        row["iteration_env_steps_per_s"] = {"mean": rate / 2, "std": 1.0, "cycles": 3}
        row["collect_s"], row["train_s"], row["publish_s"] = 400.0, 400.0, 0.1
        overlapped_row = dict(row)
        overlapped_row["layout"] = "l40_only_overlapped"
        overlapped_row["wandb_run"] = f"ahead{row['num_envs']}"
        overlapped_row["iteration_env_steps_per_s"] = {
            "mean": rate * 0.9,
            "std": 1.0,
            "cycles": 3,
        }
        (
            overlapped_row["collect_s"],
            overlapped_row["train_s"],
            overlapped_row["publish_s"],
        ) = 405.0, 400.0, 5.0
        data["rows"].append(overlapped_row)
    path = tmp_path / "throughput.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _with_two_servers_layout(tmp_path: Path) -> Path:
    """The single-card file plus the L40 layout with its environments spread over two servers."""
    data = json.loads(THROUGHPUT.read_text(encoding="utf-8"))
    data["layouts"]["l40_only_2srv"] = {
        **data["layouts"]["l40_only"],
        "label": "L40 only, two servers",
        "simulator_servers": 2,
    }
    for row in [row for row in data["rows"] if row["layout"] == "l40_only"][:2]:
        two = dict(row)
        two["layout"] = "l40_only_2srv"
        two["wandb_run"] = f"twosrv{row['num_envs']}"
        two["env_steps_per_s"] = {
            **row["env_steps_per_s"],
            "mean": row["env_steps_per_s"]["mean"] * 1.4,
        }
        data["rows"].append(two)
    path = tmp_path / "throughput.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_servers_table_pairs_a_several_server_layout_with_its_one_server_counterpart(
    tmp_path: Path,
) -> None:
    results = throughput.load_throughput(_with_two_servers_layout(tmp_path))
    two = results.layout("l40_only_2srv")
    assert results.server_layouts == [two]
    one = results.one_server_counterpart(two)
    assert one is not None and one.key == "l40_only"
    # A server count is neither a runtime nor a schedule: those comparisons hold it fixed.
    assert (
        results.native_counterpart(two) is two and results.sync_counterpart(two) is two
    )
    assert results.single_layouts_for(two) == [two]
    table = throughput.render_servers_table(results)
    lines = [line for line in table.splitlines() if line.startswith("| L40 only |")]
    assert len(lines) == 2 and all(
        "| 2 |" in line and "1.40×" in line for line in lines
    ), table
    rendered = throughput.render(results)
    assert (
        "several simulator servers" in rendered and "L40 only, two servers" in rendered
    )


def test_every_layout_declares_where_its_rollout_worker_runs_its_schedule_and_its_servers() -> (
    None
):
    for path in (THROUGHPUT, DGX_STATION, SCHEDULES):
        results = throughput.load_throughput(path)
        # A layout on another runtime is only meaningful next to the native layout on the same cards, and a
        # layout on several servers next to the one-server layout on the same cards, runtime and schedule.
        for layout in results.runtime_layouts:
            if layout.simulator_servers == 1:
                assert results.native_counterpart(layout) is not None, layout
        for layout in results.server_layouts:
            assert results.one_server_counterpart(layout) is not None, layout
    single = throughput.load_throughput(THROUGHPUT)
    assert (
        not single.server_layouts
        and "several simulator servers" not in throughput.render(single)
    )
    for path in (THROUGHPUT, DGX_STATION):
        results = throughput.load_throughput(path)
        assert not results.schedule_layouts, "every layout here ran the sync schedule"
        assert "overlapped schedule" not in throughput.render(results)


def test_schedule_table_pairs_an_overlapped_layout_with_its_sync_counterpart(
    tmp_path: Path,
) -> None:
    results = throughput.load_throughput(_with_overlapped_layout(tmp_path))
    overlapped = results.layout("l40_only_overlapped")
    assert results.schedule_layouts == [overlapped]
    sync = results.sync_counterpart(overlapped)
    assert sync is not None and sync.key == "l40_only"
    # A schedule is not a runtime: the native counterpart of an overlapped layout must run overlapped too.
    assert results.native_counterpart(overlapped) is overlapped
    assert results.single_layouts_for(overlapped) == [overlapped]
    table = throughput.render_schedule_table(results)
    lines = [line for line in table.splitlines() if line.startswith("| L40 only |")]
    assert len(lines) == 2, table
    assert all(
        "1.80×" in line and "| 400 / 400 / 0 | 405 / 400 / 5 |" in line
        for line in lines
    ), table
    assert all(line.count(" / ") == 6 for line in lines), (
        table
    )  # two wall triples, a decode pair, a SAC update pair
    rendered = throughput.render(results)
    assert (
        "overlapped schedule" in rendered
        and "L40 only, overlapped schedule" in rendered
    )
    # The rows without the whole-iteration rate (older runs) are left out of the table, not rendered as zeros.
    for row in results.series("l40_only")[2:]:
        assert row.iteration_rate is None


def test_rollout_runtime_table_pairs_a_runtime_layout_with_its_native_counterpart(
    tmp_path: Path,
) -> None:
    results = throughput.load_throughput(_with_rlinf_layout(tmp_path))
    rlinf = results.layout("l40_only_rlinf")
    assert rlinf in results.runtime_layouts
    native = results.native_counterpart(rlinf)
    assert native is not None and native.key == "l40_only"
    assert results.native_counterpart(native) is native
    table = throughput.render_rollout_runtime_table(results)
    lines = [line for line in table.splitlines() if line.startswith("| L40 only |")]
    assert len(lines) == 2, table
    assert all("0.95×" in line for line in lines), table
    rendered = throughput.render(results)
    assert "RLinf runtime" in rendered and "L40 only, RLinf runtime" in rendered


def test_schematics_draw_and_are_the_ones_the_readme_shows(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from example_workflows.dgx_example import schematics

    drawn = schematics.draw_all(tmp_path)
    assert [path.name for path in drawn] == list(schematics.SCHEMATICS)
    readme = (WORKFLOW / "README.md").read_text(encoding="utf-8")
    for path in drawn:
        assert path.read_text(encoding="utf-8").lstrip().startswith("<?xml")
        assert f"figures/{path.name}" in readme, path.name
        assert (WORKFLOW / "figures" / path.name).exists(), (
            "regenerate with python -m example_workflows.dgx_example.schematics --out figures/"
        )


def test_figures_draw_from_each_results_file(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from example_workflows.dgx_example import figures

    results = throughput.load_throughput(THROUGHPUT)
    drawn = figures.draw_all(results, tmp_path)
    expected = ["cycle_breakdown.svg", "gpu_memory.svg"]
    if results.runtime_layouts:
        expected.append(figures.RUNTIME_LAYER_FIGURE)
    assert [path.name for path in drawn] == expected
    dgx_station = figures.draw_all(
        throughput.load_throughput(DGX_STATION), tmp_path, prefix="dgx_station_"
    )
    assert [path.name for path in dgx_station] == [
        f"dgx_station_{name}" for name in figures.FIGURES
    ] + [f"dgx_station_{figures.RUNTIME_LAYER_FIGURE}"]
    for path in drawn + dgx_station:
        text = path.read_text(encoding="utf-8")
        assert text.lstrip().startswith("<?xml") and "<svg" in text
    with_rlinf = figures.draw_all(
        throughput.load_throughput(_with_rlinf_layout(tmp_path)), tmp_path / "rlinf"
    )
    assert [path.name for path in with_rlinf][-1] == figures.RUNTIME_LAYER_FIGURE
