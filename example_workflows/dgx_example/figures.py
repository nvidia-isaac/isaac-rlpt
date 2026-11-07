# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The workflow's figures, drawn from ``results/throughput.json`` with matplotlib as text-mode SVG.

    python -m example_workflows.dgx_example.figures results/throughput.json --out figures/

``throughput_speedup.svg``
    collection throughput against N for every layout, and the split layout's speed-up over each
    single card;
``cycle_breakdown.svg``
    what one policy decision costs per layout and N: the simulator steps it drives and the
    policy decode;
``gpu_memory.svg``
    peak GPU memory against N per card, the split's two cards apart;
``runtime_layer_comparison.svg``
    the collection rate with the rollout worker placed by the RLinf runtime against the native
    runtime on the same cards, drawn when the file has such a layout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from example_workflows.dgx_example.throughput import (
    Layout,
    Throughput,
    load_throughput,
)

INK, INK2, GRID = "#1A1A1A", "#5A5A5A", "#ECEFF1"
FIGURES = ("throughput_speedup.svg", "cycle_breakdown.svg", "gpu_memory.svg")
RUNTIME_LAYER_FIGURE = "runtime_layer_comparison.svg"
OFFSET_POINTS, UPPER_LEFT = "offset points", "upper left"

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        # A fixed salt keeps the SVG ids stable between runs, so a regenerated
        # figure only changes where the data did.
        "svg.hashsalt": "dgx_example",
    }
)


def _log2_axis(
    ax: plt.Axes, env_counts: list[int], xlabel: str = "environments N"
) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(env_counts)
    ax.set_xticklabels([str(n) for n in env_counts])
    ax.minorticks_off()
    ax.set_xlim(env_counts[0] / 1.5, env_counts[-1] * 1.4)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(xlabel)


def _weight(layout: Layout, results: Throughput) -> bool:
    """Station layouts (those on the split's cards) draw bold; other cards are references."""
    station_gpus = {
        gpu
        for split in results.split_layouts
        for gpu in (split.simulator_gpu, split.learner_gpu)
    }
    return layout.simulator_gpu in station_gpus


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def _annotate_points(
    ax: plt.Axes, points: list[tuple[int, float]], fmt: str, color: str, bold: bool
) -> None:
    for x, y in points:
        ax.annotate(
            fmt.format(y),
            (x, y),
            textcoords=OFFSET_POINTS,
            xytext=(0, 7),
            ha="center",
            fontsize=8.5 if bold else 7.5,
            color=color,
            fontweight="bold" if bold else "normal",
        )


def _plot_rates(ax: plt.Axes, results: Throughput) -> float:
    """Each layout's collection rate against N; returns the highest mean drawn."""
    top = 0.0
    for layout in results.layouts:
        rows = [row for row in results.series(layout.key) if row.rate is not None]
        if not rows:
            continue
        bold = _weight(layout, results)
        xs = [row.num_envs for row in rows]
        ys = [row.rate.mean for row in rows if row.rate is not None]
        errors = [row.rate.std for row in rows if row.rate is not None]
        top = max(top, max(ys))
        ax.errorbar(
            xs,
            ys,
            yerr=errors,
            color=layout.color,
            marker="o",
            markersize=6 if bold else 4.5,
            linewidth=2.2 if bold else 1.4,
            capsize=2,
            label=layout.label,
            zorder=3 if bold else 2,
        )
        _annotate_points(ax, list(zip(xs, ys)), "{:.0f}", layout.color, bold)
    return top


def _speedup_points(
    results: Throughput, split: Layout, layout: Layout
) -> list[tuple[int, float]]:
    """The split's rate over ``layout``'s at every N both measured."""
    points = []
    for row in results.series(layout.key):
        numerator = results.row(split.key, row.num_envs)
        if (
            row.rate is not None
            and numerator is not None
            and numerator.rate is not None
        ):
            points.append((row.num_envs, numerator.rate.mean / row.rate.mean))
    return points


def _plot_speedups(ax: plt.Axes, results: Throughput) -> None:
    splits = [
        split for split in results.split_layouts if results.single_layouts_for(split)
    ]
    for split in splits:
        for layout in results.single_layouts_for(split):
            points = _speedup_points(results, split, layout)
            if not points:
                continue
            bold = _weight(layout, results)
            counts, ratios = zip(*points)
            ax.plot(
                counts,
                ratios,
                color=layout.color,
                marker="o",
                markersize=6 if bold else 4.5,
                linewidth=2.2 if bold else 1.4,
                label=f"vs {layout.label}",
                zorder=3,
            )
            _annotate_points(ax, points, "{:.2f}×", layout.color, bold)
        ax.axhline(1.0, color=INK2, linewidth=1, linestyle=":")
        ax.set_title(
            f"{split.label}: speed-up over each single card", loc="left", fontsize=11
        )


def throughput_speedup(results: Throughput, out: Path, prefix: str = "") -> Path:
    env_counts = results.env_counts
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    top = _plot_rates(left, results)
    _log2_axis(left, env_counts)
    left.set_ylim(0, top * 1.18)
    left.set_ylabel("collection env-steps per second (mean ± std over cycles)")
    left.set_title("Collection throughput by layout", loc="left", fontsize=11)
    left.legend(frameon=False, fontsize=8, loc=UPPER_LEFT)

    _plot_speedups(right, results)
    _log2_axis(right, env_counts)
    right.set_ylabel("split throughput ÷ single-card throughput")
    right.legend(frameon=False, fontsize=8, loc=UPPER_LEFT)
    path = out / f"{prefix}{FIGURES[0]}"
    _save(fig, path)
    return path


def cycle_breakdown(results: Throughput, out: Path, prefix: str = "") -> Path:
    steps_per_decision = int(results.recipe["action_exec_len"])
    # With a split the station layouts carry the story; a single-card file shows every layout.
    if results.split_layouts:
        layouts = [layout for layout in results.layouts if _weight(layout, results)]
    else:
        layouts = list(results.layouts)
    env_counts = results.env_counts
    fig, ax = plt.subplots(figsize=(11, 4.6))
    width = 0.8 / max(len(layouts), 1)
    for index, layout in enumerate(layouts):
        for position, num_envs in enumerate(env_counts):
            row = results.row(layout.key, num_envs)
            if row is None or row.env_step_s is None or row.decode_s is None:
                continue
            x = position + (index - (len(layouts) - 1) / 2) * width
            steps = row.env_step_s * steps_per_decision
            ax.bar(
                x,
                steps,
                width * 0.9,
                color=layout.color,
                label=layout.label if position == 0 else None,
                zorder=3,
            )
            ax.bar(
                x,
                row.decode_s,
                width * 0.9,
                bottom=steps,
                color=layout.color,
                alpha=0.45,
                hatch="//",
                zorder=3,
            )
            ax.annotate(
                f"{steps + row.decode_s:.1f}",
                (x, steps + row.decode_s),
                textcoords=OFFSET_POINTS,
                xytext=(0, 3),
                ha="center",
                fontsize=8,
                color=INK,
            )
    ax.set_xticks(range(len(env_counts)))
    ax.set_xticklabels([f"N = {n}" for n in env_counts])
    ax.set_ylabel("seconds per decision")
    ax.set_title(
        f"Where one policy decision's time goes: {steps_per_decision} simulator steps (solid) "
        "and the policy decode (hatched)",
        loc="left",
        fontsize=11,
    )
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc=UPPER_LEFT)
    path = out / f"{prefix}{FIGURES[1]}"
    _save(fig, path)
    return path


def gpu_memory(results: Throughput, out: Path, prefix: str = "") -> Path:
    env_counts = results.env_counts
    fig, ax = plt.subplots(figsize=(7, 4.6))
    gpu_labels = {gpu.key: gpu.label for gpu in results.gpus}
    for layout in results.layouts:
        rows = results.series(layout.key)
        learner = [
            (row.num_envs, row.peak_gib_learner)
            for row in rows
            if row.peak_gib_learner is not None
        ]
        if not learner:
            continue
        if layout.shape == "single":
            series = [(f"{layout.label} (whole stack)", learner, "-")]
        else:
            simulator = [
                (row.num_envs, row.peak_gib_simulator)
                for row in rows
                if row.peak_gib_simulator is not None
            ]
            series = [
                (
                    f"{layout.label}: {gpu_labels[layout.learner_gpu]}, learner",
                    learner,
                    "-",
                ),
                (
                    f"{layout.label}: {gpu_labels[layout.simulator_gpu]}, simulator",
                    simulator,
                    "--",
                ),
            ]
        for label, points, style in series:
            counts, peaks = zip(*points)
            ax.plot(
                counts,
                peaks,
                color=layout.color,
                linestyle=style,
                marker="o",
                markersize=5,
                linewidth=2,
                label=label,
            )
            for x, y in points:
                ax.annotate(
                    f"{y:.0f}",
                    (x, y),
                    textcoords=OFFSET_POINTS,
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8,
                    color=layout.color,
                )
    _log2_axis(ax, env_counts)
    ax.set_ylabel("peak GPU memory (GiB)")
    ax.set_title("Peak GPU memory by layout", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc=UPPER_LEFT)
    path = out / f"{prefix}{FIGURES[2]}"
    _save(fig, path)
    return path


def rollout_runtime_comparison(
    results: Throughput, out: Path, prefix: str = ""
) -> Path:
    """The RLinf-placed rollout worker against the native runtime on the same cards, per layout and N.

    Sync-schedule layouts only, like the table: under the overlapped schedule the round shares the learner's card
    with the burst, so its rate is not the placement's alone.
    """
    pairs = [
        (native, layout)
        for layout in results.runtime_layouts
        for native in [results.native_counterpart(layout)]
        if native is not None and layout.schedule == "sync"
    ]
    fig, axes = plt.subplots(
        1, len(pairs), figsize=(5.5 * len(pairs), 4.4), squeeze=False
    )
    for ax, (native, layout) in zip(axes[0], pairs):
        rows = [
            (baseline, row)
            for row in results.series(layout.key)
            for baseline in [results.row(native.key, row.num_envs)]
            if row.rate is not None
            and baseline is not None
            and baseline.rate is not None
        ]
        positions = list(range(len(rows)))
        width = 0.38
        for offset, index, color, label, weight in (
            (-width / 2, 0, native.color, "native runtime", "normal"),
            (width / 2, 1, INK2, f"{layout.rollout_runtime} runtime", "bold"),
        ):
            series = [pair[index] for pair in rows]
            ax.bar(
                [p + offset for p in positions],
                [row.rate.mean for row in series if row.rate is not None],
                width,
                yerr=[row.rate.std for row in series if row.rate is not None],
                color=color,
                capsize=2,
                label=label,
                zorder=3,
            )
            for p, row in zip(positions, series):
                if row.rate is not None:
                    ax.annotate(
                        f"{row.rate.mean:.0f}",
                        (p + offset, row.rate.mean + row.rate.std),
                        textcoords=OFFSET_POINTS,
                        xytext=(0, 4),
                        ha="center",
                        fontsize=8,
                        color=INK,
                        fontweight=weight,
                    )
        ax.set_xticks(positions)
        ax.set_xticklabels([f"N = {row.num_envs}" for _, row in rows])
        ax.set_title(native.label, loc="left", fontsize=11)
        ax.set_ylabel("collection env-steps per second")
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc=UPPER_LEFT)
    fig.suptitle(
        "Same cards, same recipe: the rollout worker placed by the RLinf runtime against the native runtime",
        x=0.01,
        ha="left",
        fontsize=11.5,
    )
    path = out / f"{prefix}{RUNTIME_LAYER_FIGURE}"
    _save(fig, path)
    return path


def draw_all(results: Throughput, out: Path, prefix: str = "") -> list[Path]:
    """Every figure the file supports: the speed-up needs a split layout, each comparison its baseline."""
    out.mkdir(parents=True, exist_ok=True)
    drawn = []
    if results.split_layouts:
        drawn.append(throughput_speedup(results, out, prefix))
    drawn.append(cycle_breakdown(results, out, prefix))
    if any(row.peak_gib_learner is not None for row in results.rows):
        drawn.append(gpu_memory(results, out, prefix))
    if any(
        results.native_counterpart(layout) is not None and layout.schedule == "sync"
        for layout in results.runtime_layouts
    ):
        drawn.append(rollout_runtime_comparison(results, out, prefix))
    return drawn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("throughput", type=Path)
    parser.add_argument(
        "--out", type=Path, required=True, help="directory the SVGs are written to"
    )
    parser.add_argument("--prefix", default="", help="prefix for the SVG file names")
    args = parser.parse_args(argv)
    for path in draw_all(load_throughput(args.throughput), args.out, args.prefix):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
