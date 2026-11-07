# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two schematics of what the runtime stages change, drawn with matplotlib as text-mode SVG.

    python -m example_workflows.dgx_example.schematics --out figures/

``schedule_schematic.svg``
    the sync schedule (collect a round, train on it, publish) against the overlapped schedule
    (the next round collects while the learner trains): what an iteration costs and where the
    policy lag comes from;
``servers_schematic.svg``
    one Arena server stepping all N environments against two servers with N/2 each, one rollout
    worker per server: how the two fall out of phase so the simulator card stays busy while the
    other server's frames decode.

The proportions are the DGX Station's split at N = 256 (``results/throughput_schedules.json``,
``results/throughput_dgx_station.json``): a 390 s round and a 430 s burst of 310 SAC updates, a
simulator step of 0.54 s for 256 environments and 0.30 s for 128, a decode of 2.3 s for 256
environments and 1.2 s for 128. The two-server panel is the intended steady state, not a measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCHEMATICS = ("schedule_schematic.svg", "servers_schematic.svg")
INK, INK2 = "#1A1A1A", "#5A5A5A"
ROUND, BURST, PUBLISH, IDLE = "#3FA0FF", "#FF8C42", "#1A1A1A", "#E4E7EA"
STEP_A, STEP_B, DECODE_A, DECODE_B = "#3FA0FF", "#A4D65E", "#1F5FA8", "#5E8C2E"

#: The sync split at N = 256: the round, the burst of 310 updates, the publish.
ROUND_S, BURST_S, PUBLISH_S = 390.0, 430.0, 5.0
#: Under the overlapped schedule the round in flight shares the GB300 with the burst, so its decode slows and
#: the round takes about 440 s; under RLinf the burst keeps its length.
OVERLAPPED_ROUND_S = 440.0
#: One policy decision: eight simulator steps, then the decode of the next chunk.
STEPS_PER_DECISION = 8
STEP_256_S, DECODE_256_S = 0.54, 2.3
STEP_128_S, DECODE_128_S = 0.30, 1.2

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "svg.hashsalt": "dgx_example",
    }
)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, format="svg", metadata={"Date": None}, bbox_inches="tight")
    plt.close(fig)


def _timeline_axes(ax: plt.Axes, rows: list[str], xmax: float, xlabel: str) -> None:
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _bar(
    ax: plt.Axes,
    row: int,
    start: float,
    length: float,
    color: str,
    label: str = "",
    min_label: float = 0.0,
) -> None:
    """One block on a timeline row; the label is dropped on a block too short to hold it."""
    ax.broken_barh(
        [(start, length)],
        (row - 0.32, 0.64),
        facecolors=color,
        edgecolors="white",
        linewidth=0.8,
    )
    if label and length >= min_label:
        ink = INK2 if color == IDLE else "white"
        ax.text(
            start + length / 2,
            row,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color=ink,
        )


def _publish(ax: plt.Axes, at: float, rows: int) -> None:
    ax.vlines(at, -0.45, rows - 0.55, color=PUBLISH, linewidth=1.6)
    ax.text(at, -0.5, "publish", ha="center", va="bottom", fontsize=8, color=INK)


def schedule_schematic(out: Path) -> Path:
    """Sync against overlapped over two iterations, both starting with a first round collected alone."""
    fig, (sync, overlapped) = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True)
    rows = ["collection\n(simulator + decode)", "learner\n(SAC burst)"]
    xmax = 2 * (ROUND_S + BURST_S + PUBLISH_S) + 60

    _timeline_axes(sync, rows, xmax, "")
    t = 0.0
    for k in (1, 2):
        _bar(sync, 0, t, ROUND_S, ROUND, f"round {k}")
        _bar(sync, 1, t + ROUND_S, BURST_S, BURST, f"burst {k}")
        _publish(sync, t + ROUND_S + BURST_S + PUBLISH_S, len(rows))
        t += ROUND_S + BURST_S + PUBLISH_S
    _bar(sync, 0, t, min(ROUND_S, xmax - t), ROUND, "round 3", min_label=120)
    sync.set_title(
        f"sync (the reference recipe): one iteration = round + burst = {ROUND_S + BURST_S:.0f} s; "
        "while one side works, the other waits",
        loc="left",
        fontsize=10,
    )

    _timeline_axes(
        overlapped, rows, xmax, "time (s), the DGX Station's split at N = 256"
    )
    _bar(overlapped, 0, 0, ROUND_S, ROUND, "round 1")
    t = ROUND_S
    for k in (1, 2):
        iteration = max(OVERLAPPED_ROUND_S, BURST_S) + PUBLISH_S
        _bar(overlapped, 0, t, OVERLAPPED_ROUND_S, ROUND, f"round {k + 1}")
        _bar(overlapped, 1, t, BURST_S, BURST, f"burst {k}")
        _publish(overlapped, t + iteration, len(rows))
        t += iteration
    _bar(
        overlapped,
        0,
        t,
        min(OVERLAPPED_ROUND_S, xmax - t),
        ROUND,
        "round 4",
        min_label=120,
    )
    _bar(overlapped, 1, t, min(BURST_S, xmax - t), BURST, "burst 3", min_label=120)
    overlapped.set_title(
        f"overlapped: round k+1 collects while burst k trains; one iteration = the longer of the two "
        f"= {max(OVERLAPPED_ROUND_S, BURST_S) + PUBLISH_S:.0f} s\n"
        "the publish waits for the round in flight, then hands over the new weights: round k+1 was "
        "collected with the weights from before burst k (policy lag 1)",
        loc="left",
        fontsize=10,
    )
    fig.legend(
        handles=[
            Patch(color=ROUND, label="collection round"),
            Patch(color=BURST, label="SAC burst"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = out / SCHEMATICS[0]
    _save(fig, path)
    return path


def servers_schematic(out: Path) -> Path:
    """One server against two servers within a round, two policy decisions of each."""
    fig, (one, two) = plt.subplots(
        2, 1, figsize=(11, 5.2), sharex=True, height_ratios=[2, 3]
    )
    stepping_256 = STEPS_PER_DECISION * STEP_256_S
    cycle_256 = stepping_256 + DECODE_256_S
    xmax = 2 * cycle_256 + 0.3

    rows_one = ["RTX Pro 6000\nserver, 256 envs", "GB300\ndecode, 256 envs"]
    _timeline_axes(one, rows_one, xmax, "")
    for k in range(2):
        t = k * cycle_256
        _bar(one, 0, t, stepping_256, STEP_A, f"8 steps × {STEP_256_S} s")
        _bar(one, 0, t + stepping_256, DECODE_256_S, IDLE, "idle")
        _bar(
            one, 1, t + stepping_256, DECODE_256_S, DECODE_A, f"decode {DECODE_256_S} s"
        )
    one.set_title(
        f"one server: the simulator waits {DECODE_256_S / cycle_256:.0%} of every cycle for the decode "
        f"({cycle_256:.1f} s per decision of 256 environments; measured 246 env-steps/s under RLinf)",
        loc="left",
        fontsize=10,
    )

    stepping_128 = STEPS_PER_DECISION * STEP_128_S
    cycle_128 = stepping_128 + DECODE_128_S
    offset = cycle_128 / 2
    rows_two = [
        "RTX Pro 6000\nserver A, 128 envs",
        "RTX Pro 6000\nserver B, 128 envs",
        "GB300\ndecode A / B",
    ]
    _timeline_axes(two, rows_two, xmax, "time within a round (s)")
    decisions = int(xmax / cycle_128) + 2
    for k in range(decisions):
        for row, start, step_color, decode_color, name in (
            (0, k * cycle_128, STEP_A, DECODE_A, "A"),
            (1, offset + k * cycle_128, STEP_B, DECODE_B, "B"),
        ):
            if start >= xmax:
                continue
            _bar(
                two,
                row,
                start,
                min(stepping_128, xmax - start),
                step_color,
                f"8 × {STEP_128_S} s",
            )
            decode_at = start + stepping_128
            if decode_at < xmax:
                _bar(two, row, decode_at, min(DECODE_128_S, xmax - decode_at), IDLE, "")
                _bar(
                    two,
                    2,
                    decode_at,
                    min(DECODE_128_S, xmax - decode_at),
                    decode_color,
                    name,
                )
    two.set_title(
        "two servers, one rollout worker each: half a cycle apart, one server steps while the other's "
        f"frames decode ({cycle_128:.1f} s per decision of 128 environments each; the intended steady state)\n"
        "whichever worker gets ahead finds the GB300 free for its decode, so the two drift apart until they "
        "alternate; the round ends when both servers have finished",
        loc="left",
        fontsize=10,
    )
    fig.legend(
        handles=[
            Patch(color=STEP_A, label="server A steps"),
            Patch(color=STEP_B, label="server B steps"),
            Patch(color=DECODE_A, label="decode for A"),
            Patch(color=DECODE_B, label="decode for B"),
            Patch(color=IDLE, label="waiting"),
        ],
        loc="upper right",
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out / SCHEMATICS[1]
    _save(fig, path)
    return path


def draw_all(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    return [schedule_schematic(out), servers_schematic(out)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="directory the SVGs go to"
    )
    args = parser.parse_args(argv)
    for path in draw_all(args.out):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
