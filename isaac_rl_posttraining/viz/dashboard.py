# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render run summaries as one self-contained HTML dashboard (inline CSS and SVG, no external files)."""

from __future__ import annotations

import html
import math
from collections.abc import Sequence

import numpy as np

from isaac_rl_posttraining.viz.data import RunSummary, Series

__all__ = ["render_dashboard"]

#: Categorical colors in fixed order (one per run); runs past the eighth reuse none, see ``_color``.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
#: One color per timing component (fixed order, independent of the runs).
TIMING_COLORS = {"policy inference": "#2a78d6", "simulator": "#1baf7a", "learner": "#eb6834", "other": "#b4b2a9"}
WIDTH, HEIGHT = 480, 250
LEFT, RIGHT, TOP, BOTTOM = 64, 16, 16, 44
MAX_LEARNER_PLOTS = 12

_CSS = """
body { background: #ffffff; color: #0b0b0b; font: 14px/1.4 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
       margin: 24px; }
h1 { font-size: 20px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 28px 0 8px; }
p.note { color: #52514e; margin: 0 0 12px; max-width: 900px; }
.grid { display: flex; flex-wrap: wrap; gap: 16px; }
figure { margin: 0; border: 1px solid #e6e5e0; border-radius: 6px; padding: 8px; background: #fff; }
figcaption { font-weight: 600; margin: 0 0 4px 4px; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 8px 0 4px; }
.legend span.swatch { display: inline-block; width: 14px; height: 4px; border-radius: 2px; margin-right: 6px;
                      vertical-align: middle; }
table { border-collapse: collapse; margin: 4px 0; } td, th { padding: 4px 10px; text-align: right;
        border-bottom: 1px solid #ecebe6; } th { color: #52514e; font-weight: 600; }
td:first-child, th:first-child { text-align: left; }
svg text { font-size: 11px; fill: #52514e; } svg .empty { font-size: 13px; }
"""


def _color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def _ticks(low: float, high: float, count: int = 5) -> list[float]:
    if not math.isfinite(low) or not math.isfinite(high):
        return []
    if high <= low:
        return [low]
    raw = (high - low) / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    start = math.ceil(low / step) * step
    ticks = []
    value = start
    while value <= high + step * 1e-9:
        ticks.append(round(value, 12))
        value += step
    return ticks


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1e6:
        return f"{value / 1e6:g}M"
    if magnitude >= 1e4:
        return f"{value / 1e3:g}k"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 1:
        return f"{value:.3g}"
    return f"{value:.2g}"


def _frame(x_ticks: list[float], y_ticks: list[float], sx, sy, x_label: str, y_label: str) -> list[str]:
    parts = []
    for tick in y_ticks:
        y = sy(tick)
        parts.append(f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y:.1f}" y2="{y:.1f}" stroke="#ecebe6"/>')
        parts.append(f'<text x="{LEFT - 6}" y="{y + 4:.1f}" text-anchor="end">{_fmt(tick)}</text>')
    for tick in x_ticks:
        x = sx(tick)
        parts.append(f'<text x="{x:.1f}" y="{HEIGHT - BOTTOM + 16}" text-anchor="middle">{_fmt(tick)}</text>')
    parts.append(
        f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{HEIGHT - BOTTOM}" y2="{HEIGHT - BOTTOM}" stroke="#8a8984"/>'
    )
    parts.append(
        f'<text x="{(LEFT + WIDTH - RIGHT) / 2}" y="{HEIGHT - 6}" text-anchor="middle">{html.escape(x_label)}</text>'
    )
    parts.append(
        f'<text transform="translate(14 {(TOP + HEIGHT - BOTTOM) / 2}) rotate(-90)" '
        f'text-anchor="middle">{html.escape(y_label)}</text>'
    )
    return parts


def _svg(body: list[str], label: str) -> str:
    return (
        f'<svg width="{WIDTH}" height="{HEIGHT}" role="img" '
        f'aria-label="{html.escape(label)}">' + "".join(body) + "</svg>"
    )


def _empty(label: str) -> str:
    return _svg(
        [f'<text class="empty" x="{WIDTH / 2}" y="{HEIGHT / 2}" text-anchor="middle">no data</text>'], label
    )


def line_chart(
    series: Sequence[tuple[str, str, Series]],
    *,
    x_label: str,
    y_label: str,
    y_range: tuple[float, float] | None = None,
    markers: bool = False,
) -> str:
    """Lines (``(name, color, points)``) on shared axes; every point has a hover title."""
    points = [point for _, _, data in series for point in data]
    if not points:
        return _empty(y_label)
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    x_low, x_high = min(xs), max(xs)
    y_low, y_high = y_range if y_range is not None else (min(ys), max(ys))
    if y_range is None:
        pad = (y_high - y_low) * 0.05 or abs(y_high) * 0.1 or 1.0
        y_low, y_high = y_low - pad, y_high + pad
    if x_high == x_low:
        x_low, x_high = x_low - 1, x_high + 1
    def sx(x: float) -> float:
        return LEFT + (x - x_low) / (x_high - x_low) * (WIDTH - LEFT - RIGHT)

    def sy(y: float) -> float:
        return HEIGHT - BOTTOM - (y - y_low) / (y_high - y_low) * (HEIGHT - TOP - BOTTOM)

    body = _frame(_ticks(x_low, x_high), _ticks(y_low, y_high), sx, sy, x_label, y_label)
    for name, color, data in series:
        if not data:
            continue
        path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in data)
        if len(data) > 1:
            body.append(
                f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2" '
                'stroke-linejoin="round" stroke-linecap="round"/>'
            )
        show_markers = markers or len(data) == 1
        for x, y in data:
            title = f"<title>{html.escape(name)}: {_fmt(y)} at {x_label.lower()} {_fmt(x)}</title>"
            radius = 4 if show_markers else 6
            fill = color if show_markers else "transparent"
            ring = ' stroke="#ffffff" stroke-width="2"' if show_markers else ""
            body.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{radius}" fill="{fill}"{ring}>{title}</circle>')
    return _svg(body, y_label)


def histogram(
    groups: Sequence[tuple[str, str, Sequence[float]]], *, x_label: str, integer: bool = False, bins: int = 20
) -> str:
    """Histograms (fraction of samples per bin), one bar per run in every bin."""
    values = [value for _, _, data in groups for value in data]
    if not values:
        return _empty(x_label)
    low, high = float(min(values)), float(max(values))
    if integer and high - low + 1 <= 40:
        edges = np.arange(low - 0.5, high + 1.5)
    else:
        edges = np.linspace(low, high if high > low else low + 1, bins + 1)
    fractions = []
    for name, color, data in groups:
        counts, _ = np.histogram(np.asarray(data, dtype=float), bins=edges)
        total = max(counts.sum(), 1)
        fractions.append((name, color, counts / total))
    y_high = max(float(f.max()) for _, _, f in fractions) * 1.1 or 1.0

    def sx(x: float) -> float:
        return LEFT + (x - edges[0]) / (edges[-1] - edges[0]) * (WIDTH - LEFT - RIGHT)

    def sy(y: float) -> float:
        return HEIGHT - BOTTOM - y / y_high * (HEIGHT - TOP - BOTTOM)

    x_ticks = _ticks(float(edges[0]), float(edges[-1]))
    if integer:
        x_ticks = [tick for tick in x_ticks if float(tick).is_integer()]
    body = _frame(x_ticks, _ticks(0.0, y_high), sx, sy, x_label, "fraction of samples")
    # Grouped bars: each bin is split into one slot per run, so runs never hide each other.
    slots = len(fractions)
    for slot, (name, color, fraction) in enumerate(fractions):
        for index, value in enumerate(fraction):
            if value <= 0:
                continue
            left, right = sx(edges[index]) + 1, sx(edges[index + 1]) - 1
            width = (right - left) / slots
            x0 = left + slot * width
            title = (
                f"<title>{html.escape(name)}: {value:.1%} in [{_fmt(edges[index])}, {_fmt(edges[index + 1])})</title>"
            )
            body.append(
                f'<rect x="{x0:.1f}" y="{sy(value):.1f}" width="{max(width - 1, 1):.1f}" '
                f'height="{sy(0) - sy(value):.1f}" fill="{color}" rx="2">{title}</rect>'
            )
    return _svg(body, x_label)


def timing_bars(runs: Sequence[tuple[str, dict[str, float]]]) -> str:
    """One stacked horizontal bar per run: the share of wall time per component."""
    if not runs:
        return _empty("timing")
    row_height, gap = 26, 14
    height = TOP + len(runs) * (row_height + gap) + 40
    label_width = 150
    span = WIDTH - label_width - RIGHT
    body = []
    for row, (name, timing) in enumerate(runs):
        total = sum(timing.values()) or 1.0
        y = TOP + row * (row_height + gap)
        body.append(f'<text x="{label_width - 8}" y="{y + 17}" text-anchor="end">{html.escape(name[:24])}</text>')
        x = float(label_width)
        for component, seconds in timing.items():
            width = seconds / total * span
            if width <= 0:
                continue
            title = f"<title>{html.escape(name)}: {component} {seconds:.2f} s ({seconds / total:.1%})</title>"
            body.append(
                f'<rect x="{x:.1f}" y="{y}" width="{max(width - 2, 0.5):.1f}" height="{row_height}" '
                f'fill="{TIMING_COLORS[component]}" rx="2">{title}</rect>'
            )
            x += width
    for index, component in enumerate(TIMING_COLORS):
        lx = 16 + index * 115
        body.append(f'<rect x="{lx}" y="{height - 22}" width="12" height="12" rx="2" fill="{TIMING_COLORS[component]}"/>')
        body.append(f'<text x="{lx + 16}" y="{height - 12}">{component}</text>')
    return (
        f'<svg width="{WIDTH}" height="{height}" role="img" '
        'aria-label="time breakdown">' + "".join(body) + "</svg>"
    )


def _figure(title: str, svg: str) -> str:
    return f"<figure><figcaption>{html.escape(title)}</figcaption>{svg}</figure>"


def _worker_table(runs: Sequence[RunSummary]) -> str:
    rows = [
        (
            "<tr><th>run</th><th>worker</th><th>decisions</th><th>env steps</th>"
            "<th>inference ms / decision</th><th>simulator ms / decision</th>"
            "<th>last report (s before run end)</th></tr>"
        )
    ]
    for run in runs:
        end = max((worker["last"] for worker in run.workers), default=0.0)
        for worker in run.workers:
            decisions = max(worker["decisions"], 1)
            rows.append(
                f"<tr><td>{html.escape(run.name)}</td><td>{worker['worker']}</td><td>{worker['decisions']}</td>"
                f"<td>{worker['env_steps']}</td><td>{1000 * worker['inference'] / decisions:.2f}</td>"
                f"<td>{1000 * worker['env'] / decisions:.2f}</td><td>{end - worker['last']:.1f}</td></tr>"
            )
    return "<table>" + "".join(rows) + "</table>"


def _totals_table(runs: Sequence[RunSummary]) -> str:
    keys = list(runs[0].totals) if runs else []
    rows = ["<tr><th>run</th>" + "".join(f"<th>{html.escape(key)}</th>" for key in keys) + "</tr>"]
    for run in runs:
        rows.append(
            f"<tr><td>{html.escape(run.name)}</td>"
            + "".join(f"<td>{_fmt(run.totals.get(key, 0.0))}</td>" for key in keys)
            + "</tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def _learner_metrics(runs: Sequence[RunSummary]) -> list[str]:
    names = sorted({name for run in runs for name in run.learner})
    losses = [name for name in names if name.endswith("loss")]
    return (losses + [name for name in names if name not in losses])[:MAX_LEARNER_PLOTS]


def render_dashboard(runs: Sequence[RunSummary], *, title: str = "RLPT run dashboard") -> str:
    """The dashboard HTML for ``runs`` (overlaid, one color per run)."""
    colored = [(run, _color(index)) for index, run in enumerate(runs)]
    legend = "".join(
        f'<span><span class="swatch" style="background:{color}"></span>{html.escape(run.name)}</span>'
        for run, color in colored
    )
    steps = "environment steps"
    sections = [
        f"<h1>{html.escape(title)}</h1>",
        (
            '<p class="note">Read from each run\'s telemetry tables. Training curves are trailing means over the '
            "last 20 episodes; evaluation points are means over one evaluation round. Hover a mark for its value.</p>"
        ),
        f'<div class="legend">{legend}</div>',
        _totals_table(runs),
        "<h2>Progress</h2>",
        '<div class="grid">',
        _figure(
            "Success rate (training episodes)",
            line_chart([(r.name, c, r.train_success) for r, c in colored], x_label=steps,
                       y_label="success rate", y_range=(0.0, 1.0)),
        ),
        _figure(
            "Success rate (evaluation)",
            line_chart([(r.name, c, r.eval_success) for r, c in colored], x_label=steps,
                       y_label="success rate", y_range=(0.0, 1.0), markers=True),
        ),
        _figure(
            "Episode return (training episodes)",
            line_chart([(r.name, c, r.train_return) for r, c in colored], x_label=steps, y_label="return"),
        ),
        _figure(
            "Episode return (evaluation)",
            line_chart([(r.name, c, r.eval_return) for r, c in colored], x_label=steps, y_label="return",
                       markers=True),
        ),
        "</div>",
        "<h2>System</h2>",
        '<div class="grid">',
        _figure(
            "Collection throughput per round",
            line_chart([(r.name, c, r.throughput) for r, c in colored], x_label=steps,
                       y_label="env steps per second"),
        ),
        _figure("Where the wall time went", timing_bars([(r.name, r.timing) for r, _ in colored])),
        _figure(
            "Policy lag of training decisions",
            histogram([(r.name, c, r.policy_lags) for r, c in colored], x_label="policy lag (versions)",
                      integer=True),
        ),
        _figure(
            "Training episode length",
            histogram([(r.name, c, r.episode_lengths) for r, c in colored], x_label="episode length (env steps)",
                      integer=True),
        ),
        "</div>",
        "<h2>Rollout workers</h2>",
        _worker_table(runs),
        "<h2>Learner</h2>",
        '<div class="grid">',
    ]
    for metric in _learner_metrics(runs):
        sections.append(
            _figure(metric, line_chart([(r.name, c, r.learner.get(metric, [])) for r, c in colored],
                                       x_label=steps, y_label=metric))
        )
    sections.append("</div>")
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>"
        + "\n".join(sections)
        + "</body></html>\n"
    )
