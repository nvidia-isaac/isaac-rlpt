# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render an audit findings JSON into a self-contained HTML risk report.

The findings JSON is the audit's single output artifact; this module turns it
into one standalone ``.html`` (inline CSS, no external assets) that a reviewer
can open directly. Every key below is required and validated once up front —
a missing or misspelled key raises with a clear message rather than silently
rendering a blank cell.

Schema::

    {
      "repo": str,                 # what was audited (shown in the header)
      "overall_grade": str,        # a letter grade: A B C D F, optional +/- suffix
      "verdict": str,              # "Ready" | "Ready-with-nits" | "Not-ready"
      "dimensions": [              # one row per audit dimension
        {"name": str, "grade": str, "summary": str}
      ],
      "findings": [                # rows of the ranked risk table
        {"severity": str,          # "high" | "medium" | "low"
         "dimension": str,
         "file": str,
         "line": int,
         "issue": str,
         "suggested_fix": str}
      ]
    }

Usage::

    python render_report.py --findings findings.json --out audit_report.html
"""

import argparse
import html
import json
from typing import TypedDict


class Dimension(TypedDict):
    name: str
    grade: str
    summary: str


class Finding(TypedDict):
    severity: str
    dimension: str
    file: str
    line: int
    issue: str
    suggested_fix: str


class Report(TypedDict):
    repo: str
    overall_grade: str
    verdict: str
    dimensions: list[Dimension]
    findings: list[Finding]


_ALLOWED_VERDICTS = ("Ready", "Ready-with-nits", "Not-ready")
_ALLOWED_SEVERITIES = ("high", "medium", "low")
_GRADE_LETTERS = ("A", "B", "C", "D", "F")
_GRADE_SUFFIXES = ("", "+", "-")

#: Severity ordering for the ranked risk table (high risk first).
_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}

# Grades and severities are an ordinal status axis, so each maps to a reserved
# status color from the validated data-viz palette (good -> critical) and is
# always drawn with its letter/word label: status is never carried by color
# alone. Each pair is (background, ink) chosen to stay legible on that fill.
_GRADE_COLORS = {
    "A": ("#006300", "#ffffff"),
    "B": ("#0ca30c", "#0b0b0b"),
    "C": ("#fab219", "#0b0b0b"),
    "D": ("#ec835a", "#0b0b0b"),
    "F": ("#d03b3b", "#ffffff"),
}
# (background, ink, glyph); the glyph is a distinct shape per level so the cue
# survives color-blindness and grayscale printing.
_SEVERITY_STYLE = {
    "high": ("#d03b3b", "#ffffff", "●"),
    "medium": ("#fab219", "#0b0b0b", "◆"),
    "low": ("#52514e", "#ffffff", "▪"),
}
_VERDICT_COLORS = {
    "Ready": ("#006300", "#ffffff"),
    "Ready-with-nits": ("#fab219", "#0b0b0b"),
    "Not-ready": ("#d03b3b", "#ffffff"),
}


# ---------------------------------------------------------------------------
# Validation: fail loud, once, before any rendering.
# ---------------------------------------------------------------------------


def _require_object(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a JSON object")
    return value


def _require_keys(value: object, keys: tuple[str, ...], where: str) -> dict[str, object]:
    obj = _require_object(value, where)
    for key in keys:
        if key not in obj:
            raise ValueError(f"{where} missing required key: {key!r}")
    return obj


def _require_nonempty_str(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _require_grade(value: object, where: str) -> str:
    if (
        not isinstance(value, str)
        or value[:1] not in _GRADE_LETTERS
        or value[1:] not in _GRADE_SUFFIXES
    ):
        raise ValueError(
            f"{where} must be a letter grade (one of {_GRADE_LETTERS}, "
            f"optional +/- suffix), got {value!r}"
        )
    return value


def _require_choice(value: object, allowed: tuple[str, ...], where: str) -> str:
    if value not in allowed:
        raise ValueError(f"{where} must be one of {allowed}, got {value!r}")
    return value  # type: ignore[return-value]  # narrowed by membership above


def validate_report(payload: object) -> Report:
    """Validate the parsed JSON against the schema, or raise ``ValueError``.

    Returns the same object once every required key and enumerated value has
    been checked, so callers can index it directly without further guards.
    """
    top = _require_keys(
        payload, ("repo", "overall_grade", "verdict", "dimensions", "findings"), "findings JSON"
    )
    _require_nonempty_str(top["repo"], "repo")
    _require_grade(top["overall_grade"], "overall_grade")
    _require_choice(top["verdict"], _ALLOWED_VERDICTS, "verdict")

    dimensions = top["dimensions"]
    if not isinstance(dimensions, list):
        raise ValueError("dimensions must be a list")
    for i, entry in enumerate(dimensions):
        dim = _require_keys(entry, ("name", "grade", "summary"), f"dimensions[{i}]")
        _require_nonempty_str(dim["name"], f"dimensions[{i}].name")
        _require_grade(dim["grade"], f"dimensions[{i}].grade")

    findings = top["findings"]
    if not isinstance(findings, list):
        raise ValueError("findings must be a list")
    for i, entry in enumerate(findings):
        finding = _require_keys(
            entry,
            ("severity", "dimension", "file", "line", "issue", "suggested_fix"),
            f"findings[{i}]",
        )
        _require_choice(finding["severity"], _ALLOWED_SEVERITIES, f"findings[{i}].severity")
        _require_nonempty_str(finding["file"], f"findings[{i}].file")
        # bool is an int subclass; a line number written as true/false is a mistake.
        if not isinstance(finding["line"], int) or isinstance(finding["line"], bool):
            raise ValueError(f"findings[{i}].line must be an integer")

    return payload  # type: ignore[return-value]  # shape verified above


# ---------------------------------------------------------------------------
# Rendering: small named helpers assembled into one document.
# ---------------------------------------------------------------------------


def _esc(value: str) -> str:
    return html.escape(value, quote=True)


def _severity_counts(findings: list[Finding]) -> dict[str, int]:
    counts = {severity: 0 for severity in _ALLOWED_SEVERITIES}
    for finding in findings:
        counts[finding["severity"]] += 1
    return counts


def _sorted_findings(findings: list[Finding]) -> list[Finding]:
    """Highest-risk first; ties broken by dimension, file, then line."""
    return sorted(
        findings,
        key=lambda f: (_SEVERITY_RANK[f["severity"]], f["dimension"], f["file"], f["line"]),
    )


def _grade_badge(grade: str, *, large: bool = False) -> str:
    background, ink = _GRADE_COLORS[grade[:1]]
    css_class = "grade-badge grade-badge--lg" if large else "grade-badge"
    return (
        f'<span class="{css_class}" style="background:{background};color:{ink}">'
        f"{_esc(grade)}</span>"
    )


def _verdict_pill(verdict: str) -> str:
    background, ink = _VERDICT_COLORS[verdict]
    return (
        f'<span class="verdict-pill" style="background:{background};color:{ink}">'
        f"{_esc(verdict)}</span>"
    )


def _severity_pill(severity: str) -> str:
    background, ink, glyph = _SEVERITY_STYLE[severity]
    return (
        f'<span class="sev-pill" style="background:{background};color:{ink}">'
        f'<span class="sev-glyph" aria-hidden="true">{glyph}</span>{_esc(severity)}</span>'
    )


def _render_banner(report: Report) -> str:
    counts = _severity_counts(report["findings"])
    chips = "".join(
        f'<span class="count-chip">{counts[severity]} {severity}</span>'
        for severity in _ALLOWED_SEVERITIES
    )
    return (
        '<section class="card banner">'
        '<div class="banner-left">'
        "<h1>Release-readiness audit</h1>"
        '<p class="sub">Graded against the repository\'s AGENTS.md invariants, '
        "DO-NOTs, and coding principles.</p>"
        f'<div class="repo">{_esc(report["repo"])}</div>'
        f'<div class="chips">{chips}</div>'
        "</div>"
        '<div class="banner-right">'
        '<div class="overall">'
        '<div class="overall-label">Overall</div>'
        f'{_grade_badge(report["overall_grade"], large=True)}'
        "</div>"
        f'{_verdict_pill(report["verdict"])}'
        "</div>"
        "</section>"
    )


def _dimension_row(dimension: Dimension) -> str:
    return (
        "<tr>"
        f'<td>{_esc(dimension["name"])}</td>'
        f'<td class="cell-grade">{_grade_badge(dimension["grade"])}</td>'
        f'<td class="cell-summary">{_esc(dimension["summary"])}</td>'
        "</tr>"
    )


def _render_dimension_table(dimensions: list[Dimension]) -> str:
    if dimensions:
        body = "".join(_dimension_row(dimension) for dimension in dimensions)
    else:
        body = '<tr><td colspan="3" class="empty">No dimensions recorded.</td></tr>'
    return (
        "<section>"
        "<h2>Per-dimension grades</h2>"
        "<table>"
        "<thead><tr><th>Dimension</th><th>Grade</th><th>Summary</th></tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
        "</section>"
    )


def _finding_row(finding: Finding) -> str:
    location = f'{finding["file"]}:{finding["line"]}'
    return (
        "<tr>"
        f'<td>{_severity_pill(finding["severity"])}</td>'
        f'<td>{_esc(finding["dimension"])}</td>'
        f'<td class="cell-loc"><code>{_esc(location)}</code></td>'
        f'<td>{_esc(finding["issue"])}</td>'
        f'<td class="cell-summary">{_esc(finding["suggested_fix"])}</td>'
        "</tr>"
    )


def _render_risk_table(findings: list[Finding]) -> str:
    if findings:
        body = "".join(_finding_row(finding) for finding in _sorted_findings(findings))
    else:
        body = '<tr><td colspan="5" class="empty">No risks recorded &mdash; clean sweep.</td></tr>'
    return (
        "<section>"
        "<h2>Ranked risks</h2>"
        "<table>"
        "<thead><tr>"
        "<th>Severity</th><th>Dimension</th><th>Location</th>"
        "<th>Issue</th><th>Suggested fix</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
        "</section>"
    )


def _render_footnote() -> str:
    return (
        "<footer>"
        "<p>How this was produced: the automated guardrail tests "
        "(<code>test_repo_hygiene</code>, <code>test_gr00t_embodiment_layering</code>, "
        "<code>test_doc_references</code>) were run, then each dimension was swept with "
        "grep/AST and read-and-judge review.</p>"
        "<p>Grep and AST hits are <strong>signals, not verdicts</strong>: every finding "
        "below was read in context and confirmed to be a real issue before it was recorded. "
        "A dimension with no confirmed issue can still be graded down for a judgment call "
        "the tooling cannot see.</p>"
        "</footer>"
    )


_CSS = """
:root{
  --surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink-2:#52514e;
  --muted:#898781;--grid:#e1e0d9;--border:rgba(11,11,11,.10);
}
@media (prefers-color-scheme:dark){
  :root{
    --surface:#1a1a19;--plane:#0d0d0d;--ink:#ffffff;--ink-2:#c3c2b7;
    --muted:#898781;--grid:#2c2c2a;--border:rgba(255,255,255,.10);
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.55}
.page{max-width:1080px;margin:0 auto;padding:32px 24px 72px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;letter-spacing:.03em;text-transform:uppercase;
  color:var(--ink-2);margin:38px 0 12px}
.sub{color:var(--ink-2);font-size:14px;margin:0 0 12px;max-width:52ch}
.card{background:var(--surface);border:1px solid var(--border);
  border-radius:14px;padding:22px 24px}
.banner{display:flex;justify-content:space-between;align-items:center;
  gap:24px;flex-wrap:wrap}
.banner-right{display:flex;align-items:center;gap:18px}
.overall{text-align:center}
.overall-label{font-size:11px;text-transform:uppercase;letter-spacing:.05em;
  color:var(--muted);margin-bottom:6px}
.repo{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  color:var(--ink-2);font-size:13.5px}
.chips{margin-top:14px;display:flex;gap:8px;flex-wrap:wrap}
.count-chip{font-size:12.5px;font-weight:600;padding:3px 11px;border-radius:999px;
  border:1px solid var(--border);color:var(--ink-2);background:var(--plane)}
.grade-badge{display:inline-flex;align-items:center;justify-content:center;
  min-width:30px;height:30px;padding:0 9px;border-radius:8px;
  font-weight:700;font-size:15px}
.grade-badge--lg{min-width:66px;height:66px;font-size:34px;border-radius:16px}
.verdict-pill{display:inline-flex;align-items:center;padding:8px 18px;
  border-radius:999px;font-weight:700;font-size:15px}
table{width:100%;border-collapse:collapse;background:var(--surface);
  border:1px solid var(--border);border-radius:14px;overflow:hidden;font-size:14px}
thead th{text-align:left;font-size:11.5px;text-transform:uppercase;letter-spacing:.03em;
  color:var(--muted);background:var(--plane);padding:11px 14px;
  border-bottom:1px solid var(--grid)}
tbody td{padding:12px 14px;border-bottom:1px solid var(--grid);vertical-align:top}
tbody tr:last-child td{border-bottom:none}
.cell-grade{width:70px}
.cell-summary{color:var(--ink-2)}
.cell-loc{white-space:nowrap;font-variant-numeric:tabular-nums}
.cell-loc code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
  background:var(--plane);border:1px solid var(--border);border-radius:6px;padding:1px 6px}
.sev-pill{display:inline-flex;align-items:center;gap:6px;padding:3px 11px;border-radius:999px;
  font-weight:600;font-size:12.5px;text-transform:capitalize}
.sev-glyph{font-size:9px;line-height:1}
.empty{color:var(--muted);text-align:center;padding:24px}
footer{margin-top:34px;color:var(--muted);font-size:12.5px;line-height:1.65;
  border-top:1px solid var(--grid);padding-top:16px}
footer strong{color:var(--ink-2)}
footer code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
"""


def _render_head(repo: str) -> str:
    return (
        "<head>"
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Audit &mdash; {_esc(repo)}</title>"
        f"<style>{_CSS}</style>"
        "</head>"
    )


def render_html(report: Report) -> str:
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            _render_head(report["repo"]),
            "<body>",
            '<main class="page">',
            _render_banner(report),
            _render_dimension_table(report["dimensions"]),
            _render_risk_table(report["findings"]),
            _render_footnote(),
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an audit findings JSON into a self-contained HTML risk report."
    )
    parser.add_argument("--findings", required=True, help="path to the findings JSON")
    parser.add_argument("--out", required=True, help="path to write the HTML report")
    args = parser.parse_args()

    with open(args.findings, encoding="utf-8") as handle:
        payload = json.load(handle)
    report = validate_report(payload)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(render_html(report))
    print(
        f"wrote {args.out} (overall grade {report['overall_grade']}, "
        f"verdict {report['verdict']}, {len(report['findings'])} findings)"
    )


if __name__ == "__main__":
    main()
