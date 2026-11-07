---
name: isaac-rlpt-audit-repo
description: Audits the Isaac RLPT repository for pull-request or public-release readiness against its own AGENTS.md invariants, DO-NOTs and coding principles, runs the guardrail tests and per-dimension sweeps, and renders a self-contained HTML risk report. Use when a contributor asks to "audit the repo", "check PR/release readiness", "run a release-readiness audit" or "is this repo ready to open-source". Not for reviewing a single change, extending the library with a simulator, base model or algorithm, or running experiments.
license: CC-BY-4.0 AND Apache-2.0
---

# Audit Isaac RLPT for release readiness

Audit `isaac_rl_posttraining` for public-release / PR readiness and emit a single self-contained HTML report.
The audit grades six dimensions A–F, each mirroring a rule the repo already sets for itself in `AGENTS.md` (the invariants, the DO-NOTs, and the coding principles), and enforced in part by three automated guardrail tests.
The output is `audit_report.html`: an overall grade, a release-readiness verdict, a per-dimension grade table, and a severity-ranked risk table.

The greps below are **signals, not verdicts**.
Every hit must be opened and read in context before it becomes a finding — most `dict.get(...)`, `getattr(...)`, and `except Exception` hits in this repo are legitimate (compat boundaries, genuinely optional attributes), and recording them blindly produces a false, alarmist report.
Grade honestly: do not inflate grades to look agreeable, and do not penalize documented, intentional exceptions.

## When to Use

- A contributor wants the repository graded before a pull request or a public release.
- A contributor wants a severity-ranked list of risks against the repository's own rules.

Not for: reviewing one diff, extending the library, or customer-facing tasks; this is a contributor skill.

## Ground truth to read first

Read these before sweeping, because they define what "ready" means here and keep the audit aligned with the repo's real rules:

- `AGENTS.md` — the invariants, the DO-NOTs, the coding principles. The six dimensions mirror these.
- `tests/unit/test_repo_hygiene.py`, `tests/unit/test_gr00t_embodiment_layering.py`, `tests/unit/test_doc_references.py` — the guardrails the audit runs and builds on.

All commands below assume the repository root as the working directory:

```bash
cd "$(git rev-parse --show-toplevel)"
PKG=isaac_rl_posttraining
```

## Workflow

### 1. Run the automated guardrails

Run them in the project's environment (`uv sync --extra cpu`, as `.github/workflows/ci.yml` does).
The two layering tests and the doc-reference test are pure filesystem/AST checks and run under any Python; the hygiene suite imports the package, so it needs the base dependencies.

```bash
# The full CPU-safe release gate (what CI runs):
uv run --extra cpu pytest tests/ simulators/*/tests/ example_workflows/*/tests/ -m "not gpu and not arena and not libero" -q

# The three guardrails specifically:
uv run --extra cpu pytest tests/unit/test_repo_hygiene.py \
    tests/unit/test_gr00t_embodiment_layering.py \
    tests/unit/test_doc_references.py -q
```

A failing guardrail is a hard release blocker: the dimension it defends caps at F and the overall verdict is Not-ready.
Capture the failing test name and message as a `high` finding.

### 2. Sweep each dimension (grep/AST), then read-and-judge

Run the sweeps for a dimension, open each hit, and keep only the confirmed problems.
Prefer `rg` (ripgrep) if available; `grep -rnE` fallbacks are given.

**Dimension 1 — Layering.**
Core imports zero simulators; the GR00T policy stays embodiment-agnostic.
The two layering tests are authoritative; these greps locate any offender to cite.

```bash
grep -rnE "^[[:space:]]*(from|import)[[:space:]]+simulators([. ]|$)" "$PKG" --include='*.py'   # core -> simulators (must be empty)
grep -rn  "$PKG.envs" "$PKG/policies/gr00t" --include='*.py'                                    # gr00t -> env layer (must be empty)
grep -rniE "\b(wbc|teleop)\b|\bg1\b|\bleg\b" "$PKG/policies/gr00t" --include='*.py'             # embodiment tokens (comments/docstrings are allowed; flag only live code/strings)
```

**Dimension 2 — Dead code & provenance.**
No TODO/FIXME left casually, no commented-out code, no conversation/branch/task provenance narrated in comments (genuine design rationale is fine).

```bash
grep -rnE "\b(TODO|FIXME|XXX|HACK)\b" "$PKG" simulators scripts --include='*.py' --include='*.sh'
grep -rnE "^[[:space:]]*#[[:space:]]*(return|import |from |def |class |if |for |while |elif |else:|self\.|print\(|[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=[^=])" "$PKG" simulators --include='*.py'
grep -rniE "#.*(parity|as requested|per (the )?task|deprecated as|revert|keep in sync with (develop|main)|for now|used to|previously|temporarily)" "$PKG" simulators --include='*.py'
```

**Dimension 3 — Defensive-programming smells.**
Fallback defaults for required values, guards that hide bugs, catch-all `else` for a known mode set.
High-noise sweeps — verify aggressively.

```bash
grep -rnE "\.get\([^,)]+,[^)]+\)" "$PKG" --include='*.py'                       # dict.get(k, default)
grep -rnE "getattr\([^,)]+,[^,)]+,[^)]+\)" "$PKG" --include='*.py'              # 3-arg getattr fallback
grep -rnE "[)A-Za-z0-9_\]]+[[:space:]]+or[[:space:]]+(\{\}|\[\]|None|0|0\.0|''|\"\"|False|-1)\b" "$PKG" --include='*.py'   # x or fallback
grep -rnE "except([[:space:]]+(Exception|BaseException))?[[:space:]]*:" "$PKG" --include='*.py'   # broad/bare except
grep -rn  "hasattr(" "$PKG" --include='*.py'                                    # existence guards
```

For catch-all mode branches, scan `else:`/`return` tails near a known mode dispatch and judge whether a real mode is hiding in the default; the tooling cannot decide this.

**Dimension 4 — Typing & style.**
Built-in generics and `|` unions; no bare `dict`/`list`, no `Any`/`object` dodges, no `typing.List/Dict/Optional/Union`, no needlessly stringized annotations.

```bash
grep -rnE "\b(List|Dict|Optional|Union|Tuple|Set|FrozenSet)\[" "$PKG" simulators --include='*.py'
grep -rnE "from typing import.*\b(List|Dict|Optional|Union|Tuple)\b" "$PKG" simulators --include='*.py'
grep -rnE "(:|->)[[:space:]]*(Any|object)\b" "$PKG" --include='*.py'                 # Any/object dodges (Any at the JSON/duck-typed boundary can be legit)
rg -n -P '(:|->)\s*(dict|list|tuple|set|frozenset)(?![\[\w])' "$PKG" --glob '*.py'   # bare container annotations (rg); grep fallback below
grep -rnE "(:|->)[[:space:]]*(dict|list|tuple|set)([[:space:],)=]|$)" "$PKG" --include='*.py'
grep -rnE "(:|->)[[:space:]]*['\"][A-Za-z_]" "$PKG" --include='*.py'                 # stringized annotations (genuine forward refs / TYPE_CHECKING are fine)
```

**Dimension 5 — Naming & consistency.**
No leftover codenames; `POSTTRAIN_*` (pipeline-level) vs `DSRL_*` (steering knobs) used consistently; validated simulators structurally parallel.

```bash
uv run --extra cpu pytest tests/unit/test_repo_hygiene.py::test_no_legacy_codename_anywhere -q   # codename gate (needle assembled at runtime)
grep -rohE "\b(POSTTRAIN|DSRL)_[A-Z][A-Z0-9_]*\b" "$PKG" simulators scripts --include='*.py' --include='*.sh' --include='*.yaml' | sort | uniq -c | sort -rn
for d in simulators/*/; do echo "== $d =="; ls "$d"; done   # validated sims (g1_arena, libero) share env.py/README.md/tests/configs; robolab is experimental and may differ
```

**Dimension 6 — Docs & tests.**
Every doc-referenced path resolves; CI covers the CPU-safe suite; each simulator ships a `conftest.py`.

```bash
uv run --extra cpu pytest tests/unit/test_doc_references.py -q
for d in simulators/*/tests; do test -f "$d/conftest.py" && echo "OK      $d" || echo "MISSING $d"; done
```

Read `.github/workflows/ci.yml` and the `markers`/`addopts` in `pyproject.toml`: confirm the CPU-safe lane runs on the supported Python versions and that markers are strict.

### 3. Verify before recording

For every candidate hit, open the file at the line and confirm it is a real problem under `AGENTS.md`.
Discard the legitimate ones: a `dict.get(k, default)` for a genuinely optional value, an `except` at a documented compatibility boundary, a `hasattr` on a duck-typed interface, an `Any` at the JSON parse edge, embodiment words inside a gr00t comment/docstring (the layering test already permits those).
A finding must name a concrete file, line, and the specific rule it breaks.

### 4. Assemble the findings JSON

Write a JSON file (e.g. `/tmp/audit_findings.json`) matching the schema in `scripts/render_report.py` (its module docstring is the source of truth).
All keys are required; a missing or misspelled key raises a clear error at render time.

```json
{
  "repo": "isaac_rl_posttraining @ <branch>",
  "overall_grade": "B",
  "verdict": "Ready-with-nits",
  "dimensions": [
    {"name": "Layering", "grade": "A", "summary": "one-line justification"},
    {"name": "Dead code & provenance", "grade": "B", "summary": "..."},
    {"name": "Defensive-programming smells", "grade": "C", "summary": "..."},
    {"name": "Typing & style", "grade": "B", "summary": "..."},
    {"name": "Naming & consistency", "grade": "A", "summary": "..."},
    {"name": "Docs & tests", "grade": "B", "summary": "..."}
  ],
  "findings": [
    {"severity": "high", "dimension": "Layering",
     "file": "isaac_rl_posttraining/policies/gr00t/wrapper.py", "line": 88,
     "issue": "what is wrong and which rule it breaks",
     "suggested_fix": "the concrete fix"}
  ]
}
```

`overall_grade` and each dimension `grade` are `A`/`B`/`C`/`D`/`F` (optional `+`/`-`).
`verdict` is `Ready`, `Ready-with-nits`, or `Not-ready`.
`severity` is `high`, `medium`, or `low`.
`line` is an integer.

### 5. Render the HTML report

```bash
uv run python .agents/contributor-skills/isaac-rlpt-audit-repo/scripts/render_report.py --findings /tmp/audit_findings.json --out audit_report.html
```

### 6. Report in chat

State the overall grade, the verdict, and the top three to five risks (highest severity first) in the chat.
Point at `audit_report.html` for the full breakdown.

## Grading rubric

Grade each dimension on confirmed findings, not raw grep counts:

- **A** — clean; the dimension's guardrail test (if any) passes; no confirmed findings.
- **B** — only `low`-severity nits.
- **C** — at least one confirmed `medium`; no `high`.
- **D** — several `medium`s, or one `high` in a non-invariant dimension.
- **F** — a guardrail test for this dimension fails, or a `high` finding that breaks a stated release rule.

Overall verdict:

- **Not-ready** — any guardrail test fails, or any `high` finding exists.
- **Ready-with-nits** — all guardrails pass and only `low`/`medium` findings remain.
- **Ready** — all guardrails pass and nothing above `low` is confirmed.

The overall grade is a holistic read of the dimension grades, weighted toward the invariants (Layering, and the codename/import/doc gates), not a mechanical average.

## Files in this skill

- `evals/evals.json` — task and trigger evaluations; `BENCHMARK.md` — how often agents load this skill.
- `scripts/render_report.py` — validates the findings JSON (fail-loud on any missing/invalid key) and writes the self-contained HTML report (`--findings <json> --out <html>`).
