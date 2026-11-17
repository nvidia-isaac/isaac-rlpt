# `_common`: shared workflow helpers

Small, dependency-free helpers so every workflow reports and checks itself the same way.
This folder holds no training components; those are in the library.

| File | Purpose | Used by |
|---|---|---|
| `EVALUATION.md` | the evaluation protocol every reported number follows, and the `results/results.json` schema | READMEs, reviewers |
| `results_table.py` | renders `results/results.json` into the README's results table (`--write`) and checks that they agree (`--check`) | workflow READMEs, the layout test |
| `smoke.py` | finds and runs every workflow's `smoke.sh` | CI, `tests/unit/test_example_workflows.py` |
| `launch.sh` | shell helpers for launchers: print the overrides a launch passes (`launch.override=` lines) and run the library's dry run on them | every shell launcher's dry run |
| `golden.py` | golden-config parity: resolves each workflow's recipes and compares them with the config pinned in its tests folder (`--write` re-pins after an intended change) | each workflow's `smoke.sh` and `test_golden_config.py` |
| `release.py` | the `internal/` policy: what counts as internal, where it may live, and the scan that enforces it | `tools/strip_internal.py`, the layout test |
| `internal/` | cluster material shared by several workflows; removed at release with every other `internal/` directory | workflows' `internal/` files |

```bash
uv run python -m example_workflows._common.smoke                 # run every workflow's smoke check
uv run python -m example_workflows._common.smoke --list          # just list them
uv run python -m example_workflows._common.results_table example_workflows/_template/results/results.json
uv run python -m example_workflows._common.golden --all          # every workflow's recipes against their pinned configs
```
