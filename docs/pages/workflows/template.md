# Start a new workflow

`example_workflows/_template/` is a skeleton of the workflow layout.
Copy it to post-train on a new simulator, base policy or recipe in a way someone else can reproduce.
If the new part is a reusable simulator, algorithm or base model, it belongs in the library instead: see [add a simulator](../extending/add_a_simulator.md), [add an algorithm](../extending/add_an_algorithm.md) and [add a base model](../extending/add_a_base_model.md).

## Folder layout

```
example_workflows/<workflow>/
  README.md               the required sections (below) and the results table
  smoke.sh                a check that runs in under two minutes with no GPU and no external repository
  results/results.json    every reported number with its base, episode count, evaluator and date
  externals.lock.yaml     other repositories it depends on, pinned by commit (omit when there are none)
  scripts/, configs/      what a user runs and overrides
  <module>.py             its own Python, importable as example_workflows.<workflow>
  tests/                  what runs without GPUs or external repositories, including the pinned recipe configs
  internal/               everything that only works inside the company; removed at release
```

## Rules

- **Imports point one way.** The library never imports a workflow.
- **One data format.** Data a workflow hands the library is the saved replay-buffer format; `uv run python -m isaac_rl_posttraining.data.inspect <dir>` must read it.
- **One evaluation protocol.** Every success rate is reported next to the frozen policy's rate under the same evaluator, with the episode count.
- **Smoke checks never need hardware or external repositories.**
- **Results carry provenance.** Every reported number states its date, evaluator, base and episode count.
- **Graduation.** When a piece of a workflow stops being specific to it, it moves into the library behind a config group, and the workflow keeps its recipe and results.

## Steps

Run every command from the repository root.

1. Copy the skeleton, naming the folder after what the workflow does:

   ```bash
   cp -r example_workflows/_template example_workflows/<workflow>
   ```

2. Add the launch and evaluation scripts, configs and task lists to the folder.
   Scripts call `scripts/train.py` and the simulator packages by repository-relative path and must not depend on the current directory.

3. Write `smoke.sh`: a dry run of each recipe (`ISAAC_RL_POSTTRAINING_DRYRUN=1`) and a mock-base-model run where the simulator has one, exiting non-zero on any failure.
   Check it passes:

   ```bash
   uv run bash example_workflows/<workflow>/smoke.sh
   ```

4. Write the README in the template's section order: `## Overview`, `## Requirements`, `## Quickstart`, `## Results`, `## Method`, `## Files`, plus a `**Status:**` line.
   A test enforces the order.
   Keep it short and link to a page under `docs/pages/workflows/` for the full guide.

5. Fill `results/results.json` with the numbers you can source (the schema of `example_workflows/_template/results/results.json`), then render the README table:

   ```bash
   uv run python -m example_workflows._common.results_table \
       example_workflows/<workflow>/results/results.json \
       --write example_workflows/<workflow>/README.md
   ```

   Leave out any number without a base and an episode count.
   The template's single row (date `1970-01-01`) is illustrative; replace it.

6. Put anything testable without hardware in `<workflow>/tests/`: mock-base-model runs, dry runs, converters.
   The skeleton copied in step 1 brings `tests/` with `golden_cases.py` and `test_golden_config.py` (in the template: `example_workflows/_template/tests/golden_cases.py` and `example_workflows/_template/tests/test_golden_config.py`).
   List every recipe your launchers run in your copy of `golden_cases.py`, then pin the config each resolves to:

   ```bash
   uv run python -m example_workflows._common.golden <workflow> --write
   ```

   Your copy of `test_golden_config.py` then fails whenever a recipe's resolved config moves.

7. Add a row to the table in `example_workflows/README.md` and in [index.md](index.md).

8. Verify before opening the pull request:

   ```bash
   uv run bash example_workflows/<workflow>/smoke.sh
   uv run python -m example_workflows._common.results_table \
       example_workflows/<workflow>/results/results.json \
       --check example_workflows/<workflow>/README.md
   uv run ruff check example_workflows tools
   uv run ruff format --check example_workflows tools
   uv run mypy --ignore-missing-imports --follow-imports=skip example_workflows tools
   uv run pytest tests/unit/test_example_workflows.py tests/unit/test_doc_references.py tests/unit/test_repo_hygiene.py -q
   uv run pytest tests/ simulators/*/tests/ example_workflows/*/tests/ -m "not gpu and not arena and not libero and not slow" -q
   ```

   All must pass.
