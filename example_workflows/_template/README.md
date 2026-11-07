# <Workflow name>

<One sentence: what running this workflow gives you.>
The full step-by-step guide is [docs/pages/workflows/<page>.md](../../docs/pages/workflows/<page>.md).

**Status:** exploratory | validated on <date>

## Overview

<Two or three sentences: the question it answers, what goes in and what comes out, and what the numbers below mean.>

## Requirements

| | |
|---|---|
| Hardware | <e.g. 2 GPUs (48 GB each) per run> |
| Software | this repository (`uv sync`) and the external repositories pinned in `externals.lock.yaml` |
| Data and checkpoints | <what, and where to get them> |

```bash
export <VARIABLE>=/path/to/...      # <what the launchers read>
```

## Quickstart

From the repository root:

```bash
uv run bash example_workflows/<workflow>/smoke.sh                      # under two minutes, no GPU
uv run bash example_workflows/<workflow>/scripts/train.sh <TASK>
uv run bash example_workflows/<workflow>/scripts/eval.sh <TASK> <checkpoint>
```

<One sentence each: where training writes its checkpoints, and what evaluation prints.>

The same training run from Python:

```python
from isaac_rl_posttraining import launch
from isaac_rl_posttraining.algorithms import DSRL, BaseModel

run = launch(
    DSRL(base=BaseModel("<base model>", "<checkpoint>")),
    env="<simulator>/<task>",
    overrides=["<key>=<value>"],
)
run.loop()
```

<One sentence: the shell recipe this run equals; `tests/test_golden_config.py` checks that both compose the same config.>

## Results

<!-- results:start -->
| Task | Method | Base | Peak | Final | Note |
|---|---|---|---|---|---|
| ExampleTask | ExampleMethod | 50.0% n=100 | 60.0% n=100 | 60.0% n=100 | illustrative |

Success rates; n is the episode count. Evaluator: illustrative numbers; replace with the evaluator, scoring rule and policy mode that produced yours. Measured 1970-01-01.
<!-- results:end -->

<One sentence reading the table.>
The table is rendered from `results/results.json` by `uv run python -m example_workflows._common.results_table example_workflows/<workflow>/results/results.json --write example_workflows/<workflow>/README.md`.

## Method

1. **Rollout.** <The policy and the simulator: what one decision drives.>
2. **Scoring.** <The reward, and how success is scored.>
3. **Optimization.** <The algorithm; link to `docs/pages/algorithms/dsrl.md` or `docs/runtimes.md` for anything general.>

## Files

| Path | Purpose |
|---|---|
| `smoke.sh` | the setup check CI runs |
| `scripts/` | what a user runs |
| `configs/` | overrides of the library config groups |
| `results/results.json` | every reported number with its provenance |
| `externals.lock.yaml` | pinned external repositories |
| `tests/` | what runs without hardware; `golden_cases.py` lists the recipes and `golden_config.json` pins the config each resolves to |
| `internal/` | material that only works inside the company; removed at release |
