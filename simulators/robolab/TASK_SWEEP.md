# RoboLab task sweep

`simulators/robolab/scripts/sweep.py` checks task names before a RoboLab launch and summarizes results after it.
It parses task files without importing Isaac Sim or RoboLab, so a typo fails before it takes a GPU.
It only understands the pinned RoboLab layout, where each task file directly declares its concrete `Task` subclass.

Before a launch, list the tasks and validate the ones you will run (an unknown or ambiguous class fails; `--json` prints the task-file paths for another launcher):

```bash
export POSTTRAIN_ROOT=/path/to/posttrain
export ROBOLAB_ROOT=/path/to/external/robot-runtime-deps/robolab

uv run --project "$POSTTRAIN_ROOT" python "$POSTTRAIN_ROOT/simulators/robolab/scripts/sweep.py" discover \
  --task-dir "$ROBOLAB_ROOT/robolab/tasks/benchmark"

uv run --project "$POSTTRAIN_ROOT" python "$POSTTRAIN_ROOT/simulators/robolab/scripts/sweep.py" validate \
  --task-dir "$ROBOLAB_ROOT/robolab/tasks/benchmark" \
  --task BagelsOnPlateTask \
  --task BananaInBowlTask
```

After an evaluation, summarize native `episode_results.jsonl` files or normalized harness traces:

```bash
uv run --project "$POSTTRAIN_ROOT" python "$POSTTRAIN_ROOT/simulators/robolab/scripts/sweep.py" report \
  --result /path/to/episode_results.jsonl \
  --min-success-rate 0.25 \
  --max-success-rate 0.80 \
  --min-episodes 10
```

The report keeps each source path and any model or evaluation metadata in the rows, and lists `missing`, `invalid` and `empty` sources separately from completed zero-success results.
Its candidate list is a queue for follow-up evaluation chosen by observed success rate, not a significance test or a DSRL result.
