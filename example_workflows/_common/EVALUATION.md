# Evaluation protocol

Every success rate in an `example_workflows` README follows these rules, so numbers from different workflows mean the same thing.

1. **Pair every number with its base.** The post-trained policy and the frozen base policy are scored under the same evaluator, task, episode budget and scoring rule.
2. **State the denominator.** Every rate carries its episode count. Rates from fewer than 50 episodes are not reported as results.
3. **Report the peak and the final, and say which is which.** `peak` is the best evaluation the run reached, `final` is where it ended. A `peak` taken from in-training evaluations is the maximum over several noisy evaluations, so it is biased upward; say so in the evaluator line. A column with no measurement is `null`, rendered `—`, never omitted.
4. **Name the evaluator.** One sentence: which simulator build, how success is scored (terminal or latched), stochastic or deterministic policy, image resolution when it matters.
5. **Carry provenance.** The date and the pinned externals (`externals.lock.yaml`) that produced the numbers.

## `results/results.json`

```json
{
  "workflow": "g1_arena_posttraining",
  "evaluator": "IsaacLab-Arena evaluator, terminal success, stochastic policy, native image resolution",
  "date": "2026-09-16",
  "rows": [
    {
      "task": "DrillLift",
      "method": "DSRL",
      "base": {"successes": 173, "episodes": 200},
      "peak": {"successes": 91, "episodes": 100},
      "final": {"successes": 84, "episodes": 100},
      "note": "final checkpoint, 1 seed"
    }
  ]
}
```

`successes` and `episodes` are integer counts (pooled over seeds when several seeds are reported; say so in `note`).
`note` is optional.
`uv run python -m example_workflows._common.results_table <results.json> --write <README.md>` renders the table between the `<!-- results:start -->` and `<!-- results:end -->` markers of the README; `--check` verifies they still agree.
