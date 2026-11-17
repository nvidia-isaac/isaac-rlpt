# RoboLab native baselines

The protocol for native RoboLab evaluations of the served policies, and their status.
These are single-task native baselines and smoke tests, not a benchmark.

## Status

| Policy | Native server | RoboLab smoke | Reproduction suite | Harness behavior |
|---|---:|---:|---:|---:|
| GR00T N1.7 DROID | validated on L40 | passed on BananaOnPlateTask and BananaInBowlTask | single-task native reproductions match upstream | disabled |

## GR00T N1.7

The upstream reference reports 412 successes from 4,800 episodes across 120 RoboLab tasks with `--open-loop-horizon 8`, including 40 of 40 on `BananaOnPlateTask` and 31 of 40 on `BananaInBowlTask`.
A local run reproduces it when it matches the upstream per-task counts under the same open-loop horizon, camera composition and action decoding, on a renderer-compatible host.
The 40-episode runs on the two banana tasks have matched both references.
The next planned step is a small suite of medium-difficulty tasks selected from the local native sweep ([`TASK_SWEEP.md`](TASK_SWEEP.md)).

## Result records

Pin the external revisions in [`externals.lock.yaml`](externals.lock.yaml), and record for every result: task, model ID, server revision, RoboLab revision, seed or run index, environment count, open-loop horizon, success count, success rate, subtask score, event counts, p50 and p95 policy latency, output directory, and trace path.
Keep run artifacts outside this repository ([`SETUP.md`](SETUP.md#artifacts)); commit only the summary and the trace metadata needed to reproduce it.
