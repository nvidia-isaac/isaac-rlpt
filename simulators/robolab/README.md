# RoboLab

RoboLab is an Isaac Sim DROID-manipulation integration for exploring RL post-training against served robot foundation policies such as GR00T N1.7.
RoboLab and each policy run in their own environments, and this repository drives the policies as remote servers over their published wire protocols.
It has no reproduction suite and is not part of the benchmarked release; every number here is diagnostic, and the API may change.

## Reading order

1. [`SETUP.md`](SETUP.md): pinned external checkouts, the RoboLab environment, and the host preflight. Start here.
2. [`POLICY_SERVERS.md`](POLICY_SERVERS.md): start the GR00T N1.7 DROID server and run a native RoboLab smoke evaluation.
3. [`TASK_SWEEP.md`](TASK_SWEEP.md): validate task names before a launch and summarize results after it.
4. [`BASELINES.md`](BASELINES.md): the native-baseline protocol and what to record for a result.
5. [`GR00T.md`](GR00T.md): the opt-in GR00T latent endpoint for steering experiments.

## Status

| Policy | Native server and smoke | Reproduction | Steering |
|---|---|---|---|
| GR00T N1.7 DROID | validated | single-task native runs match the upstream per-task references | exploratory latent endpoint only; no DSRL success-rate result |

## Layout

- `runtime/`: the harness runtime (executive, arbiter, trace, capabilities, steering) the RoboLab policies build on.
- `env.py`, `policies/`, `scripts/`, `tests/`: the environment adapter, policy bridges, tooling and tests.
