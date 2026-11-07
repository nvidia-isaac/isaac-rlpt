# GR00T N1.7 latent endpoint

`simulators/robolab/scripts/gr00t_latent_server.py` is an opt-in GR00T server that adds `options.harness_latent` to the stock server contract, for steering experiments.
It clips the requested latent once at the server boundary, uses it in place of the flow head's initial noise, and returns the clipping metadata with the decoded action chunk.
The stock GR00T RoboLab server and the no-op harness are unchanged.
It verifies the integration only; it has produced no benchmark or post-training results.

## Setup

The checkpoint needs the gated `nvidia/Cosmos-Reason2-2B` backbone.
Before starting the server, set the cache location and authenticate for any missing gated files (an authenticated cache with every snapshot file present is enough):

```bash
export POLICY_HARNESS_ROOT=/path/to/policy_harness
export HF_HOME="$POLICY_HARNESS_ROOT/cache/huggingface"
```

Never put a Hugging Face token in source, artifacts, commands or logs.

RoboLab's environment also needs the harness logging dependency:

```bash
uv pip install --python "$EXT_DEPS_ROOT/robolab/.venv/bin/python" loguru==0.7.3
```

## Recorded evidence

- `$POLICY_HARNESS_ROOT/artifacts/robolab_native/gr00t_latent_probe_v1/run_v7`: `probe_v7.json` records seeded zero-latent parity, a changed action for a bounded nonzero latent, and server-side clipping of all eight entries of a `100.0` probe; `environment_fix.json` records the cache precondition that made the model load reproducibly.
- `$POLICY_HARNESS_ROOT/artifacts/robolab_native/gr00t_dsrl_smoke_banana_in_bowl_v5`: one exploratory DSRL smoke, a single `BananaInBowlTask` episode (101 environment steps, 13 latent decision transitions, one SAC update) with its 14-row JSONL trace. The actor saw only a 15-D DROID proprioceptive vector and a fixed language vector. The serial collector that drove it is no longer in the repository, so this smoke cannot be rerun as is, and it is not a success-rate claim.
