# Base models

The frozen policies a run can steer, and the switch between them.
Each one sits behind the `BaseModelPolicy` contract ([add a base model](pages/extending/add_a_base_model.md)).

## Selection

The frozen policy is the `base_model` config group (`isaac_rl_posttraining/configs/base_model/`, default `gr00t`).

```bash
BASE_MODEL=mock uv run bash example_workflows/g1_arena_posttraining/scripts/eval.sh DrillLift base 50   # launcher knob
ARENA_SERVER_URL=tcp://127.0.0.1:7300 ARENA_EVAL_SERVER_URL=tcp://127.0.0.1:7301 \
ISAAC_RL_POSTTRAINING_DRYRUN=1 uv run python scripts/train.py base_model=mock                          # Hydra override
```

With `ISAAC_RL_POSTTRAINING_DRYRUN=1` a launcher prints the selection as `launch.override=base_model=<name>`, and the library's dry run that follows prints the resolved config, whose `base_model:` section names it.
Per-base-model fields (checkpoint paths, action geometry) live in the group file and are overridable as `base_model.<field>=...`.

## Available base models

| Base model | Steering | Status | Install | Checkpoints |
|---|---|---|---|---|
| `gr00t` | yes | the validated reference; results in `example_workflows/g1_arena_posttraining/README.md` | the Isaac-GR00T package in the same environment (`uv pip install -e /path/to/Isaac-GR00T`) | GR00T-N1.7-3B and its fine-tunes (Hugging Face) |
| `gr00t_n15`, `gr00t_n17` | yes (the same frozen GR00T base model) | names the generation, so fine-tuning (`algorithm=ppo`, `sft`, `iql`) picks its adapter without opening the checkpoint; `gr00t_n15` is the validated PPO reference | as `gr00t` | GR00T N1.5 / N1.7 checkpoints |
| `mock` | yes | test double; runs the whole pipeline with no GPU | nothing | none |
| `pi0` | yes | integrated on LIBERO; the setting of the DSRL paper | the [dsrl_openpi](https://github.com/jesbu1/dsrl_openpi) fork installed from source | openpi checkpoints (`gs://openpi-assets`) |
| `dreamzero_local` | yes | experimental; loads the frozen DreamZero world-action model in this process | `uv sync --extra dreamzero-local`, then `uv pip install -e /path/to/dreamzero --no-deps` | DreamZero checkpoint directories (`config.json`, `model.safetensors`, `experiment_cfg/metadata.json`) |
| `dreamzero_server` | no (native evaluation only) | experimental; client to a served DreamZero model | `uv sync --inexact --extra dreamzero` (`--inexact` keeps packages installed with `uv pip`, such as GR00T and openpi) | served by an external DreamZero server |

Fine-tuning uses the same `base_model=` selection; `gr00t_n15`, `gr00t_n17` and `pi0` have a trainable adapter (`isaac_rl_posttraining/policies/trainable.py`), the others do not.
A base model with `supports_steering=False` can only run native evaluation, and the training entry point enforces that up front.

## DreamZero notes

- The DreamZero action head draws one initial action-noise tensor (`[B, action_horizon, 32]`) to seed its denoising; steering that draw is DSRL. `dreamzero_local` intercepts only that draw with a sampler patch (`isaac_rl_posttraining/policies/dreamzero/patch.py`); the video latent keeps the model's own seeded behavior. A consumed-flag check makes "the sample never reached the sampler" a hard error, and the patch target is checked against the installed `groot` version at load time.
- In-process inference is stateless per chunk: observations enter as single frames, which resets the head's autoregressive video context on every call. Native mode (`noise=None`) keeps the model's fixed-seed draw, the base policy's own evaluation protocol.
- The served protocol has no field for an injected sample, so `dreamzero_server` is native-only. Served DreamZero keeps autoregressive state within an episode; a harness using it must call `base_model.start_episode()` after each environment reset.
