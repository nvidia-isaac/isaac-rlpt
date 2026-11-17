# Replay data and typed records

What one recorded decision looks like, how a saved replay buffer is laid out, and how to inspect, save and reuse one.

## Buffer inspection

```bash
uv run python -m isaac_rl_posttraining.data.inspect /path/to/replay_buffers/buffer_latest
```

Prints the buffer's schema (version, reward channels, observation spec), the episode counts by outcome, the transitions by who acted, the mean return per reward channel over the complete episodes, and the shapes of the first transition.
The buffer directory is the only argument; a folder of `part-NNNNN` buffers (what `data.persist.path` writes) is read as one buffer.

## Episode persistence

```bash
uv run bash example_workflows/g1_arena_posttraining/scripts/train.sh DrillLift data.persist.path=collected
```

Streams each completed training episode to `<run dir>/collected/` as it arrives, in the saved replay-buffer format, a `part-NNNNN` folder every `data.persist.every_episodes` episodes (100 by default) and the rest when the run ends.
It works for every algorithm (DSRL, RFS, SAC-edit, PPO, DSRL-PPO) and runtime, including the on-policy ones that keep no replay buffer; an absolute path is used as given.
Camera frames are dropped unless `data.persist.keep_frames=true`; evaluation episodes are not saved.
`load_buffer("<run dir>/collected")` reads everything back, and the `inspect` command above summarizes it.
From Python, `launch(..., persist="collected")` or `persist=PersistSpec(path, every_episodes=..., keep_frames=...)` sets the same keys.

## Buffer saving

```bash
uv run bash example_workflows/g1_arena_posttraining/scripts/train.sh DrillLift \
  buffer.save_buffer_on_exit=true buffer.save_buffer_every=10000
```

A DSRL run that writes its replay buffer every 10,000 environment steps and once more when it ends or is killed.
Buffers land in `replay_buffers/` next to the run's `checkpoints/`: `buffer_latest/` for the periodic save (overwritten each time), `buffer_final_<timestamp>/` when the run ends, and `buffer_signal_exit_<timestamp>/` when it is killed (SIGINT or SIGTERM).
The final and signal-exit saves need `buffer.save_buffer_on_exit=true`; `buffer.save_buffer_every` alone writes only `buffer_latest/`.
A run killed before it stores its first transition writes no signal-exit buffer.
Every off-policy run of `scripts/train.py` (`algorithm=dsrl_sac`, `rfs`, `sac_edit`) takes the same two keys.

## Demonstrations from a saved buffer

```bash
uv run bash example_workflows/g1_arena_posttraining/scripts/train.sh DrillLift \
  buffer.demonstrations=/path/to/replay_buffers/buffer_latest buffer.demonstration_fraction=0.25
```

Mixes a saved buffer into every training batch: a quarter of each batch comes from the demonstrations, the rest from the run's own replay.
`buffer.demonstrations` and `buffer.demonstration_fraction` are set together, and the fraction lies strictly between 0 and 1; the run refuses either key alone.
The demonstrations are read with the learner's reward channel and aggregation, and every decision in them must have the recipe's chunk size.

## Notes

- When the buffer is full, the oldest complete episodes are evicted whole; an episode still running is never evicted. The `buffer` keys are listed under [configuration](#configuration).
- PPO (`algorithm=ppo`, `algorithm=dsrl_ppo`) trains on fresh episodes and ignores `buffer.demonstrations` and `buffer.demonstration_fraction`; `dsrl_ppo` logs a warning when they are set.
- A key the schema does not declare is refused at launch, in any group, with the settings that group does have, so a typo fails immediately instead of silently running the default.
- The dry run prints `plan.data_specs=...`: what the simulator declares it emits and accepts, what the policy reads, which of them were checked and which could not be (for example a width only known once the checkpoint loads). A definite disagreement fails the launch with `SpecMismatch`; `plan.checks=` closes the plan with the counts and every failure at once. The dry-run command for each shipped simulator is in [add a simulator](pages/extending/add_a_simulator.md#verify).
- A simulator that declares its observation as a small dataclass ([the records](#the-records)) gets the launch check, the load-time check and the transport handshake with it.

## Reference

### The records

Every record (`isaac_rl_posttraining/data/`) is a frozen dataclass inheriting from `Record`.
Building one coerces its arrays with NumPy, checks each field against its annotation and runs the cross-field rules in `validate()`.
The package depends on NumPy only, so every layer can import it.

**Observation.** What a simulator emits at one step; a concrete type declares its components as annotated fields, as the G1 Factory client does (`ArenaObservation` in `simulators/g1_arena/env.py`):

```python
from dataclasses import dataclass
import numpy as np
from isaac_rl_posttraining.data import Array, Observation, Text, dim

@dataclass(frozen=True)
class MyObservation(Observation):
    image: Array[np.uint8, dim.H, dim.W, 3]
    state: Array[np.float32, dim.state_dim]
    instruction: Text | None = None
```

The class gains `spec()`, `declared_spec(bindings)`, `bindings()`, `validate()`, `to_mapping()` and `from_mapping()`.

| Annotation | Meaning |
|---|---|
| `Array[dtype, *dims]` | an array; each dim is a fixed size (`3`), a named size (`dim.H`) or `None` for any size |
| `Text` | a string, such as a language instruction |
| `ArrayMap` | named arrays whose keys are only known at run time, such as cached activations |
| another `Observation` subclass | a nested record, flattened as `field/component` |
| `X \| None = None` | optional; left out of the declared spec unless asked for |

A named size binds once, from config or from the first instance, and every field sharing the name must agree, so a `state` of 31 next to a `previous_state` of 43 is rejected.
`DynamicObservation` wraps an observation without a declared type (`Observation.from_mapping(dict)`); its spec is measured from the instance and carries no named sizes.

**LearnerAction and EnvAction.** A decision produces two actions.

| Record | Fields | Rules |
|---|---|---|
| `LearnerAction` | `kind` (`latent`, `edit`, `chunk` or `chain`), `value` | a floating-point array of rank at least one |
| `EnvAction` | `commands` `[steps, act_dim]`, `execution_length` | rank two; `execution_length` is at least `steps` |

`kind` says what the learner chose: the steering latent (DSRL, RFS, steering PPO), the edit around the frozen decode (SAC-edit), the action chunk itself, or the denoising chain of a flow policy fine-tuned with PPO.
`steps` is how many commands actually ran, shorter than `execution_length` when the episode ended inside the chunk.

**PolicyOutputs.** Named arrays a policy produced alongside its action: `log_prob`, `value`, `next_value` and `entropy` (scalars PPO reads), `action_mean` and `action_std` (`[chunk_steps, act_dim]`), `denoising_log_prob` (`[denoising_steps]`) and `features` (cached activations of frozen sub-modules, `isaac_rl_posttraining/policies/feature_cache.py`).
Every field is optional.

**Transition.** One decision: the observation the policy saw, what it chose, the steps the simulator executed, and the observation after them.

| Field | Type | Meaning |
|---|---|---|
| `episode_id` | int or str | the episode this decision belongs to |
| `observation`, `next_observation` | `Observation` | before the decision and after its last executed step |
| `learner_action` | `LearnerAction` or None | None for a native rollout |
| `env_action` | `EnvAction` or None | the executed commands |
| `rewards` | mapping of `float32[T]` | named reward channels; `env` is always present |
| `terminated`, `truncated`, `success` | `bool[T]` | Gymnasium's two ending flags plus the task outcome |
| `behavior_source` | `policy`, `warmup`, `native` or `expert` | who produced the action |
| `metadata` | `TransitionMetadata` | `step_in_episode`, `decision_index`, `policy_version`, `environment_index`, `reset_seed`, `reset_seed_scope`, `wall_time`, `max_steps_in_episode`, `intervention`, `extra` |
| `policy_outputs` | `PolicyOutputs` | the policy's extra outputs |

`T` is the number of executed steps.
Every per-step array has length `T`, only the last step may set an ending flag, `success` counts only on the ending step, and every channel is `float32[T]`.
Views: `ended`, `succeeded`, `outcome` (`ongoing`, `success`, `failure` or `timeout`) and `decision_reward(channel, aggregation=...)`.
`Transition.from_legacy_fields(fields, kind=..., env_action=...)` and `to_legacy_fields(aggregation=..., reward_channel=...)` convert to and from the flat field mapping (`obs`, `action`, `reward`, `done`, ...) of older buffers.

**Episode.** The ordered transitions of one episode and its `EpisodeMetadata` (`task`, `instruction`, `seed`, `initial_state_id`, `embodiment`, `behavior_source`, `policy_version_at_start`).
Building one checks contiguous steps and that only the last transition ends the episode; `length`, `complete`, `outcome` and `return_(channel)` summarize it.

### Specs and checks

| Spec | Describes |
|---|---|
| `ArraySpec(shape, dtype)` | one array; shape entries are fixed sizes, named sizes or None |
| `ObservationSpec(arrays, text)` | the components and text entries of an observation; `ObservationSpec.from_space(space)` reads a Gymnasium `Dict` space |
| `ActionSpec(width, dtype, horizon)` | the per-step action width a policy emits or a simulator accepts |

Comparing two specs yields one of three outcomes: a mismatch, a checked entry, or an unverified entry.
Fixed sizes are compared, a fixed size against a bound named size is checked, and a free or unbound size is reported as unverified rather than accepted.

*At plan time*, `train_plan` (`isaac_rl_posttraining/runners/build.py`) calls `check_data_specs` (`isaac_rl_posttraining/runners/spec_checks.py`) before anything loads and prints `plan.data_specs=`.
A `SimulatorEntry` (`isaac_rl_posttraining/envs/simulator_entry.py`) declares what it emits and accepts by overriding `observation_spec(spec)` and `action_spec(spec)`; the base class returns `None`, which the plan reports as unverified.
The G1 Factory entry (`simulators/g1_arena/runtime.py`) binds `H`, `W` and `state_dim` from the configured image resolution and state width, so its plan line reads `simulator g1_arena emits {ego_view_image: uint8[480, 640, 3], state: float32[31]} and accepts float32[35]`.
The LIBERO entries (`simulators/libero/runtime.py`) declare each convention's cameras, the 8-dim end-effector state and the 7-dim action; the Isaac Lab entry (`simulators/isaaclab/runtime.py`) declares the task mapping's cameras at `simulator.image_resolution`, its state groups and `simulator.action_dim`.
The policy side comes from the entry's `steering_key_mapping`, `base_model.action_dim`, `dsrl.noise_dim` and the base model's geometry (`isaac_rl_posttraining/policies/geometry.py`): the mock base model's configured widths, and for GR00T the checkpoint's JSON metadata (`isaac_rl_posttraining/policies/gr00t/checkpoint_layout.py`: state width, decoded action width and steps, flow horizon and head width), read without loading weights.
What neither side states before load is listed as unverified with the reason.

*At load time*, `BaseModelPolicy.input_spec()` (`isaac_rl_posttraining/policies/base.py`) returns the observation entries a loaded base model reads, with the widths its checkpoint fixes; `check_policy_inputs_at_load` compares them with the environment's observation space as soon as the sizing environment exists and raises on a definite disagreement.

*On the transport*, a remote simulator server answers the `SPEC` request (`isaac_rl_posttraining/envs/transport.py`) with a `specs` entry (`spec_payload`), and the client runs `handshake(...)` when it connects; a mismatch closes the connection with `SpecMismatch` before the first episode.
The Arena server (`simulators/g1_arena/env_server.py`) declares its specs and both Arena clients (`simulators/g1_arena/env.py`) check them; a server without a declaration is reported as unverified.

### Reward channels

`Transition.rewards` maps channel names to `float32[T]`.
The `env` channel is what the simulator returned and is never rewritten, so a run can be re-scored without losing the ground truth.

| Channel | Source |
|---|---|
| `env` | the simulator, recorded by the worker |
| `sparse_outcome` | `SparseOutcomeReward` (`isaac_rl_posttraining/runtimes/gym_worker.py`, selected by `env.sparse_success_reward`): the outcome value on the step that ended the episode, the step value elsewhere |
| `reward_model/<id>`, or any other name | a relabel pass that scores stored episodes: `buffer.relabel(channel, score)` |

The buffer is built with the channel and the per-decision aggregation the learner trains on (`reward_channel`, `reward_aggregation`), every batch carries that channel as `reward`, and the collection metrics report it per episode.
Evaluation scores the channel its own workers record (`reward_transform_for` in `isaac_rl_posttraining/runtimes/policy_families.py`): the raw `env` reward for the steering family and the shaped reward for the edit family, so with a sparse outcome reward the evaluation and training returns are on different scales while the success rates compare directly.

```python
buffer.relabel("reward_model/progress", lambda episode: [model(t) for t in episode.transitions])
```

Relabeling replaces each transition of every complete episode with a copy carrying the new channel; `env` and every other field are untouched, and an episode still running is left alone until it ends.

### Replay buffer and samplers

`ReplayBuffer(capacity, drop_components=(), reward_channel="env", reward_aggregation="last")` (`isaac_rl_posttraining/buffers/replay.py`) stores whole episodes; the transitions of an episode in progress are sampleable as they arrive.
`add(transition)` checks that steps are contiguous; `indices(outcome=..., exclude_outcome=..., complete=..., source=...)`, `count(...)` and `draw(rng, count, ...)` select and sample; `episodes()`, `transitions()`, `transition(index)`, `get_statistics()`, `reward_statistics()`, `relabel(...)` and `collate(indices)` complete the surface.
`drop_components` names the observation components the buffer does not store, such as raw camera frames the learner never reads.

A learner receives a `Sampler` (`sample(batch_size) -> Batch`, `can_sample(batch_size)`), never a buffer layout; every sampler owns a seeded NumPy generator, so draws are reproducible.

| Sampler (`isaac_rl_posttraining/buffers/samplers.py`) | Draws |
|---|---|
| `UniformSampler(buffer, **select)` | uniformly over every stored transition, or over the ones `indices(**select)` selects (`outcome="success"`) |
| `SuccessFractionSampler(buffer, fraction)` | a fixed share of every batch from successful episodes and the rest from the other finished ones; when one side is empty, the other supplies the whole batch |
| `MixedSampler({"online": ..., "demos": ...}, fractions)` | fixed shares from several samplers or buffers, renormalized over the members that can supply |
| `SequenceSampler(buffer, length)` | windows of consecutive decisions of one episode, stacked along a time axis |

`create_replay` (`isaac_rl_posttraining/utils/training_utils.py`) maps the `buffer` config onto them: the defaults give `UniformSampler`, `success_fraction: p` gives `SuccessFractionSampler`, and `demonstrations` with `demonstration_fraction: q` wraps either in a `MixedSampler` with a `UniformSampler` over the loaded demonstrations.
`BaseAlgorithm.learner_batch()` packs the draws of one `train_step` (for SAC: `replay`, plus `expert` when `expert_bc_coef` is on and `online_success` when `online_success_bc_coef` is on) into one `LearnerBatch`, and `ReplayLearner` (`isaac_rl_posttraining/algorithms/replay_learner.py`) unpacks it on the learner, so the learner never samples for itself.

A `Batch` is built by `collate`; `to_tensors(batch, device)` moves it to torch.

| Key | Content |
|---|---|
| `obs`, `next_obs` | dict of stacked observation components; floats are narrowed to float32, any other dtype is kept, so a `uint8` camera frame stays bytes |
| `action` | the learner action (float32) |
| `reward` | the buffer's reward channel aggregated per decision |
| `done`, `truncated` | the last step's flags; `done` is the terminal flag (terminated wins when both are set, as in the on-policy loop); a time-limit cut alone is never a terminal in a TD target |
| `bootstrap_steps` | executed steps of the decision (int64), for chunk discounting |
| `reward_sequence`, `behavior_source`, `info` | per-row lists: the per-step rewards of the channel, who acted, and the metadata's `extra` |

### Saved buffer format

A saved buffer is a directory (`isaac_rl_posttraining/buffers/storage.py`):

```text
<buffer>/
  schema.json      # schema version, the buffer's read settings, the reward channels, the observation spec
  metadata.json    # one entry per episode and per transition: ids, sources, metadata, text
  episodes.npz     # every array field over all transitions, episode by episode
```

Arrays only some transitions carry (a learner action, a relabeled channel, a policy output) are stored with a presence mask next to their values.
`save_buffer(buffer, path)` writes a buffer; `load_buffer(path, capacity=..., reward_channel=..., reward_aggregation=...)` rebuilds one, validating every transition and taking the read settings from the file unless given.
The schema version lets a newer reader load an older file: fields a newer writer adds are read with defaults when a file lacks them.

### Configuration

The `buffer` section (`ReplayBufferConfig` in `isaac_rl_posttraining/configs/schema.py`; the composed defaults are in `isaac_rl_posttraining/configs/train.yaml`):

| Key | Default | Meaning |
|---|---|---|
| `capacity` | 200000 | transitions kept; the oldest complete episodes are evicted whole |
| `reward_aggregation` | `last` | a decision's training reward: its outcome step (`last`) or the sum of its steps (`sum`) |
| `success_fraction` | null | share of every batch drawn from successful episodes (null = uniform) |
| `demonstrations` | null | saved buffer directory mixed into every batch |
| `demonstration_fraction` | null | share of every batch drawn from the demonstrations |
| `save_buffer_on_exit` | false | write the buffer when the run ends or is killed |
| `save_buffer_every` | 0 | also write it every N environment steps (0 = never) |
| `save_buffer_dir` | `replay_buffers` | subdirectory of the run for saved buffers |

Tests: `tests/unit/test_data_model.py` (records and specs), `tests/unit/test_replay_inspect.py` (loading an older file), `tests/unit/test_replay_learner.py` (the learner draws through samplers only) and `tests/unit/test_transports.py` (the handshake refusal over each transport).
