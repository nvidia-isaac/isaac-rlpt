# Base model internals

How the registry, the run builder and the fine-tuning path use a base model.
The step-by-step guide for adding one, including the `BaseModelPolicy` contract and its five rules, is [add a base model](pages/extending/add_a_base_model.md).
Nothing in `isaac_rl_posttraining/algorithms/` touches model internals; everything goes through `BaseModelPolicy` in `isaac_rl_posttraining/policies/base.py`.

## Registry

`isaac_rl_posttraining/policies/__init__.py` holds two tables: `_BASE_MODELS` (name to module and class) and `_OPTIONAL_REQUIREMENTS` (name to the packages construction needs).
`get_base_model(name)` imports the module only on first use, and `available_base_models()` lists only the base models whose requirements are installed.
`register_base_model(...)` adds an entry at run time for code that cannot edit the table; it does not fire on its own, so the caller's package must call it during setup.
`gr00t`, `gr00t_n15` and `gr00t_n17` name the same frozen class; the generation-specific names let fine-tuning pick its adapter by `base_model.name` alone.

## Frozen policy construction

The run builder (`isaac_rl_posttraining/runners/off_policy_launch.py`) reads `base_model.name`, resolves the class through `get_base_model` and calls `from_dsrl_config(cfg, device, dataset_adapter=...)`.
A registered base model with a config group under `isaac_rl_posttraining/configs/base_model/` is therefore picked up without editing dispatch code.
`BaseModelConfig` (`isaac_rl_posttraining/configs/base_model_schema.py`) is the union of every base model's fields, stored as the `base_base_model` schema each group file extends.
The launchers select a base model with `BASE_MODEL=<name>`; the SAC networks are sized from `dsrl.noise_dim`, `dsrl.noise_action_bound` and `training.chunk_size`.

Once loaded, `BaseModelPolicy.input_spec()` returns the observation entries the base model reads, which the launch checks against the environment ([data_model.md](data_model.md#specs-and-checks)).

## Contract tests

`tests/unit/test_base_model_policy.py` runs a base-model-generic suite over the `any_base_model` fixture.
It pins the output shape, native mode, both noise modes, shape validation, clipped-latent stability and the capability flag.

## Trainable adapters

PPO fine-tuning (`algorithm=ppo`, [the PPO page](pages/algorithms/ppo.md)) requires three things a frozen policy does not provide: sampling an action chunk with a tractable likelihood, sending the learner what it needs to recompute that likelihood, and training the model's weights on the learner ranks.
That contract is `TrainablePolicy` in `isaac_rl_posttraining/policies/trainable.py`; the offline launcher (`algorithm=sft`, `algorithm=iql`, `isaac_rl_posttraining/runners/offline_launch.py`) loads the same adapters.
The shared implementation for flow-matching heads is `FlowSDEActorCritic` in `isaac_rl_posttraining/policies/flow_sde.py`: one stochastic denoising step, the chunk log-density, the value head and transport of inputs.
A base model adapter subclasses it and supplies its model's pieces:

| Member | Meaning |
|---|---|
| `frozen_submodules` | dotted paths (relative to the adapter) that never train; kept whole on every learner rank, absent from published policy states |
| `cached_feature_names`, `requires_cached_features` | the frozen part's outputs a decision may ship instead of the inputs they came from (empty when nothing is cached); `requires_cached_features` when the raw inputs have no per-sample layout at all |
| `build_inputs(frames, states, instruction)` | a batch of camera frames and state parts to model inputs |
| `condition(inputs, cached_features=None)` | run the frozen part (or take its cached outputs) and the head's input encoders once per decode |
| `velocity(conditioning, x_t, step_index)` | the head's velocity at a chain state on the shared grid (noise at step 0, data at step `K`); a flow parameterized the other way maps its time here (the pi0 adapter negates its velocity) |
| `value_features(conditioning)` | what the value head pools |
| `decode(normalized, inputs=, states=)` | normalized chunks to raw environment actions |
| `learner_inputs`, `pack_inputs`, `unpack_inputs` | override only when the model's inputs need their own transport (GR00T N1.5 ships its pixels as bfloat16 bits) |
| `spec_options(base_model)`, `load(checkpoint, *, device, flow_sde, value_head_seed, **options)` | class methods: extract and validate the base model's own fields from the `base_model` config section before any weights load, and build the adapter on one process |

Adapters register in `_TRAINABLE_POLICIES` (plus `_OPTIONAL_REQUIREMENTS` when loading needs an optional package) in `isaac_rl_posttraining/policies/trainable.py`, or at run time with `register_trainable_policy(...)`.
The key is the `base_model.name` of the config group, so one `base_model=` selection serves both steering and fine-tuning.
