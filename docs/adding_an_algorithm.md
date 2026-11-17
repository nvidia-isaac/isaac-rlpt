# Algorithm layer

How the algorithm layer is organized.
The step-by-step procedure for landing a new method (package, registration, config file, launcher, tests, verification) is [Adding an algorithm](pages/extending/add_an_algorithm.md); what each shipped method does is on its algorithm page ([DSRL](pages/algorithms/dsrl.md), [RFS](pages/algorithms/rfs.md), [SAC-edit](pages/algorithms/sac_edit.md), [PPO](pages/algorithms/ppo.md), [SFT](pages/algorithms/sft.md), [IQL](pages/algorithms/iql.md)).

## Layering

- **`isaac_rl_posttraining/algorithms/sac/` is the generic core.**
  Ensemble soft actor-critic (randomized TD-target subsets, target networks, entropy auto-tuning) with a small set of override hooks: critic outputs and TD target, actor observations and regularization, reward shaping, and the log-probability reduction.
  Every hook defaults to the identity, so the DSRL recipe is the plain core.
  Nothing here imports a base model.
- **`isaac_rl_posttraining/algorithms/registry.py` lists every available method.**
  One `AlgorithmSpec` per method names its loop, action space, config class (a `BaseAlgorithmConfig` subclass) and launcher by lazy `"module:attr"` import, mirroring the base-model registry.
  `isaac_rl_posttraining/configs/registry.py` iterates it to register the Hydra `algorithm` group.
- **Method packages hold only method-specific code.**
  `isaac_rl_posttraining/algorithms/dsrl/` holds the steering geometry (the `dsrl:` config section); the receding-horizon discount it folds into `gamma` is the shared `chunk_discounted_gamma` in `isaac_rl_posttraining/algorithms/discounting.py`.
  `isaac_rl_posttraining/algorithms/rfs/` adds a deterministic bounded residual head next to the flow latent (the `rfs:` section).
  `isaac_rl_posttraining/algorithms/sac_edit/` holds the `ReferenceEditActor` (a bounded edit around the frozen policy's decoded chunk) and the SAC subclass that scales the actor's Q term and anchors the edit to the reference (the `direct_action:` section).
  It also carries optional reward and input normalizers, off in the shipped recipe.
  `isaac_rl_posttraining/algorithms/sft/` and `isaac_rl_posttraining/algorithms/iql/` train the base model itself on the offline loop ([sft.md](sft.md), [iql.md](iql.md)).
- **`isaac_rl_posttraining/policies/<name>/`** owns how a latent becomes the model's initial noise and how the frozen policy decodes a reference chunk (`infer_reference`), behind the [`BaseModelPolicy`](adding_a_base_model.md) contract.
- **`isaac_rl_posttraining/runtimes/`** owns chunked execution: the steering rollout policy (`isaac_rl_posttraining/runtimes/steering_policy.py`) serves every method that acts through the flow latent (DSRL, RFS, steering PPO) and the reference-edit policy (`isaac_rl_posttraining/runtimes/edit_policy.py`) serves direct-action methods (SAC-edit); `isaac_rl_posttraining/rollouts/steering_step.py` holds the decision step they share and `isaac_rl_posttraining/training/evaluation.py` scores episodes.
- **`isaac_rl_posttraining/algorithms/base.py`** is the shared scaffolding: component registry, the learner modules and optimizers the runtime manages, and the trainer-facing hooks (`on_evaluation`, `observe_rewards`, `target_update_count`); it assumes nothing about actors or critics, so the Q-ensemble builder lives in the SAC core.

## Routing

The registered launcher (`isaac_rl_posttraining/runners/launchers.py`) runs the method's `loop` (`off_policy`, `on_policy`, `offline`), and `isaac_rl_posttraining/runners/build.py` checks the selected schedule against that loop.
Within the off-policy loop, the registered `action_space` (`latent`, `latent_residual`, `edit`) selects the learner's environment stand-in and networks in `isaac_rl_posttraining/runners/off_policy_models.py`; nothing there names algorithms (the table is on [Adding an algorithm](pages/extending/add_an_algorithm.md#intervention-types)).
`setup_dsrl_training` in `isaac_rl_posttraining/runners/off_policy_launch.py` builds the algorithm config from the composed group, injects the runtime objects (`env`, `actor`, `critic`, `buffer`, `action_space`, `logger`), and keeps method-specific glue to a few lines.
The entry point round-trips the composed config through the dataclasses and refuses, before anything loads, a key the schema does not declare, naming the settings it does have.

## Algorithm responsibilities

Subclass `BaseAlgorithm` (or the SAC core) and own exactly the learning state:

- networks, target networks, optimizers, and learned coefficients, registered in `component_names` (the shared names live in `isaac_rl_posttraining/algorithms/constants.py`);
- which of them the learner runtime places and checkpoints itself: `learner_modules()` (exactly one marked `publish`, the policy the rollout workers run) and `learner_optimizers()`; every other component travels in `learner_state()`;
- `train_step(...) -> dict`, one training call returning metrics and a no-op below `config.learning_starts`;
- a config dataclass subclassing `BaseAlgorithmConfig` (or `SACConfig`) with the hyperparameters plus the runtime wiring slots the entry point injects.

The learner runtime writes and restores checkpoints from these declarations; `load()` only reads the older per-component layout (`<name>.pt` files), with strict shapes.
Settings keyed on the environment step (the SAC actor window, actor Q-loss start and switch, behavior anchor) act only with `algorithm.respect_step_schedules=true`, which passes the collected step count to every update; without it, setting them fails at construction.

## Harness responsibilities

`OffPolicyTrainer` (`isaac_rl_posttraining/training/off_policy.py`) owns the loop: collection, per-episode training bursts, evaluation scheduling at `eval_freq` boundaries, boundary-aligned checkpoints with save-failure resilience and weight fingerprints, buffer persistence, and publishing the trained weights to the rollout workers.
The rollout workers and evaluators own chunked execution, the terminal-outcome reward, N-episode scoring, and the native-policy base mode.
The learner draws its batches through samplers ([data_model.md](data_model.md)); it never touches the buffer layout.
