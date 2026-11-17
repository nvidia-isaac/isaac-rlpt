# Concepts

This page defines the terms the rest of the site uses.

## Configuration choices

Every run is one command with four selections.

| Choice | Example | What it picks |
|---|---|---|
| `simulator=` | `g1_arena` | the task family, its task list and its per-task step caps |
| `base_model=` | `gr00t` | the frozen policy |
| `algorithm=` | `dsrl_sac` | the RL method |
| `runtime=` | `native` | where the rollout workers run |

```bash
ARENA_SERVER_URL=tcp://127.0.0.1:7300 ARENA_EVAL_SERVER_URL=tcp://127.0.0.1:7301 \
uv run python scripts/train.py algorithm=dsrl_sac simulator=g1_arena base_model=gr00t runtime=native
```

G1 Factory needs the two addresses, which its launchers set for you; a simulator that runs inside the training process, such as LIBERO, needs neither.
Any other setting is a dotted override on the same line, for example `dsrl.noise_action_bound=2.0`.
An unknown name is refused at launch with a list of the names that group does have.
[cli_reference.md](cli_reference.md) lists them all.

## Frozen base model

A **base model** is a pretrained robot policy: it reads camera images, joint readings and a language instruction, and produces actions.
The ones used here are GR00T N1.7, pi0 and DreamZero, each trained by supervised learning on teleoperation demonstrations.

**Frozen** means the steering methods never change its weights.
They train a separate, much smaller learner around it, and a failed run leaves the policy unchanged.

`base_model=` picks it: `gr00t` (the default), `gr00t_n15`, `gr00t_n17`, `pi0`, `dreamzero_local`, `dreamzero_server`, or `mock` for a check that needs no GPU.
In Python it is `BaseModel("gr00t_n17", ckpt=...)`, passed to an algorithm as `base=`.
The code that loads and runs one kind of base model is a `BaseModelPolicy` subclass, registered under the name `base_model=` selects.

The **base part** is where the base model runs inside a steering policy, next to the learner's actor: in every rollout worker, or in a policy server with `placement.parts.base.mode=served`.

## Action chunks and the steering latent

These policies emit an **action chunk**, a short sequence of future actions, in one forward pass.
They build it by refining a random noise sample over a few denoising steps.
That sample is the only source of randomness, so choosing it chooses the behaviour without touching a weight.
A sample chosen by a learner is the **steering latent**.

`dsrl.noise_dim` is its width per step: 35 for the G1 policy, 32 for the Isaac Lab stack-cube checkpoint, 7 for GR00T on LIBERO.
Two lengths describe a sample; the workflow launchers set both.

- `training.chunk_size`: how many per-step latents one sample holds, matching the policy's noise window (40 for the G1 policy).
- `dsrl.action_exec_len`: how many decoded actions run before the policy is called again (8 of the G1 policy's 16).

## Algorithms

| Method | What the learner emits | Optimizer |
|---|---|---|
| DSRL (`algorithm=dsrl_sac`) | the steering latent | SAC |
| DSRL-PPO (`algorithm=dsrl_ppo`) | the steering latent | PPO |
| RFS (`algorithm=rfs`) | the steering latent, plus a small bounded residual added to the decoded actions | SAC |
| SAC-edit (`algorithm=sac_edit`) | a bounded edit of the chunk the frozen policy proposed, and no latent | SAC |
| PPO (`algorithm=ppo`) | nothing: it fine-tunes the policy's action head from live rollouts, with the vision-language backbone frozen | PPO |
| IQL (`algorithm=iql`) | nothing: it fine-tunes the action head from demonstrations, weighted by token critics (untested) | IQL |
| SFT (`algorithm=sft`) | nothing: it fine-tunes the action head from demonstrations by behavior cloning (untested) | AdamW |

The first four are **steering**: the policy's weights never change.
PPO learns from live collection; IQL and SFT learn from a fixed demonstration dataset.
SAC stores every transition in a replay buffer and reuses it many times; PPO collects a batch with the current policy, updates, and discards it.
Each method has its own page: [DSRL](algorithms/dsrl.md), [RFS](algorithms/rfs.md), [SAC-edit](algorithms/sac_edit.md), [PPO](algorithms/ppo.md), [IQL](algorithms/iql.md), [SFT](algorithms/sft.md).

## Decision budget

A run's budget is counted in environment steps: `training.num_rollouts`, 40,000 in the reference recipe.
One **decision** is one call to the frozen policy:

1. the learner emits one steering latent;
2. the frozen policy decodes it into an action chunk;
3. the first `dsrl.action_exec_len` actions run in the simulator;
4. one transition is stored, with the latent as the action and the chunk's outcome as the reward.

Because one transition covers several simulator steps, `algorithm.gamma` is raised to that power once at setup and the learner discounts per decision.

Three separate counts set the parallelism:

- `training.num_envs`: simulator scenes collecting in parallel.
- `placement.envs.workers`: processes advancing them.
- `placement.learner.ranks`: learner processes.

`runtime=native` (the default) runs everything in the training process, or on Ray actors when the placement asks for several processes; `rlinf` spreads the processes out on its own framework.

## Reward

The shipped recipes use a sparse terminal reward: 0 when the episode succeeds, -1 when it ends without succeeding, 0 on every other step.
The shipped environments end only on success, so the step cap ends a failed episode.
The simulator's own reward is kept as a separate channel in the replay buffer, so a run can be re-scored later.

## Evaluation

An **episode** is one attempt at the task, from a reset until success or the task's **step cap**, which is defined only in the simulator config.
A **success rate** is the fraction of episodes that succeeded, written with its episode count as `<rate>% n=<episodes>`.

Two settings change what a number means, so the protocol fixes both:

- Evaluate by **sampling** the steering policy (`eval.action_selection=sample`, the default). The mean latent understates it.
- Give the policy the **native camera frame**. A pre-resized image is the usual cause of a task scoring zero that the policy otherwise handles.

Training evaluates every `eval.eval_freq` environment steps over `eval.eval_num_episodes` episodes and appends each result to `checkpoints/evaluation_summary.json`.
Report a retained checkpoint's fresh score, not the evaluation that selected it, next to the frozen policy's score under the same evaluator, with the episode count.
`example_workflows/_common/EVALUATION.md` states the full protocol, and each workflow page's Results section holds its measured numbers.

## Replay buffer

SAC keeps up to `buffer.capacity` transitions (200,000 by default) and drops the oldest complete episodes when full.
`buffer.save_buffer_on_exit=true` writes it to disk; read one back with:

```bash
uv run python -m isaac_rl_posttraining.data.inspect /path/to/replay_buffers/buffer_latest
```

`buffer.demonstrations=<dir>` with `buffer.demonstration_fraction=0.25` mixes a saved buffer into a later run.
The PPO paths ignore both.
`data.persist.path=<dir>` saves every completed training episode in the same format, for every algorithm including PPO.
