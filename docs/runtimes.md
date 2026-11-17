# Runtimes

A runtime decides where the rollout workers, policy servers and learner ranks run; the algorithm, the simulator and the replay data stay the same whichever you pick.
This page covers choosing, installing, running and configuring one; [runtime-class-interactions.md](runtime-class-interactions.md) describes the classes behind them.

## Available runtimes

| `runtime=` | Rollout workers | Use it when |
|---|---|---|
| `native` (the default) | in the training process, stepped in turn; Ray actors when the placement asks for several processes | one simulator on one GPU (the simplest setup), or several GPUs and nodes with the `ray` extra |
| `rlinf` | RLInf worker groups on Ray: parallel processes, shared batched policy servers, FSDP learner ranks | several simulators or GPUs, or a multi-rank PPO learner |

| Runtime | Parallel rollout workers | Non-blocking collection (`collect_async`) | Batched policy servers | Distributed FSDP learner |
|---|---|---|---|---|
| `native` | on several processes | yes | one in-process server, or replica actors | on several processes |
| `rlinf` | yes | yes | yes | yes |

## Setup

```bash
uv sync                          # native
uv sync --extra ray              # native on several processes: Ray, the same pin as rlinf
uv sync --extra rlinf            # rlinf: RLInf and its Ray pin
```

Run anything that starts RLInf with `uv run --extra rlinf`.
Under plain `uv run`, Ray 2.58 rebuilds every worker's environment from the project without the extra, so the workers die with `No module named ray` and the run fails with `ActorUnschedulableError`.
If the environment is not the project's `.venv` (for example `UV_PROJECT_ENVIRONMENT` points elsewhere), set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` instead, so the workers reuse the launching interpreter.

## Training

DSRL on one LIBERO task with eight simulators as parallel RLInf rollout workers:

```bash
ISAAC_RL_POSTTRAINING_DRYRUN=1 uv run python scripts/train.py algorithm=dsrl_sac simulator=libero_spatial runtime=rlinf \
  env.env_name=libero/libero_spatial/8 'env.max_episode_steps=${simulator.gr00t_eval.max_episode_steps}' \
  'env.image_keys=[video.image]' base_model.embodiment_tag=libero_sim dsrl.noise_dim=7 training.chunk_size=40 \
  training.num_envs=8 base_model.ckpt=/path/to/gr00t-checkpoint
```

`training.num_envs` is the number of environments, and the LIBERO entry gives each its own worker.
The dry run prints the plan (`plan.runtime=rlinf`, `plan.layout=8 rollout worker(s) x 1 environment(s)`, `plan.placement=...`) and exits; drop `ISAAC_RL_POSTTRAINING_DRYRUN=1` to launch.

By default `rlinf` keeps the learner on GPU 0 and starts the rollout workers and policy servers at GPU 1, so a launch needs at least two GPUs; the dry run passes on any machine because it plans onto the GPUs the run will have.
To launch on one GPU, append `runtime.rlinf.first_accelerator=0 runtime.rlinf.exclusive_accelerators=false`, which puts every role on GPU 0 (about 16 GiB of GPU memory for a tiny LIBERO DSRL run), or use `runtime=native`.
Two FSDP learner ranks (`placement.learner.ranks=2`) need two GPUs.

The same run through the LIBERO launcher, inside the LIBERO environment of `example_workflows/libero_posttraining/README.md` with the RLInf extra installed as that README describes (`RUNTIME=` is the launchers' knob for the runtime):

```bash
SUITE=libero_spatial RUNTIME=rlinf NUM_ENVS=8 bash example_workflows/libero_posttraining/scripts/train.sh 8
```

Steering PPO on G1 Factory with two FSDP learner ranks on GPUs 0 and 1 and the rollout workers on GPU 2, so three GPUs (`PPO_LEARNER_RANKS` becomes `placement.learner.ranks`, [DSRL with PPO](pages/algorithms/dsrl.md#dsrl-with-ppo)):

```bash
NUM_ENVS=32 RUNTIME=rlinf PPO_LEARNER_RANKS=2 LEARNER_GPUS=0,1,2 \
  uv run --extra rlinf bash example_workflows/g1_arena_posttraining/scripts/train_ppo.sh DrillLift
```

## Dry run

The dry run prints `plan.runtime`, `plan.placement`, `plan.layout`, `plan.evaluation_mode`, `plan.evaluation` and `plan.schedule_kind` before anything loads, and ends with `plan.checks` (every failed check at once); a simulator without a registered entry fails there.
`plan.placement` shows `learner_first_accelerator`, `first_accelerator` and `num_nodes` only; the `plan.placement.*` lines that follow show how the policy is served (`local` or `served by ...`) and the planned GPUs of every role.
The run writes the same plan to `<run_dir>/telemetry/placement.json`, and logs the resolved serving when the workers start, as `RLInf rollout topology: ... policy serving=shared, <n> policy server(s)`.
Explicit GPU lists that overlap or name a GPU the node lacks are reported with the roles and GPUs involved (`envs (placement.envs.gpus) and learner (placement.learner.gpus) both ask for GPU(s) node0:4, node0:5`).

## Placement

The `placement` section says which role runs where; the planner chooses unset values, and the section applies to every runtime.

- **Rollout workers.** `placement.envs.workers=null` lets the simulator entry lay out the workers: one environment per worker for LIBERO, one batched worker for G1 Factory and for Isaac Lab steering with several scenes; the `ppo` recipe sets it explicitly. `placement.envs.gpus` is a count or a list of `node:index` GPUs.
- **Policy serving.** `placement.policy.mode=local` gives every rollout worker its own policy; `served` sends every worker's requests to policy servers that batch them; `auto` (the default) lets the runtime decide. Under `rlinf`, `auto` serves the policy when the runtime runs on GPUs and the simulator's workers support a shared policy (LIBERO does), with one server per four environments, capped by the available GPUs. On `native`, `auto` serves it to unbatched workers whose policy may be shared, from one in-process server. `placement.policy.replicas` sets the number of servers.
- **Batching.** A policy server waits up to `placement.policy.max_wait_ms` (5 ms) after the first request, so the requests of rollout workers that act together form one batch of at most `placement.policy.max_batch`, and answers at once when every rollout worker it serves has a request waiting. Evaluation requests are never batched across rollout workers, so batching cannot change a score. On `native`, served workers of one group collect on threads so the server can batch them (`runtime.native.concurrent_collection`, `auto` by default; `false` keeps one worker after another, which keeps a seeded run's draws in a fixed order).
- **Serving only the base model.** `placement.parts.base.mode=served` serves only the frozen base model of a steering or edit policy (DSRL, RFS, SAC-edit), shared by the training and evaluation groups, while every rollout worker keeps its own copy of the actor and receives the actor weights. `placement.parts.base.replicas` and `.gpus` size the servers as `placement.policy` does. `native` and `rlinf` support it. A seeded evaluation session's random streams travel with each decode request, so a served base model decodes the same numbers as a local one. Serving the actor without its base model, or the two parts on different servers, is refused before the run starts.
- **Flow-SDE PPO.** `placement.policy.mode=served` also serves the trainable policy of `algorithm=ppo` on `native`: the learner's weights reach the servers after every update, and each decision carries the version that produced it ([PPO](pages/algorithms/ppo.md)).
- **Learner.** `placement.learner.ranks` is the number of data-parallel learner processes (FSDP ranks under `rlinf`, or `native` on several processes); 0 evaluates an on-policy checkpoint without a learner. A single rank trains in the launching process on GPU 0.

### GPUs under RLInf

Several learner ranks occupy GPUs `0` to `placement.learner.ranks - 1`.
The rollout workers and policy servers start at `runtime.rlinf.first_accelerator`, by default `${placement.learner.ranks}`, counted from GPU 0 whatever `learner_first_accelerator` says; moving the learner with `learner_first_accelerator` therefore needs `first_accelerator` moved as well.
While `runtime.rlinf.exclusive_accelerators=true`, the dry run refuses learner ranks that would share a card with the rollout workers or policy servers.

From `first_accelerator` on, one rule places the policy servers and rollout workers (`isaac_rl_posttraining/runtimes/rlinf/layout.py`, which the dry run also applies):

- Policy servers take the first cards: one per four environments, at most one per rollout worker, and, when the workers need cards too, two fifths of them.
- Rollout workers need a card when they hold a policy part (a local policy, or the actor next to a served base model) or simulate on one (Isaac Lab). They take one card each from those the servers leave, share them evenly when there are fewer cards than workers, and share the servers' cards when none is left. Each worker sees only its own card.
- Workers of a served policy whose simulator runs on the CPU or behind a server (LIBERO, G1 Factory) stay on the CPU.
- A run's evaluation group follows the same rule and shares the training workers' cards; the two never collect at once.

`runtime.rlinf.rollout_workers_share_accelerator=true` runs every rollout worker on `first_accelerator` instead, which is how several batched workers (one per simulator server, G1 Factory with a comma-separated `ARENA_SERVER_URL`) share the learner's card.

### Native on several processes

`runtime=native` runs every role in the training process unless the placement asks for more: several learner ranks (`placement.learner.ranks`), several policy-server replicas or GPUs for them (`placement.policy.replicas`, `placement.policy.gpus`), GPUs for the rollout workers (`placement.envs.gpus`), or several nodes (`runtime.native.num_nodes`).
Then it runs each role as Ray actors on the GPUs of the plan the dry run prints (needs the `ray` extra); `runtime.native.processes=single` keeps one process, and `ray` forces actors.

Roles the placement leaves open split the GPUs (per 8: envs 3, policy 2, learner 3; a single learner rank takes one), and no role takes more GPUs than its processes use (one per rollout worker, at most one replica per rollout worker); the dry run says how many stay free.
With too few GPUs for each such role to have one, they share: on one GPU every role runs on it; on two the learner keeps GPU 0 and the rollout workers and policy servers share GPU 1.
The run joins a running Ray cluster (`ray.init(address="auto")`, or `runtime.native.ray_address`) and otherwise starts a local one.

Two nodes: start Ray on each, then launch on the head.

```bash
ray start --head --port=6379             # on the first node
ray start --address=<head>:6379          # on every other node
uv run python scripts/train.py ... runtime.native.num_nodes=2 placement.gpus_per_node=8
```

Checkpoints of several learner ranks need a directory every node can write.

## Schedule

The `schedule` config group says when collection and updates happen. `native` and `rlinf` run every schedule.

| `schedule=` | What happens | Policy lag of training data |
|---|---|---|
| `sync` (the default) | collect a round, train on it, publish the new weights; the reference recipe | 0 |
| `overlapped` | the next round collects while the learner trains on the last one | 1 |
| `async` | rollout workers collect continuously; the learner trains on what has arrived and publishes once per training burst | unbounded unless `schedule.max_policy_lag` is set |

Policy lag is the number of weight publishes between the weights that collected a decision and the learner's current weights; the run logs it as `system/policy_lag`.
`schedule.max_policy_lag` bounds it: under `async` the data source drops older decisions (`system/dropped_stale_decisions`); under `sync` it must be 0 and under `overlapped` at least 1.
`schedule.publish_every=<N>` (`async` only) publishes every `N` learner updates instead of once per burst.
On-policy algorithms (PPO) accept data at most one version old, so `async` defaults to `schedule.max_policy_lag=1` for them and refuses any other bound.
Evaluation never overlaps training: an iteration followed by an evaluation starts no round ahead.
The dry run prints the result as `plan.schedule_kind`, for example `plan.schedule_kind=overlapped, max policy lag 1`.

## Evaluation

Evaluation follows the simulator entry, and `plan.evaluation` shows which mode applies:

- G1 Factory evaluates on a dedicated simulator server (`ARENA_EVAL_SERVER_URL`).
- LIBERO evaluates on fresh workers seeded from their own stream; with a served policy they join the training run's policy servers, so they claim no accelerator of their own.
- Isaac Lab evaluates on the training workers at episode boundaries when `eval.eval_num_envs` equals `training.num_envs`, and otherwise on a simulation of its own with `eval.eval_num_envs` scenes.

`eval.reset_seed=<int>` replays the same initial scenes at every evaluation.
A new simulator or robot plugs into every runtime through one `SimulatorEntry` subclass ([add a simulator](pages/extending/add_a_simulator.md)).

## Measured performance

Each table is one measurement under the stated conditions, not a guarantee for other models, tasks or machines.
PPO learner scaling over FSDP ranks is on [the PPO page](pages/algorithms/ppo.md#learner-scaling-over-fsdp-ranks), and the batched GR00T decode on G1 Factory in `simulators/g1_arena/README.md`.

### Shared GR00T policy servers

Eight rollout workers sent 32 decisions each (256 requests) to 1, 2 or 4 shared policy servers holding GR00T N1.7 (LIBERO-10 checkpoint).
Every condition replayed the same fixed observations of LIBERO-10 task 8 (393,279 bytes per request) with a microbatch of up to 32 requests and a 5 ms wait.
One NVIDIA L40S per policy server, one trial per condition; 2026-09-11, `scripts/bench_gr00t_policy_services.py`.

| Policy servers | Requests/s, total | Requests/s per server | Service latency p50 / p95 | Mean batch per server | GPU memory per server | Group startup |
|---|---|---|---|---|---|---|
| 1 | 32.66 | 32.66 | 203.4 / 210.3 ms | 7.11 | 7,934 MiB | 172.6 s |
| 2 | 52.22 | 26.11 | 126.0 / 138.8 ms | 4.00 | 6,918 MiB | 174.0 s |
| 4 | 55.02 | 13.75 | 109.7 / 115.5 ms | 2.00 | up to 6,792 MiB | 178.9 s |

Two servers deliver 80% of twice the one-server throughput; four add only 5.4% over two, because eight workers fill each of four servers with batches of two.
The automatic count of `placement.policy.mode=auto` under `rlinf` (one server per four environments) gives two servers for eight environments.
A heavier policy or more concurrent workers moves the best count, so set `placement.policy.replicas` from a measurement on your own workload.

### Time breakdown of a 128-environment G1 Factory PPO iteration

Steering PPO (`algorithm=dsrl_ppo`) on G1 Factory DrillLift with `runtime=rlinf`: 128 environments in one batched simulator, one rollout worker holding GR00T N1.7, one learner rank, and a second 128-environment simulator for evaluation.
Two NVIDIA L40 GPUs: both simulators and the learner on GPU 0, the rollout worker on GPU 1.
15 training iterations with learning rate 1e-6, 5,750,819 environment steps in total; 2026-09-13.

| Phase | Mean (s) | Median (s) | Share of the mean iteration |
|---|---|---|---|
| Simulator step and transport | 212.35 | 212.55 | 50.1% |
| GR00T action inference | 155.89 | 156.58 | 36.8% |
| Bootstrap-value inference | 36.53 | 37.15 | 8.6% |
| Rollout-worker refresh after an evaluation | 8.07 | 0.00 | 1.9% |
| PPO learner update | 3.18 | 3.11 | 0.75% |
| Reset, trajectory and batch build, policy publication | 1.24 | 1.22 | 0.29% |

An iteration took 424.09 s on average (median 420.39 s), at 113.6 environment steps per second on average (103.2 to 121.2).
The refresh happens only after the three intermediate evaluations, about 40 s each, so it shows in the mean and not in the median.
In a separate instrumented run at the same revision, a 128-episode evaluation took 521 to 528 s, about half of it GR00T action inference.
The simulator and the frozen policy take almost all of the time and the learner update under 1%, so more learner GPUs do not speed up this workload; more simulator capacity and faster policy serving do.

### Scheduling cost of the runtimes (synthetic)

This measures only the runtime layer: the simulator is a 10 ms sleep per step and the policy a 2 ms sleep per call, so it says nothing about Isaac Lab, LIBERO or GR00T throughput.
128 logical environments, 32 rollout workers with four environments each (the native runtime steps them in turn), 2 decisions after 1 warm-up, median of 5 repeats, a 4 MiB policy state.
AMD Ryzen Threadripper PRO 7965WX (48 logical CPUs), Ray 2.58.0, RLInf 0.2; measured 2026-09-09.

| `runtime=` | Environment steps/s | Relative to native | Startup | Policy-state sync (median) |
|---|---|---|---|---|
| `native` | 382.5 | 1.0x | under 1 ms | 0.004 ms |
| `rlinf` | 11,129.2 | 29.1x | 8.05 s | 57.81 ms |

The distributed runtimes run the 32 workers concurrently and spend 7.1 to 7.5% of collection time in dispatch and transport.
They pay a few seconds of startup and about 57 ms per policy-state publication, a cost that pays off only when several simulators run in parallel.

## Reference

### Terms

| Name | Meaning |
|---|---|
| `training.num_envs` | simulator environments collecting experience |
| `placement.envs.workers` | rollout workers: processes (or actors) that advance environments; each owns a set of environment indices |
| `placement.policy.replicas` | policy servers: processes that batch and serve policy inference for the rollout workers |
| `placement.learner.ranks` | data-parallel learner processes |
| `world_size` | PyTorch's name for the size of one learner process group; equals `placement.learner.ranks` and never means environments |

### Configuration

`runtime.name` selects the runtime ([Available runtimes](#available-runtimes) lists them).
The `placement` and `schedule` sections are in `isaac_rl_posttraining/configs/train.yaml`; each runtime's own section is in `isaac_rl_posttraining/configs/runtime/`.

| Key | Default | Meaning |
|---|---|---|
| `placement.policy.mode`, `placement.parts.base.mode` | `auto`, `auto` | `local`, `served` or `auto` for the whole policy and for the frozen base model alone |
| `placement.policy.replicas` | null | policy servers (`rlinf`: null = one per four environments; `native`: more than one runs them as Ray actors) |
| `placement.policy.max_batch`, `placement.policy.max_wait_ms` | null, null | batch size and wait of a policy server (null = the runtime's: 32 and 5 ms under `rlinf`, unbounded and 5 ms on `native`; the dry run prints the values in `plan.placement.policy`) |
| `placement.policy.routing` | `modulo` | how a worker's environments reach a replica: by environment index (`modulo`) or the replica with the fewest clients (`least_loaded`) |
| `placement.gpus_per_node` | null | GPUs per node the planner assumes (null = the GPUs visible here) |
| `schedule.max_policy_lag`, `schedule.publish_every` | null, null | see [Schedule](#schedule) |
| `runtime.rollout_timeout_seconds` | 1800 | how long one collection may take per episode each rollout worker was asked for (every runtime); a longer one is treated as a frozen simulator: the run fails with a `RolloutTimeoutError` naming the late rollout workers and their GPUs, writes its `aborted` checkpoint and exits (in one process, if the simulator never returns, with status 3); null: no limit |
| `runtime.native.device` | `auto` | device of the in-process policy |
| `runtime.native.concurrent_collection` | `auto` | rollout workers collect on threads, so a served policy batches their requests (auto: on when the placement serves the policy and a group has several workers) |
| `runtime.native.serve_evaluation_unbatched`, `runtime.rlinf.serve_evaluation_unbatched` | true | a policy server never batches evaluation requests of different workers |
| `runtime.native.collect_policy_server_metrics` | `auto` | report policy-server batch sizes and timings (`policy_service/*`; auto: when the placement serves the policy) |
| `runtime.native.processes` | `auto` | `auto`: Ray actors when the placement asks for several processes; `single`: one process; `ray`: always actors |
| `runtime.native.num_nodes`, `runtime.native.ray_address` | 1, null | Ray nodes the placement plans onto; the cluster to join (null: the running one, else a local one) |
| `runtime.native.node_join_timeout_seconds` | 120 | how long the run waits for every node of the placement to join the Ray cluster |
| `runtime.native.startup_timeout_seconds`, `control_timeout_seconds`, `call_timeout_seconds` | 900, 300, null | how long the run waits for an actor to start; for a weight update, hook or close; for an update or checkpoint (a dead actor fails a call at once) |
| `runtime.native.policy_request_timeout_seconds`, `collective_timeout_seconds` | 600, 1800 | one rollout worker's call to a policy server; one collective among learner ranks |
| `runtime.native.learner_mixed_precision`, `learner_seed`, `seed` | `none`, `${training.seed}`, `${training.seed}` | FSDP precision and seed of the learner ranks; seed of every actor's generators |
| `runtime.rlinf.num_nodes`, `runtime.rlinf.node_ranks` | 1, null | Ray nodes to place on |
| `runtime.rlinf.device` | `gpu` | `cpu` or `gpu` workers |
| `runtime.rlinf.first_accelerator` | `${placement.learner.ranks}` | first GPU of the rollout and policy-server groups |
| `runtime.rlinf.learner_first_accelerator` | 0 | first GPU of several learner ranks; also set `first_accelerator` (for two ranks on GPUs 1-2: `learner_first_accelerator=1 first_accelerator=3`) |
| `runtime.rlinf.exclusive_accelerators` | true | learner ranks refuse a card the rollout workers or policy servers use |
| `runtime.rlinf.rollout_workers_share_accelerator` | false | all rollout workers on `first_accelerator` |
| `runtime.rlinf.learner_mixed_precision` | `none` | `none` or `bf16` for the FSDP learner |
| `runtime.rlinf.learner_seed` | 0 | seed of the learner ranks |
| `runtime.rlinf.policy_request_timeout_seconds` | 600 | how long a rollout worker waits for its policy server |
| `runtime.rlinf.collect_policy_server_metrics` | `auto` | report policy-server batch sizes and timings (`policy_service/*`; auto: when the runtime serves the policy or its base model) |
| `runtime.rlinf.shutdown_ray_on_close` | false | stop Ray when the run ends |
| `runtime.rlinf.serialize_specs_by_value` | true | pickle worker specs by value instead of by module reference |

`runtime.rlinf.policy_serving`, `num_policy_servers`, `policy_batch_size` and `policy_batch_wait_seconds` are deprecated aliases of `placement.policy.mode`, `replicas`, `max_batch` and `max_wait_ms`.
