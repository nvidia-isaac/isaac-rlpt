# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parallel rollout collection on RLinf's scheduler, behind our collector surface.

:class:`ParallelRolloutAdapter` runs N environment workers as RLinf
``Worker`` processes (Ray actors placed by ``NodePlacementStrategy``), each
owning one environment and one policy replica. To the caller it is a
collector: ``run()`` honors ``num_rollouts`` / ``count_by`` (``"episode"``
collects that many episodes *per environment*, ``"step"`` that many
environment steps *in total*), ``collect(n_steps)`` is the explicit
step-counted form, transitions land in the replay buffer via
``buffer.add(...)``, and metrics come from the shared
:class:`~isaac_rl_posttraining.rollouts.episode_tracker.EpisodeTracker`.

What runs where
---------------

* **Worker processes** (:class:`_EnvWorker`): build ``make_env(rank)``
  and ``make_policy(rank)`` locally (both must be picklable callables,
  and the package itself must be importable inside Ray workers -- install
  it, or register the defining module for cloudpickle by-value
  serialization), step their environment, and return their transition
  chunk as the remote-call result (the Ray object plane).
* **Learner process** (the adapter): dispatches per-rank budgets
  concurrently, gathers the chunks, and writes transitions **rank-major**
  (all of rank 0's chunk, then rank 1's, ...) so buffer contents are
  deterministic for a given set of seeds regardless of worker timing.
  Episode ids are assigned at write time and are unique per
  (worker, episode); for a single worker they count up from zero in
  collection order.

Environment/policy protocol (deliberately minimal -- this adapter is the
multi-process *pattern proof*, mock-scale by design): environments follow
the unbatched Gymnasium API (``reset() -> (obs, info)``,
``step(a) -> (obs, reward, terminated, truncated, info)``); policies are
either callables ``obs -> action`` or objects with ``act(obs) -> action``.

Why the Ray object plane and not an RLinf ``Channel``
-----------------------------------------------------

RLinf's own streaming primitive (``rlinf.scheduler.Channel``) would be the
natural transition transport, but at the pinned version its worker-side
``put`` rides the collective plane, whose process groups are wired for the
node's *detected hardware*: on a machine with NVIDIA GPUs it builds NCCL
process groups regardless of ``CUDA_VISIBLE_DEVICES``, which requires a
CUDA torch build (``ProcessGroupNCCL``) -- under a CPU-only torch the
first worker-to-channel ``put`` fails inside the lazy collective-group
setup and the remaining ranks hang (no error surfaces without a stack
dump). There is no per-group opt-out at v0.2. Worker *return values*
travel over Ray's object plane instead, which works on any build; the
trade-off and the debugging story are documented in
``docs/infrastructure.md``.

Operational weight (measured; see ``docs/infrastructure.md``): RLinf's
scheduler needs a live Ray runtime. The adapter boots one on first use
(``Cluster(num_nodes=1)`` -- several seconds and a fleet of Ray daemon
processes) and each worker-group launch adds actor spawn time
(``startup_seconds`` records both). RLinf requires being the one to
initialize Ray: constructing an adapter in a process where ``ray.init``
already ran outside RLinf raises. Worker-group teardown uses
``WorkerGroup._close`` (no public equivalent at the pinned version), and
the Ray runtime itself stays up for the process lifetime.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Optional

import numpy as np
from loguru import logger

from isaac_rl_posttraining.data import (
    DynamicObservation,
    EnvAction,
    LearnerAction,
    Transition,
    TransitionMetadata,
    episode_success,
)
from isaac_rl_posttraining.rollouts.episode_tracker import EpisodeTracker

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker

__all__ = ["ParallelRolloutAdapter", "ensure_cluster"]


def ensure_cluster(num_nodes: int = 1) -> "Cluster":
    """Return the process-wide RLinf cluster, booting Ray on first use.

    Args:
        num_nodes: Nodes to wait for when booting a fresh cluster.

    Returns:
        The singleton ``rlinf.scheduler.Cluster``.
    """
    if Cluster.has_initialized():
        return Cluster()
    start = time.monotonic()
    cluster = Cluster(num_nodes=num_nodes)
    logger.info(
        f"[compat.rlinf] RLinf cluster (Ray runtime) started in "
        f"{time.monotonic() - start:.1f}s"
    )
    return cluster


class _EnvWorker(Worker):
    """One environment + policy replica inside an RLinf worker process."""

    def __init__(self, make_env: Callable, make_policy: Callable) -> None:
        super().__init__()
        self._env = make_env(self._rank)
        self._policy = make_policy(self._rank)
        self._obs: Any = None
        self._local_episode = 0

    def ping(self) -> int:
        """Readiness probe; returns this worker's rank."""
        return self._rank

    def _act(self, obs: Any) -> Any:
        if hasattr(self._policy, "act"):
            return self._policy.act(obs)
        return self._policy(obs)

    @staticmethod
    def _within_budget(
        steps: int, episodes: int, num_steps: int | None, num_episodes: int | None
    ) -> bool:
        if num_steps is not None and steps >= num_steps:
            return False
        if num_episodes is not None and episodes >= num_episodes:
            return False
        # No budget: collect nothing rather than loop forever.
        return not (num_steps is None and num_episodes is None)

    def _step_once(self) -> dict[str, Any]:
        """Step the environment once, resetting first if needed, and advance the episode state."""
        if self._obs is None:
            self._obs, _ = self._env.reset()
        action = self._act(self._obs)
        next_obs, reward, terminated, truncated, info = self._env.step(action)
        done = bool(terminated)
        trunc = bool(truncated)
        transition = {
            "obs": self._obs,
            "action": action,
            "reward": float(reward),
            "next_obs": next_obs,
            "done": done,
            "truncated": trunc,
            "info": dict(info) if info else {},
            "local_episode": self._local_episode,
        }
        if done or trunc:
            self._local_episode += 1
            self._obs = None
        else:
            self._obs = next_obs
        return transition

    def collect(
        self,
        num_steps: Optional[int] = None,
        num_episodes: Optional[int] = None,
    ) -> dict[str, Any]:
        """Collect transitions until the step/episode budget is exhausted.

        Collection state (current observation) persists across calls, so a
        step budget may pause mid-episode and resume on the next call, like
        the native collector.

        Args:
            num_steps: Stop after this many environment steps (``None`` for
                episode-counted collection).
            num_episodes: Stop after this many completed episodes (``None``
                for step-counted collection).

        Returns:
            ``{"transitions": [...], "steps": int, "episodes": int}``; each
            transition dict carries obs / action / reward / next_obs / done
            / truncated / info / local_episode.
        """
        steps = 0
        episodes = 0
        transitions: list[dict[str, Any]] = []

        while self._within_budget(steps, episodes, num_steps, num_episodes):
            transition = self._step_once()
            transitions.append(transition)
            steps += 1
            if transition["done"] or transition["truncated"]:
                episodes += 1

        return {"transitions": transitions, "steps": steps, "episodes": episodes}


class ParallelRolloutAdapter:
    """N parallel environment workers behind a collector surface.

    Args:
        make_env: Picklable callable ``rank -> env`` (unbatched
            Gymnasium API). Seed per-rank inside it for
            deterministic collection.
        make_policy: Picklable callable ``rank -> policy`` where the
            policy is an ``obs -> action`` callable or has ``act(obs)``.
        buffer: Replay buffer receiving the transitions (``add(...)``).
        num_envs: Number of parallel environment workers.
        num_rollouts: Episodes per environment (``count_by="episode"``) or
            total environment steps (``count_by="step"``) per :meth:`run`.
        count_by: ``"episode"`` or ``"step"``.
        cluster: Existing RLinf cluster; ``None`` boots one via
            :func:`ensure_cluster`.
        name: Worker-group name prefix; a unique suffix is appended because
            RLinf registers groups as named Ray actors.
    """

    def __init__(
        self,
        make_env: Callable,
        make_policy: Callable,
        buffer: Any,
        num_envs: int = 1,
        num_rollouts: int = 1,
        count_by: str = "episode",
        cluster: Optional["Cluster"] = None,
        name: str = "isaacrl_rollout",
    ) -> None:
        if count_by not in ("episode", "step"):
            raise ValueError(f"count_by must be 'episode' or 'step', got {count_by!r}")
        self.buffer = buffer
        self.num_envs = int(num_envs)
        self.num_rollouts = int(num_rollouts)
        self.count_by = count_by
        self.episode_tracker = EpisodeTracker(self.num_envs)
        self.total_episodes = 0
        # Per rank: the worker's local episode counter, the global id it maps to, and the
        # next step index, kept across calls so a step budget can pause mid-episode.
        self._rank_state: dict[int, tuple[int | None, int | None, int]] = {}
        self._next_episode_id = 0

        start = time.monotonic()
        self._cluster = cluster if cluster is not None else ensure_cluster()
        self._group = _EnvWorker.create_group(make_env, make_policy).launch(
            cluster=self._cluster,
            name=f"{name}_{uuid.uuid4().hex[:8]}",
            placement_strategy=NodePlacementStrategy([0] * self.num_envs),
        )
        self._group.ping().wait()  # workers built their env/policy replicas
        self.startup_seconds = time.monotonic() - start
        self._closed = False

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def run(self, can_train: bool = True) -> dict[str, float]:
        """Collect ``num_rollouts`` steps/episodes and return rollout metrics.

        Args:
            can_train: Accepted for interface parity with the training
                loop's data sources (which switch to warm-up exploration
                while ``False``); worker policies own their exploration here.
        """
        del can_train
        if self.count_by == "episode":
            return self._collect(num_episodes=self.num_rollouts)
        return self._collect(num_steps=self.num_rollouts)

    def collect(self, n_steps: int) -> dict[str, float]:
        """Collect exactly ``n_steps`` environment steps across all workers."""
        return self._collect(num_steps=n_steps)

    def _collect(
        self, num_steps: Optional[int] = None, num_episodes: Optional[int] = None
    ) -> dict[str, float]:
        """Dispatch per-worker budgets, gather results, write rank-major."""
        if self._closed:
            raise RuntimeError("ParallelRolloutAdapter is closed")

        if num_steps is not None:
            # Exact total: spread the step budget, remainder to low ranks.
            # execute_on dispatches asynchronously, so the per-rank calls
            # (with their differing quotas) still run concurrently.
            base, extra = divmod(int(num_steps), self.num_envs)
            handles = [
                self._group.execute_on(rank).collect(num_steps=base + (1 if rank < extra else 0))
                for rank in range(self.num_envs)
            ]
            per_rank = [handle.wait()[0]["transitions"] for handle in handles]
        else:
            results = self._group.collect(num_episodes=int(num_episodes)).wait()
            per_rank = [result["transitions"] for result in results]

        for rank in range(self.num_envs):
            self._ingest_rank(rank, per_rank[rank])
        return self.episode_tracker.get_metrics()

    def _ingest_rank(self, rank: int, transitions: list[dict[str, Any]]) -> None:
        """Write one worker's transitions into the buffer + episode tracker.

        Episode ids are unique across ranks and calls; an episode a step
        budget paused keeps its id and its step count on the next call.
        """
        current_local, episode_id, step_in_episode = self._rank_state.get(rank, (None, None, 0))
        for t in transitions:
            if t["local_episode"] != current_local:
                current_local = t["local_episode"]
                episode_id = self._next_episode_id
                self._next_episode_id += 1
                step_in_episode = 0
            info = t["info"]
            done, truncated = t["done"], t["truncated"]
            ended = bool(done or truncated)
            succeeded = ended and episode_success(info)
            action = np.asarray(t["action"], dtype=np.float32).reshape(1, -1)
            self.buffer.add(
                Transition(
                    episode_id=episode_id,
                    observation=DynamicObservation.from_mapping(t["obs"]),
                    next_observation=DynamicObservation.from_mapping(t["next_obs"]),
                    learner_action=LearnerAction("chunk", action),
                    env_action=EnvAction(commands=action, execution_length=1),
                    rewards={"env": np.asarray([t["reward"]], dtype=np.float32)},
                    terminated=np.asarray([bool(done)]),
                    truncated=np.asarray([bool(truncated)]),
                    success=np.asarray([succeeded]),
                    behavior_source="policy",
                    metadata=TransitionMetadata(step_in_episode=step_in_episode, extra=info),
                )
            )
            step_in_episode += 1
            self.episode_tracker.add_step(rank, t["reward"], info, done, truncated, success=succeeded)
            if done or truncated:
                self.episode_tracker.end_episode(rank)
                self.total_episodes += 1
        self._rank_state[rank] = (current_local, episode_id, step_in_episode)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Tear down the worker group (idempotent).

        Uses ``WorkerGroup._close`` -- the pinned RLinf version has no
        public teardown -- and leaves the Ray runtime itself up (it is
        process-wide and other adapters may share it).
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._group._close()
        # Teardown is best-effort.
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[compat.rlinf] worker-group teardown failed: {exc}")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass  # interpreter shutdown: scheduler internals may be gone
