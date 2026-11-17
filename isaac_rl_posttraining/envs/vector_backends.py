# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vectorized-environment backends for parallel rollout collection.

The rollout collector steps ``N`` environments through one ``gym.vector``-style
interface::

    reset(seed=, options=) -> (obs[N], info)
    step(actions[N])       -> (obs[N], rew[N], term[N], trunc[N], info)
    num_envs, single_observation_space, single_action_space
    # NEXT_STEP auto-reset on done + optional masked reset via
    #   reset(options={"reset_mask": bool[N]})

The ``vector_backend`` argument of a simulator's env builder selects the backend:
  * ``sync``    — ``SyncVectorEnv``: envs stepped sequentially in-process.
  * ``async``   — N envs in N worker processes (CPU single-robot sims:
                  LIBERO / robosuite / mujoco).
  * ``batched`` — one natively ``[N]``-batched sim (IsaacLab / MJX / Brax).

Workers ship raw observations; per-batch learner-side work (e.g. the DINO
embedding) runs once over all N, not per worker. Every backend supports
``gymnasium.vector`` NEXT_STEP auto-reset and masked reset
(``options={"reset_mask": mask}``) of only the envs that finished an episode.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

try:  # gymnasium is always present; imported lazily-friendly for clarity
    import gymnasium as gym
except Exception:  # pragma: no cover - gym is a hard dep in practice
    gym = None

# Subclass the real ``gymnasium.vector.VectorEnv`` so downstream vector
# wrappers (e.g. the DINO embedding wrapper, a ``gymnasium.vector.VectorWrapper``)
# accept these backends -- they assert ``isinstance(env, VectorEnv)``, so the
# gym.vector API alone is not enough. gymnasium 1.x ``VectorEnv`` declares its
# spaces as plain attributes (no properties, no required ``__init__``), which
# the backends already set, so subclassing is safe without calling ``super()``.
_VectorEnvBase = gym.vector.VectorEnv if gym is not None else object


VECTOR_BACKENDS = ("sync", "async", "batched")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _CloudpickleWrapper:
    """Pickle an env constructor (often a closure) so it survives ``spawn`` workers."""

    def __init__(self, fn: Callable):
        self.fn = fn

    def __getstate__(self):
        try:
            import cloudpickle

            return cloudpickle.dumps(self.fn)
        except Exception:
            import pickle

            return pickle.dumps(self.fn)

    def __setstate__(self, ob):
        try:
            import cloudpickle  # noqa: F401
            import pickle

            self.fn = pickle.loads(ob)
        except Exception:
            import pickle

            self.fn = pickle.loads(ob)

    def __call__(self):
        return self.fn()


def _batch_obs(obs_list: List[Any]) -> Any:
    """Stack a list of per-env observations into one batched observation.

    Dict observations are batched key-by-key.  Numeric leaves are ``np.stack``-ed
    (giving the ``[N, ...]`` arrays the learner batches over); non-numeric leaves
    (e.g. string prompts) are kept as an object array so nothing is lost.
    """
    first = obs_list[0]
    if isinstance(first, dict):
        out: Dict[str, Any] = {}
        for k in first.keys():
            vals = [o[k] for o in obs_list]
            if all(isinstance(v, np.ndarray) for v in vals):
                out[k] = np.stack(vals, axis=0)
            else:
                out[k] = np.array(vals, dtype=object)
        return out
    if all(isinstance(v, np.ndarray) for v in obs_list):
        return np.stack(obs_list, axis=0)
    return np.array(obs_list, dtype=object)


def _split_reset_mask(options: Optional[dict], num_envs: int) -> Optional[np.ndarray]:
    if not options or "reset_mask" not in options:
        return None
    mask = np.asarray(options["reset_mask"], dtype=np.bool_)
    assert mask.shape == (num_envs,), f"reset_mask must be shape ({num_envs},), got {mask.shape}"
    return mask


# ---------------------------------------------------------------------------
# Mode 2 — N robots in N processes (async / subprocess vector env)
# ---------------------------------------------------------------------------
def _worker_step(env, action, autoreset: bool) -> tuple:
    """Step ``env``, or reset it (ignoring ``action``) when the previous step ended the episode."""
    if autoreset:
        obs, info = env.reset()
        reward, terminated, truncated = 0.0, False, False
    else:
        obs, reward, terminated, truncated, info = env.step(action)
    return obs, reward, terminated, truncated, info


def _worker_reset(env, seed: int | None, options: dict | None) -> tuple:
    """Reset ``env``, forwarding only the ``seed``/``options`` that were given."""
    kwargs = {}
    if seed is not None:
        kwargs["seed"] = seed
    if options is not None:
        kwargs["options"] = options
    obs, info = env.reset(**kwargs)
    return obs, info


def _subproc_worker(remote, parent_remote, env_fn_wrapper):
    """One worker process: owns a single single-robot env, ships raw obs.

    Auto-reset is NEXT_STEP: after a ``done`` step the *next* ``step`` resets the
    env (ignoring the action) and returns the fresh obs, exactly like
    ``gymnasium.vector.SyncVectorEnv``.
    """
    parent_remote.close()
    env = env_fn_wrapper()
    autoreset = False
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                result = _worker_step(env, data, autoreset)
                autoreset = bool(result[2]) or bool(result[3])
                remote.send(result)
            elif cmd == "reset":
                seed, options = data
                result = _worker_reset(env, seed, options)
                autoreset = False
                remote.send(result)
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "get_attr":
                remote.send(getattr(getattr(env, "unwrapped", env), data, None))
            elif cmd == "close":
                remote.send(True)
                break
            else:  # pragma: no cover - defensive
                raise RuntimeError(f"SubprocVectorEnv worker got unknown cmd {cmd!r}")
    except (KeyboardInterrupt, EOFError):  # pragma: no cover
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass
        remote.close()


class SubprocVectorEnv(_VectorEnvBase):
    """Async vector env: N single-robot envs in N worker processes (Mode 2).

    Presents the same ``gym.vector`` API as ``SyncVectorEnv`` (``reset``/``step``
    return batched ``[N, ...]`` results, NEXT_STEP auto-reset, masked reset) but
    steps the workers in parallel.  Raw observations are shipped over pipes so
    the learner's ONE batched pi0/DINO forward stays on the main process.

    Worker ``i`` is seeded ``base_seed + i`` on the first (unmasked) reset.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], "gym.Env"]],
        context: str = "spawn",
        copy: bool = True,
    ):
        self.num_envs = len(env_fns)
        self.copy = copy
        self.closed = False
        self._base_seed: Optional[int] = None

        ctx = mp.get_context(context)
        self._remotes, self._work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self._procs = []
        for work_remote, remote, fn in zip(self._work_remotes, self._remotes, env_fns):
            proc = ctx.Process(
                target=_subproc_worker,
                args=(work_remote, remote, _CloudpickleWrapper(fn)),
                daemon=True,
            )
            proc.start()
            self._procs.append(proc)
            work_remote.close()

        self._remotes[0].send(("get_spaces", None))
        self.single_observation_space, self.single_action_space = self._remotes[0].recv()
        # Batched spaces mirror gymnasium's vector spaces when available.
        if gym is not None:
            try:
                self.observation_space = gym.vector.utils.batch_space(
                    self.single_observation_space, self.num_envs
                )
                self.action_space = gym.vector.utils.batch_space(
                    self.single_action_space, self.num_envs
                )
            except Exception:
                self.observation_space = self.single_observation_space
                self.action_space = self.single_action_space
        self._env_obs: List[Any] = [None] * self.num_envs

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError("SubprocVectorEnv is closed")

    def get_attr(self, name: str) -> Any:
        """Fetch attribute ``name`` from worker 0's underlying env.

        Per-env attributes (e.g. ``language_instruction``) live inside the
        worker processes; callers that walk the wrapper chain for them (the
        pi0 prompt sync) reach this instead of a dead end.
        """
        self._ensure_open()
        self._remotes[0].send(("get_attr", name))
        return self._remotes[0].recv()

    @property
    def language_instruction(self) -> Any:
        return self.get_attr("language_instruction")

    # -- gym.vector API ----------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._ensure_open()
        mask = _split_reset_mask(options, self.num_envs)
        # Strip reset_mask before forwarding to the single envs (they don't know it).
        sub_options = None
        if options:
            sub_options = {k: v for k, v in options.items() if k != "reset_mask"} or None

        if seed is not None:
            self._base_seed = int(seed)

        if mask is None:
            idxs = list(range(self.num_envs))
        else:
            idxs = [i for i in range(self.num_envs) if mask[i]]

        infos: List[dict] = [{} for _ in range(self.num_envs)]
        for i in idxs:
            worker_seed = None if self._base_seed is None else self._base_seed + i
            # Re-seeding on every masked reset would replay the same episode; seed
            # only when the caller explicitly passes a seed.
            send_seed = worker_seed if seed is not None else None
            self._remotes[i].send(("reset", (send_seed, sub_options)))
        for i in idxs:
            obs, info = self._remotes[i].recv()
            self._env_obs[i] = obs
            infos[i] = info if isinstance(info, dict) else {}

        return self._batched_obs(), infos

    def step(self, actions):
        self._ensure_open()
        for remote, action in zip(self._remotes, actions):
            remote.send(("step", action))

        rewards, terms, truncs = [], [], []
        infos: List[dict] = [{} for _ in range(self.num_envs)]
        for i, remote in enumerate(self._remotes):
            obs, reward, terminated, truncated, info = remote.recv()
            self._env_obs[i] = obs
            rewards.append(reward)
            terms.append(terminated)
            truncs.append(truncated)
            infos[i] = info if isinstance(info, dict) else {}

        return (
            self._batched_obs(),
            np.asarray(rewards, dtype=np.float64),
            np.asarray(terms, dtype=np.bool_),
            np.asarray(truncs, dtype=np.bool_),
            infos,
        )

    def close(self):
        if self.closed:
            return
        for remote in self._remotes:
            try:
                remote.send(("close", None))
                remote.recv()
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():  # pragma: no cover
                proc.terminate()
        self.closed = True

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    # -- internals ---------------------------------------------------------
    def _batched_obs(self):
        return _batch_obs(list(self._env_obs))


# ---------------------------------------------------------------------------
# Mode 1 — N robots in ONE simulator (natively batched env adapter)
# ---------------------------------------------------------------------------
class BatchedEnvAdapter(_VectorEnvBase):
    """Adapter for an *already* ``[N]``-batched env (Mode 1).

    A natively-batched sim (IsaacLab / MJX / Brax) computes ``step(actions[N])``
    in one call.  This thin adapter presents such an env through the exact same
    ``gym.vector`` API the collector expects — including NEXT_STEP auto-reset and
    masked reset — so the DSRL loop is identical to the sync/async cases.

    Contract for the wrapped ``batched_env``:
      * ``num_envs`` (or pass ``num_envs=``)
      * ``reset(seed=, options=) -> (obs[N], info)``; ideally honors
        ``options={"reset_mask": mask}`` to reset only the flagged envs.  If it
        cannot, the adapter falls back to a full reset (fine for sims that
        auto-reset internally, but note the other envs restart too).
      * ``step(actions[N]) -> (obs[N], rew[N], term[N], trunc[N], info)`` and must
        NOT auto-reset internally — the adapter emulates NEXT_STEP auto-reset so
        every backend behaves identically.
      * ``single_observation_space`` / ``single_action_space`` if available.
    """

    def __init__(self, batched_env: Any, num_envs: Optional[int] = None, copy: bool = True):
        self._env = batched_env
        self.copy = copy
        self.closed = False
        self.num_envs = int(num_envs if num_envs is not None else getattr(batched_env, "num_envs"))
        self.single_observation_space = getattr(batched_env, "single_observation_space", None)
        self.single_action_space = getattr(batched_env, "single_action_space", None)
        self.observation_space = getattr(batched_env, "observation_space", self.single_observation_space)
        self.action_space = getattr(batched_env, "action_space", self.single_action_space)
        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

    @property
    def env(self) -> Any:
        """The wrapped batched env, in wrapper convention (``.env`` chain).

        Callers that walk the wrapper chain (e.g. the collector's ``need_obs``
        toggle) traverse ``.env`` attributes; without this the walk dead-ends
        at the adapter and never reaches the batched client.
        """
        return self._env

    @property
    def language_instruction(self):
        be = getattr(self._env, "unwrapped", self._env)
        return getattr(be, "language_instruction", None)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        mask = _split_reset_mask(options, self.num_envs)
        obs, info = self._env.reset(seed=seed, options=options)
        if mask is None:
            self._autoreset_envs[:] = False
        else:
            self._autoreset_envs[mask] = False
        return obs, info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self._env.step(actions)
        reward = np.asarray(reward, dtype=np.float64)
        terminated = np.asarray(terminated, dtype=np.bool_)
        truncated = np.asarray(truncated, dtype=np.bool_)

        # NEXT_STEP auto-reset: envs that were done last step are reset now
        # (their action this step is ignored), mirroring SyncVectorEnv.
        if self._autoreset_envs.any():
            mask = self._autoreset_envs.copy()
            reset_obs, _ = self._env.reset(options={"reset_mask": mask})
            obs = _overwrite_masked(obs, reset_obs, mask)
            reward = np.where(mask, 0.0, reward)
            terminated = np.where(mask, False, terminated)
            truncated = np.where(mask, False, truncated)

        self._autoreset_envs = np.logical_or(terminated, truncated)
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.closed:
            return
        if hasattr(self._env, "close"):
            self._env.close()
        self.closed = True

    def __getattr__(self, name):
        # Forward anything we don't define to the wrapped batched env.
        return getattr(self.__dict__["_env"], name)


def _overwrite_masked(obs, reset_obs, mask):
    if isinstance(obs, dict):
        for k in obs:
            obs[k][mask] = reset_obs[k][mask]
        return obs
    obs[mask] = reset_obs[mask]
    return obs


# ---------------------------------------------------------------------------
# constructor — the single, obvious knob
# ---------------------------------------------------------------------------
def build_vector_env(
    backend: str,
    *,
    env_fns: Optional[Sequence[Callable[[], "gym.Env"]]] = None,
    batched_env: Any = None,
    num_envs: Optional[int] = None,
    copy: bool = True,
    context: str = "spawn",
):
    """Build the vectorized env for a collection strategy.

    ``backend`` selects the strategy:

        "sync"    -> gymnasium SyncVectorEnv (sequential; the default)
        "async"   -> SubprocVectorEnv (N worker processes; e.g. LIBERO)
        "batched" -> BatchedEnvAdapter over a natively-batched env

    Args:
        backend: one of ``{"sync", "async", "batched"}``.
        env_fns: list of single-env factories (required for sync/async).
        batched_env: an already-``[N]``-batched env (required for batched).
        num_envs: sanity-checked against ``len(env_fns)`` / ``batched_env.num_envs``.
    """
    backend = (backend or "sync").lower()
    if backend not in VECTOR_BACKENDS:
        raise ValueError(
            f"Unknown vector_backend {backend!r}. Valid options: {VECTOR_BACKENDS}. "
            f"Use 'sync' for one sequential sim (default), 'async' for N CPU sims in "
            f"N processes (e.g. LIBERO), or 'batched' for a natively GPU-batched sim."
        )

    if backend in ("sync", "async"):
        if not env_fns:
            raise ValueError(f"vector_backend={backend!r} requires env_fns (single-env factories).")
        if num_envs is not None and len(env_fns) != num_envs:
            raise ValueError(
                f"num_envs={num_envs} but got {len(env_fns)} env_fns; they must match."
            )
        if backend == "sync":
            return gym.vector.SyncVectorEnv(list(env_fns))
        return SubprocVectorEnv(list(env_fns), context=context, copy=copy)

    # backend == "batched"
    if batched_env is None:
        raise ValueError(
            "vector_backend='batched' requires a natively-batched env (batched_env=...), "
            "i.e. an env whose step(actions[N]) returns obs[N] in one call (IsaacLab / MJX / "
            "Brax). This CPU env stack is single-robot per instance — use 'async' instead."
        )
    return BatchedEnvAdapter(batched_env, num_envs=num_envs, copy=copy)
