# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RLinf compat-layer tests: import guards, version pin, and rollout parity.

Two groups:

* **Guard tests (always run):** ``compat.rlinf`` imports without rlinf/ray
  installed, adapter access raises ``ImportError`` with the ``[rlinf]``
  extra hint, the version pin constants agree with ``pyproject.toml``, and
  core packages never import ``rlinf``/``ray`` (only ``compat/rlinf/``
  may).
* **Adapter tests (skip without rlinf):** the ``ParallelRolloutAdapter``
  produces, per environment worker, exactly the transition stream stepping
  the scripted environment directly produces on the same seeds — plus
  per-env episode accounting and exact step budgets.

The rollout tests boot a session-scoped RLinf cluster (a local Ray
runtime, several seconds); scripted env/policy classes are registered for
cloudpickle by-value serialization so Ray workers need not import this
test module.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pytest
from conftest import PACKAGE_NAME, REPO_ROOT

from isaac_rl_posttraining.compat import rlinf as compat_rlinf

try:
    RLINF_AVAILABLE = version("rlinf") == compat_rlinf.RLINF_PINNED_VERSION
except PackageNotFoundError:
    RLINF_AVAILABLE = False

requires_rlinf = pytest.mark.skipif(
    not RLINF_AVAILABLE,
    reason="adapters require the [rlinf] extra",
)


# ---------------------------------------------------------------------------
# Guard tests (run in every environment)
# ---------------------------------------------------------------------------


def test_compat_imports_and_guards_without_rlinf():
    """``compat.rlinf`` imports with rlinf/ray absent; adapters raise the hint."""
    code = textwrap.dedent(
        f"""
        import sys

        class _Blocker:
            blocked = ("rlinf", "ray")

            def find_spec(self, name, path=None, target=None):
                if any(name == b or name.startswith(b + ".") for b in self.blocked):
                    raise ImportError(f"blocked optional dependency {{name!r}}")
                return None

        sys.meta_path.insert(0, _Blocker())
        for mod in list(sys.modules):
            if any(mod == b or mod.startswith(b + ".") for b in _Blocker.blocked):
                del sys.modules[mod]

        import {PACKAGE_NAME}.compat
        import {PACKAGE_NAME}.compat.rlinf as compat_rlinf
        assert compat_rlinf.is_available() is False
        try:
            compat_rlinf.ParallelRolloutAdapter
        except ImportError as exc:
            assert "[rlinf]" in str(exc), f"install hint missing: {{exc}}"
        else:
            raise AssertionError("ParallelRolloutAdapter access must raise ImportError without rlinf")
        print("GUARD_OK")
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0 and "GUARD_OK" in result.stdout, (
        f"compat.rlinf must import-guard cleanly.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_core_never_imports_rlinf_or_ray():
    """Only ``compat/rlinf/`` may import rlinf (or ray, its runtime).

    RLinf is a library behind our interfaces, never a framework hosting us:
    if core code imported it, the optional extra would stop being optional
    and the native path would stop being the default.
    """
    package_dir = os.path.join(REPO_ROOT, PACKAGE_NAME)
    allowed_prefix = os.path.join(package_dir, "compat", "rlinf") + os.sep
    import_rx = re.compile(r"^\s*(?:import|from)\s+(rlinf|ray)\b", re.MULTILINE)
    offenders = []
    for root, dirs, names in os.walk(package_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in names:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            if path.startswith(allowed_prefix):
                continue
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            for match in import_rx.finditer(text):
                line_no = text[: match.start()].count("\n") + 1
                offenders.append(
                    f"{os.path.relpath(path, REPO_ROOT)}:{line_no}: {match.group(0).strip()}"
                )
    assert not offenders, "core code imports rlinf/ray:\n  " + "\n  ".join(offenders)


def test_pin_constants_match_pyproject():
    """The compat pin constants and the rlinf extra agree."""
    with open(os.path.join(REPO_ROOT, "pyproject.toml"), encoding="utf-8") as fh:
        pyproject = fh.read()
    assert f'"rlinf=={compat_rlinf.RLINF_PINNED_VERSION}"' in pyproject, (
        "pyproject [rlinf] must pin the validated adapter version"
    )
    assert f'"ray[default]=={compat_rlinf.RAY_PINNED_VERSION}"' in pyproject, (
        "pyproject [rlinf] must pin the validated Ray version"
    )
    assert re.fullmatch(r"[0-9a-f]{40}", compat_rlinf.RLINF_PINNED_COMMIT)


def test_require_rlinf_rejects_dependency_drift(monkeypatch):
    """Adapter startup fails before connecting with a mismatched Ray client."""
    from importlib import metadata

    from isaac_rl_posttraining.compat import rlinf as compat_rlinf

    versions = {"rlinf": compat_rlinf.RLINF_PINNED_VERSION, "ray": "2.56.1"}
    monkeypatch.setattr(compat_rlinf, "is_available", lambda: True)
    monkeypatch.setattr(metadata, "version", versions.__getitem__)

    with pytest.raises(RuntimeError, match=r"ray==2\.56\.1.*ray==2\.58\.0"):
        compat_rlinf.require_rlinf()


# ---------------------------------------------------------------------------
# Scripted env / policy shared by the parity tests (picklable by value)
# ---------------------------------------------------------------------------


class ScriptedEnv:
    """Deterministic single (unbatched) env: obs encodes (rank, episode, t).

    Episode ``e`` on rank ``r`` runs ``3 + (r + e) % 2`` steps and ends
    terminated, with success on even episodes; rewards are position-coded
    so any reordering or loss of transitions is visible in comparisons.

    The env auto-resets on termination (``step`` returns the next episode's
    initial observation, vector-env style) and ``reset`` is idempotent
    given the episode counter — so the native collector (which relies on
    auto-reset) and the adapter workers (which reset explicitly) observe
    byte-identical streams.
    """

    def __init__(self, rank: int):
        self.rank = int(rank)
        self.t = 0
        self.ep = 0

    def _obs(self):
        return {"state": np.array([self.rank, self.ep, self.t], dtype=np.float32)}

    def reset(self, **kwargs):
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        horizon = 3 + (self.rank + self.ep) % 2
        done = self.t >= horizon
        success = bool(done and self.ep % 2 == 0)
        reward = float(self.rank * 100 + self.ep * 10 + self.t)
        info = {"success": success, "is_success": success}
        if done:  # auto-reset: report the next episode's initial observation
            self.ep += 1
            self.t = 0
        return self._obs(), reward, done, False, info


class Batch1Env:
    """The same scripted env behind the batch-1 vectorized API the native
    collector consumes (numpy-batched values, list-of-dicts infos)."""

    def __init__(self, rank: int):
        self._env = ScriptedEnv(rank)

    def reset(self, **kwargs):
        obs, info = self._env.reset()
        return {k: v[None] for k, v in obs.items()}, info

    def step(self, action):
        obs, reward, done, truncated, info = self._env.step(np.asarray(action)[0])
        return (
            {k: v[None] for k, v in obs.items()},
            np.array([reward], dtype=np.float32),
            np.array([done]),
            np.array([truncated]),
            [info],
        )


def scripted_policy_action(obs) -> np.ndarray:
    """The shared deterministic policy: action = 0.5 * state."""
    return (np.asarray(obs["state"], dtype=np.float32) * 0.5).astype(np.float32)


class ScriptedPolicy:
    """Adapter-side policy protocol (``act(obs) -> action``)."""

    def act(self, obs):
        return scripted_policy_action(obs)


class RecordingBuffer:
    """Records every transition ``add`` files, for content comparison."""

    def __init__(self):
        self.adds = []

    def add(self, transition):
        self.adds.append(transition)


def make_env(rank: int) -> ScriptedEnv:
    return ScriptedEnv(rank)


def make_policy(rank: int) -> ScriptedPolicy:
    del rank
    return ScriptedPolicy()


def native_reference_stream(rank: int, num_episodes: int) -> list[dict]:
    """The buffer stream one scripted env yields when stepped directly with the scripted policy.

    Episode ids count up from zero per env; the terminal transition of an
    episode carries the env's success flag, exactly as the adapter writes it.
    """
    env = ScriptedEnv(rank)
    stream: list[dict] = []
    obs, _ = env.reset()
    episode_id = 0
    step_in_episode = 0
    while episode_id < num_episodes:
        action = scripted_policy_action(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        transition = {
            "obs": obs,
            "action": np.asarray(action).reshape(-1),
            "reward": float(reward),
            "next_obs": next_obs,
            "done": bool(done),
            "truncated": bool(truncated),
            "episode_id": episode_id,
            "step_in_episode": step_in_episode,
        }
        step_in_episode += 1
        if done or truncated:
            transition["is_success"] = info["is_success"]
            episode_id += 1
            step_in_episode = 0
        stream.append(transition)
        obs = next_obs
    return stream


def _stream_fields(transition) -> dict:
    """The typed transition the adapter filed, in the reference stream's field form."""
    fields = {
        "obs": transition.observation.arrays(),
        "action": np.asarray(transition.learner_action.value).reshape(-1),
        "reward": float(transition.reward("env")[-1]),
        "next_obs": transition.next_observation.arrays(),
        "done": bool(transition.terminated[-1]),
        "truncated": bool(transition.truncated[-1]),
        "episode_id": transition.episode_id,
        "step_in_episode": transition.metadata.step_in_episode,
    }
    if transition.ended:
        fields["is_success"] = transition.succeeded
    return fields


def _assert_streams_equal(adapter_stream: list, native_stream: list[dict]) -> None:
    """``adapter_stream`` holds the filed transitions, or their field forms already normalized."""
    assert len(adapter_stream) == len(native_stream)
    adapter_stream = [
        item if isinstance(item, dict) else _stream_fields(item)
        for item in adapter_stream
    ]
    for i, (got, want) in enumerate(zip(adapter_stream, native_stream)):
        assert set(got.keys()) == set(want.keys()), f"transition {i}: field sets differ"
        for key in want:
            got_v, want_v = got[key], want[key]
            if isinstance(want_v, dict):
                assert set(got_v.keys()) == set(want_v.keys()), f"transition {i}.{key}"
                for k in want_v:
                    np.testing.assert_array_equal(
                        np.asarray(got_v[k]),
                        np.asarray(want_v[k]),
                        err_msg=f"transition {i}.{key}.{k}",
                    )
            elif isinstance(want_v, np.ndarray):
                np.testing.assert_array_equal(
                    np.asarray(got_v), want_v, err_msg=f"transition {i}.{key}"
                )
            else:
                assert got_v == want_v, f"transition {i}.{key}: {got_v!r} != {want_v!r}"


# ---------------------------------------------------------------------------
# Rollout adapter (needs a Ray runtime; session-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rlinf_cluster():
    if not RLINF_AVAILABLE:
        pytest.skip("adapters require the [rlinf] extra")
    from ray import cloudpickle

    # Ray workers must not need to import this test module: serialize its
    # scripted classes by value.
    cloudpickle.register_pickle_by_value(sys.modules[__name__])

    from isaac_rl_posttraining.compat.rlinf.rollout import ensure_cluster

    yield ensure_cluster()
    # The Ray runtime is process-wide; it is left up for the session.


@requires_rlinf
@pytest.mark.rlinf_cluster
@pytest.mark.timeout(600)
def test_parallel_rollout_matches_the_scripted_reference(rlinf_cluster):
    """One worker, same seeds: the adapter's buffer stream is byte-identical
    to stepping the scripted env directly (fields, values, episode numbering)."""
    from isaac_rl_posttraining.compat.rlinf import ParallelRolloutAdapter

    buffer = RecordingBuffer()
    adapter = ParallelRolloutAdapter(
        make_env,
        make_policy,
        buffer,
        num_envs=1,
        num_rollouts=3,
        count_by="episode",
        cluster=rlinf_cluster,
    )
    try:
        metrics = adapter.run()
    finally:
        adapter.close()

    native = native_reference_stream(rank=0, num_episodes=3)
    _assert_streams_equal(buffer.adds, native)
    assert metrics["num_episodes"] == 3
    assert adapter.total_episodes == 3


@requires_rlinf
@pytest.mark.rlinf_cluster
@pytest.mark.timeout(600)
def test_parallel_rollout_per_env_accounting(rlinf_cluster):
    """N workers: per-rank streams match per-rank native references, buffer
    writes are rank-major, and episode accounting covers every env."""
    from isaac_rl_posttraining.compat.rlinf import ParallelRolloutAdapter

    num_envs, episodes_each = 3, 2
    buffer = RecordingBuffer()
    adapter = ParallelRolloutAdapter(
        make_env,
        make_policy,
        buffer,
        num_envs=num_envs,
        num_rollouts=episodes_each,
        count_by="episode",
        cluster=rlinf_cluster,
    )
    try:
        metrics = adapter.run()
    finally:
        adapter.close()

    assert metrics["num_episodes"] == num_envs * episodes_each
    assert adapter.episode_tracker.min_episodes_collected() == episodes_each

    # Rank-major buffer order: split the stream by the rank encoded in obs.
    streams: dict[int, list[dict]] = {r: [] for r in range(num_envs)}
    seen_ranks = []
    for add in (_stream_fields(transition) for transition in buffer.adds):
        rank = int(add["obs"]["state"][0])
        streams[rank].append(add)
        if not seen_ranks or seen_ranks[-1] != rank:
            seen_ranks.append(rank)
    assert seen_ranks == sorted(seen_ranks), "buffer writes must be rank-major"

    for rank in range(num_envs):
        native = native_reference_stream(rank=rank, num_episodes=episodes_each)
        # Episode ids are globally unique across ranks; compare them per-rank
        # by normalizing to per-rank numbering like the native reference.
        normalized = []
        rank_episode_base = None
        for add in streams[rank]:
            add = dict(add)
            if rank_episode_base is None:
                rank_episode_base = add["episode_id"]
            add["episode_id"] -= rank_episode_base
            normalized.append(add)
        _assert_streams_equal(normalized, native)

    # Success flags reached the tracker: even-numbered episodes succeed.
    assert metrics["ep_overall_success_rate"] == pytest.approx(0.5)


@requires_rlinf
@pytest.mark.rlinf_cluster
@pytest.mark.timeout(600)
def test_parallel_rollout_exact_step_budget(rlinf_cluster):
    """``collect(n_steps)`` yields exactly n transitions across all workers."""
    from isaac_rl_posttraining.compat.rlinf import ParallelRolloutAdapter

    buffer = RecordingBuffer()
    adapter = ParallelRolloutAdapter(
        make_env,
        make_policy,
        buffer,
        num_envs=3,
        num_rollouts=1,
        count_by="step",
        cluster=rlinf_cluster,
    )
    try:
        metrics = adapter.collect(10)
        assert len(buffer.adds) == 10
        assert metrics["total_steps"] == 10
        # State persists: the next budget resumes mid-episode, no resets lost.
        adapter.collect(5)
        assert len(buffer.adds) == 15
    finally:
        adapter.close()


@requires_rlinf
@pytest.mark.rlinf_cluster
@pytest.mark.timeout(600)
def test_step_budgets_feed_a_real_replay_buffer_with_unique_contiguous_episodes(
    rlinf_cluster,
):
    """Two step-counted calls pause both workers mid-episode; the real buffer must accept the stream:
    one id per (rank, episode), contiguous steps across the calls."""
    from isaac_rl_posttraining.buffers.replay import ReplayBuffer
    from isaac_rl_posttraining.compat.rlinf import ParallelRolloutAdapter

    buffer = ReplayBuffer(capacity=1000)
    adapter = ParallelRolloutAdapter(
        make_env,
        make_policy,
        buffer,
        num_envs=2,
        num_rollouts=1,
        count_by="step",
        cluster=rlinf_cluster,
    )
    try:
        adapter.collect(7)
        adapter.collect(6)
    finally:
        adapter.close()

    assert len(buffer) == 13
    stored = [buffer.transition(i) for i in range(len(buffer))]
    by_episode: dict = {}
    for transition in stored:
        by_episode.setdefault(transition.episode_id, []).append(transition)
    for transitions in by_episode.values():
        ranks = {int(t.observation.to_mapping()["state"][0]) for t in transitions}
        assert len(ranks) == 1, "an episode id must belong to one worker"
        assert [t.metadata.step_in_episode for t in transitions] == list(
            range(len(transitions))
        )
    assert len(by_episode) == adapter._next_episode_id
    assert adapter.total_episodes == sum(
        1 for t in stored if bool(t.terminated[-1]) or bool(t.truncated[-1])
    )
