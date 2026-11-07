# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from isaac_rl_posttraining.envs.simulator_entry import EnvironmentSpec, requested_layout
from isaac_rl_posttraining.envs.transport import (
    PROTOCOL_VERSION,
    EnvTransport,
    spec_payload,
)
from isaac_rl_posttraining.runtimes import RolloutWorkerContext
from simulators.v2d.contract import (
    ACTION_KEYS,
    IMAGE_KEYS,
    JOINT_DIM,
    STATE_KEYS,
    batched_observation_from_record,
    v2d_action_spec,
    v2d_observation_spec,
)
from simulators.v2d.env import V2DBatchedEnv, V2DEnv
from simulators.v2d.runtime import V2D


@pytest.mark.parametrize("phase,num_envs", [("train", 128), ("eval", 16)])
def test_model_sizing_queries_the_phase_batch_without_collecting(
    monkeypatch, phase, num_envs
):
    from isaac_rl_posttraining.runners.simulator_entries import sizing_environment

    transport = _Transport(num_envs)
    monkeypatch.setattr(
        "simulators.v2d.env.transport_from_uri", lambda *args, **kwargs: transport
    )
    monkeypatch.setenv("V2D_SERVER_URL", "tcp://127.0.0.1:7000")
    monkeypatch.setenv("V2D_EVAL_SERVER_URL", "tcp://127.0.0.1:7001")
    cfg = OmegaConf.create(
        {
            "env": {
                "env_name": "v2d/tissue_box_lift_hold",
                "max_episode_steps": 519,
                "image_keys": list(IMAGE_KEYS),
            },
            "training": {"num_envs": 128, "seed": 2, "chunk_size": 16},
            "eval": {"eval_num_envs": 16},
            "simulator": {"action_dim": 58, "image_resolution": 0},
        }
    )
    env = sizing_environment(V2D(), cfg, phase=phase, device="cpu")
    assert env.num_envs == 1
    assert env.single_observation_space["state"].shape == (58,)
    assert env.single_observation_space[IMAGE_KEYS[0]].shape == (2, 3, 3)
    assert env.single_action_space.shape == (58,)
    assert env.chunk_size == 16
    assert not transport.connected
    assert transport.reset_calls == []
    assert transport.step_calls == []
    env.close()


def _record(num_envs: int) -> dict[str, np.ndarray]:
    return {
        "arm_joint_pos": np.arange(num_envs * 14, dtype=np.float32).reshape(
            num_envs, 14
        ),
        "finger_joint_pos": np.arange(num_envs * 44, dtype=np.float32).reshape(
            num_envs, 44
        ),
        "image": np.zeros((num_envs, 2, 3, 4), dtype=np.float32),
        "image_right_wrist": np.zeros((num_envs, 2, 3, 4), dtype=np.float32),
        "image_left_wrist": np.zeros((num_envs, 2, 3, 4), dtype=np.float32),
    }


class _Transport(EnvTransport):
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self._connected = False
        self.reset_calls: list[tuple[int | None, list[int] | None]] = []
        self.step_calls: list[tuple[Any, bool]] = []

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def spec(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL_VERSION,
            "task_id": "tissue_box_lift_hold",
            "episode_horizon": 519,
            "action_dim": JOINT_DIM,
            "num_envs": self.num_envs,
            "prompt": "lift the box and hold it",
            "image_shapes": {key: [2, 3, 3] for key in IMAGE_KEYS},
            "specs": spec_payload(v2d_observation_spec(), v2d_action_spec()),
        }

    def reset(
        self, seed: int | None = None, env_ids: list[int] | None = None
    ) -> dict[str, Any]:
        self.reset_calls.append((seed, env_ids))
        observation = batched_observation_from_record(_record(self.num_envs))
        if self.num_envs == 1:
            observation = {key: value[0] for key, value in observation.items()}
        return {
            **observation,
            "info": {
                "initial_state_fingerprint": [
                    f"case-{index}" for index in range(self.num_envs)
                ],
                "case_seed": list(range(self.num_envs)),
            },
        }

    def step(self, action: Any, need_obs: bool = True) -> dict[str, Any]:
        self.step_calls.append((action, need_obs))
        observation = batched_observation_from_record(_record(self.num_envs))
        metrics = {"success": True, "lift_hold_success": True, "final_height_ok": True}
        if self.num_envs == 1:
            requested = len(action) if action and isinstance(action[0], list) else 1
            return {
                **{key: value[0] for key, value in observation.items()},
                "reward": 1.0,
                "done": True,
                "truncated": False,
                "success": True,
                "num_steps": requested,
                "info": {"episode_steps": requested, "task_metrics": metrics},
            }
        if not need_obs:
            for key in IMAGE_KEYS:
                del observation[key]
        return {
            **observation,
            "reward": np.ones(self.num_envs, dtype=np.float32),
            "done": np.ones(self.num_envs, dtype=np.bool_),
            "truncated": np.zeros(self.num_envs, dtype=np.bool_),
            "success": np.ones(self.num_envs, dtype=np.bool_),
            "num_steps": np.ones(self.num_envs, dtype=np.int64),
            "info": [
                {"episode_steps": 1, "task_metrics": dict(metrics)}
                for _ in range(self.num_envs)
            ],
        }

    def close(self) -> None:
        self._connected = False


def test_joint_and_tiled_camera_contract() -> None:
    assert JOINT_DIM == 58
    assert STATE_KEYS == ("right_arm", "left_arm", "right_finger", "left_finger")
    assert ACTION_KEYS == ("right_arm", "right_finger", "left_arm", "left_finger")
    observation = batched_observation_from_record(_record(2))
    assert observation["state"].shape == (2, 58)
    assert tuple(key for key in observation if key != "state") == IMAGE_KEYS
    assert all(observation[key].dtype == np.uint8 for key in IMAGE_KEYS)


def test_scalar_client_preserves_terminal_metrics_and_reset_fingerprint() -> None:
    transport = _Transport(1)
    env = V2DEnv(
        max_episode_steps=519,
        task_id="tissue_box_lift_hold",
        transport=transport,
    )
    observation, reset_info = env.reset(seed=17)
    assert observation["state"].shape == (58,)
    assert reset_info["initial_state_fingerprint"] == ["case-0"]
    _, reward, terminated, truncated, info = env.step(np.zeros(58, dtype=np.float32))
    assert (reward, terminated, truncated) == (1.0, True, False)
    assert info["task_metrics"]["lift_hold_success"] is True
    assert transport.reset_calls == [(17, None)]


def test_scalar_client_separates_single_action_and_chunk_contracts() -> None:
    transport = _Transport(1)
    env = V2DEnv(
        max_episode_steps=519,
        task_id="tissue_box_lift_hold",
        transport=transport,
    )
    env.reset()
    actions = np.zeros((3, JOINT_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="must have shape"):
        env.step(actions)
    _, rewards, terminated, truncated, info = env.step_chunk(actions)
    assert rewards == [0.0, 0.0, 1.0]
    assert terminated and not truncated
    assert info["num_steps"] == 3
    assert len(transport.step_calls) == 1


def test_batched_client_preserves_per_environment_terminal_metrics() -> None:
    transport = _Transport(3)
    env = V2DBatchedEnv(
        num_envs=3,
        max_episode_steps=519,
        task_id="tissue_box_lift_hold",
        transport=transport,
    )
    observation, info = env.reset(seed=23)
    assert observation["state"].shape == (3, 58)
    assert info["initial_state_fingerprint"] == ["case-0", "case-1", "case-2"]
    env.need_obs = False
    next_observation, reward, terminated, truncated, infos = env.step(
        np.zeros((3, 58), dtype=np.float32)
    )
    assert reward.tolist() == [1.0, 1.0, 1.0]
    assert terminated.all() and not truncated.any()
    assert all(item["task_metrics"]["final_height_ok"] for item in infos)
    assert all(not next_observation[key].any() for key in IMAGE_KEYS)


def test_entry_runs_one_rollout_worker_on_its_one_server() -> None:
    entry = V2D()
    assert requested_layout(entry, 128, workers=1).workers == 1
    with pytest.raises(ValueError, match="placement.envs.workers=2"):
        requested_layout(entry, 128, workers=2)


def test_entry_uses_one_native_batch_and_rejects_non_joint_geometry(
    monkeypatch,
) -> None:
    entry = V2D()
    assert entry.worker_layout(32).batched and entry.worker_layout(32).workers == 1
    with pytest.raises(ValueError, match="one server; 2 addresses"):
        entry.worker_layout(32, num_servers=2)
    assert entry.evaluation == "dedicated"
    spec = EnvironmentSpec(
        env_name="v2d/tissue_box_lift_hold",
        num_envs=32,
        max_episode_steps=519,
        seed=0,
        server_url="tcp://127.0.0.1:7000",
        action_dim=58,
        image_resolution=0,
    )
    recorded = {}

    def client(server_url, **kwargs):
        recorded.update(server_url=server_url, **kwargs)
        return object()

    monkeypatch.setattr("simulators.v2d.runtime.V2DBatchedEnv", client)
    result = entry.make_environment(
        spec,
        "eval",
        RolloutWorkerContext(
            worker_index=0,
            num_rollout_workers=1,
            environment_indices=tuple(range(32)),
            device="cpu",
        ),
    )
    assert result is not None
    assert recorded == {
        "server_url": "tcp://127.0.0.1:7000",
        "num_envs": 32,
        "max_episode_steps": 519,
        "task_id": "tissue_box_lift_hold",
    }
    with pytest.raises(ValueError, match="action_dim=58"):
        entry.action_spec(
            EnvironmentSpec(
                env_name=spec.env_name,
                num_envs=1,
                max_episode_steps=519,
                seed=0,
                action_dim=35,
                image_resolution=0,
            )
        )
