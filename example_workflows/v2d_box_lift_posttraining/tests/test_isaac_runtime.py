# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the tensor boundary used by Isaac's indexed scene reset."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from example_workflows.v2d_box_lift_posttraining.capability import (
    TERMINAL_CAPTURE_KEY,
    IsaacRuntime,
    evaluation_case_seed,
)
from example_workflows.v2d_box_lift_posttraining.reset_events import BatchedTextureReset


@pytest.mark.parametrize("selected", [None, [1], [0, 2]])
def test_reset_uses_device_tensor_indices_and_preserves_seed(selected):
    calls = []
    observation = {"record": {"state": torch.arange(6).reshape(3, 2)}}

    def reset(*, seed, env_ids):
        if env_ids is not None:
            assert isinstance(env_ids, torch.Tensor)
            assert env_ids.dtype == torch.int64
            assert env_ids.device == torch.device("cpu")
            assert env_ids.tolist() == selected
        else:
            assert selected is None
        calls.append(seed)
        return observation, {}

    command = SimpleNamespace(
        virtual_object_controller_scale_factor=torch.tensor(0.0),
        virtual_object_controller_scale_factor_per_env=torch.zeros(3),
    )
    env = SimpleNamespace(
        device="cpu",
        reset=reset,
        observation_manager=SimpleNamespace(compute=lambda: observation),
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )
    runtime = IsaacRuntime(
        env,
        render_ticks=0,
        termination_reasons=lambda env, index: (),
        texture_resets=(),
    )

    record = runtime.reset_envs(selected, seed=2834126987)

    assert calls == [2834126987]
    np.testing.assert_array_equal(record["state"], np.arange(6).reshape(3, 2))


@pytest.mark.parametrize("selected", [tuple(range(128)), (3, 127), (3,)])
def test_seeded_resets_capture_once_after_all_selected_states_are_reset(selected):
    states = torch.zeros(128, dtype=torch.float64)
    render = Mock()
    cases = [(index, evaluation_case_seed(2, 0, index)) for index in selected]
    reset_calls = []
    texture_resets = (BatchedTextureReset(Mock()), BatchedTextureReset(Mock()))

    def reset(*, seed, env_ids):
        render.assert_not_called()
        assert env_ids.dtype == torch.int64
        index = env_ids.item()
        for event in texture_resets:
            assert event.apply_materials == (index == selected[-1])
        reset_calls.append((index, seed))
        states[index] = seed
        return {"record": {"state": states}}, {}

    def compute():
        assert reset_calls == cases
        assert render.call_count == 10
        return {"record": {"state": states}}

    command = SimpleNamespace(
        virtual_object_controller_scale_factor=torch.tensor(0.0),
        virtual_object_controller_scale_factor_per_env=torch.zeros(128),
    )
    capture = Mock(side_effect=compute)
    env = SimpleNamespace(
        device="cpu",
        reset=reset,
        sim=SimpleNamespace(render=render),
        observation_manager=SimpleNamespace(compute=capture),
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )
    runtime = IsaacRuntime(
        env,
        render_ticks=10,
        termination_reasons=lambda env, index: (),
        texture_resets=texture_resets,
    )

    record = runtime.reset_cases(cases)

    capture.assert_called_once_with()
    assert all(event.apply_materials for event in texture_resets)
    expected = np.zeros(128)
    for index, seed in cases:
        expected[index] = seed
    np.testing.assert_array_equal(record["state"], expected)


def test_reset_failure_restores_material_writes():
    texture = BatchedTextureReset(Mock())

    def reset(*, seed, env_ids):
        assert not texture.apply_materials
        raise RuntimeError("reset failed")

    env = SimpleNamespace(device="cpu", reset=reset)
    runtime = IsaacRuntime(
        env,
        render_ticks=0,
        termination_reasons=lambda env, index: (),
        texture_resets=(texture,),
    )

    with pytest.raises(RuntimeError, match="reset failed"):
        runtime.reset_cases([(0, 2), (1, 3)])

    assert texture.apply_materials


def test_reset_statistics_are_captured_before_native_autoreset():
    attempts = torch.tensor([3, 1])
    collision_reset = Mock()
    collision_reset.diagnostics.side_effect = lambda: {"reset_attempts": attempts}
    record = {"state": torch.zeros(2, 3)}

    def step(actions):
        attempts[0] = 1
        return (
            {"record": record},
            torch.zeros(2),
            torch.tensor([True, False]),
            torch.zeros(2, dtype=torch.bool),
            {TERMINAL_CAPTURE_KEY: record},
        )

    runtime = IsaacRuntime(
        SimpleNamespace(num_envs=2, device="cpu", step=step),
        render_ticks=0,
        termination_reasons=lambda env, index: (),
        texture_resets=(),
        collision_reset=collision_reset,
    )
    result = runtime.step_batch(np.zeros((2, 58)))
    np.testing.assert_array_equal(result.reset_diagnostics["reset_attempts"], [3, 1])
    assert attempts.tolist() == [1, 1]


@pytest.mark.parametrize("seed", [None, 123])
def test_explicit_seed_reaches_collision_sampler_before_selected_reset(seed):
    collision_reset = Mock()

    def reset(*, seed, env_ids):
        if seed is None:
            collision_reset.reseed.assert_not_called()
        else:
            call = collision_reset.reseed.call_args
            assert call.args[0].tolist() == [1]
            assert call.args[1] == seed

    command = SimpleNamespace(
        virtual_object_controller_scale_factor=torch.tensor(0.0),
        virtual_object_controller_scale_factor_per_env=torch.zeros(2),
    )
    env = SimpleNamespace(
        device="cpu",
        reset=reset,
        observation_manager=SimpleNamespace(compute=lambda: {"record": {}}),
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )
    runtime = IsaacRuntime(
        env,
        render_ticks=0,
        termination_reasons=lambda env, index: (),
        texture_resets=(),
        collision_reset=collision_reset,
    )
    runtime.reset_envs([1], seed=seed)
