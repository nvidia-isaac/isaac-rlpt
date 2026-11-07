# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from example_workflows.v2d_box_lift_posttraining.compare_native import compare
from example_workflows.v2d_box_lift_posttraining.evaluation_verifier import verify
from example_workflows.v2d_box_lift_posttraining.launch import (
    _wait_for_server,
    parser,
    resolve_plan,
)


@pytest.mark.parametrize("cadence,cap", [("step", 128), ("episode", 17664)])
@pytest.mark.parametrize("minibatch", [64, 128, 256])
@pytest.mark.parametrize("target_reduction", ["mean", "min"])
def test_utd4_training_plan_preserves_horizon_and_cadence(
    tmp_path, cadence, cap, minibatch, target_reduction
):
    args = parser().parse_args(
        [
            "--v2d-root",
            "/unread/v2d",
            "--checkpoint",
            "/unread/checkpoint",
            "--motion-file",
            "/unread/motion.parquet",
            "--sim-python",
            "/unread/python",
            "--output-dir",
            str(tmp_path),
            "--evaluation-mode",
            "disabled",
            "--training-cadence",
            cadence,
            "--updates-per-transition",
            "1",
            "--replay-batch-size",
            str(minibatch),
            "--max-updates-per-burst",
            str(cap),
            "--critic-target-reduction",
            target_reduction,
        ]
    )
    plan = resolve_plan(args)
    settings = plan["resolved_settings"]
    assert len(plan["servers"]) == 1
    assert settings["env.max_episode_steps"] == 550
    assert settings["algorithm.learning_starts"] == 70400
    assert settings["algorithm.batch_size"] == minibatch
    assert settings["algorithm.critic_target_reduction"] == target_reduction
    assert settings["algorithm.num_updates_per_train_step"] == 1
    assert settings["algorithm.num_critic_updates_per_actor_update"] == 4
    assert settings["training.max_updates_per_burst"] == cap
    assert settings["training.rollout_count_by"] == cadence
    assert settings["training.train_after_episode"] == (cadence == "episode")
    assert settings["eval.eval_freq"] == 0
    assert not settings["eval.eval_on_first_step"]


def _record(
    *,
    reset_seed: int,
    environment_index: int,
    success: bool,
    policy_version: int,
) -> dict[str, object]:
    return {
        "evaluation_index": 0,
        "policy_version": policy_version,
        "environment_index": environment_index,
        "episode_id": environment_index,
        "reset_seed": reset_seed,
        "reset_seed_scope": "worker_batch",
        "policy_seed": reset_seed + environment_index,
        "case_seed": reset_seed * 100 + environment_index,
        "round_index": 0,
        "initial_state_fingerprint": f"case-{reset_seed}-{environment_index}",
        "terminal_reason": "success" if success else "task_horizon",
        "decisions": 138,
        "total_reward": float(success),
        "success": success,
        "task_metrics": {
            "success": success,
            "combined_success": success,
            "lift_hold_success": True,
            "final_height_ok": success,
            "evaluator": "lift_hold",
        },
    }


def _write(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_resolved_plan_separates_training_and_evaluation_batching(
    tmp_path: Path,
) -> None:
    args = parser().parse_args(
        [
            "--v2d-root",
            "/unread/v2d",
            "--checkpoint",
            "/unread/checkpoint",
            "--motion-file",
            "/unread/motion.parquet",
            "--sim-python",
            "/unread/python",
            "--output-dir",
            str(tmp_path / "output"),
            "--evaluation-mode",
            "enabled",
            "--dry-run",
        ]
    )
    plan = resolve_plan(args)
    settings = plan["resolved_settings"]
    assert settings["training.num_envs"] == 128
    assert settings["training.policy_decode_batch_size"] == 16
    assert settings["eval.eval_num_envs"] == 32
    assert settings["eval.eval_num_episodes"] == 32
    assert settings["env.max_episode_steps"] == 550
    assert settings["eval.eval_max_steps"] == 138
    assert settings["+simulator.training_fixed_reset"] is True
    assert settings["+simulator.evaluation_fixed_reset"] is True
    native_settings = plan["native_baseline_settings"]
    assert native_settings is not None
    assert native_settings["training.num_rollouts"] == 0
    assert native_settings["eval.eval_base"] is True
    assert native_settings["eval.eval_num_envs"] == 32
    assert native_settings["eval.eval_num_episodes"] == 32
    assert plan["native_baseline_learner"] != plan["learner"]
    assert len(plan["servers"]) == 2
    assert plan["server_visible_devices"] == ["1", "2"]
    assert plan["server_receipts"] == [
        str((tmp_path / "output/runtime/train_server_receipt.json").resolve()),
        str((tmp_path / "output/runtime/eval_server_receipt.json").resolve()),
    ]
    assert all(
        command[command.index("--device") + 1] == "cuda:0"
        for command in plan["servers"]
    )


def test_resolved_plan_rejects_shared_simulator_gpu(tmp_path: Path) -> None:
    args = parser().parse_args(
        [
            "--v2d-root",
            "/unread/v2d",
            "--checkpoint",
            "/unread/checkpoint",
            "--motion-file",
            "/unread/motion.parquet",
            "--sim-python",
            "/unread/python",
            "--output-dir",
            str(tmp_path / "output"),
            "--evaluation-mode",
            "enabled",
            "--train-device",
            "cuda:1",
            "--eval-device",
            "cuda:1",
            "--dry-run",
        ]
    )
    with pytest.raises(ValueError, match="distinct GPUs"):
        resolve_plan(args)


@pytest.mark.parametrize("load_checkpoint", [False, True])
def test_evaluation_only_starts_one_simulator_on_one_gpu(
    tmp_path: Path, load_checkpoint: bool
) -> None:
    arguments = [
        "--v2d-root",
        "/unread/v2d",
        "--checkpoint",
        "/unread/checkpoint",
        "--motion-file",
        "/unread/motion.parquet",
        "--sim-python",
        "/unread/python",
        "--output-dir",
        str(tmp_path / "output"),
        "--evaluation-mode",
        "enabled",
        "--training-steps",
        "0",
        "--eval-num-envs",
        "2",
        "--eval-num-episodes",
        "4",
        "--train-device",
        "cuda:0",
        "--eval-device",
        "cuda:0",
    ]
    if load_checkpoint:
        arguments.extend(["--steering-checkpoint", "/unread/steering"])

    plan = resolve_plan(parser().parse_args(arguments))

    assert len(plan["servers"]) == 1
    assert plan["server_visible_devices"] == ["0"]
    assert plan["server_receipts"] == [
        str((tmp_path / "output/runtime/eval_server_receipt.json").resolve())
    ]
    assert plan["native_baseline_learner"] is None
    settings = plan["resolved_settings"]
    assert settings["training.num_rollouts"] == 0
    assert settings["eval.eval_base"] is False
    assert settings["eval.eval_on_first_step"] is True
    assert settings["eval.eval_num_envs"] == 2
    assert settings["eval.eval_num_episodes"] == 4
    if load_checkpoint:
        assert settings["training.load_dir"] == "/unread/steering"
    else:
        assert settings["training.load_dir"] is None


def test_default_training_plan_disables_all_evaluation(tmp_path: Path) -> None:
    args = parser().parse_args(
        [
            "--v2d-root",
            "/unread/v2d",
            "--checkpoint",
            "/unread/checkpoint",
            "--motion-file",
            "/unread/motion.parquet",
            "--sim-python",
            "/unread/python",
            "--output-dir",
            str(tmp_path / "output"),
            "--dry-run",
        ]
    )

    plan = resolve_plan(args)
    settings = plan["resolved_settings"]
    assert settings["training.num_envs"] == 128
    assert settings["algorithm.learning_starts"] == 128 * 550
    assert settings["training.num_rollouts"] == 704000
    assert settings["training.rollout_count_by"] == "step"
    assert settings["training.train_after_episode"] is False
    assert settings["training.max_updates_per_burst"] == 128
    assert settings["training.save_interval"] == 10000
    assert settings["algorithm.batch_size"] == 64
    assert settings["algorithm.num_updates_per_train_step"] == 1
    assert settings["algorithm.num_critic_updates_per_actor_update"] == 4
    assert settings["algorithm.ent_coef"] == pytest.approx(0.005)
    assert settings["algorithm.td_target_clip"] == [0.0, 1.0]
    assert settings["eval.eval_on_first_step"] is False
    assert settings["eval.eval_freq"] == 0
    assert plan["eval_url"] is None
    assert plan["eval_capability_path"] is None
    assert plan["eval_capability"] is None
    assert plan["native_baseline_learner"] is None
    assert plan["native_baseline_settings"] is None
    assert len(plan["servers"]) == 1
    assert plan["server_visible_devices"] == ["1"]
    assert plan["server_receipts"] == [
        str((tmp_path / "output/runtime/train_server_receipt.json").resolve())
    ]


def test_native_mode_rejects_disabled_evaluation(tmp_path: Path) -> None:
    args = parser().parse_args(
        [
            "--v2d-root",
            "/unread/v2d",
            "--checkpoint",
            "/unread/checkpoint",
            "--motion-file",
            "/unread/motion.parquet",
            "--sim-python",
            "/unread/python",
            "--output-dir",
            str(tmp_path / "output"),
            "--native",
            "--evaluation-mode",
            "disabled",
            "--dry-run",
        ]
    )

    with pytest.raises(ValueError, match="requires evaluation-mode=enabled"):
        resolve_plan(args)


class _ProcessStub:
    def __init__(self, return_codes: Iterator[int | None]) -> None:
        self._return_codes = return_codes

    def poll(self) -> int | None:
        return next(self._return_codes)


def test_wait_for_server_accepts_live_process_with_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    log_path = tmp_path / "server.log"
    log_path.write_text("ready\n", encoding="utf-8")
    process = cast(subprocess.Popen[str], _ProcessStub(iter([None])))

    _wait_for_server(
        process,
        receipt=receipt,
        log_path=log_path,
        timeout_seconds=1,
    )


def test_wait_for_server_reports_early_process_exit(tmp_path: Path) -> None:
    receipt = tmp_path / "missing.json"
    log_path = tmp_path / "server.log"
    log_path.write_text("extension setup failed\n", encoding="utf-8")
    process = cast(subprocess.Popen[str], _ProcessStub(iter([7])))

    with pytest.raises(RuntimeError, match="extension setup failed"):
        _wait_for_server(
            process,
            receipt=receipt,
            log_path=log_path,
            timeout_seconds=1,
        )


def test_resolved_plan_rejects_nonmultiple_evaluation_count(tmp_path: Path) -> None:
    args = parser().parse_args(
        [
            "--v2d-root",
            "/unread/v2d",
            "--checkpoint",
            "/unread/checkpoint",
            "--motion-file",
            "/unread/motion.parquet",
            "--sim-python",
            "/unread/python",
            "--output-dir",
            str(tmp_path / "output"),
            "--eval-num-envs",
            "32",
            "--eval-num-episodes",
            "50",
            "--dry-run",
        ]
    )
    with pytest.raises(ValueError, match="exact multiple"):
        resolve_plan(args)


def test_evaluation_verifier_and_paired_comparison(tmp_path: Path) -> None:
    native_path = tmp_path / "native.jsonl"
    steered_path = tmp_path / "steered.jsonl"
    native = [
        _record(
            reset_seed=10000,
            environment_index=index,
            success=index == 0,
            policy_version=0,
        )
        for index in range(2)
    ]
    steered = [
        _record(
            reset_seed=10000, environment_index=index, success=True, policy_version=4
        )
        for index in range(2)
    ]
    for record in (*native, *steered):
        record["initial_state_fingerprint"] = "same-nominal-state"
    _write(native_path, native)
    _write(steered_path, steered)

    verified = verify(native_path, num_envs=2, num_episodes=2)
    compared = compare(native_path, steered_path)

    assert verified["success_rate"] == pytest.approx(0.5)
    assert compared["native_successes"] == 1
    assert compared["steered_successes"] == 2
    assert compared["success_rate_change"] == pytest.approx(0.5)


@pytest.mark.parametrize("case_seed", [True, "10000", 1.5, None])
def test_verifier_rejects_non_integer_case_seed(tmp_path: Path, case_seed) -> None:
    path = tmp_path / "evaluation.jsonl"
    record = _record(
        reset_seed=10000, environment_index=0, success=True, policy_version=0
    )
    record["case_seed"] = case_seed
    _write(path, [record])

    with pytest.raises(TypeError, match="case_seed must be an integer"):
        verify(path, num_envs=1, num_episodes=1)


def test_comparison_rejects_unpaired_initial_states(tmp_path: Path) -> None:
    native_path = tmp_path / "native.jsonl"
    steered_path = tmp_path / "steered.jsonl"
    native = [
        _record(reset_seed=10000, environment_index=0, success=True, policy_version=0)
    ]
    steered = [
        _record(reset_seed=10000, environment_index=0, success=True, policy_version=2)
    ]
    steered[0]["initial_state_fingerprint"] = "different"
    _write(native_path, native)
    _write(steered_path, steered)

    with pytest.raises(ValueError, match="fingerprint differs"):
        compare(native_path, steered_path)
