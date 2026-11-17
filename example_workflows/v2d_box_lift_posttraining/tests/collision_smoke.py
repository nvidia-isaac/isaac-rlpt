# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local simulator checks for rejection, fixed candidate replay, and autoreset."""

from __future__ import annotations

import argparse
import inspect
import json
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import torch

from example_workflows.v2d_box_lift_posttraining.config import (
    load_capability_launch,
    load_task,
)
from simulators.v2d.contract import JsonValue

if TYPE_CHECKING:
    from example_workflows.v2d_box_lift_posttraining.capability import IsaacRuntime


def snapshot(runtime: IsaacRuntime, target_index: int) -> dict[str, torch.Tensor]:
    robot = runtime.env.scene["robot"]
    box = runtime.env.command_manager.get_term("motion").objects[target_index]
    return {
        "root": robot.data.root_state_w.clone(),
        "joint_pos": robot.data.joint_pos.clone(),
        "joint_vel": robot.data.joint_vel.clone(),
        "box": box.data.root_state_w.clone(),
    }


def check_fixtures(launch_config: Path, device: str) -> dict[str, JsonValue]:
    from example_workflows.v2d_box_lift_posttraining.capability import _build_runtime

    runtime, _, _, target_index, receipt = _build_runtime(launch_config, device, 2, 2)
    try:
        # Runtime construction installs the pinned V2D import paths first.
        from robotic_grounding.tasks.v2d_whole_body.mdp.events import (
            reset_robot_to_trajectory_start,
        )

        from example_workflows.v2d_box_lift_posttraining.collision_reset import (
            CollisionValidatedReset,
        )
        from example_workflows.v2d_box_lift_posttraining.reset_collision import (
            ResetCollisionChecker,
        )
        from example_workflows.v2d_box_lift_posttraining.reset_config import (
            ResetValidationConfig,
        )

        reset_source = Path(inspect.getfile(reset_robot_to_trajectory_start)).resolve()
        assert reset_source.is_relative_to(
            load_capability_launch(launch_config).v2d_root.resolve()
        ), f"Native reset loaded outside pinned V2D checkout: {reset_source}"
        record = runtime.reset_envs(None, seed=2)
        env = runtime.env
        config = ResetValidationConfig(True, 2, 0.001, ())
        checker = ResetCollisionChecker(env, config)
        nominal = snapshot(runtime, target_index)
        nominal_collisions = checker.check(0)
        assert nominal_collisions, "Nominal regression fixture no longer intersects"
        actions = np.concatenate(
            (record["arm_joint_pos"], record["finger_joint_pos"]), axis=1
        )
        # Obtain a valid test fixture; the production reset sampler never steps physics.
        for _ in range(5):
            runtime.step_batch(actions)
            if not checker.check(0) and not checker.check(1):
                break
        else:
            raise AssertionError("Could not obtain a valid test fixture")
        settled = snapshot(runtime, target_index)
        clock = env.sim.current_time
        steps = env.episode_length_buf.clone()
        event_cfg = env.event_manager.get_term_cfg("reset_to_trajectory_frame")
        event_cfg = replace(
            event_cfg,
            params={
                **event_cfg.params,
                "reset_validation": config,
                "fixed_reset": True,
                "seed": 2,
            },
        )
        event = CollisionValidatedReset(event_cfg, env)
        env.event_manager.set_term_cfg(
            "reset_to_trajectory_frame", replace(event_cfg, func=event)
        )
        runtime.collision_reset = event

        def restore(selected: torch.Tensor, state: dict[str, torch.Tensor]) -> None:
            robot = env.scene["robot"]
            box = env.command_manager.get_term("motion").objects[target_index]
            robot.write_root_state_to_sim(state["root"][selected], env_ids=selected)
            robot.write_joint_state_to_sim(
                state["joint_pos"][selected],
                state["joint_vel"][selected],
                env_ids=selected,
            )
            box.write_root_state_to_sim(state["box"][selected], env_ids=selected)

        def fixture_reset(environment, selected, **kwargs):
            state = nominal if torch.initial_seed() == 2 else settled
            restore(selected, state)

        with patch(
            "example_workflows.v2d_box_lift_posttraining.collision_reset.reset_robot_to_trajectory_start",
            fixture_reset,
        ):
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state(env.device).clone()
            event(env, torch.tensor([0], device=env.device), **event_cfg.params)
            assert event.sampler.attempts.tolist() == [2, 0]
            assert event.sampler.rejections.tolist() == [1, 0]
            assert not checker.check(0)
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state(env.device))
            after = snapshot(runtime, target_index)
            assert all(
                torch.equal(value, settled[name]) for name, value in after.items()
            )
            assert env.sim.current_time == clock
            assert env.episode_length_buf[1] == steps[1]

            accepted = event.sampler.accepted_seeds[0]
            runtime.reset_envs([0], seed=999)
            assert event.sampler.accepted_seeds[0] == accepted
            assert event.sampler.attempts.tolist() == [1, 0]
            assert all(
                torch.equal(value, settled[name])
                for name, value in snapshot(runtime, target_index).items()
            )

            env.episode_length_buf[0] = env.max_episode_length - 1
            result = runtime.step_batch(actions)
            assert result.completed.tolist() == [True, False]
            assert event.sampler.accepted_seeds[0] == accepted
            assert event.sampler.attempts.tolist() == [1, 0]
            assert not checker.check(0)
            assert all(
                torch.equal(value[0], settled[name][0])
                for name, value in snapshot(runtime, target_index).items()
            )

        return {
            "passed": True,
            "v2d_revision": receipt["v2d_revision"],
            "loaded_reset_implementation": str(reset_source),
            "nominal_collisions": [asdict(contact) for contact in nominal_collisions],
            "fresh_collision_queries": True,
            "selective_retries_preserve_other_env_and_time": True,
            "fixed_accepted_candidate_repeats": True,
            "native_autoreset_reuses_accepted_candidate": True,
            "cpu_and_cuda_rng_preserved": True,
            "fixture_used_settled_state": True,
            "training_performed": False,
        }
    finally:
        runtime.close()


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability-config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--strict-startup", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app
    try:
        if args.strict_startup:
            from example_workflows.v2d_box_lift_posttraining.capability import (
                create_capability,
            )

            task = load_task(load_capability_launch(args.capability_config).task_config)
            assert task.reset_validation is not None and task.reset_validation.enabled
            try:
                capability = create_capability(
                    args.capability_config, args.device, 2, 2
                )
            except RuntimeError as error:
                assert "No valid reset" in str(error) and "after 32 attempts" in str(
                    error
                )
                report = {
                    "strict_startup_rejected_invalid_nominal_pose": True,
                    "error": str(error),
                }
            else:
                capability.close()
                raise AssertionError("Expected nominal reset rejection")
        else:
            report = check_fixtures(args.capability_config, args.device)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print("COLLISION_SMOKE " + json.dumps(report), flush=True)
    except Exception as error:
        args.report.write_text(
            json.dumps(
                {
                    "passed": False,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
            + "\n"
        )
        traceback.print_exc()
        raise
    finally:
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
