# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GR00T-LIBERO integration seams, tested without the LIBERO or GR00T stacks.

Four layers, all local and deterministic:

1. :class:`LiberoGr00tAdapter` -- the observation/action conventions of the
   released GR00T-N1.7-LIBERO checkpoints over a scripted inner environment:
   camera-view renaming, the 180-degree training orientation, the exact
   reference gripper transform, and chunk pass-through.
2. ``example_workflows/libero_posttraining/scripts/eval.py`` helpers -- success-rate accounting through the
   real rollout worker and evaluator over a mock base model and a scripted task
   environment, plus the DSRL-target selection rule and the report format.
3. The documented LIBERO override set composed onto the canonical Hydra
   config (task variants are CLI overrides, not config files).
4. The ``example_workflows/libero_posttraining/scripts/train.sh`` launcher dry-run plan.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import REPO_ROOT, run_launcher_dryrun

gym = pytest.importorskip("gymnasium")

from simulators.libero.policies.gr00t import (
    LiberoGr00tAdapter,
    gripper_to_env,
)

CONFIG_DIR = Path(REPO_ROOT) / "isaac_rl_posttraining" / "configs"
LIBERO_LAUNCHER = (
    Path(REPO_ROOT)
    / "example_workflows"
    / "libero_posttraining"
    / "scripts"
    / "train.sh"
)
EVAL_SCRIPT = (
    Path(REPO_ROOT)
    / "example_workflows"
    / "libero_posttraining"
    / "scripts"
    / "eval.py"
)

STATE_DIM = 8
IMG = 8  # stub camera resolution


def _load_eval_libero():
    spec = importlib.util.spec_from_file_location("eval_libero", EVAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("eval_libero", module)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. The conventions adapter over a scripted inner environment
# ---------------------------------------------------------------------------


class _ScriptedLiberoEnv(gym.Env):
    """Stands in for LiberoEnv: two upright camera frames + eef state."""

    def __init__(self, succeed_on_episode=(), max_chunks=3):
        image_space = gym.spaces.Box(
            low=0, high=255, shape=(IMG, IMG, 3), dtype=np.uint8
        )
        self.observation_space = gym.spaces.Dict(
            {
                "agentview_image": image_space,
                "robot0_eye_in_hand_image": image_space,
                "state": gym.spaces.Box(
                    -np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.language_instruction = "scripted task"
        self.task_name = "scripted_task"
        self.succeed_on_episode = set(succeed_on_episode)
        self.max_chunks = max_chunks
        self.episode = -1
        self.chunks = 0
        self.stepped_actions: list[np.ndarray] = []

    def _obs(self):
        scene = np.zeros((IMG, IMG, 3), np.uint8)
        scene[:, 0, :] = 255  # LEFT edge marked: orientation is observable
        wrist = np.full((IMG, IMG, 3), 7, np.uint8)
        wrist[:, 0, :] = 99
        return {
            "agentview_image": scene,
            "robot0_eye_in_hand_image": wrist,
            "state": np.arange(STATE_DIM, dtype=np.float32),
            "prompt": self.language_instruction,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.chunks = 0
        return self._obs(), {"supports_action_chunking": True}

    def step(self, action):
        self.stepped_actions.append(np.array(action, copy=True))
        self.chunks += 1
        success = self.episode in self.succeed_on_episode and self.chunks >= 2
        truncated = not success and self.chunks >= self.max_chunks
        info = {"success": success, "is_success": success}
        return self._obs(), float(success), success, truncated, info


def test_adapter_renames_views_and_restores_training_orientation():
    env = LiberoGr00tAdapter(_ScriptedLiberoEnv())
    assert set(env.observation_space.spaces) == {
        "video.image",
        "video.wrist_image",
        "state",
    }
    obs, info = env.reset()
    assert info["supports_action_chunking"] is True
    assert set(obs) == {"video.image", "video.wrist_image", "state", "prompt"}
    # The inner env marks the upright frame's LEFT edge; the training frames
    # are the raw buffer rotated 180 degrees = the upright frame mirrored
    # horizontally, so the mark must land on the RIGHT edge.
    assert (obs["video.image"][:, -1, :] == 255).all()
    assert (obs["video.image"][:, 0, :] == 0).all()
    assert (obs["video.wrist_image"][:, -1, :] == 99).all()
    np.testing.assert_allclose(obs["state"], np.arange(STATE_DIM))
    assert obs["prompt"] == "scripted task"
    assert env.steering_key_mapping["image"] == ["video.image", "video.wrist_image"]


def test_adapter_requires_both_checkpoint_cameras():
    class _OneCamera(gym.Env):
        observation_space = gym.spaces.Dict(
            {
                "agentview_image": gym.spaces.Box(0, 255, (IMG, IMG, 3), np.uint8),
                "state": gym.spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32),
            }
        )
        action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)

    with pytest.raises(ValueError, match="robot0_eye_in_hand"):
        LiberoGr00tAdapter(_OneCamera())


def _reference_gripper(action: np.ndarray) -> np.ndarray:
    """The reference evaluation's composition, transcribed verbatim:
    normalize_gripper_action (binarize=True) then invert_gripper_action."""
    out = np.array(action, dtype=np.float32, copy=True)
    out[..., -1] = 2.0 * (out[..., -1] - 0.0) / (1.0 - 0.0) - 1.0
    out[..., -1] = np.sign(out[..., -1])
    out[..., -1] = out[..., -1] * -1.0
    return out


def test_gripper_transform_matches_reference_exactly():
    grippers = np.array([0.0, 0.1, 0.49, 0.5, 0.51, 0.9, 1.0], np.float32)
    chunk = np.zeros((len(grippers), 7), np.float32)
    chunk[:, 6] = grippers
    np.testing.assert_array_equal(
        gripper_to_env(grippers), _reference_gripper(chunk)[:, 6]
    )
    # Model-open (1.0) commands the simulator's open (-1); model-close (0.0)
    # commands close (+1).
    assert gripper_to_env(np.float32(1.0)) == pytest.approx(-1.0)
    assert gripper_to_env(np.float32(0.0)) == pytest.approx(1.0)


def test_adapter_converts_gripper_and_passes_arm_dims_through():
    inner = _ScriptedLiberoEnv()
    env = LiberoGr00tAdapter(inner)
    env.reset()
    chunk = np.tile(np.linspace(-0.5, 0.5, 7, dtype=np.float32), (4, 1))
    chunk[:, 6] = [0.0, 0.4, 0.6, 1.0]
    env.step(chunk)
    executed = inner.stepped_actions[0]
    assert executed.shape == (4, 7)
    np.testing.assert_allclose(executed[:, :6], chunk[:, :6])  # untouched
    np.testing.assert_array_equal(executed[:, 6], [1.0, 1.0, -1.0, -1.0])
    # The caller's chunk is never mutated in place.
    np.testing.assert_array_equal(
        chunk[:, 6], np.array([0.0, 0.4, 0.6, 1.0], dtype=np.float32)
    )


# ---------------------------------------------------------------------------
# 2. Eval-runner success accounting and target selection
# ---------------------------------------------------------------------------


def test_eval_runner_counts_successes_through_the_real_evaluator():
    """evaluate_task drives the rollout worker with the frozen policy's native decode
    and reports the terminal per-episode successes -- scripted here as episodes 0 and 2 of 4."""
    pytest.importorskip("torch")
    eval_libero = _load_eval_libero()
    from isaac_rl_posttraining.policies.base import MockBaseModelPolicy

    env = LiberoGr00tAdapter(
        _ScriptedLiberoEnv(succeed_on_episode={0, 2}, max_chunks=3)
    )
    successes = eval_libero.evaluate_task(
        MockBaseModelPolicy(action_horizon=16, action_dim=7),
        env,
        episodes=4,
        action_exec_len=8,
        max_decision_steps=10,
    )
    assert successes == 2


def _result(eval_libero, task_id, sr, episodes=20):
    return eval_libero.TaskResult(
        task_id=task_id,
        task_name=f"task_{task_id}",
        description=f"description {task_id}",
        episodes=episodes,
        successes=round(sr * episodes),
    )


def test_pick_dsrl_target_worst_in_band():
    eval_libero = _load_eval_libero()
    results = [
        _result(eval_libero, 0, 0.90),
        _result(eval_libero, 1, 0.45),
        _result(eval_libero, 2, 0.35),
        _result(eval_libero, 3, 0.75),
    ]
    pick, note = eval_libero.pick_dsrl_target(results)
    assert pick.task_id == 2  # worst INSIDE the band
    assert "worst" in note


def test_pick_dsrl_target_notes_near_zero_worst():
    """A ~0 task is never picked; the note names it next to the band pick."""
    eval_libero = _load_eval_libero()
    results = [
        _result(eval_libero, 0, 0.0),
        _result(eval_libero, 1, 0.55),
        _result(eval_libero, 2, 0.95),
    ]
    pick, note = eval_libero.pick_dsrl_target(results)
    assert pick.task_id == 1
    assert "task_0" in note and "0%" in note


def test_pick_dsrl_target_without_band_candidates_falls_back():
    eval_libero = _load_eval_libero()
    results = [
        _result(eval_libero, 0, 0.02),
        _result(eval_libero, 1, 0.85),
        _result(eval_libero, 2, 0.95),
    ]
    pick, note = eval_libero.pick_dsrl_target(results)
    assert pick.task_id == 1  # weakest non-zero task
    assert "no task inside" in note
    none_pick, none_note = eval_libero.pick_dsrl_target([_result(eval_libero, 0, 0.0)])
    assert none_pick is None and "no viable" in none_note


def test_markdown_report_has_table_and_pick():
    eval_libero = _load_eval_libero()
    results = [_result(eval_libero, 0, 0.9), _result(eval_libero, 1, 0.4)]
    text = eval_libero.format_markdown(results, "/ckpt", "libero_10", "20 eps")
    assert "| Task | Description | Success rate |" in text
    assert "| 0 | description 0 | 18/20 (90%) |" in text
    assert "| 1 | description 1 | 8/20 (40%) |" in text
    assert "DSRL target: task 1" in text


# ---------------------------------------------------------------------------
# 3. The documented LIBERO override set composes on the canonical config
# ---------------------------------------------------------------------------

#: The override set the LIBERO launcher assembles (task variants are CLI
#: overrides on the one canonical config, never new config files).
LIBERO_OVERRIDES = [
    "base_model=gr00t",
    "simulator=libero_10",
    "env.env_name=libero/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "env.max_episode_steps=720",
    "env.image_keys=[video.image]",
    "base_model.embodiment_tag=libero_sim",
    "base_model.task_instruction='put both moka pots on the stove'",
    "dsrl.noise_dim=7",
    "dsrl.action_exec_len=8",
    "training.chunk_size=40",
]


@pytest.fixture()
def libero_cfg():
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        yield compose(config_name="train", overrides=LIBERO_OVERRIDES)


def test_libero_override_set_composes(libero_cfg):
    assert libero_cfg.simulator.provider == "libero"
    assert libero_cfg.env.env_name.startswith("libero/libero_10/")
    assert int(libero_cfg.env.max_episode_steps) == 720
    assert int(libero_cfg.dsrl.noise_dim) == 7
    assert int(libero_cfg.dsrl.action_exec_len) == 8
    assert int(libero_cfg.training.chunk_size) == 40
    assert list(libero_cfg.env.image_keys) == ["video.image"]


def test_base_model_group_receives_embodiment_and_instruction(libero_cfg):
    """The base model group carries embodiment and instruction (config-driven
    embodiment: no code fork selects the LIBERO checkpoint layout)."""
    assert str(libero_cfg.base_model.embodiment_tag) == "libero_sim"
    assert (
        str(libero_cfg.base_model.task_instruction) == "put both moka pots on the stove"
    )
    # Baked-statistics path: no dataset directories are configured.
    assert libero_cfg.base_model.dataset_dir is None
    assert libero_cfg.base_model.multitask_dirs is None


def test_simulator_config_pins_gr00t_protocol(libero_cfg):
    protocol = libero_cfg.simulator.gr00t_eval
    assert int(protocol.max_episode_steps) == 720
    assert int(protocol.image_size) == 256
    assert int(protocol.action_horizon) == 16
    assert int(protocol.n_action_steps) == 8


def test_arena_composition_still_resolves_new_base_model_fields():
    """The arena config declares the new dsrl keys, so the base model group's
    interpolations resolve to the wrapper defaults (no LIBERO leakage)."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="train")
    assert OmegaConf.select(cfg, "base_model.embodiment_tag") is None
    assert OmegaConf.select(cfg, "base_model.task_instruction") is None


# ---------------------------------------------------------------------------
# 4. Launcher dry-run plan
# ---------------------------------------------------------------------------


def _plan_value(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}=(.*)$", text, re.MULTILINE)
    assert match, f"dry-run plan is missing {key}:\n{text}"
    return match.group(1).strip()


def test_libero_launcher_dryrun_plan(launcher_sandbox):
    task = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
    result = run_launcher_dryrun(LIBERO_LAUNCHER, [task], launcher_sandbox)
    assert result.returncode == 0, (
        f"dry run must exit 0; rc={result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    text = result.stdout
    assert _plan_value(text, "launch.task") == task
    # The launcher names the simulator config group (default suite), as the g1_arena launcher does.
    assert "launch.override=simulator=libero_10" in text
    assert (
        "max_episode_steps: 720" in text
    )  # gr00t_eval block, as the library's dry run resolves it
    assert f"launch.override=env.env_name=libero/libero_10/{task}" in text
    assert "launch.override=base_model.embodiment_tag=libero_sim" in text
    assert "launch.override=dsrl.noise_dim=7" in text
    assert "launch.override=dsrl.action_exec_len=8" in text
    assert "launch.override=training.chunk_size=40" in text
    assert "launch.override=eval.eval_freq=5000" in text
    assert "launch.override=eval.eval_num_episodes=50" in text
    # Secrets never appear in the plan.
    assert "HF_TOKEN=" not in text.replace("secrets.HF_TOKEN=set", "").replace(
        "secrets.HF_TOKEN=unset", ""
    )


def test_libero_launcher_dryrun_respects_maxsteps_override(launcher_sandbox):
    result = run_launcher_dryrun(
        LIBERO_LAUNCHER,
        ["KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"],
        launcher_sandbox,
        extra_env={"TRAIN_MAXSTEPS": "512"},
    )
    assert result.returncode == 0
    assert "launch.override=env.max_episode_steps=512" in result.stdout
    assert "max_episode_steps: 512" in result.stdout


def test_libero_ppo_launcher_pins_base_policy_flow_prior(launcher_sandbox):
    result = run_launcher_dryrun(
        LIBERO_LAUNCHER,
        ["KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"],
        launcher_sandbox,
        extra_env={"TRAINER": "ppo"},
    )
    assert result.returncode == 0, result.stderr
    text = result.stdout
    # dsrl_ppo.yaml carries the recipe: one decision is one whole 16-step chunk of the checkpoint.
    assert "launch.override=algorithm=dsrl_ppo" in text
    assert "action_horizon: 16" in text and "chunk_size: 16" in text
    assert "official_policy_init: true" in text and "use_tanh_output: false" in text
