# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The typed records: annotations are the schema, records validate themselves, specs compare.

A simulator author declares an observation type with annotated fields and gets
its spec, validation and the launch-time comparison without other code; a
rollout worker's legacy transition converts to the typed record and back
without losing a field; the launch plan refuses a policy whose action is wider
than the simulator accepts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from isaac_rl_posttraining.configs import register_configs
from isaac_rl_posttraining.data import (
    ActionSpec,
    Array,
    ArrayMap,
    ArraySpec,
    DynamicObservation,
    EnvAction,
    Episode,
    EpisodeMetadata,
    LearnerAction,
    Observation,
    ObservationSpec,
    PolicyOutputs,
    SpecMismatch,
    Text,
    Transition,
    TransitionMetadata,
    compare_action_specs,
    compare_observation_specs,
    dim,
)
from isaac_rl_posttraining.envs.simulator_entry import EnvironmentSpec
from isaac_rl_posttraining.policies.base import MockBaseModelPolicy
from isaac_rl_posttraining.runners.build import train_plan
from isaac_rl_posttraining.runners.spec_checks import (
    check_chunk_size_at_load,
    check_policy_inputs_at_load,
)
from simulators.g1_arena.env import ArenaEnv, arena_action_spec, arena_observation_spec
from simulators.g1_arena.runtime import G1Arena

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "isaac_rl_posttraining" / "configs"


def _compose(*overrides: str):
    register_configs()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="train", overrides=list(overrides))


# ---------------------------------------------------------------------------
# A simulator author's observation type: annotations only, no other code.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraAndState(Observation):
    ego_view_image: Array[np.uint8, dim.H, dim.W, 3]
    state: Array[np.float32, dim.state_dim]
    instruction: Text | None = None


@dataclass(frozen=True)
class WithFeatures(Observation):
    raw: CameraAndState
    dino_embedding: Array[np.float32, dim.feat_dim]
    activations: ArrayMap | None = None


@dataclass(frozen=True)
class TwoStates(Observation):
    state: Array[np.float32, dim.state_dim]
    previous_state: Array[np.float32, dim.state_dim]


def _camera_and_state(
    state_dim: int = 31, size: tuple[int, int] = (4, 5)
) -> CameraAndState:
    return CameraAndState(
        ego_view_image=np.zeros((*size, 3), np.uint8),
        state=np.zeros(state_dim, np.float32),
    )


def test_an_undecorated_observation_type_fails_instead_of_declaring_nothing() -> None:
    class Undecorated(Observation):
        image: Array[np.uint8, dim.H, dim.W, 3]
        state: Array[np.float32, dim.state_dim]

    with pytest.raises(
        TypeError,
        match=r"Undecorated annotates \['image', 'state'\].*@dataclass\(frozen=True\)",
    ):
        Undecorated.declared_spec()


def test_declaring_an_observation_type_takes_only_annotations() -> None:
    declared = CameraAndState.declared_spec()
    assert declared.arrays == {
        "ego_view_image": ArraySpec(("H", "W", 3), "uint8"),
        "state": ArraySpec(("state_dim",), "float32"),
    }
    assert declared.text == frozenset()
    assert CameraAndState.declared_spec(include_optional=True).text == frozenset(
        {"instruction"}
    )
    assert CameraAndState.declared_spec({"state_dim": 31}).arrays["state"] == ArraySpec(
        (31,), "float32"
    )

    observation = _camera_and_state()
    assert observation.bindings() == {"H": 4, "W": 5, "state_dim": 31}
    assert observation.spec().arrays["state"] == ArraySpec((31,), "float32")
    assert observation.spec().images == ("ego_view_image",)
    assert CameraAndState.from_mapping(observation.to_mapping()) == observation


def test_observation_types_compose_and_flatten_their_components() -> None:
    nested = WithFeatures(
        raw=_camera_and_state(),
        dino_embedding=np.ones(8, np.float32),
        activations={"backbone": np.zeros((2, 3), np.float32)},
    )
    assert sorted(nested.arrays()) == [
        "activations/backbone",
        "dino_embedding",
        "raw/ego_view_image",
        "raw/state",
    ]
    assert nested.bindings() == {"H": 4, "W": 5, "state_dim": 31, "feat_dim": 8}
    assert WithFeatures.declared_spec().arrays["raw/state"] == ArraySpec(
        ("state_dim",), "float32"
    )
    assert "activations" not in WithFeatures.declared_spec(include_optional=True).arrays


def test_observation_rejects_wrong_dtype_rank_and_inconsistent_named_dimensions() -> (
    None
):
    with pytest.raises(SpecMismatch, match="state: dtype float64, expected float32"):
        CameraAndState(
            ego_view_image=np.zeros((4, 5, 3), np.uint8), state=np.zeros(31, np.float64)
        )
    with pytest.raises(
        SpecMismatch, match=r"ego_view_image: shape \[4, 5\], expected uint8\[H, W, 3\]"
    ):
        CameraAndState(
            ego_view_image=np.zeros((4, 5), np.uint8), state=np.zeros(31, np.float32)
        )
    with pytest.raises(SpecMismatch, match="previous_state"):
        TwoStates(
            state=np.zeros(31, np.float32), previous_state=np.zeros(43, np.float32)
        )
    with pytest.raises(ValueError, match="CameraAndState.state is required"):
        CameraAndState(ego_view_image=np.zeros((4, 5, 3), np.uint8), state=None)
    with pytest.raises(ValueError, match="no components"):
        CameraAndState.from_mapping(
            {
                "ego_view_image": np.zeros((4, 5, 3), np.uint8),
                "state": np.zeros(31, np.float32),
                "extra": 1,
            }
        )


def test_named_dimensions_bind_from_config_or_from_the_first_instance() -> None:
    observation = _camera_and_state()
    from_config = CameraAndState.declared_spec({"state_dim": 31, "H": 4, "W": 5})
    from_instance = CameraAndState.declared_spec(observation.bindings())
    assert from_config == from_instance
    # A consumer that binds the width to the policy's layout catches a wider state.
    comparison = compare_observation_specs(
        _camera_and_state(43).spec(), CameraAndState.declared_spec({"state_dim": 31})
    )
    assert comparison.mismatches == (
        "observation 'state': shape float32[43] vs float32[31]",
    )
    # Left unbound, the width stays unverified rather than silently accepted.
    unbound = compare_observation_specs(
        observation.spec(), CameraAndState.declared_spec()
    )
    assert unbound.ok and any(
        "state_dim unbound" in item for item in unbound.unverified
    )


def test_array_spec_compare_handles_fixed_named_and_free_dimensions() -> None:
    fixed = ArraySpec((480, 640, 3), "uint8")
    # A size one side leaves free is accepted, but the summary names the axis nobody compared.
    assert fixed.compare(ArraySpec((None, None, 3), "uint8")) == (
        [],
        [
            "axis 0 free on one side (uint8[480, 640, 3] vs uint8[?, ?, 3])",
            "axis 1 free on one side (uint8[480, 640, 3] vs uint8[?, ?, 3])",
        ],
    )
    assert ArraySpec((None, 3), "uint8").compare(ArraySpec((None, 3), "uint8")) == (
        [],
        [],
    )
    assert fixed.compare(
        ArraySpec(("H", "W", 3), "uint8"), bindings={"H": 480, "W": 640}
    ) == ([], [])
    mismatches, _ = fixed.compare(
        ArraySpec(("H", "W", 3), "uint8"), bindings={"H": 224, "W": 224}
    )
    assert mismatches == ["shape uint8[480, 640, 3] vs uint8[H, W, 3]"]
    assert ArraySpec((31,), "float32").compare(ArraySpec((31,), "float64")) == (
        ["dtype float32 vs float64"],
        [],
    )
    assert ArraySpec((31,), "float32").mismatch(np.zeros(31, np.float32)) is None
    assert (
        ArraySpec(("d",), "float32").mismatch(
            np.zeros(4, np.float32), bindings={"d": 3}
        )
        is not None
    )


def test_observation_spec_comparison_names_missing_and_mismatched_components() -> None:
    emitted = ObservationSpec(
        {
            "state": ArraySpec((31,), "float32"),
            "ego_view_image": ArraySpec((480, 640, 3), "uint8"),
        }
    )
    consumed = ObservationSpec(
        {
            "state": ArraySpec((43,), "float32"),
            "wrist_image": ArraySpec((None, None, 3), "uint8"),
        },
        frozenset({"instruction"}),
    )
    comparison = compare_observation_specs(emitted, consumed)
    assert (
        "observation 'state': shape float32[31] vs float32[43]" in comparison.mismatches
    )
    assert any(
        item.startswith("observation 'wrist_image' is read but not emitted")
        for item in comparison.mismatches
    )
    assert "text 'instruction' is read but not emitted" in comparison.mismatches
    with pytest.raises(SpecMismatch, match="state"):
        comparison.raise_on_mismatch("hand-off")


def test_action_spec_comparison_catches_a_wider_policy_action() -> None:
    assert compare_action_specs(ActionSpec(width=35), ActionSpec(width=35)).ok
    wide = compare_action_specs(ActionSpec(width=43), ActionSpec(width=35))
    assert wide.mismatches == ("action: emitted float32[43] vs accepted float32[35]",)
    unbound = compare_action_specs(ActionSpec(width="act_dim"), ActionSpec(width=35))
    assert unbound.ok and unbound.unverified
    assert ActionSpec(width=35).mismatch(np.zeros((8, 43), np.float32)) is not None


def test_dynamic_observation_keeps_runtime_named_components_and_strings() -> None:
    observation = Observation.from_mapping(
        {
            "state": np.zeros(4, np.float32),
            "dino_embedding": np.ones(8, np.float32),
            "instruction": "pick",
        }
    )
    assert isinstance(observation, DynamicObservation)
    assert (
        observation.spec().describe()
        == "{state: float32[4], dino_embedding: float32[8], instruction: text}"
    )
    assert observation.text() == {"instruction": "pick"}
    assert observation.bindings() == {}


def test_policy_outputs_flatten_features_and_share_named_dimensions() -> None:
    outputs = PolicyOutputs(
        log_prob=np.float32(-1.5),
        value=np.float32(0.25),
        action_mean=np.zeros((8, 35), np.float32),
        action_std=np.ones((8, 35), np.float32),
        features={"backbone": np.zeros((3, 2), np.float32)},
    )
    assert sorted(outputs.arrays()) == [
        "action_mean",
        "action_std",
        "features/backbone",
        "log_prob",
        "value",
    ]
    assert outputs.bindings() == {"chunk_steps": 8, "act_dim": 35}
    with pytest.raises(SpecMismatch, match="action_std"):
        PolicyOutputs(
            action_mean=np.zeros((8, 35), np.float32),
            action_std=np.ones((8, 34), np.float32),
        )
    assert PolicyOutputs().arrays() == {}


def test_learner_and_environment_actions_validate_their_geometry() -> None:
    latent = LearnerAction("latent", np.zeros(35, np.float32))
    assert latent.value.shape == (35,)
    with pytest.raises(ValueError, match="unknown action kind"):
        LearnerAction("noise", np.zeros(35, np.float32))
    with pytest.raises(TypeError, match="floating point"):
        LearnerAction("chunk", np.zeros((8, 35), np.int64))
    executed = EnvAction(np.zeros((3, 35), np.float32), execution_length=8)
    assert (executed.steps, executed.width) == (3, 35)
    assert executed.spec() == ActionSpec(width=35)
    with pytest.raises(ValueError, match="execution_length"):
        EnvAction(np.zeros((9, 35), np.float32), execution_length=8)
    with pytest.raises(ValueError, match=r"\[steps, act_dim\]"):
        EnvAction(np.zeros(35, np.float32), execution_length=1)


def _transition(
    *,
    steps: int = 3,
    ended: bool = True,
    truncated: bool = False,
    success: bool = True,
    step_in_episode: int = 0,
    episode_id: int = 0,
) -> Transition:
    observation = Observation.from_mapping(
        {"state": np.zeros(4, np.float32), "dino_embedding": np.ones(8, np.float32)}
    )
    terminated_flags = np.zeros(steps, bool)
    truncated_flags = np.zeros(steps, bool)
    success_flags = np.zeros(steps, bool)
    if ended:
        truncated_flags[-1] = truncated
        terminated_flags[-1] = not truncated
        success_flags[-1] = success
    return Transition(
        episode_id=episode_id,
        observation=observation,
        next_observation=observation,
        learner_action=LearnerAction("latent", np.full(6, 0.5, np.float32)),
        env_action=EnvAction(np.zeros((steps, 5), np.float32), execution_length=8),
        rewards={"env": np.arange(steps, dtype=np.float32)},
        terminated=terminated_flags,
        truncated=truncated_flags,
        success=success_flags,
        behavior_source="policy",
        metadata=TransitionMetadata(
            step_in_episode=step_in_episode, policy_version=3, reset_seed=7
        ),
        policy_outputs=PolicyOutputs(
            log_prob=np.float32(-0.5),
            features={"backbone": np.zeros((2, 2), np.float32)},
        ),
    )


def test_transition_checks_step_consistency_and_reports_the_outcome() -> None:
    transition = _transition()
    assert (transition.steps, transition.ended, transition.outcome) == (
        3,
        True,
        "success",
    )
    assert transition.decision_reward() == pytest.approx(3.0)
    assert transition.decision_reward(aggregation="last") == pytest.approx(2.0)
    assert _transition(ended=False).outcome == "ongoing"
    assert _transition(success=False).outcome == "failure"
    assert _transition(truncated=True, success=False).outcome == "timeout"
    with pytest.raises(ValueError, match="executed 2 steps but 3 rewards"):
        Transition(
            **{
                **_fields_of(_transition()),
                "env_action": EnvAction(
                    np.zeros((2, 5), np.float32), execution_length=8
                ),
            }
        )
    with pytest.raises(ValueError, match="only the last executed step"):
        Transition(
            **{**_fields_of(_transition()), "terminated": np.array([True, False, True])}
        )
    with pytest.raises(ValueError, match="'env' channel"):
        Transition(
            **{
                **_fields_of(_transition()),
                "rewards": {"sparse": np.zeros(3, np.float32)},
            }
        )
    with pytest.raises(KeyError, match="reward channel"):
        transition.reward("reward_model/v1")


def _fields_of(transition: Transition) -> dict:
    return {
        "episode_id": transition.episode_id,
        "observation": transition.observation,
        "next_observation": transition.next_observation,
        "learner_action": transition.learner_action,
        "env_action": transition.env_action,
        "rewards": dict(transition.rewards),
        "terminated": transition.terminated,
        "truncated": transition.truncated,
        "success": transition.success,
        "behavior_source": transition.behavior_source,
        "metadata": transition.metadata,
        "policy_outputs": transition.policy_outputs,
    }


def test_episode_requires_contiguous_decisions_and_one_ending() -> None:
    first = _transition(ended=False, step_in_episode=0)
    second = _transition(ended=True, step_in_episode=3)
    episode = Episode.from_transitions(
        [first, second], EpisodeMetadata(task="LMDrillLift", seed=1)
    )
    assert (episode.length, episode.complete, episode.outcome, episode.return_()) == (
        6,
        True,
        "success",
        6.0,
    )
    assert Episode.from_transitions([first]).outcome == "ongoing"
    with pytest.raises(ValueError, match="starts at step 5, expected 3"):
        Episode.from_transitions([first, _transition(step_in_episode=5)])
    with pytest.raises(ValueError, match="ends the episode but 1 follow"):
        Episode.from_transitions(
            [_transition(step_in_episode=0), _transition(step_in_episode=3)]
        )
    with pytest.raises(ValueError, match="belongs to episode 9"):
        Episode(id=0, transitions=(_transition(episode_id=9),))


def _legacy_fields(*, sparse: bool = False) -> dict:
    """The fields a gym rollout worker pushes for one DSRL decision of three executed steps."""
    sequence = (
        np.array([0.0, 0.0, 1.0], np.float32)
        if sparse
        else np.array([0.5, 0.25, 1.0], np.float32)
    )
    return {
        "obs": {
            "state": np.arange(4, dtype=np.float32),
            "dino_embedding": np.ones(8, np.float32),
        },
        "action": np.full(6, 0.5, np.float32),
        "reward": float(sequence[-1]) if sparse else float(sequence.sum()),
        "next_obs": {
            "state": np.arange(4, dtype=np.float32) + 1,
            "dino_embedding": np.zeros(8, np.float32),
        },
        "done": True,
        "truncated": False,
        "episode_id": 4,
        "step_in_episode": 24,
        "max_steps_in_episode": 350,
        "timestamp": 1_700_000_000.5,
        "language_instruction": "Walk to the table and pick up the power drill",
        "info": {
            "success": True,
            "is_success": True,
            "action_source": "policy",
            "policy_version": 3,
            "reset_seed": 11,
            "reset_seed_scope": "environment",
            "num_steps": 3,
            "behavior_log_prob": np.float32(-1.25),
        },
        "reward_sequence": sequence,
        "bootstrap_steps": 3,
        "behavior_source": "policy",
        "intervention_phase": "none",
        "policy_tick_mask": None,
        "expert_tick_mask": None,
        "pre_intervention_tick_mask": None,
        "post_intervention_tick_mask": None,
        "cached_features": {"backbone": np.zeros((2, 3), np.float32)},
    }


def _assert_same_fields(actual: dict, expected: dict) -> None:
    assert set(actual) == set(expected)
    for key, value in expected.items():
        if isinstance(value, dict):
            assert set(actual[key]) == set(value), key
            for name, entry in value.items():
                np.testing.assert_array_equal(
                    actual[key][name], entry, err_msg=f"{key}.{name}"
                )
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual[key], value, err_msg=key)
        else:
            assert actual[key] == value, key


def test_legacy_transition_round_trips_through_the_typed_record() -> None:
    fields = _legacy_fields()
    transition = Transition.from_legacy_fields(fields, kind="latent")
    assert transition.steps == 3 and transition.outcome == "success"
    assert transition.learner_action.kind == "latent" and transition.env_action is None
    assert (
        transition.metadata.policy_version == 3 and transition.metadata.reset_seed == 11
    )
    assert transition.observation.text() == {
        "instruction": fields["language_instruction"]
    }
    assert float(transition.policy_outputs.log_prob) == pytest.approx(-1.25)
    np.testing.assert_array_equal(
        transition.policy_outputs.features["backbone"],
        fields["cached_features"]["backbone"],
    )
    _assert_same_fields(transition.to_legacy_fields(), fields)


def test_sparse_outcome_rewards_round_trip_with_last_step_aggregation() -> None:
    fields = _legacy_fields(sparse=True)
    transition = Transition.from_legacy_fields(fields, kind="latent")
    np.testing.assert_array_equal(transition.reward(), fields["reward_sequence"])
    exported = transition.to_legacy_fields(aggregation="last")
    assert exported["reward"] == pytest.approx(1.0)
    _assert_same_fields(exported, fields)


def test_legacy_conversion_keeps_the_executed_chunk_when_the_worker_supplies_it() -> (
    None
):
    fields = _legacy_fields()
    executed = EnvAction(np.full((3, 35), 0.1, np.float32), execution_length=8)
    transition = Transition.from_legacy_fields(
        fields, kind="latent", env_action=executed
    )
    assert transition.env_action == executed
    with pytest.raises(ValueError, match="executed 2 steps"):
        Transition.from_legacy_fields(
            fields,
            kind="latent",
            env_action=EnvAction(np.zeros((2, 35), np.float32), execution_length=8),
        )
    chunk_fields = {**fields, "action": np.full((3, 35), 0.1, np.float32)}
    as_chunk = Transition.from_legacy_fields(chunk_fields, kind="chunk")
    assert as_chunk.env_action is not None and as_chunk.env_action.steps == 3
    ended_without_flag = {**fields, "info": {"action_source": "policy"}}
    with pytest.raises(
        ValueError, match="ended without an 'is_success' or 'success' flag"
    ):
        Transition.from_legacy_fields(ended_without_flag, kind="latent")


# ---------------------------------------------------------------------------
# The simulator side declares what it emits; the launch plan checks it.
# ---------------------------------------------------------------------------


def test_arena_client_declares_the_spaces_it_builds() -> None:
    for resolution in (0, 224):
        client = ArenaEnv(image_resolution=resolution)
        assert ObservationSpec.from_space(
            client.observation_space
        ) == arena_observation_spec(image_resolution=resolution)
        assert ArraySpec.from_space(client.action_space) == ArraySpec(
            (arena_action_spec(35).width,), "float32"
        )
    spec = EnvironmentSpec(
        env_name="g1_arena/LMDrillLift",
        num_envs=1,
        max_episode_steps=10,
        seed=0,
        action_dim=35,
        image_resolution=0,
    )
    entry = G1Arena()
    assert entry.observation_spec(spec).arrays["state"] == ArraySpec((31,), "float32")
    assert entry.action_spec(spec) == ActionSpec(width=35)


def test_train_plan_reports_the_data_specs_and_refuses_a_wider_action() -> None:
    plan = train_plan(_compose("algorithm=dsrl_sac", "base_model=mock"))
    assert (
        "simulator g1_arena emits {ego_view_image: uint8[480, 640, 3], state: float32[31]} and accepts float32[35]"
        in plan.data_specs
    )
    assert (
        "checked: steering chunk, executed steps per decision, steering latent width, observation 'state', "
        "observation 'ego_view_image', camera keys, action width" in plan.data_specs
    )
    assert plan.image_keys == "ego_view_image from g1_arena.steering_key_mapping"
    assert (
        "the policy's 'state' width binds at load: base model mock reads a state of any width"
        in plan.data_specs
    )
    assert "plan.data_specs=" in "\n".join(plan.lines())
    with pytest.raises(
        SpecMismatch,
        match=r"action: emitted float32\[43\] x 40 steps vs accepted float32\[35\]",
    ):
        train_plan(
            _compose(
                "algorithm=dsrl_sac", "base_model=mock", "base_model.action_dim=43"
            )
        )
    with pytest.raises(
        SpecMismatch,
        match="dsrl.noise_dim=43 but base model mock config steers exactly 35",
    ):
        train_plan(
            _compose("algorithm=dsrl_sac", "base_model=mock", "dsrl.noise_dim=43")
        )


def test_train_plan_refuses_an_image_key_the_simulator_does_not_emit() -> None:
    plan = train_plan(
        _compose(
            "algorithm=dsrl_sac", "base_model=mock", "env.image_keys=[ego_view_image]"
        )
    )
    assert plan.image_keys == "ego_view_image from env.image_keys"
    with pytest.raises(
        SpecMismatch,
        match=r"image keys \['nope'\] from env.image_keys are not camera entries of the observation "
        r"\(cameras: \['ego_view_image'\]\); set env.image_keys",
    ):
        train_plan(
            _compose("algorithm=dsrl_sac", "base_model=mock", "env.image_keys=[nope]")
        )


def test_train_plan_refuses_a_steering_chunk_longer_than_the_base_model_horizon() -> (
    None
):
    fits = train_plan(
        _compose(
            "algorithm=dsrl_sac",
            "base_model=mock",
            "base_model.action_horizon=8",
            "training.chunk_size=8",
        )
    )
    assert "steering chunk" in fits.data_specs
    with pytest.raises(
        SpecMismatch,
        match=r"training.chunk_size=40 exceeds base_model.action_horizon=8 for base model mock; "
        r"set training.chunk_size<=8 or a base model with a longer horizon",
    ):
        train_plan(
            _compose(
                "algorithm=dsrl_sac", "base_model=mock", "base_model.action_horizon=8"
            )
        )
    # A base model that reads its horizon from the checkpoint settles the chunk at load time.
    deferred = train_plan(
        _compose("algorithm=dsrl_sac", "base_model=gr00t", "base_model.ckpt=/ckpt/x")
    )
    assert (
        "the policy's steering window, decoded horizon, flow-head width, 'state' width, action width bind at load: "
        "base_model.ckpt=/ckpt/x is not a local directory" in deferred.data_specs
    )


def test_load_time_check_refuses_a_steering_chunk_longer_than_the_latent_window() -> (
    None
):
    cfg = OmegaConf.create(
        {"base_model": {"name": "mock"}, "training": {"chunk_size": 40}}
    )
    check_chunk_size_at_load(cfg, MockBaseModelPolicy(action_horizon=40))
    check_chunk_size_at_load(
        OmegaConf.create(
            {"base_model": {"name": "mock"}, "training": {"chunk_size": None}}
        ),
        MockBaseModelPolicy(action_horizon=8),
    )
    with pytest.raises(
        SpecMismatch,
        match=r"training.chunk_size=40 exceeds the loaded policy's latent_horizon=8 for base model mock",
    ):
        check_chunk_size_at_load(cfg, MockBaseModelPolicy(action_horizon=8))


class _LayoutBaseModel(MockBaseModelPolicy):
    def __init__(self, width: int | None) -> None:
        super().__init__()
        self._width = width

    def input_spec(self) -> ObservationSpec | None:
        if self._width is None:
            return None
        return ObservationSpec({"state": ArraySpec((self._width,), "float32")})


def test_load_time_check_compares_the_policy_layout_with_the_environment_state() -> (
    None
):
    class StubEnv(gym.Env):
        # The steering contract names the entry the policy's "state" role reads.
        steering_key_mapping: ClassVar[dict[str, str | None | list[str]]] = {
            "state": "proprio",
            "language": None,
            "image": ["ego_view_image"],
        }
        observation_space = gym.spaces.Dict(
            {
                "proprio": gym.spaces.Box(-np.inf, np.inf, (43,), np.float32),
                "ego_view_image": gym.spaces.Box(0, 255, (4, 4, 3), np.uint8),
            }
        )
        action_space = gym.spaces.Box(-1.0, 1.0, (35,), np.float32)

    for env in (StubEnv(), gym.vector.SyncVectorEnv([StubEnv])):
        assert "proprio: float32[43]" in check_policy_inputs_at_load(
            _LayoutBaseModel(43), env
        )
        assert "unknown" in check_policy_inputs_at_load(_LayoutBaseModel(None), env)
        with pytest.raises(
            SpecMismatch, match=r"'proprio': shape float32\[43\] vs float32\[31\]"
        ):
            check_policy_inputs_at_load(_LayoutBaseModel(31), env)
