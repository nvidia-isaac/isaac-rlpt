# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare, before anything loads, what the simulator emits with what the policy reads.

The hand-offs that have silently corrupted runs are the state vector the
frozen policy slices, the action width the simulator executes, the cameras
the learner embeds and the steering chunk the base model decodes. Both sides
state their side before any weights load: a simulator entry declares the
observation it emits and the per-step action it accepts, and the policy side
comes from the config and the checkpoint's metadata
(:func:`~isaac_rl_posttraining.policies.geometry.policy_geometry`: state
width, decoded action width and steps, steering window and latent width),
read through the entry's steering key mapping. The launch plan runs the
comparison and prints it, with every size that stays unbound until load; a
definite disagreement fails the dry run and the launch.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import gymnasium as gym
from omegaconf import DictConfig, OmegaConf

from isaac_rl_posttraining.algorithms.registry import algorithm_spec
from isaac_rl_posttraining.data.spec import (
    ActionSpec,
    ArraySpec,
    ObservationSpec,
    SpecComparison,
    compare_action_specs,
    compare_observation_specs,
)
from isaac_rl_posttraining.envs.simulator_entry import EnvironmentSpec, SimulatorEntry
from isaac_rl_posttraining.policies.geometry import PolicyGeometry, policy_geometry
from isaac_rl_posttraining.runners.simulator_entries import (
    environment_spec,
    selected_entry,
)
from isaac_rl_posttraining.runtimes.steering_rollouts import resolve_image_keys

if TYPE_CHECKING:
    from isaac_rl_posttraining.policies.base import BaseModelPolicy

__all__ = [
    "SpecReport",
    "check_chunk_size_at_load",
    "check_data_specs",
    "check_policy_inputs_at_load",
    "plan_geometry",
    "steering_input_spec",
]

@dataclass(frozen=True)
class SpecReport:
    """What the plan compared, what disagreed, and what only a loaded checkpoint can settle."""

    simulator: str
    emitted: ObservationSpec | None
    accepted: ActionSpec | None
    consumed: ObservationSpec | None
    comparison: SpecComparison

    def summary(self) -> str:
        """One line for the launch plan."""
        if self.emitted is None:
            simulator = f"simulator {self.simulator} declares no observation spec"
        else:
            simulator = f"simulator {self.simulator} emits {self.emitted.describe()}"
        if self.accepted is not None:
            simulator += f" and accepts {self.accepted}"
        policy = "policy reads: unknown before load" if self.consumed is None else f"policy reads {self.consumed.describe()}"
        return " | ".join([simulator, policy, self.comparison.summary()])


_CHUNK_DEFERRED = "steering chunk length binds when the checkpoint loads"
_HORIZON_DEFERRED = "executed steps per decision: the decoded horizon binds at load"
_LATENT_DEFERRED = "steering latent width: the flow head's width binds at load"


def steering_input_spec(
    entry: SimulatorEntry,
    spec: EnvironmentSpec,
    geometry: PolicyGeometry | None = None,
    emitted: ObservationSpec | None = None,
) -> ObservationSpec:
    """The observation entries the frozen policy and the steering learner read.

    The entry's key mapping names them. The state width is the policy's
    (``geometry.state_width`` when its metadata states it, else a named
    dimension bound at load); a policy that resizes its camera frames reads
    any size, so its cameras take the emitted size.
    """
    mapping = dict(entry.steering_key_mapping(spec))
    state_width = None if geometry is None else geometry.state_width
    arrays = {str(mapping["state"]): ArraySpec(("state_dim" if state_width is None else state_width,), "float32")}
    resizes = geometry is not None and geometry.resizes_images
    for image_key in mapping["image"]:
        key = str(image_key)
        if resizes and emitted is not None and key in emitted.arrays:
            arrays[key] = emitted.arrays[key]
        else:
            arrays[key] = ArraySpec((None, None, 3), "uint8")
    if mapping.get("language") is not None:
        arrays[str(mapping["language"])] = ArraySpec(("language_dim",), "float32")
    return ObservationSpec(arrays)


def _emitted_action(cfg: DictConfig, geometry: PolicyGeometry) -> tuple[ActionSpec, SpecComparison]:
    """The action the base model emits per step: ``base_model.action_dim`` or the checkpoint's width, else bound at load."""
    configured = OmegaConf.select(cfg, "base_model.action_dim")
    width: int | str = "act_dim"
    comparison = SpecComparison()
    if configured is not None:
        width = int(configured)
        if geometry.action_width is not None and geometry.action_width != width:
            comparison = SpecComparison(
                ("base model action width",),
                (
                    (f"base_model.action_dim={width} but {geometry.source} decodes {geometry.action_width}-wide "
                    f"actions; set base_model.action_dim={geometry.action_width} or null"),
                ),
            )
    elif geometry.action_width is not None:
        width = geometry.action_width
    return ActionSpec(width=width, horizon=geometry.decoded_horizon), comparison


def _latent_comparison(cfg: DictConfig, geometry: PolicyGeometry) -> SpecComparison:
    """A steering latent cannot be wider than the base model's flow head.

    The simulator's action width does not bound it: pi0 and DreamZero steer
    their padded internal action space (32 wide on LIBERO's 7-wide action).
    """
    noise_dim = OmegaConf.select(cfg, "dsrl.noise_dim")
    if noise_dim is None:
        return SpecComparison()
    if geometry.latent_width is None:
        return SpecComparison(unverified=(_LATENT_DEFERRED,))
    if geometry.exact_latent_width and int(noise_dim) != geometry.latent_width:
        return SpecComparison(
            ("steering latent width",),
            (
                (f"dsrl.noise_dim={int(noise_dim)} but {geometry.source} steers exactly {geometry.latent_width} "
                f"dimensions per step; set dsrl.noise_dim={geometry.latent_width}"),
            ),
        )
    if int(noise_dim) > geometry.latent_width:
        return SpecComparison(
            ("steering latent width",),
            (
                (f"dsrl.noise_dim={int(noise_dim)} exceeds the {geometry.latent_width}-wide flow head of "
                f"{geometry.source}; set dsrl.noise_dim<={geometry.latent_width}"),
            ),
        )
    return SpecComparison(("steering latent width",))


def _camera_comparison(cfg: DictConfig, entry: SimulatorEntry, spec: EnvironmentSpec, emitted: ObservationSpec) -> SpecComparison:
    """The cameras the learner's image encoder embeds must be camera entries the simulator emits."""
    mismatch = resolve_image_keys(cfg, entry, spec).mismatch(emitted.images)
    return SpecComparison(("camera keys",), () if mismatch is None else (mismatch,))


def _chunk_comparison(cfg: DictConfig, *, horizon: int | None, horizon_name: str) -> SpecComparison:
    """``training.chunk_size`` steering latents feed one decode; the base model accepts at most its horizon."""
    chunk_size = OmegaConf.select(cfg, "training.chunk_size")
    if chunk_size is None:
        return SpecComparison()
    if horizon is None:
        return SpecComparison(unverified=(_CHUNK_DEFERRED,))
    if int(chunk_size) > int(horizon):
        return SpecComparison(
            ("steering chunk",),
            (
                (f"training.chunk_size={int(chunk_size)} exceeds {horizon_name}={int(horizon)} for base model "
                f"{cfg.base_model.name}; set training.chunk_size<={int(horizon)} or a base model with a longer horizon"),
            ),
        )
    return SpecComparison(("steering chunk",))


def _executed_steps_comparison(cfg: DictConfig, geometry: PolicyGeometry) -> SpecComparison:
    """The rollout workers execute ``dsrl.action_exec_len`` steps of each decoded chunk; a decode emits only so many."""
    executed = OmegaConf.select(cfg, "dsrl.action_exec_len")
    if executed is None:
        return SpecComparison()
    if geometry.decoded_horizon is None:
        return SpecComparison(unverified=(_HORIZON_DEFERRED,))
    if int(executed) > geometry.decoded_horizon:
        return SpecComparison(
            ("executed steps per decision",),
            (
                (f"dsrl.action_exec_len={int(executed)} exceeds the {geometry.decoded_horizon} steps one decode of "
                f"{geometry.source} emits; set dsrl.action_exec_len<={geometry.decoded_horizon}"),
            ),
        )
    return SpecComparison(("executed steps per decision",))


def plan_geometry(cfg: DictConfig) -> PolicyGeometry:
    """The frozen policy's geometry from its config and checkpoint metadata, as the launcher would load it."""
    from isaac_rl_posttraining.policies import is_gr00t_base_model

    adapter = None
    if is_gr00t_base_model(str(cfg.base_model.name)):
        from isaac_rl_posttraining.envs.embodiments import get_dataset_adapter

        adapter = get_dataset_adapter(OmegaConf.select(cfg, "base_model.dataset_adapter", default="g1_wbc"))
    return policy_geometry(cfg, dataset_adapter=adapter)


def check_data_specs(cfg: DictConfig, *, algorithm: str) -> SpecReport:
    """Compare the simulator's declared specs with the policy's; the report lists every disagreement.

    The policy side comes from the config and the checkpoint's metadata
    (:func:`~isaac_rl_posttraining.policies.geometry.policy_geometry`); what
    neither states before load (a width only the loaded weights know) is
    reported as unverified, and the launcher checks it again once the policy
    has loaded. Whether the acting policy reads the entry's mapped state,
    cameras and language comes from the algorithm's registered action space.
    """
    reads_steering_inputs = algorithm_spec(algorithm).reads_base_model_inputs
    geometry = plan_geometry(cfg) if reads_steering_inputs else None
    comparison = SpecComparison()
    if geometry is not None:
        comparison = _chunk_comparison(cfg, horizon=geometry.latent_horizon, horizon_name=geometry.horizon_key)
        comparison = comparison.merged(_executed_steps_comparison(cfg, geometry))
        comparison = comparison.merged(_latent_comparison(cfg, geometry))
    entry = selected_entry(cfg)
    if entry is None:
        comparison = comparison.merged(SpecComparison(unverified=("no simulator entry declares specs",)))
        report = SpecReport(str(cfg.env.env_name), None, None, None, comparison)
    else:
        report = _entry_report(cfg, entry, comparison, geometry)
    if geometry is None:
        return report
    return dataclasses.replace(report, comparison=_explain_unbound(report.comparison, geometry))


def _explain_unbound(comparison: SpecComparison, geometry: PolicyGeometry) -> SpecComparison:
    """Fold the policy-side sizes that stay unbound into one item that says why."""
    kept: list[str] = []
    deferred: list[str] = []
    for item in comparison.unverified:
        if "state_dim unbound" in item:
            deferred.append(f"{item.split(': ', 1)[0].removeprefix('observation ')} width")
        elif "act_dim unbound" in item:
            deferred.append("action width")
        elif item == _CHUNK_DEFERRED:
            deferred.append("steering window")
        elif item == _HORIZON_DEFERRED:
            deferred.append("decoded horizon")
        elif item == _LATENT_DEFERRED:
            deferred.append("flow-head width")
        elif " free on one side " in item:
            deferred.append("camera size")
        else:
            kept.append(item)
    deferred = list(dict.fromkeys(deferred))
    if deferred:
        verb = "binds" if len(deferred) == 1 else "bind"
        kept.append(f"the policy's {', '.join(deferred)} {verb} at load: {geometry.unknown}")
    return SpecComparison(comparison.checks, comparison.mismatches, tuple(kept))


def _entry_report(
    cfg: DictConfig, entry: SimulatorEntry, comparison: SpecComparison, geometry: PolicyGeometry | None
) -> SpecReport:
    """Compare what the entry declares with what the policy reads and emits."""
    spec = environment_spec(cfg, num_envs=1, server_url=None)
    # An entry that does not declare its specs returns None from the base-class hooks.
    emitted = entry.observation_spec(spec)
    accepted = entry.action_spec(spec)
    consumed = None if geometry is None else steering_input_spec(entry, spec, geometry, emitted)
    if emitted is None:
        comparison = comparison.merged(SpecComparison(unverified=(f"{entry.name} declares no observation spec",)))
    elif consumed is not None:
        comparison = comparison.merged(compare_observation_specs(emitted, consumed))
        comparison = comparison.merged(_camera_comparison(cfg, entry, spec, emitted))
    else:
        comparison = comparison.merged(SpecComparison(unverified=("policy observation keys load with the checkpoint",)))
    if accepted is None:
        comparison = comparison.merged(SpecComparison(unverified=(f"{entry.name} declares no action spec",)))
    elif geometry is not None:
        emitted_action, width_check = _emitted_action(cfg, geometry)
        comparison = comparison.merged(width_check)
        comparison = comparison.merged(compare_action_specs(emitted_action, accepted))
    else:
        comparison = comparison.merged(SpecComparison(unverified=("policy action width loads with the checkpoint",)))
    return SpecReport(entry.name, emitted, accepted, consumed, comparison)


def environment_spec_for_plan(cfg: DictConfig) -> EnvironmentSpec:
    """The one-environment spec the plan describes the simulator with."""
    return environment_spec(cfg, num_envs=1, server_url=None)


def check_chunk_size_at_load(cfg: DictConfig, base_model: BaseModelPolicy) -> None:
    """Refuse a steering chunk longer than the loaded policy's latent window.

    Base models that read their geometry from the checkpoint leave
    ``base_model.action_horizon`` null, so the plan could only defer this; the
    window a flow head denoises (``latent_horizon``) bounds the latents one
    decode accepts.
    """
    comparison = _chunk_comparison(cfg, horizon=base_model.latent_horizon, horizon_name="the loaded policy's latent_horizon")
    comparison.raise_on_mismatch("the steering chunk disagrees with the loaded policy")


def check_policy_inputs_at_load(base_model: BaseModelPolicy, env: gym.Env | gym.vector.VectorEnv) -> str:
    """Compare the loaded policy's input layout with the environment's observation, raising on a mismatch.

    The state width a GR00T checkpoint slices is known only once its modality
    layout is registered, so this runs after the frozen policy has loaded and
    before any rollout worker starts. The base model names its inputs by their
    role in the steering contract (``state``); the environment's
    ``steering_key_mapping`` says which observation entry fills each role.
    Returns one line for the launch log.
    """
    consumed = base_model.input_spec()
    if consumed is None:
        return f"policy inputs: unknown ({type(base_model).__name__} declares no input spec)"
    if isinstance(env, gym.vector.VectorEnv):
        space = env.single_observation_space
        mapping = env.get_attr("steering_key_mapping")[0]
    else:
        space = env.observation_space
        mapping = env.steering_key_mapping
    arrays: dict[str, ArraySpec] = {}
    for role, array_spec in consumed.arrays.items():
        entry = mapping[role] if isinstance(mapping.get(role), str) else role
        arrays[entry] = array_spec
    consumed = ObservationSpec(arrays, consumed.text)
    emitted = ObservationSpec.from_space(space)
    comparison = compare_observation_specs(emitted, consumed)
    comparison.raise_on_mismatch("the loaded policy's inputs disagree with the environment's observation")
    return f"policy reads {consumed.describe()} of {emitted.describe()} | {comparison.summary()}"
