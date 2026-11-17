# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""When rollouts, updates and weight publishes happen (runtime v2 schedules).

``sync``: collect a round, train on it, publish.
``overlapped``: the next round collects while the learner trains, so its data
lags the policy by one version. ``async``: rollout workers collect
continuously; the trainer trains on what has arrived and publishes once per
training burst (or every ``publish_every`` updates when set); data more than
``max_policy_lag`` publishes behind the policy is dropped. Policy lag is the
number of weight publishes between the weights that collected the data and the
learner's current weights. Evaluation never overlaps training under any schedule.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

__all__ = [
    "SCHEDULES",
    "Schedule",
    "ScheduleSpec",
    "as_schedule_spec",
    "on_policy_schedule",
    "parse_schedule",
    "schedule_from_config",
]

Schedule = Literal["sync", "overlapped", "async"]
SCHEDULES: tuple[Schedule, ...] = ("sync", "overlapped", "async")


def parse_schedule(name: str) -> Schedule:
    """The schedule ``name`` names."""
    for schedule in SCHEDULES:
        if name == schedule:
            return schedule
    raise ValueError(f"schedule must be one of {SCHEDULES}, got {name!r}")


def _non_negative(value: object, *, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


@dataclass(frozen=True)
class ScheduleSpec:
    """A schedule and its staleness bound.

    ``max_policy_lag`` bounds the policy lag of training data: the number of
    weight publishes between the weights that collected it and the learner's
    current weights. ``None`` takes the schedule's own lag (0 for ``sync``, 1
    for ``overlapped``) and leaves ``async`` unbounded.

    ``publish_every`` (``async`` only): ``None``, the default, publishes once
    per training burst, like the other schedules, so a round collected during
    one burst lags by one; ``N`` publishes every ``N`` learner updates instead
    (a round collected during a burst of ``U`` updates then lags ``U // N``).
    """

    kind: Schedule = "sync"
    max_policy_lag: int | None = None
    publish_every: int | None = None

    def __post_init__(self) -> None:
        kind = parse_schedule(self.kind)
        object.__setattr__(self, "kind", kind)
        if self.publish_every is not None:
            _non_negative(self.publish_every, name="publish_every", minimum=1)
        if self.publish_every == 1 and kind != "async":
            # The old default; outside async every schedule publishes once per burst anyway.
            object.__setattr__(self, "publish_every", None)
        if self.publish_every is not None and kind != "async":
            raise ValueError(
                f"publish_every applies to the async schedule only, not {kind!r}"
            )
        if self.max_policy_lag is None:
            return
        _non_negative(self.max_policy_lag, name="max_policy_lag", minimum=0)
        if kind == "sync" and self.max_policy_lag != 0:
            raise ValueError(
                "the sync schedule trains on current data; max_policy_lag must be 0"
            )
        if kind == "overlapped" and self.max_policy_lag < 1:
            raise ValueError(
                "the overlapped schedule's data lags one version; max_policy_lag must be >= 1"
            )

    @property
    def policy_lag_bound(self) -> int | None:
        """The largest policy lag training data may have; ``None`` is unbounded."""
        if self.max_policy_lag is not None:
            return self.max_policy_lag
        return {"sync": 0, "overlapped": 1}.get(self.kind)

    @property
    def staleness_bound(self) -> int | None:
        """The policy lag above which the data source drops collected data (async only; ``None``: keep all)."""
        return self.max_policy_lag if self.kind == "async" else None

    @property
    def updates_per_publish(self) -> int | None:
        """Learner updates between publishes inside a burst; ``None``: publish once after each burst."""
        return self.publish_every if self.kind == "async" else None

    @property
    def overlaps_collection(self) -> bool:
        """Whether rollouts run while the learner trains."""
        return self.kind != "sync"


def as_schedule_spec(value: ScheduleSpec | str) -> ScheduleSpec:
    """A :class:`ScheduleSpec`, or the one a schedule name denotes."""
    if isinstance(value, ScheduleSpec):
        return value
    if isinstance(value, str):
        return ScheduleSpec(value)  # type: ignore[arg-type]
    raise TypeError(f"schedule must be a ScheduleSpec or a schedule name, got {value!r}")


def on_policy_schedule(spec: ScheduleSpec) -> ScheduleSpec:
    """Check a schedule for an on-policy algorithm (PPO): data may lag the policy by one version at most.

    The async schedule defaults to ``max_policy_lag=1`` here; a larger bound,
    or ``0`` (every overlapped round would be dropped), is rejected.
    """
    if spec.kind != "async":
        return spec
    if spec.max_policy_lag is None:
        return replace(spec, max_policy_lag=1)
    if spec.max_policy_lag != 1:
        raise ValueError(
            "on-policy algorithms train on data at most one policy version old: the async "
            f"schedule needs schedule.max_policy_lag=1, got {spec.max_policy_lag} "
            "(use the sync schedule for strictly current data)"
        )
    return spec


def schedule_from_config(section: Mapping[str, Any] | str | None) -> ScheduleSpec:
    """The schedule of a run config's ``schedule`` entry.

    ``section`` is a schedule name, a mapping with ``kind`` (and optionally
    ``max_policy_lag``, ``publish_every``), or ``None`` (the sync schedule).
    """
    if section is None:
        return ScheduleSpec()
    if isinstance(section, str):
        return ScheduleSpec(section)  # type: ignore[arg-type]
    unknown = sorted(set(section) - {"kind", "max_policy_lag", "publish_every"})
    if unknown:
        raise ValueError(f"unknown schedule keys {unknown}; known: kind, max_policy_lag, publish_every")
    return ScheduleSpec(
        section.get("kind", "sync"),
        max_policy_lag=section.get("max_policy_lag"),
        publish_every=section.get("publish_every"),
    )
