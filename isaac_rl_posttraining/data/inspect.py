# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print what a saved replay buffer holds: ``python -m isaac_rl_posttraining.data.inspect <buffer dir>``.

``<buffer dir>`` is one saved buffer or a folder of ``part-NNNNN`` buffers
(what ``data.persist.path`` writes), read together as ``load_buffer`` does.
The summary is the schema (version, read settings, reward channels, the
observation spec), the episode counts by outcome and by who acted, the mean
return per reward channel over the complete episodes, and the spec of the
first transition, so a researcher can answer "what is in this buffer"
without a Python session.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from collections.abc import Sequence

import numpy as np

from isaac_rl_posttraining.buffers.storage import buffer_parts, load_buffer, read_schema
from isaac_rl_posttraining.data.transitions import Transition

__all__ = ["main", "summary_lines"]


def _spec_lines(title: str, arrays: dict[str, dict[str, object]], text: Sequence[str]) -> list[str]:
    lines = [f"{title}:"]
    for name, spec in arrays.items():
        shape = ", ".join("?" if size is None else str(size) for size in spec["shape"])
        lines.append(f"  {name}: {spec['dtype']}[{shape}]")
    for name in text:
        lines.append(f"  {name}: text")
    return lines


def _transition_lines(transition: Transition) -> list[str]:
    spec = transition.observation.spec()
    lines = _spec_lines(
        "first transition observation",
        {name: {"shape": list(entry.shape), "dtype": entry.dtype} for name, entry in spec.arrays.items()},
        sorted(spec.text),
    )
    if transition.learner_action is None:
        lines.append("  learner action: none")
    else:
        value = transition.learner_action.value
        lines.append(f"  learner action: {transition.learner_action.kind} {value.dtype}{list(value.shape)}")
    if transition.env_action is None:
        lines.append("  env action: none")
    else:
        lines.append(
            f"  env action: {transition.env_action.spec()} x {transition.env_action.steps} steps "
            f"(execution length {transition.env_action.execution_length})"
        )
    outputs = [name for name in ("log_prob", "value", "next_value", "entropy") if getattr(transition.policy_outputs, name) is not None]
    if transition.policy_outputs.features is not None:
        outputs.append(f"features {sorted(transition.policy_outputs.features)}")
    lines.append(f"  policy outputs: {', '.join(outputs) if outputs else 'none'}")
    return lines


def summary_lines(path: str) -> list[str]:
    """The inspection report of the saved buffer (or folder of ``part-NNNNN`` buffers) at ``path``, one line per entry."""
    parts = buffer_parts(path)
    schemas = [read_schema(part) for part in parts]
    schema = schemas[0]
    buffer = load_buffer(path)
    channels = sorted({name for part in schemas for name in part["reward_channels"]})
    dropped = sorted({name for part in schemas for name in part.get("drop_components", [])})
    lines = [f"replay buffer {path}" + (f" ({len(parts)} parts)" if len(parts) > 1 else "")]
    if len(parts) > 1:
        lines.append(f"parts: {os.path.basename(parts[0])} .. {os.path.basename(parts[-1])}")
    lines += [
        (
            f"schema version {schema['schema_version']}; capacity {buffer.capacity}; "
            f"reward channel {schema['reward_channel']!r} aggregated by {schema['reward_aggregation']!r}"
        ),
        f"reward channels: {', '.join(channels)}",
        f"dropped observation components: {', '.join(dropped) or 'none'}",
        *_spec_lines("observation spec", schema["observation"]["arrays"], schema["observation"]["text"]),
        f"transitions: {len(buffer)}; episodes: {len(buffer.episodes())}",
    ]
    episodes = buffer.episodes()
    outcomes = Counter(episode.outcome for episode in episodes)
    lines.append("episodes by outcome: " + ", ".join(f"{name} {outcomes[name]}" for name in ("success", "failure", "timeout", "ongoing")))
    sources = Counter(transition.behavior_source for transition in buffer.transitions())
    lines.append("transitions by source: " + ", ".join(f"{name} {count}" for name, count in sorted(sources.items())))
    complete = [episode for episode in episodes if episode.complete]
    for channel in channels:
        returns = [
            episode.return_(channel)
            for episode in complete
            if all(channel in transition.rewards for transition in episode.transitions)
        ]
        if returns:
            lines.append(
                f"mean return on {channel!r} over {len(returns)} complete episodes: "
                f"{float(np.mean(returns)):.4f} (min {float(np.min(returns)):.4f}, max {float(np.max(returns)):.4f})"
            )
        else:
            lines.append(f"mean return on {channel!r}: no complete episode carries it")
    lines.extend(_transition_lines(buffer.transitions()[0]))
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a saved replay buffer directory.")
    parser.add_argument(
        "path",
        help="directory written by isaac_rl_posttraining.buffers.save_buffer, or a folder of its part-NNNNN "
        "directories (data.persist.path)",
    )
    arguments = parser.parse_args(argv)
    for line in summary_lines(arguments.path):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
