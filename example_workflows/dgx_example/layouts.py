# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where one run's processes sit on a DGX Station's GPUs, derived from the simulator entry.

A post-training run has two GPU consumers: the simulator stepping N environments, and the
learner process (the frozen base policy's batched decode plus the SAC updates). A layout says
which card each one sits on:

``single``
    everything on one card.
``split``
    the simulator on one card and the learner on the other, joined over loopback TCP.

What a simulator supports follows from its :class:`SimulatorEntry`, not from a table of
simulators. A split needs a simulator that runs in its own process behind a server (the entry
declares ``server_url_variables``); an in-process simulator such as Isaac Lab or LIBERO can only
run ``single``. Whether the N environments are stepped as one batched worker follows from
``entry.worker_layout(num_envs, num_servers)``, and a batched worker collects whole episodes, so
rollouts are then counted by episode. ``--num-servers`` spreads the training environments over that
many simulator servers on the simulator's card, one batched worker each, on consecutive ports.

Command line::

    python -m example_workflows.dgx_example.layouts capabilities [--check README.md]
    python -m example_workflows.dgx_example.layouts plan --simulator arena --num-envs 64 \\
        --simulator-gpu <id> --learner-gpu <id> [--port 7300] [--num-servers 1] [--format text|shell]

``plan --format shell`` prints ``DGX_*=value`` lines for a launcher script to evaluate; the same
card for both roles is the single layout, two cards the split.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from example_workflows.dgx_example.readme_blocks import block_matches, write_block
from isaac_rl_posttraining.envs.providers import (
    available_simulator_entries,
    get_simulator_entry,
)
from isaac_rl_posttraining.envs.simulator_entry import SimulatorEntry

Shape = Literal["single", "split"]
SHAPES: tuple[Shape, ...] = ("single", "split")
CAPABILITIES_BLOCK = "capabilities"
DEFAULT_PORT = 7300


@dataclass(frozen=True)
class Capabilities:
    """What one registered simulator supports, read off its entry class.

    ``segments`` are the environment-name segments that select the entry (the
    ``arena`` of ``arena/drill_lift``); ``entry`` is the entry's own name, which
    is also its ``simulators/<entry>/`` folder.
    """

    segments: tuple[str, ...]
    entry: str
    served: bool
    server_url_variables: tuple[str, str] | None
    batched_collection: bool
    decode_batch_size: int | None
    evaluation: str

    @property
    def shapes(self) -> tuple[Shape, ...]:
        """The layouts the entry can run: a split needs a simulator in its own process."""
        return SHAPES if self.served else ("single",)


def _capabilities(segments: tuple[str, ...], entry: SimulatorEntry) -> Capabilities:
    return Capabilities(
        segments=segments,
        entry=entry.name,
        served=entry.server_url_variables is not None,
        server_url_variables=entry.server_url_variables,
        # Two environments is the smallest count that can be batched.
        batched_collection=entry.worker_layout(2).batched,
        decode_batch_size=entry.decode_batch_size,
        evaluation=entry.evaluation,
    )


def capabilities(simulator: str) -> Capabilities:
    """The capabilities of the entry registered for the environment-name segment ``simulator``."""
    return _capabilities((simulator,), get_simulator_entry(simulator))


def all_capabilities() -> list[Capabilities]:
    """One row per registered entry; segments sharing an entry (``arena``, ``isaac``) collapse into one."""
    segments_by_entry: dict[str, list[str]] = {}
    entries: dict[str, SimulatorEntry] = {}
    for segment in available_simulator_entries():
        entry = get_simulator_entry(segment)
        segments_by_entry.setdefault(entry.name, []).append(segment)
        entries[entry.name] = entry
    return [
        _capabilities(tuple(segments), entries[name])
        for name, segments in sorted(segments_by_entry.items())
    ]


def capabilities_table() -> str:
    """The per-simulator table the README carries, rendered from the registered entries."""
    lines = [
        "| Simulator (`env.env_name` segment) | Entry | Simulator process | Layouts | Collection at N > 1 | Decode batch cap | Evaluation |",
        "|---|---|---|---|---|---|---|",
    ]
    for caps in all_capabilities():
        process = "its own, behind a server" if caps.served else "the learner's"
        collection = (
            "one batched worker, whole episodes"
            if caps.batched_collection
            else "one worker per environment"
        )
        cap = "none" if caps.decode_batch_size is None else str(caps.decode_batch_size)
        lines.append(
            f"| {', '.join(f'`{segment}`' for segment in caps.segments)} | `{caps.entry}` | {process} "
            f"| {', '.join(caps.shapes)} | {collection} | {cap} | {caps.evaluation} |"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class Plan:
    """One run's placement and the launch inputs it implies."""

    capabilities: Capabilities
    shape: Shape
    num_envs: int
    simulator_gpu: str
    learner_gpu: str
    batched: bool
    port: int
    num_servers: int

    @property
    def hydra_overrides(self) -> tuple[str, ...]:
        """What the learner's config must say for this many environments on this entry."""
        overrides = [f"training.num_envs={self.num_envs}"]
        if self.batched:
            # A batched worker collects whole episodes; the runtime rejects
            # step-counted collection for it.
            overrides.append("training.rollout_count_by=episode")
        return tuple(overrides)

    @property
    def train_server_url(self) -> str | None:
        """Where the learner finds the training simulators when the two are launched apart: one
        address per server, on consecutive ports from ``port``, comma-separated."""
        if self.shape != "split":
            return None
        return ",".join(
            f"tcp://127.0.0.1:{self.port + index}" for index in range(self.num_servers)
        )

    def values(self) -> dict[str, str]:
        values = {
            "DGX_SIMULATOR": self.capabilities.segments[0],
            "DGX_ENTRY": self.capabilities.entry,
            "DGX_SHAPE": self.shape,
            "DGX_NUM_ENVS": str(self.num_envs),
            "DGX_NUM_SERVERS": str(self.num_servers),
            "DGX_SIMULATOR_GPU": self.simulator_gpu,
            "DGX_LEARNER_GPU": self.learner_gpu,
            "DGX_BATCHED": "1" if self.batched else "0",
            "DGX_HYDRA_OVERRIDES": " ".join(self.hydra_overrides),
        }
        if self.capabilities.server_url_variables is not None:
            train_variable, eval_variable = self.capabilities.server_url_variables
            values["DGX_TRAIN_SERVER_URL_VARIABLE"] = train_variable
            values["DGX_EVAL_SERVER_URL_VARIABLE"] = eval_variable
        if self.train_server_url is not None:
            values["DGX_TRAIN_SERVER_URL"] = self.train_server_url
        return values

    def shell_lines(self) -> list[str]:
        """``NAME=value`` lines a shell script evaluates."""
        return [f"{key}={shlex.quote(value)}" for key, value in self.values().items()]

    def text_lines(self) -> list[str]:
        return [
            f"{key.removeprefix('DGX_').lower()}: {value}"
            for key, value in self.values().items()
        ]


def plan(
    simulator: str,
    num_envs: int,
    simulator_gpu: str,
    learner_gpu: str,
    port: int = DEFAULT_PORT,
    num_servers: int = 1,
) -> Plan:
    """Place ``num_envs`` environments of ``simulator`` with its simulator and learner on the given cards.

    The same card for both is the single layout, two cards the split. ``num_servers``
    spreads the environments over that many simulator servers (one batched worker each);
    the entry decides whether it can. Fails when the entry cannot run the requested shape,
    before anything is launched.
    """
    if isinstance(num_envs, bool) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    if isinstance(num_servers, bool) or num_servers < 1:
        raise ValueError("num_servers must be a positive integer")
    if not simulator_gpu or not learner_gpu:
        raise ValueError(
            "simulator_gpu and learner_gpu must name a card (an nvidia-smi index or UUID)"
        )
    if not 0 < port < 65536 - num_servers:
        raise ValueError(
            "port must be a TCP port number, with room for one port per server"
        )
    entry = get_simulator_entry(simulator)
    caps = _capabilities((simulator,), entry)
    shape: Shape = "single" if simulator_gpu == learner_gpu else "split"
    if shape == "split" and not caps.served:
        raise ValueError(
            f"{entry.name} runs its simulator inside the learner's process (its entry declares no "
            "simulator server), so the simulator and the learner cannot sit on different cards; "
            "give both roles the same GPU for the single layout"
        )
    return Plan(
        capabilities=caps,
        shape=shape,
        num_envs=num_envs,
        simulator_gpu=simulator_gpu,
        learner_gpu=learner_gpu,
        batched=entry.worker_layout(num_envs, num_servers=num_servers).batched,
        port=port,
        num_servers=num_servers,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    table = commands.add_parser("capabilities", help="the per-simulator table")
    table.add_argument(
        "--write", type=Path, metavar="README", help="replace the README's block"
    )
    table.add_argument(
        "--check", type=Path, metavar="README", help="verify the README's block"
    )
    placement = commands.add_parser("plan", help="place one run")
    placement.add_argument(
        "--simulator", required=True, help="environment-name segment, e.g. arena"
    )
    placement.add_argument("--num-envs", type=int, required=True)
    placement.add_argument(
        "--simulator-gpu", required=True, help="nvidia-smi index or UUID"
    )
    placement.add_argument(
        "--learner-gpu", required=True, help="nvidia-smi index or UUID"
    )
    placement.add_argument("--port", type=int, default=DEFAULT_PORT)
    placement.add_argument(
        "--num-servers",
        type=int,
        default=1,
        help="simulator servers hosting the training environments, on consecutive ports",
    )
    placement.add_argument("--format", choices=("text", "shell"), default="text")
    args = parser.parse_args(argv)

    if args.command == "capabilities":
        body = capabilities_table()
        if args.write is not None:
            write_block(args.write, CAPABILITIES_BLOCK, body)
        elif args.check is not None:
            if not block_matches(args.check, CAPABILITIES_BLOCK, body):
                print(
                    f"{args.check}: the capability table is out of date; run this command with --write",
                    file=sys.stderr,
                )
                return 1
            print(f"{args.check}: capability table is current")
        else:
            print(body)
        return 0
    try:
        placed = plan(
            args.simulator,
            args.num_envs,
            args.simulator_gpu,
            args.learner_gpu,
            args.port,
            args.num_servers,
        )
    except (KeyError, ValueError) as exc:
        print(f"cannot place this run: {exc}", file=sys.stderr)
        return 2
    lines = placed.shell_lines() if args.format == "shell" else placed.text_lines()
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
