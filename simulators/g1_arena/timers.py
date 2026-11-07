# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a batched G1 Factory step's time goes, seen from the server and from the client.

Both timers keep a rolling window and hand back one summary line per window; the
env-server prints its line under the simulator's minimal Python (numpy only), the
client logs its line through loguru. The server echoes its own per-step time in
every reply, so the client can split its round trip into server work and wire time
without a shared clock.
"""

from __future__ import annotations


class StepTimer:
    """Sim-side timing of the batched step, reported every ``report_every`` steps.

    The client only sees the round trip; splitting the server's own time into
    the simulator step (physics + render + whole-body controller) and the
    payload build (image extraction) is what tells the sim cost apart from
    transport and the learner's per-decision work at a given N.
    """

    def __init__(self, num_envs: int, report_every: int = 100, log_prefix: str = "[arena_env_server]") -> None:
        if report_every < 1:
            raise ValueError(f"report_every must be >= 1, got {report_every}")
        self.num_envs = num_envs
        self.report_every = report_every
        self.log_prefix = log_prefix
        self.sim_s: list[float] = []
        self.payload_s: list[float] = []
        self.encode_s: list[float] = []
        self.send_s: list[float] = []
        self.image_steps = 0
        self.total_steps = 0

    def record(self, sim_s: float, payload_s: float, with_images: bool) -> str | None:
        """Add one step; return the summary line when a window completes, else None."""
        self.sim_s.append(sim_s)
        self.payload_s.append(payload_s)
        self.image_steps += int(with_images)
        self.total_steps += 1
        if len(self.sim_s) < self.report_every:
            return None
        summary = self.summary()
        self.sim_s.clear()
        self.payload_s.clear()
        self.encode_s.clear()
        self.send_s.clear()
        self.image_steps = 0
        return summary

    def record_send(self, encode_s: float, send_s: float) -> None:
        """Add the serialization and socket-write time of one reply (reported by the transport)."""
        self.encode_s.append(encode_s)
        self.send_s.append(send_s)

    def summary(self) -> str:
        n = len(self.sim_s)
        sim_mean = sum(self.sim_s) / n
        n_send = len(self.send_s)
        send_part = (
            f"encode {1000 * sum(self.encode_s) / n_send:.0f} + send {1000 * sum(self.send_s) / n_send:.0f} ms/reply mean, "
            if n_send
            else ""
        )
        return (
            f"{self.log_prefix} step timing over {n} steps (through step {self.total_steps}): "
            f"sim {1000 * sim_mean:.0f} ms/step mean (min {1000 * min(self.sim_s):.0f}, max {1000 * max(self.sim_s):.0f}), "
            f"payload {1000 * sum(self.payload_s) / n:.0f} ms/step mean, {send_part}images on {self.image_steps} steps; "
            f"{self.num_envs / sim_mean:.1f} env-steps/s sim-only"
        )


class RoundTripTimer:
    """Per-window log of where a batched client step's wall time goes.

    The server echoes its own step time in every reply, so the round trip
    splits into server work, wire time (pickle, transfer, latency, unpickle)
    and the client's observation formatting. Comparing the wire share of a
    co-located run with a remote one is the direct measure of what the
    network costs at a given N.
    """

    def __init__(self, num_envs: int, report_every: int = 100) -> None:
        if report_every < 1:
            raise ValueError(f"report_every must be >= 1, got {report_every}")
        self.num_envs = num_envs
        self.report_every = report_every
        self.round_trip_s: list[float] = []
        self.server_s: list[float] = []
        self.recv_s: list[float] = []
        self.decode_s: list[float] = []
        self.format_s: list[float] = []
        self.image_steps = 0
        self.total_steps = 0

    def record(
        self,
        round_trip_s: float,
        server_s: float,
        recv_s: float,
        decode_s: float,
        format_s: float,
        with_images: bool,
    ) -> str | None:
        """Add one step; return the summary line when a window completes, else None.

        ``recv_s`` is the time from sending the request to the last byte of
        the reply (server work plus transfer), ``decode_s`` the reply's
        unpickling; both come from the transport and are 0 if it does not
        measure them.
        """
        self.round_trip_s.append(round_trip_s)
        self.server_s.append(server_s)
        self.recv_s.append(recv_s)
        self.decode_s.append(decode_s)
        self.format_s.append(format_s)
        self.image_steps += int(with_images)
        self.total_steps += 1
        if len(self.round_trip_s) < self.report_every:
            return None
        summary = self.summary()
        self.round_trip_s.clear()
        self.server_s.clear()
        self.recv_s.clear()
        self.decode_s.clear()
        self.format_s.clear()
        self.image_steps = 0
        return summary

    def summary(self) -> str:
        n = len(self.round_trip_s)
        round_trip = sum(self.round_trip_s) / n
        server = sum(self.server_s) / n
        recv = sum(self.recv_s) / n
        decode = sum(self.decode_s) / n
        fmt = sum(self.format_s) / n
        return (
            f"[arena_batched_client] step timing over {n} steps (through step {self.total_steps}): "
            f"round trip {1000 * round_trip:.0f} ms/step mean = server {1000 * server:.0f} + "
            f"wire {1000 * (round_trip - server):.0f} (transfer {1000 * (recv - server):.0f}, "
            f"decode {1000 * decode:.0f}); format {1000 * fmt:.0f} ms/step; "
            f"images on {self.image_steps} steps; {self.num_envs / (round_trip + fmt):.1f} env-steps/s"
        )
