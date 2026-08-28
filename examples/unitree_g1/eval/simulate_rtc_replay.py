#!/usr/bin/env python3
"""Simulate training-time RTC action-chunk scheduling offline.

This script reads action-chunk files or generates synthetic chunks. It does not
connect to ROS, access a policy server, or publish robot actions. It provides a
quick check of hard-prefix skipping and execution continuity under fixed delay.
"""
# ruff: noqa: E402

from __future__ import annotations

import dataclasses
from pathlib import Path
import sys

import numpy as np
import tyro

OPENPI_ROOT = Path(__file__).resolve().parents[3]
if str(OPENPI_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPI_ROOT))

from examples.unitree_g1.rtc.rtc_chunker import RtcChunker


@dataclasses.dataclass
class Args:
    """Configuration for an offline RTC replay."""

    chunk_dir: Path | None = None
    output: Path = Path("./outputs/rtc_replay_smoke.npz")
    horizon: int = 50
    min_horizon: int = 8
    fixed_delay_steps: int = 4
    num_chunks: int = 12
    action_dim: int = 28


def _load_chunks(chunk_dir: Path, *, horizon: int, action_dim: int) -> list[np.ndarray]:
    """Read action chunks from deployment-log NPZ files."""
    chunks: list[np.ndarray] = []
    for path in sorted(chunk_dir.glob("*.npz")):
        with np.load(path) as data:
            key = "execution_plan" if "execution_plan" in data else "raw_actions"
            raw_action = np.asarray(data[key], dtype=np.float32)
        if raw_action.ndim != 2 or raw_action.shape[1] < action_dim or not np.isfinite(raw_action).all():
            raise ValueError(f"{key} in {path} is not a finite [H,D>={action_dim}] array")
        action = raw_action[:horizon, :action_dim]
        if len(action) > 0:
            chunks.append(action)
    if not chunks:
        raise ValueError(f"No usable NPZ action chunks found in {chunk_dir}")
    return chunks


def _make_synthetic_chunks(args: Args) -> list[np.ndarray]:
    """Generate continuous synthetic chunks with small phase changes."""
    base_t = np.linspace(0.0, 1.0, args.horizon, dtype=np.float32)
    dims = np.linspace(0.5, 1.5, args.action_dim, dtype=np.float32)
    chunks = []
    for idx in range(args.num_chunks):
        phase = idx * args.min_horizon / max(1, args.horizon)
        chunk = np.sin((base_t[:, None] + phase) * dims[None, :] * np.pi).astype(np.float32)
        chunks.append(chunk)
    return chunks


def _smoothness(actions: np.ndarray) -> dict[str, float]:
    """Compute first- and second-order difference metrics for an execution sequence."""
    if len(actions) < 3:
        return {"mean_abs_delta": 0.0, "max_abs_delta": 0.0, "mean_abs_ddelta": 0.0, "max_abs_ddelta": 0.0}
    delta = np.diff(actions, axis=0)
    ddelta = np.diff(delta, axis=0)
    return {
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "max_abs_delta": float(np.max(np.abs(delta))),
        "mean_abs_ddelta": float(np.mean(np.abs(ddelta))),
        "max_abs_ddelta": float(np.max(np.abs(ddelta))),
    }


def run(args: Args) -> None:
    """Run an offline RTC replay and save metrics."""
    if args.horizon <= 0 or args.min_horizon <= 0 or args.action_dim <= 0 or args.num_chunks <= 0:
        raise ValueError("horizon, min_horizon, action_dim, and num_chunks must be greater than zero")
    if args.min_horizon > args.horizon or not 1 <= args.fixed_delay_steps <= args.horizon:
        raise ValueError("min_horizon and fixed_delay_steps must fit within the action horizon")
    chunks = (
        _load_chunks(args.chunk_dir, horizon=args.horizon, action_dim=args.action_dim)
        if args.chunk_dir is not None
        else _make_synthetic_chunks(args)
    )
    chunker = RtcChunker(
        horizon=args.horizon,
        min_horizon=args.min_horizon,
        delay_buffer_size=8,
        initial_delay_steps=args.fixed_delay_steps,
    )

    executed: list[np.ndarray] = []
    observed_delays: list[int] = []
    fallback = np.zeros(args.action_dim, dtype=np.float32)
    request_id = 0
    chunk_iter = iter(chunks)

    while True:
        try:
            chunk = next(chunk_iter)
        except StopIteration:
            break

        prefix = chunker.make_action_prefix()
        if prefix is not None:
            action_prefix, delay = prefix
            if action_prefix.shape != (args.horizon, args.action_dim) or delay > args.fixed_delay_steps:
                raise RuntimeError("RTC hard-prefix dimension or delay constraint was violated")

        context = chunker.make_request_context(request_id)
        for _ in range(args.fixed_delay_steps):
            executed.append(chunker.consume_action(fallback))
            chunker.record_control_step()
        _, observed_delay = chunker.accept_new_chunk(context.request_id, chunk)
        observed_delays.append(observed_delay)

        for _ in range(args.min_horizon):
            executed.append(chunker.consume_action(fallback))
            fallback = executed[-1]
            chunker.record_control_step()
        request_id += 1

    executed_arr = np.stack(executed, axis=0)
    metrics = _smoothness(executed_arr)
    metrics["num_actions"] = float(len(executed_arr))
    metrics["mean_delay_steps"] = float(np.mean(observed_delays)) if observed_delays else 0.0
    metrics["max_delay_steps"] = float(np.max(observed_delays)) if observed_delays else 0.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, executed_actions=executed_arr, observed_delays=np.asarray(observed_delays), **metrics
    )
    print(f"RTC replay saved: {args.output}")
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")


if __name__ == "__main__":
    run(tyro.cli(Args))
