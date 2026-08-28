"""Deployment-side scheduler for training-time RTC.

The scheduler tracks asynchronous inference requests, latency estimates, and
the previous chunk's action prefix. The model applies RTC hard-prefix
conditioning. The deployment side sends actions that will execute during
inference as ``rtc_prefix`` and skips that already-executed prefix when the new
chunk arrives.
"""

from __future__ import annotations

from collections import deque
import dataclasses

import numpy as np


@dataclasses.dataclass
class RtcRequestContext:
    """Context associated with one RTC inference request."""

    request_id: int
    start_step: int
    executed_since_swap: int
    delay_estimate_steps: int
    previous_suffix: np.ndarray


class RtcChunker:
    """Track training-time RTC state and align hard-prefix chunks."""

    def __init__(
        self,
        *,
        horizon: int,
        min_horizon: int,
        delay_buffer_size: int,
        initial_delay_steps: int,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be greater than zero")
        if min_horizon <= 0:
            raise ValueError("min_horizon must be greater than zero")
        if delay_buffer_size <= 0:
            raise ValueError("delay_buffer_size must be greater than zero")
        if min_horizon > horizon or initial_delay_steps < 0 or initial_delay_steps > horizon:
            raise ValueError("min_horizon and initial_delay_steps must fit within the action horizon")
        self.horizon = int(horizon)
        self.min_horizon = min(int(min_horizon), self.horizon)
        self.delay_steps = deque([max(1, initial_delay_steps)], maxlen=delay_buffer_size)
        self.current_chunk: np.ndarray | None = None
        self.executed_since_swap = 0
        self.control_step = 0
        self.inflight_context: RtcRequestContext | None = None

    def has_chunk(self) -> bool:
        """Return whether an executable chunk is available."""
        return self.current_chunk is not None and len(self.current_chunk) > 0

    def record_control_step(self) -> None:
        """Record one control-loop step."""
        self.control_step += 1
        self.executed_since_swap += 1

    def should_request(self) -> bool:
        """Return whether the next inference request should start."""
        if self.inflight_context is not None:
            return False
        if self.current_chunk is None:
            return True
        remaining = len(self.current_chunk) - self.executed_since_swap
        delay_estimate = max(self.delay_steps) if self.delay_steps else self.min_horizon
        return self.executed_since_swap >= self.min_horizon or remaining <= delay_estimate

    def make_request_context(self, request_id: int) -> RtcRequestContext:
        """Create an inference-request context."""
        if request_id < 0:
            raise ValueError("request_id must be non-negative")
        if self.inflight_context is not None:
            raise RuntimeError("An RTC inference request is already in flight")
        delay_estimate = max(self.delay_steps) if self.delay_steps else self.min_horizon
        if self.current_chunk is None:
            suffix = np.empty((0, 0), dtype=np.float32)
        else:
            suffix = self.current_chunk[self.executed_since_swap :].copy()
        context = RtcRequestContext(
            request_id=request_id,
            start_step=self.control_step,
            executed_since_swap=self.executed_since_swap,
            delay_estimate_steps=delay_estimate,
            previous_suffix=suffix,
        )
        self.inflight_context = context
        return context

    def make_action_prefix(self) -> tuple[np.ndarray, int] | None:
        """Build hard-prefix actions and an estimated delay for the model.

        Returned actions are padded to ``horizon``, but only the first
        ``delay`` steps are valid.
        """
        if self.current_chunk is None:
            return None
        suffix = self.current_chunk[self.executed_since_swap :].copy()
        if suffix.size == 0 or suffix.ndim != 2:
            return None

        delay_estimate = max(self.delay_steps) if self.delay_steps else self.min_horizon
        delay = min(max(0, int(delay_estimate)), len(suffix), self.horizon)
        if delay == 0:
            return None

        prefix = np.zeros((self.horizon, suffix.shape[1]), dtype=np.float32)
        copy_len = min(len(suffix), self.horizon)
        prefix[:copy_len] = suffix[:copy_len]
        return prefix, delay

    def consume_action(self, fallback_action: np.ndarray) -> np.ndarray:
        """Return the action for the current control step."""
        fallback = np.asarray(fallback_action, dtype=np.float32)
        if fallback.ndim != 1 or not np.isfinite(fallback).all():
            raise ValueError("fallback_action must be a finite one-dimensional action")
        if self.current_chunk is not None and self.current_chunk.shape[1:] != fallback.shape:
            raise ValueError("fallback_action does not match the current chunk action dimension")
        if self.current_chunk is None or self.executed_since_swap >= len(self.current_chunk):
            return fallback
        return self.current_chunk[self.executed_since_swap].copy()

    def accept_new_chunk(self, request_id: int, new_chunk: np.ndarray) -> tuple[np.ndarray, int]:
        """Accept an inference result and create a new executable chunk."""
        new_chunk = np.asarray(new_chunk, dtype=np.float32)
        if new_chunk.ndim != 2 or len(new_chunk) == 0 or not np.isfinite(new_chunk).all():
            raise ValueError("new_chunk must be a non-empty finite [H,D] array")
        if self.inflight_context is None or self.inflight_context.request_id != request_id:
            raise ValueError("RTC response request_id does not match the in-flight request")
        previous_suffix = self.inflight_context.previous_suffix
        if previous_suffix.size and previous_suffix.shape[1] != new_chunk.shape[1]:
            raise ValueError("Old and new RTC chunks have different action dimensions")

        observed_delay = max(1, self.control_step - self.inflight_context.start_step)
        self.delay_steps.append(observed_delay)
        aligned = self._drop_executed_prefix(previous_suffix, new_chunk, observed_delay)
        self.current_chunk = aligned[: self.horizon].copy()
        self.executed_since_swap = 0
        self.inflight_context = None
        return self.current_chunk.copy(), observed_delay

    def cancel_request(self, request_id: int) -> bool:
        """Clear a failed matching request so rolling replanning can continue."""

        if self.inflight_context is None or self.inflight_context.request_id != request_id:
            return False
        self.inflight_context = None
        return True

    def _drop_executed_prefix(
        self, previous_suffix: np.ndarray, new_chunk: np.ndarray, observed_delay: int
    ) -> np.ndarray:
        """Discard the new chunk prefix that the old chunk already executed."""
        new_chunk = np.asarray(new_chunk, dtype=np.float32)
        if previous_suffix.size == 0:
            return new_chunk
        if new_chunk.ndim != 2:
            return new_chunk
        skip = min(max(0, int(observed_delay)), len(new_chunk))
        return new_chunk[skip:]
