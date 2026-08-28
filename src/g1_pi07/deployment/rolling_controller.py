"""Latency-aware rolling action-chunk execution independent of ROS2."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RequestContext:
    request_id: int
    control_step: int
    executed_in_chunk: int
    prefix: np.ndarray | None
    estimated_delay_steps: int


class RollingChunkController:
    """Maintains an asynchronous chunk cache and RTC hard prefix."""

    def __init__(
        self,
        *,
        horizon: int = 50,
        min_replan_steps: int = 2,
        initial_delay_steps: int = 2,
        delay_history: int = 16,
        blend_steps: int = 2,
    ) -> None:
        if horizon <= 0 or min_replan_steps <= 0 or delay_history <= 0:
            raise ValueError("horizon, min_replan_steps, and delay_history must be greater than zero")
        if (
            min_replan_steps > horizon
            or initial_delay_steps < 0
            or initial_delay_steps > horizon
            or blend_steps < 0
        ):
            raise ValueError("Replanning, delay, or blend parameters are outside their valid ranges")
        self.horizon = horizon
        self.min_replan_steps = min_replan_steps
        self.blend_steps = blend_steps
        self._delay_steps = deque([max(1, initial_delay_steps)], maxlen=delay_history)
        self._chunk: np.ndarray | None = None
        self._index = 0
        self._control_step = 0
        self._inflight: RequestContext | None = None
        self._last_action: np.ndarray | None = None

    @property
    def remaining_steps(self) -> int:
        return 0 if self._chunk is None else max(0, len(self._chunk) - self._index)

    def should_replan(self) -> bool:
        if self._inflight is not None:
            return False
        if self._chunk is None:
            return True
        delay = max(self._delay_steps)
        return self._index >= self.min_replan_steps and self.remaining_steps <= delay + self.min_replan_steps

    def begin_request(self, request_id: int) -> RequestContext:
        if request_id < 0:
            raise ValueError("request_id must be non-negative")
        if self._inflight is not None:
            raise RuntimeError("An inference request is already in flight")
        prefix = None
        if self._chunk is not None and self.remaining_steps:
            prefix = np.zeros((self.horizon, self._chunk.shape[1]), dtype=np.float32)
            suffix = self._chunk[self._index :]
            copy_len = min(len(prefix), len(suffix))
            prefix[:copy_len] = suffix[:copy_len]
        context = RequestContext(
            request_id=request_id,
            control_step=self._control_step,
            executed_in_chunk=self._index,
            prefix=prefix,
            estimated_delay_steps=max(self._delay_steps),
        )
        self._inflight = context
        return context

    def accept(self, request_id: int, chunk: np.ndarray) -> int:
        actions = np.asarray(chunk, dtype=np.float32)
        if actions.ndim != 2 or len(actions) == 0 or not np.isfinite(actions).all():
            raise ValueError("chunk must be a non-empty finite [H,D] array")
        if self._inflight is None or self._inflight.request_id != request_id:
            raise ValueError("Inference-result request_id does not match the current request")
        if self._last_action is not None and actions.shape[1:] != self._last_action.shape:
            raise ValueError("Chunk action dimension does not match previously executed actions")
        observed_delay = max(0, self._control_step - self._inflight.control_step)
        self._delay_steps.append(max(1, observed_delay))
        aligned = actions[min(observed_delay, len(actions)) : self.horizon]
        if len(aligned) == 0:
            aligned = actions[-1:]
        if self._last_action is not None and self.blend_steps:
            count = min(self.blend_steps, len(aligned))
            weights = np.linspace(0.0, 1.0, count + 1, dtype=np.float32)[1:, None]
            aligned[:count] = (1.0 - weights) * self._last_action[None, :] + weights * aligned[:count]
        self._chunk = aligned.copy()
        self._index = 0
        self._inflight = None
        return observed_delay

    def next_action(self, fallback: np.ndarray) -> np.ndarray:
        fallback = np.asarray(fallback, dtype=np.float32)
        if fallback.ndim != 1 or not np.isfinite(fallback).all():
            raise ValueError("fallback must be a finite one-dimensional action")
        if self._chunk is not None and self._chunk.shape[1:] != fallback.shape:
            raise ValueError("fallback does not match the current chunk action dimension")
        if self._chunk is None or self._index >= len(self._chunk):
            action = self._last_action.copy() if self._last_action is not None else fallback.copy()
        else:
            action = self._chunk[self._index].copy()
            self._index += 1
        self._control_step += 1
        self._last_action = action
        return action
