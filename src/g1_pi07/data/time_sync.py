"""Nearest-neighbour synchronization using each action command as the clock."""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
import math
from typing import Generic, Iterable, TypeVar

from g1_pi07.data.types import ActionTargetFrame
from g1_pi07.data.types import AlignedStep
from g1_pi07.data.types import CameraFrame
from g1_pi07.data.types import RobotStateFrame


T = TypeVar("T")


class AlignmentError(RuntimeError):
    """Raised when a required stream has no frame inside the tolerance."""


@dataclass(frozen=True)
class TimestampMatch(Generic[T]):
    item: T
    delta_ns: int


class TimestampBuffer(Generic[T]):
    """Small sorted buffer that also tolerates mildly out-of-order delivery."""

    def __init__(self, capacity: int = 512) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")
        self._capacity = capacity
        self._items: list[T] = []
        self._timestamps: list[int] = []

    def push(self, item: T) -> None:
        timestamp = int(getattr(item, "timestamp_ns"))
        position = bisect_left(self._timestamps, timestamp)
        self._timestamps.insert(position, timestamp)
        self._items.insert(position, item)
        overflow = len(self._items) - self._capacity
        if overflow > 0:
            del self._items[:overflow]
            del self._timestamps[:overflow]

    def nearest(self, timestamp_ns: int, *, max_delta_ns: int) -> TimestampMatch[T] | None:
        if not self._items:
            return None
        position = bisect_left(self._timestamps, timestamp_ns)
        candidates = [index for index in (position - 1, position) if 0 <= index < len(self._items)]
        index = min(candidates, key=lambda i: (abs(self._timestamps[i] - timestamp_ns), self._timestamps[i]))
        delta = self._timestamps[index] - timestamp_ns
        if abs(delta) > max_delta_ns:
            return None
        return TimestampMatch(self._items[index], delta)

    def discard_before(self, timestamp_ns: int) -> None:
        position = bisect_left(self._timestamps, timestamp_ns)
        del self._items[:position]
        del self._timestamps[:position]

    def __len__(self) -> int:
        return len(self._items)


class ActionCentricSynchronizer:
    """Matches state and three RGB frames to an action timestamp.

    State sequence identity is preferred when the teleoperation node records it;
    nearest timestamp is used only as a fallback.  Signed deltas are preserved so
    data-quality checks can detect a consistently early or late camera.
    """

    def __init__(
        self,
        camera_names: Iterable[str] = ("head", "left_wrist", "right_wrist"),
        *,
        max_state_delta_ms: float = 15.0,
        max_camera_delta_ms: float = 40.0,
        capacity: int = 512,
    ) -> None:
        self.camera_names = tuple(camera_names)
        if not self.camera_names or len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("camera_names must be non-empty and unique")
        if (
            not math.isfinite(max_state_delta_ms)
            or not math.isfinite(max_camera_delta_ms)
            or max_state_delta_ms < 0
            or max_camera_delta_ms < 0
        ):
            raise ValueError("Temporal alignment tolerances must be finite and non-negative")
        self.max_state_delta_ns = int(max_state_delta_ms * 1e6)
        self.max_camera_delta_ns = int(max_camera_delta_ms * 1e6)
        self.states: TimestampBuffer[RobotStateFrame] = TimestampBuffer(capacity)
        self.cameras = {name: TimestampBuffer[CameraFrame](capacity) for name in self.camera_names}
        self._states_by_sequence: dict[int, RobotStateFrame] = {}
        self._sequence_fifo: deque[int] = deque(maxlen=capacity)

    def push_state(self, frame: RobotStateFrame) -> None:
        self.states.push(frame)
        if frame.sequence in self._states_by_sequence:
            # Replacing a duplicate sequence must also replace its FIFO entry;
            # otherwise deque(maxlen) silently evicts an unrelated state while
            # the lookup table keeps it forever.
            self._sequence_fifo.remove(frame.sequence)
        elif len(self._sequence_fifo) == self._sequence_fifo.maxlen:
            expired = self._sequence_fifo.popleft()
            self._states_by_sequence.pop(expired, None)
        self._sequence_fifo.append(frame.sequence)
        self._states_by_sequence[frame.sequence] = frame

    def push_camera(self, frame: CameraFrame) -> None:
        if frame.camera not in self.cameras:
            raise KeyError(f"Unknown camera {frame.camera!r}; expected one of {self.camera_names}")
        self.cameras[frame.camera].push(frame)

    def align(self, action: ActionTargetFrame) -> AlignedStep:
        state = None
        state_delta = 0
        if action.source_state_sequence is not None:
            state = self._states_by_sequence.get(action.source_state_sequence)
            if state is not None:
                state_delta = state.timestamp_ns - action.timestamp_ns
                if abs(state_delta) > self.max_state_delta_ns:
                    raise AlignmentError(
                        f"State sequence={action.source_state_sequence} referenced by action "
                        f"{action.step_index} has expired"
                    )
        if state is None:
            state_match = self.states.nearest(action.timestamp_ns, max_delta_ns=self.max_state_delta_ns)
            if state_match is None:
                raise AlignmentError(f"No robot state is available near action {action.step_index}")
            state, state_delta = state_match.item, state_match.delta_ns

        matched_cameras: dict[str, CameraFrame] = {}
        camera_deltas: dict[str, int] = {}
        for name, buffer in self.cameras.items():
            match = buffer.nearest(action.timestamp_ns, max_delta_ns=self.max_camera_delta_ns)
            if match is None:
                raise AlignmentError(f"No {name} image is available near action {action.step_index}")
            matched_cameras[name] = match.item
            camera_deltas[name] = match.delta_ns

        return AlignedStep(
            action=action,
            state=state,
            cameras=matched_cameras,
            state_delta_ns=state_delta,
            camera_delta_ns=camera_deltas,
        )
