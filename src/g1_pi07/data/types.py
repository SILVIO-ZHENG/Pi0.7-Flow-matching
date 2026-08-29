"""Typed records shared by collection, conversion, and replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from g1_pi07.joints import FULL_DOF


def _vector(value, size: int, name: str, *, dtype=np.float32) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape=({size},); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _optional_timestamp(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class RobotStateFrame:
    """Canonical timestamped 43-DoF robot state plus a 10-value IMU vector.

    ``validity_mask`` marks joints populated by the source message; zero values
    alone must never be interpreted as valid measurements.
    """

    timestamp_ns: int
    sequence: int
    q: np.ndarray
    dq: np.ndarray
    tau_est: np.ndarray
    imu: np.ndarray
    validity_mask: np.ndarray
    imu_valid: bool = True

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0 or self.sequence < 0:
            raise ValueError("timestamp_ns and sequence must be non-negative")
        object.__setattr__(self, "q", _vector(self.q, FULL_DOF, "q"))
        object.__setattr__(self, "dq", _vector(self.dq, FULL_DOF, "dq"))
        object.__setattr__(self, "tau_est", _vector(self.tau_est, FULL_DOF, "tau_est"))
        object.__setattr__(self, "imu", _vector(self.imu, 10, "imu"))
        object.__setattr__(
            self,
            "validity_mask",
            _vector(self.validity_mask, FULL_DOF, "validity_mask", dtype=np.bool_),
        )
        object.__setattr__(self, "imu_valid", bool(self.imu_valid))


@dataclass(frozen=True)
class ActionTargetFrame:
    """Candidate 43-DoF target tied to the state used for its computation.

    Timing fields are monotonic pipeline markers used to measure inference and
    transport latency. ``validity_mask`` identifies commanded joint dimensions.
    """

    timestamp_ns: int
    episode_id: str
    step_index: int
    source_state_sequence: int | None
    position: np.ndarray
    compute_started_ns: int | None = None
    compute_finished_ns: int | None = None
    sent_ns: int | None = None
    source: str = ""
    validity_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0 or self.step_index < 0:
            raise ValueError("timestamp_ns and step_index must be non-negative")
        if not self.episode_id or self.episode_id in {".", ".."} or "/" in self.episode_id or "\\" in self.episode_id:
            raise ValueError("episode_id must be a single safe name")
        if self.source_state_sequence is not None and self.source_state_sequence < 0:
            raise ValueError("source_state_sequence must be non-negative or None")
        object.__setattr__(self, "position", _vector(self.position, FULL_DOF, "position"))
        validity = np.ones(FULL_DOF, dtype=np.bool_) if self.validity_mask is None else self.validity_mask
        object.__setattr__(self, "validity_mask", _vector(validity, FULL_DOF, "validity_mask", dtype=np.bool_))
        for name in ("compute_started_ns", "compute_finished_ns", "sent_ns"):
            object.__setattr__(self, name, _optional_timestamp(getattr(self, name), name))
        ordered = [
            value for value in (self.compute_started_ns, self.compute_finished_ns, self.sent_ns) if value is not None
        ]
        if ordered != sorted(ordered):
            raise ValueError("Action timing must satisfy started <= finished <= sent")


@dataclass(frozen=True)
class CameraFrame:
    """One named RGB frame represented as a non-empty HWC uint8 array."""

    timestamp_ns: int
    camera: str
    rgb: np.ndarray

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0 or not self.camera:
            raise ValueError("Camera timestamp must be non-negative and camera must not be empty")
        image = np.asarray(self.rgb)
        if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2], default=0) <= 0 or image.dtype != np.uint8:
            raise ValueError(f"rgb must be HWC uint8; got shape={image.shape}, dtype={image.dtype}")
        object.__setattr__(self, "rgb", image)


@dataclass(frozen=True)
class AlignedStep:
    """One action-clock sample with its nearest state and camera observations.

    Delta values are signed as ``observation_time - action_time`` and are
    validated against the embedded frame timestamps at construction time.
    """

    action: ActionTargetFrame
    state: RobotStateFrame
    cameras: Mapping[str, CameraFrame]
    state_delta_ns: int
    camera_delta_ns: Mapping[str, int]

    def __post_init__(self) -> None:
        if set(self.cameras) != set(self.camera_delta_ns):
            raise ValueError("cameras and camera_delta_ns must have identical keys")
        expected_state_delta = self.state.timestamp_ns - self.action.timestamp_ns
        if self.state_delta_ns != expected_state_delta:
            raise ValueError("state_delta_ns is inconsistent with state/action timestamps")
        for name, frame in self.cameras.items():
            if frame.camera != name:
                raise ValueError(f"Camera mapping key {name!r} does not match frame.camera={frame.camera!r}")
            expected_camera_delta = frame.timestamp_ns - self.action.timestamp_ns
            if self.camera_delta_ns[name] != expected_camera_delta:
                raise ValueError(f"{name} camera_delta_ns is inconsistent with frame/action timestamps")

    @property
    def timestamp_ns(self) -> int:
        return self.action.timestamp_ns
