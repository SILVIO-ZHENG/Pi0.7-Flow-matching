"""Calibrated 21-keypoint XR hand to seven-motor Dex3-1 retargeting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _bend(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = a - b
    second = c - b
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator < 1e-8:
        return 0.0
    angle = np.arccos(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.clip(1.0 - angle / np.pi, 0.0, 1.0))


@dataclass(frozen=True)
class Dex3Calibration:
    lower: np.ndarray
    upper: np.ndarray
    invert: np.ndarray

    @classmethod
    def unit_range(cls) -> "Dex3Calibration":
        return cls(np.zeros(7, np.float32), np.ones(7, np.float32), np.zeros(7, np.bool_))

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float32)
        upper = np.asarray(self.upper, dtype=np.float32)
        invert = np.asarray(self.invert, dtype=np.bool_)
        if lower.shape != (7,) or upper.shape != (7,) or invert.shape != (7,):
            raise ValueError("Dex3 calibration lower/upper/invert values must all have 7 dimensions")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("Dex3 calibration must not contain NaN or Inf")
        if np.any(upper <= lower):
            raise ValueError("Each Dex3 upper limit must be greater than its lower limit")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "invert", invert)


@dataclass(frozen=True)
class Dex3Retargeter:
    calibration: Dex3Calibration = Dex3Calibration.unit_range()

    def normalized_flexion(self, keypoints: np.ndarray) -> np.ndarray:
        points = np.asarray(keypoints, dtype=np.float32)
        if points.shape != (21, 3):
            raise ValueError(f"Hand keypoints must have shape [21,3]; got {points.shape}")
        if not np.all(np.isfinite(points)):
            raise ValueError("Hand keypoints must not contain NaN or Inf")
        # MediaPipe/OpenXR-compatible indices: wrist=0, thumb=1..4,
        # index=5..8, middle=9..12. Dex3-1 uses 3+2+2 actuators.
        values = np.asarray(
            [
                _bend(points[0], points[1], points[2]),
                _bend(points[1], points[2], points[3]),
                _bend(points[2], points[3], points[4]),
                _bend(points[0], points[5], points[6]),
                0.5 * (_bend(points[5], points[6], points[7]) + _bend(points[6], points[7], points[8])),
                _bend(points[0], points[9], points[10]),
                0.5 * (_bend(points[9], points[10], points[11]) + _bend(points[10], points[11], points[12])),
            ],
            dtype=np.float32,
        )
        return values

    def retarget(self, keypoints: np.ndarray) -> np.ndarray:
        normalized = self.normalized_flexion(keypoints)
        normalized = np.where(self.calibration.invert, 1.0 - normalized, normalized)
        return self.calibration.lower + normalized * (self.calibration.upper - self.calibration.lower)
