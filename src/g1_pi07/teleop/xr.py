"""Pure XR tracking-frame calibration used by the ROS2 UDP bridge."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize_quaternion(quaternion_xyzw: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion_xyzw must be a finite four-dimensional array")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError("Quaternion must not be zero")
    return quaternion / norm


def quaternion_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion(quaternion_xyzw)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_multiply(left_xyzw: np.ndarray, right_xyzw: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = normalize_quaternion(left_xyzw)
    rx, ry, rz, rw = normalize_quaternion(right_xyzw)
    return normalize_quaternion(
        np.asarray(
            [
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
                lw * rw - lx * rx - ly * ry - lz * rz,
            ],
            dtype=np.float64,
        )
    )


@dataclass(frozen=True)
class RigidXrCalibration:
    """Scale positions, then apply one rigid XR-frame -> robot-frame transform."""

    translation: np.ndarray
    quaternion_xyzw: np.ndarray
    position_scale: float = 1.0

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation, dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("translation must be a finite three-dimensional array")
        scale = float(self.position_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("position_scale must be positive and finite")
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "quaternion_xyzw", normalize_quaternion(self.quaternion_xyzw))
        object.__setattr__(self, "position_scale", scale)

    @property
    def rotation(self) -> np.ndarray:
        return quaternion_matrix(self.quaternion_xyzw)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != 3 or not np.isfinite(values).all():
            raise ValueError("The final points dimension must be 3 and values must not contain NaN/Inf")
        flat = values.reshape(-1, 3)
        transformed = self.translation + (self.rotation @ (self.position_scale * flat).T).T
        return transformed.reshape(values.shape)

    def transform_pose(
        self,
        position: np.ndarray,
        quaternion_xyzw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        transformed_position = self.transform_points(np.asarray(position, dtype=np.float64))
        transformed_orientation = quaternion_multiply(self.quaternion_xyzw, quaternion_xyzw)
        return transformed_position, transformed_orientation
