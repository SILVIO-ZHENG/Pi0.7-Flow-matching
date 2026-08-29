"""Backend-independent damped-least-squares inverse kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Pose:
    """Cartesian target with a normalized quaternion in ``x, y, z, w`` order."""

    position: np.ndarray
    quaternion_xyzw: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64)
        quaternion = np.asarray(self.quaternion_xyzw, dtype=np.float64)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("Pose requires position[3] and quaternion_xyzw[4]")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
            raise ValueError("Pose must not contain NaN or Inf")
        norm = np.linalg.norm(quaternion)
        if norm < 1e-8:
            raise ValueError("Quaternion must not be zero")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion_xyzw", quaternion / norm)


class ArmKinematics(Protocol):
    """Minimal forward-kinematics interface required by the numeric IK solver."""

    def forward(self, q: np.ndarray) -> Pose: ...

    def jacobian(self, q: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class IKResult:
    """Solver output with convergence state and final translation/rotation errors."""

    q: np.ndarray
    converged: bool
    iterations: int
    position_error: float
    rotation_error: float


def _quat_to_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion_xyzw
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-7:
        return np.zeros(3, dtype=np.float64)
    if np.pi - angle < 1e-5:
        # The usual skew/sin(theta) expression is singular at 180 degrees.
        # Recover an axis from the diagonal and choose signs from off-diagonal
        # terms; this is sufficient for a stable DLS error near pi.
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0))
        largest = int(np.argmax(axis))
        if axis[largest] < 1e-8:
            axis = np.asarray([1.0, 0.0, 0.0])
        else:
            for index in range(3):
                if index != largest:
                    axis[index] = np.copysign(axis[index], rotation[index, largest] + rotation[largest, index])
            axis /= np.linalg.norm(axis)
        return angle * axis
    vector = np.asarray(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]
    )
    return angle * vector / (2.0 * np.sin(angle))


def pose_error(current: Pose, target: Pose) -> np.ndarray:
    translation = target.position - current.position
    rotation = _quat_to_matrix(target.quaternion_xyzw) @ _quat_to_matrix(current.quaternion_xyzw).T
    return np.concatenate([translation, _so3_log(rotation)])


@dataclass
class DampedLeastSquaresIK:
    """Solve bounded arm IK using damped least-squares Jacobian updates.

    ``max_step_norm`` caps each joint-space update before joint-limit clipping,
    reducing instability near singular configurations.
    """

    kinematics: ArmKinematics
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    damping: float = 0.05
    max_iterations: int = 80
    position_tolerance: float = 2e-3
    rotation_tolerance: float = 2e-2
    max_step_norm: float = 0.15

    def solve(self, target: Pose, seed: np.ndarray) -> IKResult:
        q = np.asarray(seed, dtype=np.float64).copy()
        lower = np.asarray(self.lower_limits, dtype=np.float64)
        upper = np.asarray(self.upper_limits, dtype=np.float64)
        if q.ndim != 1 or q.shape != lower.shape or q.shape != upper.shape:
            raise ValueError("seed, lower_limits, and upper_limits must have identical dimensions")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("IK seed and joint limits must not contain NaN or Inf")
        if np.any(lower >= upper):
            raise ValueError("Each IK lower limit must be smaller than its upper limit")
        numeric_config = np.asarray(
            [self.damping, self.position_tolerance, self.rotation_tolerance, self.max_step_norm],
            dtype=np.float64,
        )
        if (
            not np.isfinite(numeric_config).all()
            or np.any(numeric_config <= 0)
            or not isinstance(self.max_iterations, int)
            or isinstance(self.max_iterations, bool)
            or self.max_iterations <= 0
        ):
            raise ValueError("IK damping, tolerance, max_iterations, and max_step_norm must be positive and finite")
        q = np.clip(q, lower, upper)

        last_error = np.full(6, np.inf)
        for iteration in range(1, self.max_iterations + 1):
            last_error = pose_error(self.kinematics.forward(q), target)
            if (
                np.linalg.norm(last_error[:3]) <= self.position_tolerance
                and np.linalg.norm(last_error[3:]) <= self.rotation_tolerance
            ):
                return IKResult(
                    q.astype(np.float32),
                    True,  # noqa: FBT003 - the result tuple stores convergence positionally.
                    iteration,
                    *self._error_norms(last_error),
                )
            jacobian = np.asarray(self.kinematics.jacobian(q), dtype=np.float64)
            if jacobian.shape != (6, len(q)):
                raise ValueError(f"Jacobian must have shape [6,{len(q)}]; got {jacobian.shape}")
            if not np.isfinite(jacobian).all():
                raise ValueError("Jacobian contains NaN or Inf")
            # Damping regularizes the task-space solve near singular Jacobians.
            regularized = jacobian @ jacobian.T + (self.damping**2) * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(regularized, last_error)
            delta_norm = np.linalg.norm(delta)
            if delta_norm > self.max_step_norm:
                delta *= self.max_step_norm / delta_norm
            q = np.clip(q + delta, lower, upper)
        return IKResult(
            q.astype(np.float32),
            False,  # noqa: FBT003 - the result tuple stores convergence positionally.
            self.max_iterations,
            *self._error_norms(last_error),
        )

    @staticmethod
    def _error_norms(error: np.ndarray) -> tuple[float, float]:
        return float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
