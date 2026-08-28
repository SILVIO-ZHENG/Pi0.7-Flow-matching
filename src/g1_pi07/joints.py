"""Joint ordering and 43 -> 28 -> 32 dimensional mappings for Unitree G1.

The default names follow the convention used by this project.  A deployment
must compare them with the names published by its own G1 SDK/URDF before
sending commands; :meth:`G1JointLayout.reorder_full` exists for that purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np


FULL_DOF: Final = 43
ACTIVE_ACTION_DIM: Final = 28
MODEL_ACTION_DIM: Final = 32

LEG_JOINT_NAMES: Final = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

WAIST_JOINT_NAMES: Final = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

ARM_JOINT_NAMES: Final = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

DEX3_JOINT_NAMES: Final = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)

FULL_JOINT_NAMES: Final = (*LEG_JOINT_NAMES, *WAIST_JOINT_NAMES, *ARM_JOINT_NAMES, *DEX3_JOINT_NAMES)
POLICY_JOINT_NAMES: Final = (*ARM_JOINT_NAMES, *DEX3_JOINT_NAMES)


def _as_last_dim(array: Sequence[float] | np.ndarray, expected: int, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim == 0 or value.shape[-1] != expected:
        raise ValueError(f"The final dimension of {name} must be {expected}; shape={value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must not contain NaN or Inf")
    return value


@dataclass(frozen=True)
class G1JointLayout:
    """Defines the only accepted joint order at every project boundary."""

    full_joint_names: tuple[str, ...] = FULL_JOINT_NAMES
    policy_joint_names: tuple[str, ...] = POLICY_JOINT_NAMES
    model_action_dim: int = MODEL_ACTION_DIM

    def __post_init__(self) -> None:
        if len(self.full_joint_names) != FULL_DOF:
            raise ValueError(f"full_joint_names must contain {FULL_DOF} entries")
        if len(self.policy_joint_names) != ACTIVE_ACTION_DIM:
            raise ValueError(f"policy_joint_names must contain {ACTIVE_ACTION_DIM} entries")
        if len(set(self.full_joint_names)) != FULL_DOF:
            raise ValueError("full_joint_names contains duplicate joints")
        missing = set(self.policy_joint_names) - set(self.full_joint_names)
        if missing:
            raise ValueError(f"Policy joints are missing from the 43-DoF list: {sorted(missing)}")
        if self.model_action_dim < ACTIVE_ACTION_DIM:
            raise ValueError("Model action dimension must not be smaller than 28")

    @property
    def policy_indices(self) -> np.ndarray:
        by_name = {name: index for index, name in enumerate(self.full_joint_names)}
        return np.asarray([by_name[name] for name in self.policy_joint_names], dtype=np.int64)

    @property
    def model_dim_mask(self) -> np.ndarray:
        mask = np.zeros(self.model_action_dim, dtype=np.bool_)
        mask[:ACTIVE_ACTION_DIM] = True
        return mask

    def full_to_policy(self, full: Sequence[float] | np.ndarray) -> np.ndarray:
        full_array = _as_last_dim(full, FULL_DOF, "full")
        return np.take(full_array, self.policy_indices, axis=-1)

    def policy_to_model(self, policy: Sequence[float] | np.ndarray) -> np.ndarray:
        policy_array = _as_last_dim(policy, ACTIVE_ACTION_DIM, "policy")
        output = np.zeros((*policy_array.shape[:-1], self.model_action_dim), dtype=np.float32)
        output[..., :ACTIVE_ACTION_DIM] = policy_array
        return output

    def model_to_policy(self, model: Sequence[float] | np.ndarray) -> np.ndarray:
        model_array = _as_last_dim(model, self.model_action_dim, "model")
        return model_array[..., :ACTIVE_ACTION_DIM].copy()

    def policy_to_full(
        self,
        policy: Sequence[float] | np.ndarray,
        *,
        base_full: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Insert upper-body commands while preserving legs and waist."""

        policy_array = _as_last_dim(policy, ACTIVE_ACTION_DIM, "policy")
        base_array = _as_last_dim(base_full, FULL_DOF, "base_full")
        if policy_array.shape[:-1] != base_array.shape[:-1]:
            raise ValueError("policy and base_full must have identical batch dimensions")
        output = base_array.copy()
        output[..., self.policy_indices] = policy_array
        return output

    def reorder_full(
        self,
        values: Sequence[float] | np.ndarray,
        input_joint_names: Sequence[str],
    ) -> np.ndarray:
        """Reorder a named 43-vector into the project's canonical order."""

        value_array = _as_last_dim(values, FULL_DOF, "values")
        if len(input_joint_names) != FULL_DOF or len(set(input_joint_names)) != FULL_DOF:
            raise ValueError("input_joint_names must contain 43 unique names")
        index = {name: i for i, name in enumerate(input_joint_names)}
        missing = [name for name in self.full_joint_names if name not in index]
        if missing:
            raise ValueError(f"Input joint names are missing: {missing}")
        order = np.asarray([index[name] for name in self.full_joint_names], dtype=np.int64)
        return np.take(value_array, order, axis=-1)


DEFAULT_LAYOUT: Final = G1JointLayout()
