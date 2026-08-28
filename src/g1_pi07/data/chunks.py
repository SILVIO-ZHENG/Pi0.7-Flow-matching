"""Future action-chunk construction and padding masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from g1_pi07.joints import ACTIVE_ACTION_DIM
from g1_pi07.joints import DEFAULT_LAYOUT
from g1_pi07.joints import MODEL_ACTION_DIM
from g1_pi07.joints import G1JointLayout


@dataclass(frozen=True)
class ActionChunk:
    actions: np.ndarray
    step_mask: np.ndarray
    dim_mask: np.ndarray
    source_indices: np.ndarray

    @property
    def loss_mask(self) -> np.ndarray:
        return self.step_mask[:, None] & self.dim_mask[None, :]


def make_action_chunk(
    actions: np.ndarray,
    start_index: int,
    *,
    horizon: int = 50,
    layout: G1JointLayout = DEFAULT_LAYOUT,
    pad_mode: str = "repeat_last",
) -> ActionChunk:
    """Build ``[horizon, 32]`` from a trajectory of 28-D expert actions."""

    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != ACTIVE_ACTION_DIM:
        raise ValueError(f"actions must have shape [T,{ACTIVE_ACTION_DIM}]; got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("actions must not contain NaN or Inf")
    if not 0 <= start_index < len(values):
        raise IndexError(f"start_index={start_index} is outside trajectory length {len(values)}")
    if horizon <= 0:
        raise ValueError("horizon must be greater than zero")

    valid = min(horizon, len(values) - start_index)
    chunk_28 = np.empty((horizon, ACTIVE_ACTION_DIM), dtype=np.float32)
    chunk_28[:valid] = values[start_index : start_index + valid]
    if valid < horizon:
        if pad_mode == "repeat_last":
            chunk_28[valid:] = chunk_28[valid - 1]
        elif pad_mode == "zeros":
            chunk_28[valid:] = 0.0
        else:
            raise ValueError("pad_mode must be 'repeat_last' or 'zeros'")

    step_mask = np.arange(horizon) < valid
    source_indices = np.full(horizon, -1, dtype=np.int64)
    source_indices[:valid] = np.arange(start_index, start_index + valid)
    return ActionChunk(
        actions=layout.policy_to_model(chunk_28),
        step_mask=step_mask,
        dim_mask=layout.model_dim_mask,
        source_indices=source_indices,
    )


def masked_mse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.bool_)
    if prediction.shape != target.shape or mask.shape != prediction.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("prediction and target must not contain NaN/Inf")
    if not np.any(mask):
        raise ValueError("mask must contain at least one valid element")
    return float(np.square(prediction - target)[mask].mean())


def default_action_masks(horizon: int = 50) -> tuple[np.ndarray, np.ndarray]:
    if horizon <= 0:
        raise ValueError("horizon must be greater than zero")
    return np.ones(horizon, dtype=np.bool_), np.r_[
        np.ones(ACTIVE_ACTION_DIM, dtype=np.bool_),
        np.zeros(MODEL_ACTION_DIM - ACTIVE_ACTION_DIM, dtype=np.bool_),
    ]
