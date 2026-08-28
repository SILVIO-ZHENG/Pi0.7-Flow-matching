"""Action-chunk interpolation and execution planning."""

from __future__ import annotations

import numpy as np


def resample_action_chunk(actions: np.ndarray, src_hz: float, dst_hz: float, *, method: str = "linear") -> np.ndarray:
    """Interpolate model actions to the control-loop rate."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"actions must be a two-dimensional array; shape={actions.shape}")
    if len(actions) == 0 or not np.isfinite(actions).all():
        raise ValueError("actions must be non-empty and must not contain NaN/Inf")
    if src_hz <= 0 or dst_hz <= 0:
        raise ValueError("src_hz and dst_hz must be greater than zero")
    if len(actions) < 2 or abs(src_hz - dst_hz) < 1e-6:
        return actions.copy()

    t_src = np.arange(len(actions), dtype=np.float32) / float(src_hz)
    t_end = float(t_src[-1])
    n_new = int(np.round(t_end * dst_hz)) + 1
    t_new = np.linspace(0.0, t_end, n_new, dtype=np.float32)

    if method == "linear":
        out = np.empty((n_new, actions.shape[1]), dtype=np.float32)
        for dim in range(actions.shape[1]):
            out[:, dim] = np.interp(t_new, t_src, actions[:, dim]).astype(np.float32)
        return out

    if method == "cubic":
        try:
            from scipy.interpolate import CubicSpline
        except ImportError as exc:
            raise RuntimeError("Cubic interpolation requires scipy") from exc
        out = np.empty((n_new, actions.shape[1]), dtype=np.float32)
        for dim in range(actions.shape[1]):
            out[:, dim] = CubicSpline(t_src, actions[:, dim], bc_type="natural")(t_new).astype(np.float32)
        return out

    if method == "none":
        return actions.copy()

    raise ValueError(f"Unknown interpolation method: {method}")


def make_execution_plan(
    action_chunk: np.ndarray,
    *,
    state_dim: int,
    max_action_chunk_len: int,
    policy_action_hz: float,
    control_hz: float,
    interpolation: str,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> np.ndarray:
    """Convert policy output into a trajectory executable by the control loop."""
    actions = np.asarray(action_chunk, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or len(actions) == 0 or actions.shape[1] < state_dim:
        raise ValueError(f"action_chunk must be non-empty [H,D>={state_dim}]; got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("action_chunk must not contain NaN/Inf")
    lower_limits = np.asarray(lower_limits, dtype=np.float32)
    upper_limits = np.asarray(upper_limits, dtype=np.float32)
    if lower_limits.shape != (state_dim,) or upper_limits.shape != (state_dim,):
        raise ValueError("Joint-limit dimensions must equal state_dim")
    if max_action_chunk_len <= 0 or state_dim <= 0:
        raise ValueError("max_action_chunk_len and state_dim must be greater than zero")
    actions = actions[:max_action_chunk_len, :state_dim]
    actions = np.clip(actions, lower_limits[None, :], upper_limits[None, :])
    actions = resample_action_chunk(actions, policy_action_hz, control_hz, method=interpolation)
    return np.clip(actions, lower_limits[None, :], upper_limits[None, :])
