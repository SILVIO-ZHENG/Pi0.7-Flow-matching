# SPDX-License-Identifier: Apache-2.0
"""Deployment configuration for G1 43-DoF with a 28-D policy."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from g1_pi07.joints import ACTIVE_ACTION_DIM
from g1_pi07.joints import DEFAULT_LAYOUT

# The deployed state/action vector contains only the 28 policy joints.
STATE_DIM = ACTIVE_ACTION_DIM
# All three camera streams are resized to the model's square image input.
MODEL_IMAGE_SIZE = 224


@dataclasses.dataclass
class Args:
    """Runtime configuration shared by asynchronous and RTC G1 clients.

    Rates are expressed in Hz, age limits in milliseconds, and horizons in
    action steps. Robot command publication remains controlled independently by
    ``enable_robot_commands`` and the validated joint-limit file.
    """

    # ROS graph and image-source configuration.
    config_path: str | None = None
    ros_domain_id: str = "0"
    state_topic: str = "/robot/state43"
    action_topic: str = "/control/action_target43"
    head_image_topic: str = "/camera/head/image_raw/compressed"
    left_wrist_image_topic: str = "/camera/left_wrist/image_raw/compressed"
    right_wrist_image_topic: str = "/camera/right_wrist/image_raw/compressed"

    # Remote policy service and task-conditioning configuration.
    remote_host: str = "127.0.0.1"
    remote_port: int = 8000
    prompt: str = "Use both hands to pick up the box and place it in the target area."
    plan_subtask: bool = True

    # Control-loop, interpolation, and replanning configuration.
    controller_mode: str = "rtc"
    control_hz: float = 20.0
    policy_action_hz: float = 20.0
    open_loop_horizon: int = 8
    max_action_chunk_len: int = 50
    interpolation: str = "linear"
    request_when_remaining_steps: int = 10
    request_immediately_after_chunk: bool = True
    rtc_min_horizon: int = 2
    rtc_delay_buffer_size: int = 16
    rtc_initial_delay_steps: int = 2
    # Freshness and robot-command safety configuration.
    max_state_age_ms: float = 100.0
    max_image_age_ms: float = 150.0

    joint_limits_path: str = "examples/unitree_g1/configs/g1_joint_limits.example.json"
    enable_robot_commands: bool = False
    episode_id: str = "inference"
    # Diagnostic output configuration.
    log_dir: str = "./outputs/g1_policy_logs"
    save_action_chunks: bool = True


@dataclasses.dataclass(frozen=True)
class RobotLayout:
    """Validated 28-D joint bounds and their URDF confirmation state."""

    lower_limits: np.ndarray
    upper_limits: np.ndarray
    limits_confirmed: bool


def load_json_config(config_path: str) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a JSON object")
    return data


def merge_config_file(args: Args) -> Args:
    if args.config_path is None:
        return args
    defaults = Args(config_path=args.config_path)
    config_data = load_json_config(args.config_path)
    field_names = {field.name for field in dataclasses.fields(Args)}
    unknown = sorted(set(config_data) - field_names)
    if unknown:
        raise ValueError(f"Configuration contains unknown fields: {unknown}")
    merged = dataclasses.asdict(args)
    for key, value in config_data.items():
        if key != "config_path" and getattr(args, key) == getattr(defaults, key):
            merged[key] = value
    return Args(**merged)


def build_robot_layout(args: Args) -> RobotLayout:
    if not 1 <= int(args.remote_port) <= 65535:
        raise ValueError("remote_port must be in 1..65535")
    if not str(args.remote_host).strip() or not str(args.prompt).strip():
        raise ValueError("remote_host and prompt must not be empty")
    if args.controller_mode not in {"async", "rtc"}:
        raise ValueError("controller_mode must be 'async' or 'rtc'")
    if args.interpolation not in {"none", "linear", "cubic"}:
        raise ValueError("interpolation must be 'none', 'linear', or 'cubic'")
    integer_limits = {
        "open_loop_horizon": args.open_loop_horizon,
        "max_action_chunk_len": args.max_action_chunk_len,
        "rtc_min_horizon": args.rtc_min_horizon,
        "rtc_delay_buffer_size": args.rtc_delay_buffer_size,
    }
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in integer_limits.values()):
        raise ValueError(f"Invalid positive-integer control parameters: {integer_limits}")
    if (
        not isinstance(args.request_when_remaining_steps, int)
        or isinstance(args.request_when_remaining_steps, bool)
        or not isinstance(args.rtc_initial_delay_steps, int)
        or isinstance(args.rtc_initial_delay_steps, bool)
        or args.request_when_remaining_steps < 0
        or args.rtc_initial_delay_steps < 0
    ):
        raise ValueError("Replanning remaining steps and RTC initial delay must be non-negative integers")
    if args.max_action_chunk_len > 50:
        raise ValueError("max_action_chunk_len must not exceed H=50 for the G1 model")
    if args.open_loop_horizon > args.max_action_chunk_len or args.rtc_min_horizon > args.max_action_chunk_len:
        raise ValueError("open_loop_horizon and rtc_min_horizon must not exceed the action-chunk length")
    if args.rtc_initial_delay_steps > args.max_action_chunk_len:
        raise ValueError("rtc_initial_delay_steps must not exceed the action-chunk length")
    payload = load_json_config(args.joint_limits_path)
    names = tuple(payload.get("policy_joint_names", ()))
    if names != DEFAULT_LAYOUT.policy_joint_names:
        raise ValueError("Joint-limit policy_joint_names do not match the canonical 28-dimensional order")
    lower = np.asarray(payload.get("lower"), dtype=np.float32)
    upper = np.asarray(payload.get("upper"), dtype=np.float32)
    if lower.shape != (STATE_DIM,) or upper.shape != (STATE_DIM,) or np.any(lower >= upper):
        raise ValueError("Joint limits must provide valid lower[28] and upper[28] arrays")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Joint limits must not contain NaN or Inf")
    positive_floats = np.asarray(
        [args.control_hz, args.policy_action_hz, args.max_state_age_ms, args.max_image_age_ms],
        dtype=np.float64,
    )
    if not np.isfinite(positive_floats).all() or np.any(positive_floats <= 0):
        raise ValueError("Control rates and observation freshness thresholds must be positive and finite")
    confirmed = bool(payload.get("confirmed_from_robot_urdf", False))
    if args.enable_robot_commands and not confirmed:
        raise ValueError("Live commands are enabled, but joint limits are not marked as verified from the target URDF")
    return RobotLayout(lower, upper, confirmed)
