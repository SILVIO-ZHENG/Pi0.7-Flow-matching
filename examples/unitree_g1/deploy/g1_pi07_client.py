#!/usr/bin/env python3
"""Unitree G1 43-DoF pi0.7-inspired asynchronous inference client."""
# ruff: noqa: E402

from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
import tyro

OPENPI_ROOT = Path(__file__).resolve().parents[3]
if str(OPENPI_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPI_ROOT))

from examples.unitree_g1.common.async_policy import AsyncPolicyProcess
from examples.unitree_g1.common.ros_io import COMPRESSED_IMAGE_MSG
from examples.unitree_g1.common.ros_io import G1RosIO
from examples.unitree_g1.common.g1_config import STATE_DIM
from examples.unitree_g1.common.g1_config import Args
from examples.unitree_g1.common.g1_config import build_robot_layout
from examples.unitree_g1.common.g1_config import merge_config_file
from examples.unitree_g1.common.trajectory import make_execution_plan
from examples.unitree_g1.rtc.rtc_chunker import RtcChunker


def _format_timing(result: dict) -> str:
    """Format inference timing as a single log line."""
    policy_timing = result.get("policy_timing", {})
    model_timing = result.get("model_timing", {})
    server_timing = result.get("server_timing", {})
    client_timing = result.get("client_timing", {})
    fields = [
        ("client", client_timing.get("websocket_infer_ms")),
        ("server", server_timing.get("infer_ms")),
        ("ready", policy_timing.get("action_ready_ms")),
        ("planner", policy_timing.get("subtask_planner_ms")),
        ("tokenize", policy_timing.get("observation_tokenize_ms")),
        ("vlm", model_timing.get("vlm_prefix_forward_ms")),
        ("flow", model_timing.get("flow_denoise_ms")),
        ("steps", model_timing.get("flow_denoise_steps")),
        ("out", policy_timing.get("output_transform_ms")),
    ]
    parts = []
    for name, value in fields:
        if value is None:
            continue
        if name == "steps":
            parts.append(f"{name}={int(value)}")
        else:
            parts.append(f"{name}={float(value):.1f}ms")
    return " ".join(parts)


class G1DualHandsController(Node):
    """Coordinate the control loop and asynchronous inference."""

    def __init__(self, args: Args) -> None:
        super().__init__("g1_pi07_async_client")
        self.args = args
        self.layout = build_robot_layout(args)
        self.ros_io = G1RosIO(self, args, self.layout)
        self.policy = AsyncPolicyProcess(remote_host=args.remote_host, remote_port=args.remote_port)
        self.policy.start()

        self.log_dir = Path(args.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.execution_queue: deque[np.ndarray] = deque()
        self.last_action: np.ndarray | None = None
        self.inference_idx = 0
        self.last_request_time = 0.0
        self.submit_next_after_result = False
        self.request_started_ns: dict[int, int] = {}
        self.last_compute_window: tuple[int, int] | None = None
        self.policy_failed = False

        self.create_timer(1.0 / args.control_hz, self.control_loop)
        self.get_logger().info(f"Connecting to policy server: {args.remote_host}:{args.remote_port}")
        self.get_logger().info(f"Policy action dimension: {STATE_DIM}; live commands={args.enable_robot_commands}")
        self.get_logger().info(f"Image message type: {COMPRESSED_IMAGE_MSG.__name__}")
        self.get_logger().info(
            f"Control rate {args.control_hz} Hz, policy action rate {args.policy_action_hz} Hz, "
            f"interpolation {args.interpolation}, open_loop_horizon={args.open_loop_horizon}"
        )

    def destroy_node(self) -> bool:
        """Stop the inference process when the node is destroyed."""
        self.policy.stop()
        return super().destroy_node()

    def control_loop(self) -> None:
        """Run one control tick without blocking on inference."""
        if not self.ros_io.is_ready():
            return

        self._poll_policy_result()
        self._maybe_submit_inference()
        self._publish_next_action()

    def _poll_policy_result(self) -> None:
        """Consume an asynchronous result and build a control-rate execution queue."""
        result = self.policy.poll_latest()
        if result is None:
            return
        if not result.get("ok", False):
            request_id = int(result.get("request_id", -1))
            self.request_started_ns.pop(request_id, None)
            self.policy_failed = request_id < 0
            self.get_logger().error(f"Inference process error:\n{result.get('error')}")
            return
        finished_ns = self.get_clock().now().nanoseconds
        started_ns = self.request_started_ns.pop(int(result["request_id"]), finished_ns)
        self.last_compute_window = (started_ns, finished_ns)

        try:
            plan = make_execution_plan(
                result["actions"],
                state_dim=STATE_DIM,
                max_action_chunk_len=self.args.max_action_chunk_len,
                policy_action_hz=self.args.policy_action_hz,
                control_hz=self.args.control_hz,
                interpolation=self.args.interpolation,
                lower_limits=self.layout.lower_limits,
                upper_limits=self.layout.upper_limits,
            )
        except (KeyError, TypeError, ValueError) as exc:
            self.policy_failed = True
            self.execution_queue.clear()
            self.get_logger().error(f"Invalid model action chunk; entering hold mode: {exc}")
            return
        execute_len = min(len(plan), self.args.open_loop_horizon)
        self.execution_queue = deque(plan[:execute_len])
        if self.args.save_action_chunks:
            np.savez_compressed(
                self.log_dir / f"async_action_chunk_{self.inference_idx:06d}.npz",
                raw_actions=result["actions"],
                execution_plan=plan,
                policy_timing=result.get("policy_timing", {}),
                model_timing=result.get("model_timing", {}),
                server_timing=result.get("server_timing", {}),
                client_timing=result.get("client_timing", {}),
                state=self.ros_io.get_state(),
                timestamp=time.time(),
            )
        self.submit_next_after_result = True
        self.get_logger().info(
            f"Received action chunk #{self.inference_idx}: raw={result['actions'].shape}, "
            f"plan={plan.shape}, execute={execute_len} timing=[{_format_timing(result)}]"
        )
        self.inference_idx += 1

    def _maybe_submit_inference(self) -> None:
        """Submit the latest observation before the execution queue is exhausted."""
        if self.policy.inflight:
            return
        if self.policy_failed:
            return
        should_submit_after_chunk = self.args.request_immediately_after_chunk and self.submit_next_after_result
        if not should_submit_after_chunk and len(self.execution_queue) > self.args.request_when_remaining_steps:
            return
        build_start = time.monotonic()
        observation = self.ros_io.build_observation(self.args.prompt)
        build_ms = (time.monotonic() - build_start) * 1000
        request_id = self.policy.submit_latest(observation)
        self.request_started_ns[request_id] = self.get_clock().now().nanoseconds
        self.last_request_time = time.time()
        self.submit_next_after_result = False
        reason = "chunk_ready" if should_submit_after_chunk else f"remaining={len(self.execution_queue)}"
        self.get_logger().info(
            f"Submitted asynchronous inference request #{request_id}: "
            f"reason={reason}, build_observation={build_ms:.1f} ms",
            throttle_duration_sec=1.0,
        )

    def _publish_next_action(self) -> None:
        """Publish the next action at the configured rate or hold the current position."""
        if self.execution_queue:
            action = self.execution_queue.popleft()
            self.last_action = action
        elif self.last_action is not None:
            action = self.last_action
        else:
            action = self.ros_io.get_state()
            self.last_action = action
        window = self.last_compute_window
        self.ros_io.publish_action(
            action,
            compute_started_ns=None if window is None else window[0],
            compute_finished_ns=None if window is None else window[1],
        )


class G1DualHandsRtcController(Node):
    """RTC controller with a control loop, separate inference process, and hard-prefix conditioning."""

    def __init__(self, args: Args) -> None:
        super().__init__("g1_pi07_rtc_client")
        self.args = args
        self.layout = build_robot_layout(args)
        self.ros_io = G1RosIO(self, args, self.layout)
        self.policy = AsyncPolicyProcess(remote_host=args.remote_host, remote_port=args.remote_port)
        self.policy.start()
        self.rtc = RtcChunker(
            horizon=args.max_action_chunk_len,
            min_horizon=args.rtc_min_horizon,
            delay_buffer_size=args.rtc_delay_buffer_size,
            initial_delay_steps=args.rtc_initial_delay_steps,
        )

        self.log_dir = Path(args.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.last_action: np.ndarray | None = None
        self.inference_idx = 0
        self.request_started_ns: dict[int, int] = {}
        self.last_compute_window: tuple[int, int] | None = None
        self.policy_failed = False

        self.create_timer(1.0 / args.control_hz, self.control_loop)
        self.get_logger().info(f"Connecting to policy server: {args.remote_host}:{args.remote_port}")
        self.get_logger().info(f"Policy action dimension: {STATE_DIM}; live commands={args.enable_robot_commands}")
        self.get_logger().info(f"Image message type: {COMPRESSED_IMAGE_MSG.__name__}")
        self.get_logger().info(
            "Training-time RTC mode: the control loop does not wait for inference and uses "
            "latency estimates with the previous chunk as a hard prefix. "
            f"Execution plan length={args.max_action_chunk_len}, "
            f"minimum replanning interval={args.rtc_min_horizon}; "
            "the model samples with rtc_prefix conditioning."
        )

    def destroy_node(self) -> bool:
        """Stop the inference process when the node is destroyed."""
        self.policy.stop()
        return super().destroy_node()

    def control_loop(self) -> None:
        """Run one RTC control-loop tick."""
        if not self.ros_io.is_ready():
            return

        self._poll_policy_result()
        self._maybe_submit_inference()
        self._publish_next_action()
        self.rtc.record_control_step()

    def _poll_policy_result(self) -> None:
        """Interpolate an inference result and pass it to the RTC chunker for prefix alignment."""
        result = self.policy.poll_latest()
        if result is None:
            return
        if not result.get("ok", False):
            request_id = int(result.get("request_id", -1))
            self.request_started_ns.pop(request_id, None)
            if request_id < 0 and self.rtc.inflight_context is not None:
                self.rtc.cancel_request(self.rtc.inflight_context.request_id)
                self.policy_failed = True
            else:
                self.rtc.cancel_request(request_id)
            self.get_logger().error(f"Inference process error:\n{result.get('error')}")
            return
        finished_ns = self.get_clock().now().nanoseconds
        started_ns = self.request_started_ns.pop(int(result["request_id"]), finished_ns)
        self.last_compute_window = (started_ns, finished_ns)

        try:
            plan = make_execution_plan(
                result["actions"],
                state_dim=STATE_DIM,
                max_action_chunk_len=self.args.max_action_chunk_len,
                policy_action_hz=self.args.policy_action_hz,
                control_hz=self.args.control_hz,
                interpolation=self.args.interpolation,
                lower_limits=self.layout.lower_limits,
                upper_limits=self.layout.upper_limits,
            )
        except (KeyError, TypeError, ValueError) as exc:
            request_id = int(result.get("request_id", -1))
            self.rtc.cancel_request(request_id)
            self.policy_failed = True
            self.get_logger().error(f"Invalid RTC action chunk; entering hold mode: {exc}")
            return
        try:
            merged_plan, observed_delay = self.rtc.accept_new_chunk(result["request_id"], plan)
        except ValueError as exc:
            self.get_logger().warning(f"Discarding stale or invalid RTC result: {exc}")
            return
        if self.args.save_action_chunks:
            np.savez_compressed(
                self.log_dir / f"rtc_action_chunk_{self.inference_idx:06d}.npz",
                raw_actions=result["actions"],
                execution_plan=plan,
                merged_plan=merged_plan,
                observed_delay_steps=observed_delay,
                policy_timing=result.get("policy_timing", {}),
                model_timing=result.get("model_timing", {}),
                server_timing=result.get("server_timing", {}),
                client_timing=result.get("client_timing", {}),
                state=self.ros_io.get_state(),
                timestamp=time.time(),
            )
        self.get_logger().info(
            f"RTC received chunk #{self.inference_idx}: raw={result['actions'].shape}, "
            f"plan={plan.shape}, merged={merged_plan.shape}, delay_steps={observed_delay} "
            f"timing=[{_format_timing(result)}]"
        )
        self.inference_idx += 1

    def _maybe_submit_inference(self) -> None:
        """Submit the next inference request when RTC conditions are satisfied."""
        if self.policy_failed or self.policy.inflight or not self.rtc.should_request():
            return
        build_start = time.monotonic()
        observation = self.ros_io.build_observation(self.args.prompt)
        build_ms = (time.monotonic() - build_start) * 1000
        prefix = self.rtc.make_action_prefix()
        if prefix is not None:
            action_prefix, delay = prefix
            observation["rtc_prefix"] = {
                "action_prefix": action_prefix,
                "delay": delay,
            }
        request_id = self.policy.submit_latest(observation)
        self.request_started_ns[request_id] = self.get_clock().now().nanoseconds
        context = self.rtc.make_request_context(request_id)
        self.get_logger().info(
            f"RTC submitted inference request #{request_id}: s={context.executed_since_swap}, "
            f"d_est={context.delay_estimate_steps}, "
            f"prefix_steps={0 if prefix is None else prefix[1]}, suffix={context.previous_suffix.shape}, "
            f"build_observation={build_ms:.1f}ms",
            throttle_duration_sec=1.0,
        )

    def _publish_next_action(self) -> None:
        """Publish the current RTC action."""
        fallback = self.last_action if self.last_action is not None else self.ros_io.get_state()
        action = self.rtc.consume_action(fallback)
        self.last_action = action
        window = self.last_compute_window
        self.ros_io.publish_action(
            action,
            compute_started_ns=None if window is None else window[0],
            compute_finished_ns=None if window is None else window[1],
        )


def main(args: Args) -> None:
    args = merge_config_file(args)
    os.environ.setdefault("ROS_DOMAIN_ID", args.ros_domain_id)
    rclpy.init()
    match args.controller_mode:
        case "async":
            node = G1DualHandsController(args)
        case "rtc":
            node = G1DualHandsRtcController(args)
        case _:
            raise ValueError("controller_mode must be 'async' or 'rtc'")
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
