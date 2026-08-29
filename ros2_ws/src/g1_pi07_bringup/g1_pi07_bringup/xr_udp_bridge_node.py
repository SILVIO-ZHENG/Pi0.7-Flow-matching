"""Validated UDP/JSON bridge from a vendor XR client to XRHandTargets.

The bridge intentionally keeps the wire format small and vendor neutral.  A
Quest, PICO or Vive client can send the same wrist/keypoint contract while this
node owns timestamping and the calibrated XR-tracking-frame -> G1 torso-frame
rigid transform.
"""

from __future__ import annotations

import json
import socket

from builtin_interfaces.msg import Time
from g1_pi07_interfaces.msg import XRHandTargets
import numpy as np
import rclpy
from rclpy.node import Node

from g1_pi07.teleop.xr import RigidXrCalibration


def _time_from_ns(timestamp_ns: int) -> Time:
    return Time(sec=int(timestamp_ns // 1_000_000_000), nanosec=int(timestamp_ns % 1_000_000_000))


class XrUdpBridgeNode(Node):
    """Validate bounded XR datagrams and publish calibrated ROS2 targets.

    Monotonic sequence checks reject replayed samples. Source timestamps are used
    only when explicitly enabled and calibrated within the configured clock skew.
    """

    def __init__(self) -> None:
        super().__init__("xr_udp_bridge")
        defaults = {
            "bind_host": "127.0.0.1",
            "bind_port": 7001,
            "output_topic": "/xr/hand_targets",
            "frame_id": "torso_link",
            "max_packet_bytes": 65535,
            "max_packets_per_tick": 32,
            "use_source_timestamp": False,
            "source_clock_offset_ns": 0,
            "max_clock_skew_ms": 100.0,
            "position_scale": 1.0,
            "xr_to_robot_translation": [0.0, 0.0, 0.0],
            "xr_to_robot_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        translation = np.asarray(self.get_parameter("xr_to_robot_translation").value, dtype=np.float64)
        quaternion = np.asarray(
            self.get_parameter("xr_to_robot_quaternion_xyzw").value,
            dtype=np.float64,
        )
        self._calibration = RigidXrCalibration(
            translation,
            quaternion,
            float(self.get_parameter("position_scale").value),
        )

        self._max_packet_bytes = int(self.get_parameter("max_packet_bytes").value)
        self._max_packets_per_tick = int(self.get_parameter("max_packets_per_tick").value)
        if self._max_packet_bytes <= 0 or self._max_packets_per_tick <= 0:
            raise ValueError("UDP packet limits must be greater than zero")
        bind_port = int(self.get_parameter("bind_port").value)
        if not 1 <= bind_port <= 65535:
            raise ValueError("bind_port must be in 1..65535")
        max_clock_skew_ms = float(self.get_parameter("max_clock_skew_ms").value)
        if not np.isfinite(max_clock_skew_ms) or max_clock_skew_ms < 0:
            raise ValueError("max_clock_skew_ms must be finite and non-negative")
        self._max_clock_skew_ns = int(max_clock_skew_ms * 1e6)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)  # noqa: FBT003 - socket API requires a positional flag.
        self._socket.bind((str(self.get_parameter("bind_host").value), bind_port))
        self._publisher = self.create_publisher(
            XRHandTargets,
            str(self.get_parameter("output_topic").value),
            20,
        )
        # A strictly increasing sequence protects the control path from UDP replay.
        self._last_sequence = -1
        self.create_timer(0.002, self._drain_socket)
        self.get_logger().info(
            f"XR UDP bridge listening on {self.get_parameter('bind_host').value}:"
            f"{self.get_parameter('bind_port').value}"
        )

    def _drain_socket(self) -> None:
        for _ in range(self._max_packets_per_tick):
            try:
                payload, _address = self._socket.recvfrom(self._max_packet_bytes + 1)
            except BlockingIOError:
                return
            if len(payload) > self._max_packet_bytes:
                self.get_logger().warning("XR UDP packet exceeds max_packet_bytes and was dropped")
                continue
            try:
                record = json.loads(payload.decode("utf-8"))
                message = self._to_message(record)
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                self.get_logger().warning(f"Invalid XR packet: {exc}", throttle_duration_sec=1.0)
                continue
            self._publisher.publish(message)

    def _to_message(self, record: dict) -> XRHandTargets:
        if not isinstance(record, dict):
            raise ValueError("XR packet root must be a JSON object")
        raw_sequence = record["sequence"]
        if isinstance(raw_sequence, bool) or not isinstance(raw_sequence, int):
            raise ValueError("sequence must be a JSON integer")
        sequence = raw_sequence
        if sequence <= self._last_sequence:
            raise ValueError(f"XR sequence is not increasing: {sequence} <= {self._last_sequence}")
        raw_timestamp_ns = record.get("timestamp_ns", 0)
        if isinstance(raw_timestamp_ns, bool) or not isinstance(raw_timestamp_ns, int):
            raise ValueError("timestamp_ns must be a JSON integer")
        source_timestamp_ns = raw_timestamp_ns
        if source_timestamp_ns < 0:
            raise ValueError("timestamp_ns must not be negative")

        now_ns = self.get_clock().now().nanoseconds
        # ROS receipt time is the default clock unless source time is calibrated.
        stamp_ns = now_ns
        if bool(self.get_parameter("use_source_timestamp").value):
            if source_timestamp_ns <= 0:
                raise ValueError("A positive timestamp_ns is required when use_source_timestamp=true")
            stamp_ns = source_timestamp_ns + int(self.get_parameter("source_clock_offset_ns").value)
            if stamp_ns <= 0:
                raise ValueError("Calibrated XR timestamp must be positive")
            if abs(now_ns - stamp_ns) > self._max_clock_skew_ns:
                raise ValueError("Calibrated XR source timestamp differs too much from the ROS clock")

        message = XRHandTargets()
        message.header.stamp = _time_from_ns(stamp_ns)
        message.header.frame_id = str(self.get_parameter("frame_id").value)
        message.sequence = sequence
        message.source_timestamp_ns = source_timestamp_ns
        self._fill_hand(message, record["left"], side="left")
        self._fill_hand(message, record["right"], side="right")
        self._last_sequence = sequence
        return message

    def _fill_hand(self, message: XRHandTargets, hand: dict, *, side: str) -> None:
        if not isinstance(hand, dict):
            raise ValueError(f"{side} must be a JSON object")
        valid = hand.get("valid", False)
        if not isinstance(valid, bool):
            raise ValueError(f"{side}.valid must be a JSON boolean")
        setattr(message, f"{side}_valid", valid)
        if not valid:
            return
        wrist = hand["wrist"]
        position = np.asarray(wrist["position"], dtype=np.float64)
        orientation = np.asarray(wrist["quaternion_xyzw"], dtype=np.float64)
        keypoints = np.asarray(hand["keypoints"], dtype=np.float64)
        if position.shape != (3,) or keypoints.shape != (21, 3):
            raise ValueError(f"{side} wrist.position must have shape [3] and keypoints must have shape [21,3]")
        if not np.isfinite(position).all() or not np.isfinite(keypoints).all():
            raise ValueError(f"{side} position/keypoints contain NaN or Inf")
        transformed_position, transformed_orientation = self._calibration.transform_pose(position, orientation)
        transformed_keypoints = self._calibration.transform_points(keypoints)
        pose = getattr(message, f"{side}_wrist")
        pose.position.x, pose.position.y, pose.position.z = transformed_position.tolist()
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = transformed_orientation.tolist()
        setattr(message, f"{side}_hand_keypoints", transformed_keypoints.astype(np.float32).reshape(-1).tolist())

    def destroy_node(self) -> bool:
        self._socket.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = XrUdpBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
