"""Action-centred recorder that derives aligned MP4/Parquet from ROS topics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
from g1_pi07.data.storage import EpisodeWriter
from g1_pi07.data.time_sync import ActionCentricSynchronizer
from g1_pi07.data.time_sync import AlignmentError
from g1_pi07.data.types import ActionTargetFrame
from g1_pi07.data.types import CameraFrame
from g1_pi07.data.types import RobotStateFrame
from g1_pi07.joints import DEFAULT_LAYOUT
from g1_pi07_interfaces.msg import ActionTarget43
from g1_pi07_interfaces.msg import RobotState43
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import JointState


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _imu_vector(imu) -> np.ndarray:
    return np.asarray(
        [
            imu.orientation.x,
            imu.orientation.y,
            imu.orientation.z,
            imu.orientation.w,
            imu.angular_velocity.x,
            imu.angular_velocity.y,
            imu.angular_velocity.z,
            imu.linear_acceleration.x,
            imu.linear_acceleration.y,
            imu.linear_acceleration.z,
        ],
        dtype=np.float32,
    )


def _parse_success(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"", "unknown", "none", "null"}:
        return None
    if normalized in {"true", "1", "yes", "success"}:
        return True
    if normalized in {"false", "0", "no", "failure", "failed"}:
        return False
    raise ValueError("success_label must be unknown, true, or false")


@dataclass(frozen=True)
class PendingAction:
    frame: ActionTargetFrame
    received_ns: int


class EpisodeRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("episode_recorder")
        defaults = {
            "state_topic": "/robot/state43",
            "action_topic": "/control/action_target43",
            "command_topic": "/g1/upper_body_position_command",
            "head_camera_topic": "/camera/head/image_raw/compressed",
            "left_wrist_camera_topic": "/camera/left_wrist/image_raw/compressed",
            "right_wrist_camera_topic": "/camera/right_wrist/image_raw/compressed",
            "output_root": "./data/raw",
            "episode_id": "replace_me",
            "task": "",
            "subtask": "",
            "success_label": "unknown",
            "failure_reason": "",
            "fps": 20.0,
            "max_state_delta_ms": 15.0,
            "max_camera_delta_ms": 40.0,
            "alignment_wait_ms": 60.0,
            "max_pending_actions": 256,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        episode_id = str(self.get_parameter("episode_id").value)
        if not episode_id or episode_id == "replace_me":
            raise ValueError("A unique episode_id must be set through parameters")
        output_root = Path(str(self.get_parameter("output_root").value))
        existing_episode = output_root / episode_id
        if any((existing_episode / name).exists() for name in ("metadata.json", "steps.parquet", "videos")):
            raise FileExistsError(f"Derived data already exists for episode_id; use a new ID: {existing_episode}")
        self._writer = EpisodeWriter(
            output_root,
            episode_id,
            fps=float(self.get_parameter("fps").value),
            task=str(self.get_parameter("task").value),
            subtask=str(self.get_parameter("subtask").value),
        )
        self._sync = ActionCentricSynchronizer(
            max_state_delta_ms=float(self.get_parameter("max_state_delta_ms").value),
            max_camera_delta_ms=float(self.get_parameter("max_camera_delta_ms").value),
        )
        self._dropped = 0
        self._pending_actions: deque[PendingAction] = deque()
        self._applied_commands: dict[int, np.ndarray] = {}
        self._command_stamps: deque[int] = deque()
        self._alignment_wait_ns = int(float(self.get_parameter("alignment_wait_ms").value) * 1e6)
        self._max_pending_actions = int(self.get_parameter("max_pending_actions").value)
        if self._alignment_wait_ns < 0 or self._max_pending_actions <= 0:
            raise ValueError("alignment_wait_ms must be non-negative and max_pending_actions must be positive")
        self._finalized = False
        self.create_subscription(
            RobotState43,
            self.get_parameter("state_topic").value,
            self._on_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(ActionTarget43, self.get_parameter("action_topic").value, self._on_action, 50)
        self.create_subscription(JointState, self.get_parameter("command_topic").value, self._on_command, 50)
        for name, parameter in (
            ("head", "head_camera_topic"),
            ("left_wrist", "left_wrist_camera_topic"),
            ("right_wrist", "right_wrist_camera_topic"),
        ):
            self.create_subscription(
                CompressedImage,
                self.get_parameter(parameter).value,
                lambda message, camera=name: self._on_image(camera, message),
                qos_profile_sensor_data,
            )
        self.create_timer(0.01, self._flush_pending)

    def _on_state(self, message: RobotState43) -> None:
        try:
            frame = RobotStateFrame(
                _stamp_ns(message.header.stamp),
                int(message.sequence),
                np.asarray(message.position),
                np.asarray(message.velocity),
                np.asarray(message.effort),
                _imu_vector(message.imu),
                np.asarray(message.validity_mask),
                bool(message.imu_valid),
            )
        except ValueError as exc:
            self.get_logger().warning(f"Rejected invalid RobotState43: {exc}")
            return
        self._sync.push_state(frame)

    def _on_image(self, camera: str, message: CompressedImage) -> None:
        bgr = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warning(f"Failed to decode {camera} image")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._sync.push_camera(CameraFrame(_stamp_ns(message.header.stamp), camera, rgb))

    def _on_command(self, message: JointState) -> None:
        if tuple(message.name) != DEFAULT_LAYOUT.policy_joint_names:
            self.get_logger().warning("Approved command has an invalid joint order and cannot be used as a label")
            return
        action = np.asarray(message.position, dtype=np.float32)
        if action.shape != (28,) or not np.isfinite(action).all():
            self.get_logger().warning("Approved command is not a finite 28-dimensional array")
            return
        stamp_ns = _stamp_ns(message.header.stamp)
        if stamp_ns <= 0:
            self.get_logger().warning("Approved command lacks a valid timestamp")
            return
        if stamp_ns not in self._applied_commands:
            self._command_stamps.append(stamp_ns)
        self._applied_commands[stamp_ns] = action
        while len(self._command_stamps) > self._max_pending_actions:
            expired = self._command_stamps.popleft()
            self._applied_commands.pop(expired, None)

    def _on_action(self, message: ActionTarget43) -> None:
        try:
            action = ActionTargetFrame(
                timestamp_ns=_stamp_ns(message.header.stamp),
                episode_id=message.episode_id,
                step_index=int(message.step_index),
                source_state_sequence=int(message.source_state_sequence),
                position=np.asarray(message.position),
                compute_started_ns=_stamp_ns(message.compute_started),
                compute_finished_ns=_stamp_ns(message.compute_finished),
                sent_ns=_stamp_ns(message.sent),
                source=message.source,
                validity_mask=np.asarray(message.validity_mask),
            )
        except ValueError as exc:
            self._dropped += 1
            self.get_logger().warning(f"Rejected invalid ActionTarget43: {exc}; dropped={self._dropped}")
            return
        self._pending_actions.append(PendingAction(action, self.get_clock().now().nanoseconds))
        if len(self._pending_actions) > self._max_pending_actions:
            pending = self._pending_actions.popleft()
            self.get_logger().warning("Action-alignment queue overflow; processing the oldest action immediately")
            self._record_action(pending.frame)

    def _record_action(self, action: ActionTargetFrame) -> None:
        try:
            aligned = self._sync.align(action)
        except AlignmentError as exc:
            self._dropped += 1
            self.get_logger().warning(f"Dropped unaligned action: {exc}; dropped={self._dropped}")
            return
        applied_action = self._applied_commands.pop(action.timestamp_ns, None)
        try:
            self._writer.add(aligned, applied_policy_action=applied_action)
        except ValueError as exc:
            self._dropped += 1
            self.get_logger().warning(f"Rejected invalid aligned step: {exc}; dropped={self._dropped}")

    def _flush_pending(self, *, force: bool = False) -> None:
        """Delay alignment so a slightly newer camera frame can enter the buffer."""

        now_ns = self.get_clock().now().nanoseconds
        while self._pending_actions:
            pending = self._pending_actions[0]
            if not force and now_ns - pending.received_ns < self._alignment_wait_ns:
                break
            self._pending_actions.popleft()
            self._record_action(pending.frame)

    def destroy_node(self) -> bool:
        if self._finalized:
            return True
        self._finalized = True
        self._flush_pending(force=True)
        if self._writer.rows:
            try:
                success = _parse_success(str(self.get_parameter("success_label").value))
            except ValueError as exc:
                self.get_logger().error(f"Final success_label is invalid and will be saved as unknown: {exc}")
                success = None
            failure_reason = str(self.get_parameter("failure_reason").value)
            for row in self._writer.rows:
                row["success"] = success
                row["failure_reason"] = failure_reason if success is False else ""
            episode_dir = self._writer.output_root / self._writer.episode_id
            self._writer.finalize(source_mcap=str(episode_dir / "raw_mcap"), overwrite=True)
            self.get_logger().info(f"Wrote {len(self._writer.rows)} steps and dropped {self._dropped} steps")
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = EpisodeRecorderNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
