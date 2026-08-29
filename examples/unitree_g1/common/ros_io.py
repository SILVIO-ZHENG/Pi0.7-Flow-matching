# SPDX-License-Identifier: Apache-2.0
"""ROS2 policy observation and command adapter for the canonical G1 messages."""

from __future__ import annotations

from builtin_interfaces.msg import Time
import cv2
from g1_pi07_interfaces.msg import ActionTarget43
from g1_pi07_interfaces.msg import RobotState43
import numpy as np
from openpi_client import image_tools
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage

from examples.unitree_g1.common.g1_config import MODEL_IMAGE_SIZE
from examples.unitree_g1.common.g1_config import Args
from examples.unitree_g1.common.g1_config import RobotLayout
from g1_pi07.joints import DEFAULT_LAYOUT

# Exposed for startup logging and dependency diagnostics in deployment clients.
COMPRESSED_IMAGE_MSG = CompressedImage


def _time_from_ns(timestamp_ns: int):
    return Time(sec=int(timestamp_ns // 1_000_000_000), nanosec=int(timestamp_ns % 1_000_000_000))


def decode_compressed_image(message: CompressedImage) -> np.ndarray | None:
    if not message.data:
        return None
    bgr = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE))


class G1RosIO:
    """Own ROS subscriptions and publish canonical 28-D policy actions.

    Observations are considered ready only when the robot state and all three
    images satisfy configured freshness bounds. Published actions are clipped
    to the validated layout before expansion into ``ActionTarget43``.
    """

    def __init__(self, node: Node, args: Args, layout: RobotLayout) -> None:
        self._node = node
        self._args = args
        self._layout = layout
        # The latest state and per-camera stamps form one freshness-gated snapshot.
        self._latest_state: RobotState43 | None = None
        self._images: dict[str, np.ndarray | None] = {"head": None, "left_wrist": None, "right_wrist": None}
        self._image_stamps_ns: dict[str, int | None] = {"head": None, "left_wrist": None, "right_wrist": None}
        # Command steps are monotonically increasing within one client episode.
        self._command_step = 0
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        command_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20,
        )
        node.create_subscription(RobotState43, args.state_topic, self._state_callback, sensor_qos)
        for name, topic in (
            ("head", args.head_image_topic),
            ("left_wrist", args.left_wrist_image_topic),
            ("right_wrist", args.right_wrist_image_topic),
        ):
            node.create_subscription(
                CompressedImage,
                topic,
                lambda message, camera=name: self._image_callback(camera, message),
                sensor_qos,
            )
        self._command_publisher = node.create_publisher(ActionTarget43, args.action_topic, command_qos)

    def _state_callback(self, message: RobotState43) -> None:
        if tuple(message.name) != DEFAULT_LAYOUT.full_joint_names:
            self._node.get_logger().error("RobotState43 joint order mismatch; state rejected")
            return
        self._latest_state = message

    def _image_callback(self, name: str, message: CompressedImage) -> None:
        image = decode_compressed_image(message)
        if image is not None:
            self._images[name] = image
            self._image_stamps_ns[name] = int(message.header.stamp.sec) * 1_000_000_000 + int(
                message.header.stamp.nanosec
            )

    def is_ready(self) -> bool:
        now_ns = self._node.get_clock().now().nanoseconds
        state = self._latest_state
        state_stamp_ns = (
            -1 if state is None else int(state.header.stamp.sec) * 1_000_000_000 + int(state.header.stamp.nanosec)
        )
        state_ready = (
            state is not None
            and all(state.validity_mask)
            and np.isfinite(np.asarray(state.position, dtype=np.float64)).all()
            and abs(now_ns - state_stamp_ns) <= int(self._args.max_state_age_ms * 1e6)
        )
        images_ready = all(
            self._images[name] is not None
            and stamp is not None
            and abs(now_ns - stamp) <= int(self._args.max_image_age_ms * 1e6)
            for name, stamp in self._image_stamps_ns.items()
        )
        if not state_ready or not images_ready:
            self._node.get_logger().info(
                f"Waiting for data: state={state_ready}, images={images_ready}",
                throttle_duration_sec=2.0,
            )
        return state_ready and images_ready

    def get_full_state(self) -> np.ndarray:
        if self._latest_state is None:
            raise RuntimeError("RobotState43 has not been received")
        return np.asarray(self._latest_state.position, dtype=np.float32)

    def get_state(self) -> np.ndarray:
        state = DEFAULT_LAYOUT.full_to_policy(self.get_full_state())
        return np.clip(state, self._layout.lower_limits, self._layout.upper_limits)

    def build_observation(self, prompt: str) -> dict:
        return {
            "observation/head_image": self._images["head"],
            "observation/left_wrist_image": self._images["left_wrist"],
            "observation/right_wrist_image": self._images["right_wrist"],
            "observation/state": self.get_state(),
            "prompt": prompt,
            "plan_subtask": self._args.plan_subtask,
        }

    def publish_action(
        self,
        action: np.ndarray,
        *,
        compute_started_ns: int | None = None,
        compute_finished_ns: int | None = None,
    ) -> None:
        policy = np.asarray(action, dtype=np.float32)
        if policy.shape != (28,):
            raise ValueError(f"Policy action must have 28 dimensions; got {policy.shape}")
        if not np.isfinite(policy).all():
            raise ValueError("Policy action contains NaN or Inf")
        policy = np.clip(policy, self._layout.lower_limits, self._layout.upper_limits)
        if not self._args.enable_robot_commands:
            self._node.get_logger().warning(
                "action computed; robot publishing is disabled by configuration",
                throttle_duration_sec=2.0,
            )
            return
        state = self._latest_state
        if state is None:
            return
        message = ActionTarget43()
        now_value = self._node.get_clock().now()
        now = now_value.to_msg()
        now_ns = now_value.nanoseconds
        message.header.stamp = now
        message.episode_id = self._args.episode_id
        message.step_index = self._command_step
        message.source_state_sequence = state.sequence
        message.source_state_stamp = state.header.stamp
        message.compute_started = _time_from_ns(compute_started_ns if compute_started_ns is not None else now_ns)
        message.compute_finished = _time_from_ns(compute_finished_ns if compute_finished_ns is not None else now_ns)
        message.sent = now
        message.source = "pi07_policy"
        message.name = list(DEFAULT_LAYOUT.full_joint_names)
        message.position = DEFAULT_LAYOUT.policy_to_full(policy, base_full=self.get_full_state()).tolist()
        validity = np.zeros(43, dtype=np.bool_)
        validity[DEFAULT_LAYOUT.policy_indices] = True
        message.validity_mask = validity.tolist()
        self._command_publisher.publish(message)
        self._command_step += 1
