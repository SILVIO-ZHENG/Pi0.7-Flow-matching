"""XR wrist poses -> MoveIt IK, hand keypoints -> Dex3, then ActionTarget43."""

from __future__ import annotations

from dataclasses import dataclass
import math

from builtin_interfaces.msg import Time
from g1_pi07_interfaces.msg import ActionTarget43
from g1_pi07_interfaces.msg import RobotState43
from g1_pi07_interfaces.msg import XRHandTargets
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from g1_pi07.joints import DEFAULT_LAYOUT
from g1_pi07.teleop.retarget import Dex3Calibration
from g1_pi07.teleop.retarget import Dex3Retargeter


def _now_message(node: Node) -> Time:
    return node.get_clock().now().to_msg()


def _stamp_ns(stamp: Time) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


@dataclass
class PendingIK:
    """Paired MoveIt requests and immutable inputs for one XR control sample."""

    left_future: object
    right_future: object
    state: RobotState43
    xr: XRHandTargets
    started_ns: int


class TeleopActionNode(Node):
    """Convert fresh bimanual XR targets into canonical robot action messages.

    Only one dual-arm IK pair is in flight. New XR samples replace the queued
    sample so delayed IK results cannot build an unbounded command backlog.
    """

    def __init__(self) -> None:
        super().__init__("teleop_action")
        parameters = {
            "state_topic": "/robot/state43",
            "xr_topic": "/xr/hand_targets",
            "action_topic": "/control/action_target43",
            "ik_service": "/compute_ik",
            "ik_frame": "torso_link",
            "left_group": "left_arm",
            "right_group": "right_arm",
            "ik_timeout_s": 0.08,
            "max_state_age_ms": 100.0,
            "max_xr_age_ms": 100.0,
            "episode_id": "teleop",
            "left_hand_lower": [0.0] * 7,
            "left_hand_upper": [1.0] * 7,
            "left_hand_invert": [False] * 7,
            "right_hand_lower": [0.0] * 7,
            "right_hand_upper": [1.0] * 7,
            "right_hand_invert": [False] * 7,
        }
        for name, value in parameters.items():
            self.declare_parameter(name, value)
        timing_values = np.asarray(
            [
                float(self.get_parameter("ik_timeout_s").value),
                float(self.get_parameter("max_state_age_ms").value),
                float(self.get_parameter("max_xr_age_ms").value),
            ],
            dtype=np.float64,
        )
        if not np.isfinite(timing_values).all() or np.any(timing_values <= 0):
            raise ValueError("ik_timeout_s, max_state_age_ms, and max_xr_age_ms must be positive and finite")
        self._state: RobotState43 | None = None
        self._latest_xr: XRHandTargets | None = None
        # ``_pending`` owns both arm futures; they are accepted or rejected together.
        self._pending: PendingIK | None = None
        self._step = 0
        self._left_retargeter = self._make_retargeter("left_hand")
        self._right_retargeter = self._make_retargeter("right_hand")
        self._ik_client = self.create_client(GetPositionIK, self.get_parameter("ik_service").value)
        self._publisher = self.create_publisher(ActionTarget43, self.get_parameter("action_topic").value, 20)
        self.create_subscription(
            RobotState43,
            self.get_parameter("state_topic").value,
            self._on_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            XRHandTargets,
            self.get_parameter("xr_topic").value,
            self._on_xr,
            qos_profile_sensor_data,
        )
        self.create_timer(0.005, self._poll_ik)

    def _make_retargeter(self, prefix: str) -> Dex3Retargeter:
        return Dex3Retargeter(
            Dex3Calibration(
                np.asarray(self.get_parameter(f"{prefix}_lower").value, dtype=np.float32),
                np.asarray(self.get_parameter(f"{prefix}_upper").value, dtype=np.float32),
                np.asarray(self.get_parameter(f"{prefix}_invert").value, dtype=np.bool_),
            )
        )

    def _on_state(self, message: RobotState43) -> None:
        if tuple(message.name) != DEFAULT_LAYOUT.full_joint_names:
            self.get_logger().error("RobotState43 joint order is invalid; teleoperation state rejected")
            return
        position = np.asarray(message.position, dtype=np.float64)
        validity = np.asarray(message.validity_mask, dtype=np.bool_)
        if position.shape != (43,) or validity.shape != (43,) or not validity.all() or not np.isfinite(position).all():
            self.get_logger().warning("RobotState43 is incomplete or non-finite; teleoperation is holding")
            return
        self._state = message

    def _on_xr(self, message: XRHandTargets) -> None:
        self._latest_xr = message
        self._start_latest_if_idle()

    def _start_latest_if_idle(self) -> None:
        if self._pending is not None or self._state is None or self._latest_xr is None:
            return
        if not self._ik_client.service_is_ready():
            self.get_logger().warning("Waiting for the MoveIt /compute_ik service", throttle_duration_sec=2.0)
            return
        xr = self._latest_xr
        self._latest_xr = None
        if not xr.left_valid or not xr.right_valid:
            return
        state = self._state
        now_ns = self.get_clock().now().nanoseconds
        state_age_ns = abs(now_ns - _stamp_ns(state.header.stamp))
        xr_age_ns = abs(now_ns - _stamp_ns(xr.header.stamp))
        if state_age_ns > int(float(self.get_parameter("max_state_age_ms").value) * 1e6):
            self.get_logger().warning("RobotState43 is stale; teleoperation is holding", throttle_duration_sec=1.0)
            return
        if xr_age_ns > int(float(self.get_parameter("max_xr_age_ms").value) * 1e6):
            self.get_logger().warning("XR hand target is stale; teleoperation is holding", throttle_duration_sec=1.0)
            return
        if not self._xr_values_are_valid(xr):
            self.get_logger().warning("XR wrist/keypoints contain invalid values; teleoperation is holding")
            return
        left_request = self._make_request(self.get_parameter("left_group").value, xr.left_wrist, state)
        right_request = self._make_request(self.get_parameter("right_group").value, xr.right_wrist, state)
        self._pending = PendingIK(
            self._ik_client.call_async(left_request),
            self._ik_client.call_async(right_request),
            state,
            xr,
            self.get_clock().now().nanoseconds,
        )

    @staticmethod
    def _xr_values_are_valid(message: XRHandTargets) -> bool:
        values = []
        for wrist, keypoints in (
            (message.left_wrist, message.left_hand_keypoints),
            (message.right_wrist, message.right_hand_keypoints),
        ):
            quaternion = np.asarray(
                [wrist.orientation.x, wrist.orientation.y, wrist.orientation.z, wrist.orientation.w],
                dtype=np.float64,
            )
            values.extend([wrist.position.x, wrist.position.y, wrist.position.z, *quaternion, *keypoints])
            if quaternion.shape != (4,) or np.linalg.norm(quaternion) < 1e-8:
                return False
        return (
            len(message.left_hand_keypoints) == 63
            and len(message.right_hand_keypoints) == 63
            and all(math.isfinite(float(value)) for value in values)
        )

    def _make_request(self, group: str, pose, state: RobotState43) -> GetPositionIK.Request:
        request = GetPositionIK.Request()
        request.ik_request.group_name = str(group)
        request.ik_request.pose_stamped = PoseStamped()
        request.ik_request.pose_stamped.header.frame_id = str(self.get_parameter("ik_frame").value)
        request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        request.ik_request.pose_stamped.pose = pose
        request.ik_request.robot_state.joint_state.name = list(state.name)
        request.ik_request.robot_state.joint_state.position = list(state.position)
        timeout = float(self.get_parameter("ik_timeout_s").value)
        request.ik_request.timeout.sec = int(timeout)
        request.ik_request.timeout.nanosec = int((timeout - int(timeout)) * 1e9)
        request.ik_request.avoid_collisions = True
        return request

    def _poll_ik(self) -> None:
        pending = self._pending
        if pending is None:
            self._start_latest_if_idle()
            return
        timeout_ns = int(float(self.get_parameter("ik_timeout_s").value) * 1e9)
        if self.get_clock().now().nanoseconds - pending.started_ns > timeout_ns:
            self.get_logger().warning("IK timed out; holding the previous robot command")
            self._pending = None
            self._start_latest_if_idle()
            return
        if not pending.left_future.done() or not pending.right_future.done():
            return
        try:
            left = pending.left_future.result()
            right = pending.right_future.result()
        except Exception as exc:  # ROS future transports middleware failures here.
            self.get_logger().error(f"MoveIt IK call failed: {exc}")
            self._pending = None
            self._start_latest_if_idle()
            return
        self._pending = None
        if (
            left is None
            or right is None
            or left.error_code.val != MoveItErrorCodes.SUCCESS
            or right.error_code.val != MoveItErrorCodes.SUCCESS
        ):
            self.get_logger().warning("MoveIt did not find a dual-arm IK solution")
            self._start_latest_if_idle()
            return
        self._publish_solution(pending, left.solution.joint_state, right.solution.joint_state)
        self._start_latest_if_idle()

    def _publish_solution(self, pending: PendingIK, left_state, right_state) -> None:
        if len(left_state.name) != len(left_state.position) or len(right_state.name) != len(right_state.position):
            self.get_logger().error("MoveIt IK solution has mismatched name/position lengths")
            return
        solution = dict(zip(left_state.name, left_state.position, strict=True))
        solution.update(dict(zip(right_state.name, right_state.position, strict=True)))
        # The first 14 policy joints are the two seven-joint arms.
        arm_names = DEFAULT_LAYOUT.policy_joint_names[:14]
        missing = [name for name in arm_names if name not in solution]
        if missing:
            self.get_logger().error(f"MoveIt IK solution is missing joints: {missing}")
            return
        arms = np.asarray([solution[name] for name in arm_names], dtype=np.float32)
        if not np.isfinite(arms).all():
            self.get_logger().error("MoveIt IK solution contains NaN/Inf")
            return
        try:
            left_keypoints = np.asarray(pending.xr.left_hand_keypoints, dtype=np.float32).reshape(21, 3)
            right_keypoints = np.asarray(pending.xr.right_hand_keypoints, dtype=np.float32).reshape(21, 3)
            policy = np.concatenate(
                [
                    arms,
                    self._left_retargeter.retarget(left_keypoints),
                    self._right_retargeter.retarget(right_keypoints),
                ]
            )
        except ValueError as exc:
            self.get_logger().error(f"Invalid Dex3 retargeting input: {exc}")
            return
        full = DEFAULT_LAYOUT.policy_to_full(policy, base_full=np.asarray(pending.state.position))
        message = ActionTarget43()
        message.header.stamp = _now_message(self)
        message.header.frame_id = str(self.get_parameter("ik_frame").value)
        message.episode_id = str(self.get_parameter("episode_id").value)
        message.step_index = self._step
        message.source_state_sequence = pending.state.sequence
        message.source_state_stamp = pending.state.header.stamp
        message.compute_started.sec = pending.started_ns // 1_000_000_000
        message.compute_started.nanosec = pending.started_ns % 1_000_000_000
        message.compute_finished = _now_message(self)
        message.sent = _now_message(self)
        message.source = "xr_teleop"
        message.name = list(DEFAULT_LAYOUT.full_joint_names)
        message.position = full.tolist()
        valid = np.zeros(43, dtype=np.bool_)
        valid[DEFAULT_LAYOUT.policy_indices] = True
        message.validity_mask = valid.tolist()
        self._publisher.publish(message)
        self._step += 1


def main() -> None:
    rclpy.init()
    node = TeleopActionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
