"""Safety gate from canonical ActionTarget43 to a named upper-body JointState command."""

from __future__ import annotations

from g1_pi07_interfaces.msg import ActionTarget43
from g1_pi07_interfaces.msg import RobotState43
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from g1_pi07.joints import DEFAULT_LAYOUT


class RobotCommandAdapterNode(Node):
    """Final safety gate that publishes a named upper-body JointState command.

    A command is rejected as a whole if any freshness, ordering, stability,
    limit, or per-cycle jump check fails. Partial joint commands are never sent.
    """

    def __init__(self) -> None:
        super().__init__("robot_command_adapter")
        defaults = {
            "state_topic": "/robot/state43",
            "action_topic": "/control/action_target43",
            "command_topic": "/g1/upper_body_position_command",
            "stable_topic": "/low_level/stable",
            "estop_topic": "/safety/estop",
            "enable_robot_commands": False,
            "limits_confirmed": False,
            "max_state_age_ms": 100.0,
            "max_action_age_ms": 100.0,
            "max_source_state_age_ms": 250.0,
            "max_step_rad": 0.15,
            "lower_limits": [-3.14] * 14 + [0.0] * 14,
            "upper_limits": [3.14] * 14 + [1.0] * 14,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self._state: RobotState43 | None = None
        # Fail-safe startup state: stability is unknown and emergency stop is active.
        self._stable = False
        self._estop = True
        # Monotonic action stamps prevent duplicate or replayed command publication.
        self._last_published_stamp_ns = -1
        self._publisher = self.create_publisher(
            JointState,
            self.get_parameter("command_topic").value,
            20,
        )
        self.create_subscription(RobotState43, self.get_parameter("state_topic").value, self._on_state, 20)
        self.create_subscription(ActionTarget43, self.get_parameter("action_topic").value, self._on_action, 20)
        self.create_subscription(Bool, self.get_parameter("stable_topic").value, self._on_stable, 10)
        self.create_subscription(Bool, self.get_parameter("estop_topic").value, self._on_estop, 10)

    def _on_state(self, message: RobotState43) -> None:
        if tuple(message.name) != DEFAULT_LAYOUT.full_joint_names:
            self.get_logger().error("RobotState43 joint order is invalid; state update rejected")
            return
        position = np.asarray(message.position, dtype=np.float64)
        validity = np.asarray(message.validity_mask, dtype=np.bool_)
        if position.shape != (43,) or validity.shape != (43,) or not np.isfinite(position).all():
            self.get_logger().error("RobotState43 position contains NaN/Inf; state update rejected")
            return
        self._state = message

    def _on_stable(self, message: Bool) -> None:
        self._stable = bool(message.data)

    def _on_estop(self, message: Bool) -> None:
        self._estop = bool(message.data)

    def _on_action(self, message: ActionTarget43) -> None:
        if not bool(self.get_parameter("enable_robot_commands").value):
            return
        if not bool(self.get_parameter("limits_confirmed").value):
            self.get_logger().error("Joint limits are unconfirmed; live command rejected", throttle_duration_sec=2.0)
            return
        if self._estop or not self._stable:
            self.get_logger().warning(
                "Emergency stop is active or the low-level controller is not stable; command withheld",
                throttle_duration_sec=2.0,
            )
            return
        state = self._state
        if state is None:
            return
        if not np.asarray(state.validity_mask, dtype=np.bool_)[DEFAULT_LAYOUT.policy_indices].all():
            self.get_logger().error("Current robot state lacks policy joints; action rejected")
            return
        if tuple(message.name) != DEFAULT_LAYOUT.full_joint_names:
            self.get_logger().error("ActionTarget43 joint order is invalid")
            return
        validity = np.asarray(message.validity_mask, dtype=np.bool_)
        if validity.shape != (43,) or not validity[DEFAULT_LAYOUT.policy_indices].all():
            self.get_logger().error("ActionTarget43 lacks the 28 policy joints")
            return

        now_ns = self.get_clock().now().nanoseconds
        state_ns = int(state.header.stamp.sec) * 1_000_000_000 + int(state.header.stamp.nanosec)
        action_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
        source_state_ns = int(message.source_state_stamp.sec) * 1_000_000_000 + int(message.source_state_stamp.nanosec)
        try:
            max_state_age_ns = self._positive_age_ns("max_state_age_ms")
            max_action_age_ns = self._positive_age_ns("max_action_age_ms")
            max_source_state_age_ns = self._positive_age_ns("max_source_state_age_ms")
        except ValueError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=2.0)
            return
        if abs(now_ns - state_ns) > max_state_age_ns:
            self.get_logger().error("Robot state is stale; action rejected", throttle_duration_sec=2.0)
            return
        if abs(now_ns - action_ns) > max_action_age_ns:
            self.get_logger().error("Action message is stale; action rejected", throttle_duration_sec=2.0)
            return
        if abs(now_ns - source_state_ns) > max_source_state_age_ns:
            self.get_logger().error(
                "Source state used to generate the action is stale; action rejected", throttle_duration_sec=2.0
            )
            return
        if action_ns <= self._last_published_stamp_ns:
            self.get_logger().warning("Action timestamp is duplicated or regressed; replayed command rejected")
            return

        target_full = np.asarray(message.position, dtype=np.float32)
        current_full = np.asarray(state.position, dtype=np.float32)
        if (
            target_full.shape != (43,)
            or current_full.shape != (43,)
            or not np.isfinite(target_full).all()
            or not np.isfinite(current_full).all()
        ):
            self.get_logger().error("Action or state contains NaN/Inf; command rejected")
            return
        # Safety limits apply only to the 28 joints controlled by the policy.
        target = DEFAULT_LAYOUT.full_to_policy(target_full)
        current = DEFAULT_LAYOUT.full_to_policy(current_full)
        lower = np.asarray(self.get_parameter("lower_limits").value, dtype=np.float32)
        upper = np.asarray(self.get_parameter("upper_limits").value, dtype=np.float32)
        if (
            lower.shape != (28,)
            or upper.shape != (28,)
            or not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or np.any(lower >= upper)
        ):
            self.get_logger().error("lower_limits/upper_limits configuration is invalid")
            return
        if np.any(target < lower) or np.any(target > upper):
            self.get_logger().error("Target exceeds confirmed joint limits; entire command rejected")
            return
        max_step = float(self.get_parameter("max_step_rad").value)
        if not np.isfinite(max_step) or max_step <= 0:
            self.get_logger().error("max_step_rad must be positive and finite")
            return
        if np.any(np.abs(target - current) > max_step):
            self.get_logger().error("Per-cycle joint jump exceeds the safety threshold; entire command rejected")
            return

        command = JointState()
        command.header = message.header
        command.name = list(DEFAULT_LAYOUT.policy_joint_names)
        command.position = target.astype(np.float64).tolist()
        self._publisher.publish(command)
        self._last_published_stamp_ns = action_ns

    def _positive_age_ns(self, parameter_name: str) -> int:
        value_ms = float(self.get_parameter(parameter_name).value)
        if not np.isfinite(value_ms) or value_ms <= 0:
            raise ValueError(f"{parameter_name} must be positive and finite")
        return int(value_ms * 1e6)


def main() -> None:
    rclpy.init()
    node = RobotCommandAdapterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
