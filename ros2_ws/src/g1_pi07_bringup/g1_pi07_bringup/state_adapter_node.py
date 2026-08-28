"""Convert named JointState + IMU messages into one canonical RobotState43."""

from __future__ import annotations

import copy
import math

from g1_pi07.joints import DEFAULT_LAYOUT
from g1_pi07_interfaces.msg import RobotState43
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class StateAdapterNode(Node):
    def __init__(self) -> None:
        super().__init__("state_adapter")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("output_topic", "/robot/state43")
        self.declare_parameter("max_imu_age_ms", 50.0)
        max_imu_age_ms = float(self.get_parameter("max_imu_age_ms").value)
        if not math.isfinite(max_imu_age_ms) or max_imu_age_ms < 0:
            raise ValueError("max_imu_age_ms must be finite and non-negative")
        self._max_imu_age_ns = int(max_imu_age_ms * 1e6)
        self._sequence = 0
        self._latest_imu: Imu | None = None
        self._latest_imu_ns = -1
        self._publisher = self.create_publisher(
            RobotState43,
            self.get_parameter("output_topic").value,
            20,
        )
        self.create_subscription(Imu, self.get_parameter("imu_topic").value, self._on_imu, qos_profile_sensor_data)
        self.create_subscription(
            JointState,
            self.get_parameter("joint_state_topic").value,
            self._on_joint_state,
            qos_profile_sensor_data,
        )

    def _on_imu(self, message: Imu) -> None:
        self._latest_imu = copy.deepcopy(message)
        self._latest_imu_ns = _stamp_ns(message.header.stamp)

    def _on_joint_state(self, message: JointState) -> None:
        if len(set(message.name)) != len(message.name):
            self.get_logger().error("JointState contains duplicate joint names; entire frame rejected")
            return
        by_name = {name: index for index, name in enumerate(message.name)}
        output = RobotState43()
        output.header = message.header
        output.sequence = self._sequence
        output.name = list(DEFAULT_LAYOUT.full_joint_names)
        positions = [0.0] * 43
        velocities = [0.0] * 43
        efforts = [0.0] * 43
        validity = [False] * 43
        for target_index, name in enumerate(DEFAULT_LAYOUT.full_joint_names):
            source_index = by_name.get(name)
            if (
                source_index is None
                or source_index >= len(message.position)
                or source_index >= len(message.velocity)
                or source_index >= len(message.effort)
            ):
                continue
            values = (
                float(message.position[source_index]),
                float(message.velocity[source_index]),
                float(message.effort[source_index]),
            )
            if not all(math.isfinite(value) for value in values):
                continue
            positions[target_index], velocities[target_index], efforts[target_index] = values
            validity[target_index] = True
        output.position = positions
        output.velocity = velocities
        output.effort = efforts
        output.validity_mask = validity

        joint_stamp_ns = _stamp_ns(message.header.stamp)
        imu_values = None if self._latest_imu is None else (
            self._latest_imu.orientation.x,
            self._latest_imu.orientation.y,
            self._latest_imu.orientation.z,
            self._latest_imu.orientation.w,
            self._latest_imu.angular_velocity.x,
            self._latest_imu.angular_velocity.y,
            self._latest_imu.angular_velocity.z,
            self._latest_imu.linear_acceleration.x,
            self._latest_imu.linear_acceleration.y,
            self._latest_imu.linear_acceleration.z,
        )
        if (
            self._latest_imu is not None
            and abs(joint_stamp_ns - self._latest_imu_ns) <= self._max_imu_age_ns
            and all(math.isfinite(value) for value in imu_values)
            and math.sqrt(sum(value * value for value in imu_values[:4])) > 1e-8
        ):
            output.imu = self._latest_imu
            output.imu_valid = True
        else:
            output.imu.header.stamp = message.header.stamp
            output.imu_valid = False
            self.get_logger().warning("IMU missing or stale; publishing zero IMU", throttle_duration_sec=2.0)
        self._publisher.publish(output)
        self._sequence += 1


def main() -> None:
    rclpy.init()
    node = StateAdapterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
