"""Launch G1 state adaptation, teleoperation, command gating, and recording."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    episode_id = LaunchConfiguration("episode_id")
    output_root = LaunchConfiguration("output_root")
    params_file = LaunchConfiguration("params_file")
    task = LaunchConfiguration("task")
    subtask = LaunchConfiguration("subtask")
    success_label = LaunchConfiguration("success_label")
    failure_reason = LaunchConfiguration("failure_reason")
    enable_teleop = LaunchConfiguration("enable_teleop")
    enable_xr_udp_bridge = LaunchConfiguration("enable_xr_udp_bridge")
    enable_robot_commands = LaunchConfiguration("enable_robot_commands")
    limits_confirmed = LaunchConfiguration("limits_confirmed")
    bag_path = PathJoinSubstitution([output_root, episode_id, "raw_mcap"])
    topics = [
        "/joint_states",
        "/imu/data",
        "/robot/state43",
        "/control/action_target43",
        "/g1/upper_body_position_command",
        "/low_level/stable",
        "/safety/estop",
        "/camera/head/image_raw/compressed",
        "/camera/left_wrist/image_raw/compressed",
        "/camera/right_wrist/image_raw/compressed",
        "/xr/hand_targets",
        "/parameter_events",
    ]
    return LaunchDescription(
        [
            DeclareLaunchArgument("episode_id"),
            DeclareLaunchArgument("output_root", default_value="./data/raw"),
            DeclareLaunchArgument("task", default_value="Use both hands to complete the demonstrated task."),
            DeclareLaunchArgument("subtask", default_value=""),
            DeclareLaunchArgument("success_label", default_value="unknown"),
            DeclareLaunchArgument("failure_reason", default_value=""),
            DeclareLaunchArgument("enable_teleop", default_value="true"),
            DeclareLaunchArgument("enable_xr_udp_bridge", default_value="false"),
            DeclareLaunchArgument("enable_robot_commands", default_value="false"),
            DeclareLaunchArgument("limits_confirmed", default_value="false"),
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution([FindPackageShare("g1_pi07_bringup"), "config", "g1_pipeline.yaml"]),
            ),
            Node(
                package="g1_pi07_bringup",
                executable="state_adapter",
                name="state_adapter",
                parameters=[params_file],
            ),
            Node(
                package="g1_pi07_bringup",
                executable="episode_recorder",
                name="episode_recorder",
                parameters=[
                    params_file,
                    {
                        "episode_id": episode_id,
                        "output_root": output_root,
                        "task": task,
                        "subtask": subtask,
                        "success_label": success_label,
                        "failure_reason": failure_reason,
                    },
                ],
            ),
            Node(
                package="g1_pi07_bringup",
                executable="xr_udp_bridge",
                name="xr_udp_bridge",
                parameters=[params_file],
                condition=IfCondition(enable_xr_udp_bridge),
            ),
            Node(
                package="g1_pi07_bringup",
                executable="teleop_action",
                name="teleop_action",
                parameters=[params_file, {"episode_id": episode_id}],
                condition=IfCondition(enable_teleop),
            ),
            Node(
                package="g1_pi07_bringup",
                executable="robot_command_adapter",
                name="robot_command_adapter",
                parameters=[
                    params_file,
                    {
                        "enable_robot_commands": ParameterValue(enable_robot_commands, value_type=bool),
                        "limits_confirmed": ParameterValue(limits_confirmed, value_type=bool),
                    },
                ],
            ),
            ExecuteProcess(cmd=["ros2", "bag", "record", "-s", "mcap", "-o", bag_path, *topics], output="screen"),
        ]
    )
