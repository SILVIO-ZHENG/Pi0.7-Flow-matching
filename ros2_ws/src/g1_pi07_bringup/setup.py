from setuptools import find_packages
from setuptools import setup


package_name = "g1_pi07_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/record_g1_episode.launch.py"]),
        (f"share/{package_name}/config", ["config/g1_pipeline.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="OpenPI G1 Project",
    maintainer_email="noreply@example.com",
    description="G1 43-DoF teleoperation and multimodal recording nodes",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "state_adapter = g1_pi07_bringup.state_adapter_node:main",
            "episode_recorder = g1_pi07_bringup.episode_recorder_node:main",
            "teleop_action = g1_pi07_bringup.teleop_action_node:main",
            "robot_command_adapter = g1_pi07_bringup.robot_command_adapter_node:main",
            "xr_udp_bridge = g1_pi07_bringup.xr_udp_bridge_node:main",
        ]
    },
)
