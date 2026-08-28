# End-to-End Runbook

This is the shortest reproducible path from initial recording to dry-run deployment. Run commands from the repository root unless noted otherwise.

## 1. Environment and hardware-independent tests

```bash
uv sync
uv run python -m pytest tests -q
```

The ROS2 overlay additionally requires a system ROS2 installation, MoveIt2, OpenCV, rosbag2 MCAP support, and a Python environment that can import `g1_pi07`.

## 2. Robot-side preparation

1. Compare all 43 names and their order in `src/g1_pi07/joints.py` against the target G1 and Dex3 URDF.
2. Define `left_arm` and `right_arm` in the SRDF and start `/compute_ik`.
3. Calibrate the seven limits and direction flags for each Dex3 hand.
4. Calibrate the rigid transform from the XR tracking frame to `torso_link`.
5. Start all three RGB streams, `/joint_states`, `/imu/data`, lower-body stability state, and emergency-stop state.
6. Keep `enable_robot_commands=false` until topic and data dry-runs have passed.

## 3. Build and record

```bash
python -m pip install -e . --no-deps
cd ros2_ws
colcon build --symlink-install
source install/setup.bash

ros2 launch g1_pi07_bringup record_g1_episode.launch.py \
  episode_id:=pick_box_0001 \
  output_root:=../data/raw \
  task:="Pick up the box with both hands and place it in the target area" \
  enable_xr_udp_bridge:=true
```

Set the result before stopping the launch:

```bash
ros2 param set /episode_recorder success_label true
# Or for a failed episode:
ros2 param set /episode_recorder success_label false
ros2 param set /episode_recorder failure_reason "grasp slipped during lift"
```

Press `Ctrl-C` only after labelling. The recorder then finalizes Parquet, MP4, and metadata. Preserve the raw MCAP bag as the source of truth.

## 4. QC, episode split, and LeRobot V3

```bash
uv run python scripts/validate_g1_episodes.py --raw-root ./data/raw
uv run python scripts/split_g1_episodes.py \
  --raw-root ./data/raw \
  --output ./data/splits.json

uv run python examples/unitree_g1/data/convert_g1_session_to_lerobot.py \
  --raw-root ./data/raw \
  --root ./data/lerobot/local/g1_43dof_teleop \
  --repo-id local/g1_43dof_teleop \
  --split-file ./data/splits.json \
  --split train
```

QC and conversion require each action to match a safety-gate-approved command by default. The default split excludes failed and unlabeled episodes.

## 5. Sidecars and q01/q99

```bash
uv run python scripts/make_g1_recap_sidecar.py \
  --episode-map ./data/lerobot/local/g1_43dof_teleop/meta/g1_episode_map.json \
  --annotations examples/unitree_g1/configs/episode_labels.example.json \
  --interventions examples/unitree_g1/configs/episode_labels.example.json \
  --output ./data/sidecars/g1_recap_mem.jsonl

uv run python scripts/compute_norm_stats.py pi07_g1_43dof_joint
```

Statistics must read only the training data. The step mask excludes repeated episode-tail padding.

## 6. Training order

First verify that three trajectories can be overfit:

```bash
uv run python scripts/train_pytorch.py pi07_g1_overfit_smoke \
  --exp-name overfit_3eps
```

Then run Flow-only behaviour cloning:

```bash
uv run python scripts/train_pytorch.py pi05_g1_43dof_flow \
  --exp-name g1_flow_stage1
```

Finally initialize the joint FAST-CE + Flow stage from the Flow checkpoint:

```bash
uv run python scripts/train_pytorch.py pi07_g1_43dof_joint \
  --exp-name g1_joint_stage2 \
  --pytorch-weight-path ./checkpoints/pi05_g1_43dof_flow/g1_flow_stage1/100000
```

The Knowledge Insulation path currently supports one training process. Flow-only training retains DDP.

## 7. Holdout evaluation, service, and dry-run

```bash
uv run python examples/unitree_g1/eval/eval_g1_holdout.py \
  --checkpoint ./checkpoints/pi07_g1_43dof_joint/g1_joint_stage2/100000 \
  --episode-dir ./data/raw/VALIDATION_EPISODE

uv run python examples/unitree_g1/eval/simulate_rtc_replay.py

uv run python scripts/serve_policy.py policy:checkpoint \
  --policy.config pi07_g1_43dof_joint \
  --policy.dir ./checkpoints/pi07_g1_43dof_joint/g1_joint_stage2/100000

uv run python examples/unitree_g1/deploy/g1_pi07_client.py \
  --config-path examples/unitree_g1/configs/g1_43dof.example.json
```

The example configuration is dry-run only. Before enabling robot commands, independently verify the client limits, safety-gate parameters, SDK bridge, balance controller, collision handling, and physical emergency stop.
