# Unitree G1 Examples

This directory contains examples for the G1 43-DoF path:

```text
common/   ROS2 I/O, parameters, interpolation, and asynchronous policy process
configs/  Deployment, joint limits, and episode annotations
data/     Aligned episode to LeRobot V3 conversion
deploy/   π0.7-inspired ROS2 RTC client
eval/     Holdout action replay and synthetic RTC replay
rtc/      Latency-adaptive action-chunk scheduler
scripts/  Training entry point
```

Common commands:

```bash
uv run python examples/unitree_g1/data/convert_g1_session_to_lerobot.py --help
uv run python examples/unitree_g1/eval/eval_g1_holdout.py --help
uv run python examples/unitree_g1/eval/simulate_rtc_replay.py
uv run pytest examples/unitree_g1/rtc/rtc_chunker_test.py -q
uv run python examples/unitree_g1/deploy/g1_pi07_client.py \
  --config-path examples/unitree_g1/configs/g1_43dof.example.json
```
