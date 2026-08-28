# Validation Record

Validation date: 2026-08-28

## Checks executed in the available workspace

| Check | Result |
|---|---|
| `PYTHONPATH=src:. python -m unittest discover -s tests -v` | 18 tests executed: 17 passed and 1 PyTorch-dependent test was skipped because PyTorch was unavailable in the lightweight environment |
| Python `compileall` and targeted `py_compile` | Passed |
| `git diff --check` | Passed |
| `uv lock --check --offline --python $(command -v python) --no-managed-python` | Passed with 278 locked packages |
| TOML, JSON, ROS2 `package.xml` XML, and shell syntax | Passed |
| 43/28/32 mappings, H=50, q01/q99, synchronization, XR calibration, sidecars, RTC, and non-finite action rejection | Covered by hardware-independent unit tests |

The skipped test checks the actual PyTorch gradient boundary used by Knowledge Insulation. It remains in the repository and should run after the complete training dependencies are installed.

## Validation still required

- Load complete PaliGemma and Action Expert weights and run GPU training.
- Build the ROS2 workspace under ROS2 Humble/Jazzy and replay topics in real time.
- Validate MoveIt dual-arm IK with the target G1/Dex3 URDF, SRDF, and Unitree SDK2 bridge.
- Calibrate real XR and three-camera clocks and coordinate frames.
- Validate the lower-body controller, physical emergency stop, collision handling, and hardware limits.
- Measure real-robot task success, tracking error, collisions, and long-duration stability.

Live robot publication therefore remains disabled by default, and the example limits keep `confirmed_from_robot_urdf=false`. A software implementation or unit test must not be presented as a completed GPU experiment or real-robot result.
