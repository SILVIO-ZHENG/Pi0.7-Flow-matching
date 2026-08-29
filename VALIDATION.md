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
| 43/28/32 mappings, H=50, q01/q99, synchronization, XR calibration, sidecars, RTC, and non-finite action rejection | Covered by unit tests |

The skipped test checks the actual PyTorch gradient boundary used by Knowledge Insulation. It remains in the repository and should run after the complete training dependencies are installed.
