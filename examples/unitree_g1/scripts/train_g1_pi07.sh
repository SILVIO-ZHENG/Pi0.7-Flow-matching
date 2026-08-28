#!/usr/bin/env bash
# Train the π0.7-inspired joint objective for G1 dual arms and dual Dex3 hands.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
CONFIG_NAME="${CONFIG_NAME:-pi07_g1_43dof_joint}"
EXP_NAME="${EXP_NAME:-g1_pi07_joint_baseline}"

cd "${PROJECT_ROOT}"

uv run scripts/train_pytorch.py "${CONFIG_NAME}" --exp-name "${EXP_NAME}"
