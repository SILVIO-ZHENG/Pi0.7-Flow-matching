#!/usr/bin/env bash
# Convert a local LeRobot V3 dataset to the V2.1 layout used by selected OpenPI paths.

set -euo pipefail

OPENPI_DIR="${OPENPI_DIR:-$(pwd)}"
RAW_DATASET="${RAW_DATASET:-${OPENPI_DIR}/data/lerobot_v3_example}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${OPENPI_DIR}/data/lerobot}"
REPO_ID="${REPO_ID:-example/lerobot_v3_task}"
CONVERTER_DIR="${CONVERTER_DIR:-${OPENPI_DIR}/scripts/lerobot_conversion}"
PYTHON="${PYTHON:-${OPENPI_DIR}/.venv/bin/python}"

DATASET_LINK="${LEROBOT_ROOT}/${REPO_ID}"
DATASET_PARENT="$(dirname "${DATASET_LINK}")"

mkdir -p "${DATASET_PARENT}"

if [[ -e "${DATASET_LINK}" && ! -L "${DATASET_LINK}" ]]; then
  echo "Target dataset path exists and is not a symbolic link: ${DATASET_LINK}" >&2
  echo "Confirm the directory manually before conversion to avoid overwriting data." >&2
  exit 1
fi

if [[ ! -e "${DATASET_LINK}" ]]; then
  ln -s "${RAW_DATASET}" "${DATASET_LINK}"
fi

cd "${CONVERTER_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "OpenPI Python environment not found: ${PYTHON}" >&2
  exit 1
fi

"${PYTHON}" convert_v3_to_v2.py --repo-id "${REPO_ID}" --root "${LEROBOT_ROOT}"

echo "Conversion completed: ${DATASET_LINK}"
echo "Original V3 backup/reference path: ${DATASET_LINK}_v3.0"
