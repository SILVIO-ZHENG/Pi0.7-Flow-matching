# OpenPI LeRobot V3-to-V2.1 Conversion Utility

This directory integrates the NVIDIA Isaac-GR00T `convert_v3_to_v2.py` utility so that a LeRobot V3 dataset can be converted to the V2.1 layout expected by selected OpenPI training paths.

## Convert a local V3 dataset

Prepare a LeRobot V3 dataset directory and run:

```bash
OPENPI_DIR=/path/to/openpi \
RAW_DATASET=/path/to/lerobot_v3_dataset \
LEROBOT_ROOT=/path/to/lerobot_root \
REPO_ID=owner/dataset_name \
./scripts/lerobot_conversion/convert_lerobot_v3_to_v2.sh
```

```text
OPENPI_DIR   OpenPI repository root
RAW_DATASET  Source LeRobot V3 dataset directory
LEROBOT_ROOT Local LeRobot root where owner/dataset_name will be created
REPO_ID      LeRobot repository identifier, for example owner/dataset_name
```

The shell wrapper creates the required symbolic link and then invokes:

```bash
python convert_v3_to_v2.py --repo-id "${REPO_ID}" --root "${LEROBOT_ROOT}"
```

By default, the wrapper uses `${OPENPI_DIR}/.venv/bin/python`. The integrated Python file retains the NVIDIA conversion logic and includes small local compatibility helpers, so the complete Isaac-GR00T repository is not required.
