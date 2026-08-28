# Source and Scope Manifest

## Source snapshot

- Input archive: `CBC_Pi0.7_Openpi_full.zip`
- Input archive SHA-256: `9e24e885848c33e9cb94e1c51c460dc88482985323990472f68a8bc2596004d1`
- Selective source repository: <https://github.com/Knight1112D/CBC_Pi0.7_Openpi>
- Migration baseline: `4e9878feefd3192d2395443135651021b1832df3`
- Original OpenPI upstream: <https://github.com/Physical-Intelligence/openpi>

The Apache-2.0 license, Gemma license, `UPSTREAM_OPENPI_README.md`, and existing source-file copyright notices are retained.

## Retained scope

The deliverable is limited to the G1 data-to-model-to-deployment path:

- retain the OpenPI model, PyTorch PaliGemma/Action Expert, trainer, normalization, policy server, and client dependencies;
- retain the small number of additional robot adapters required by shared configuration registration and checkpoint compatibility;
- include G1 43-DoF, dual Dex3, three-RGB, ROS2, LeRobot, joint-objective, and RTC integrations;
- exclude `.git`, caches, datasets, weights, checkpoints, runtime logs, and unrelated notebooks from the distribution archive;
- do not copy robot datasets, pretrained model weights, credentials, or vendor SDK binaries.

## Project-specific directories

- `src/g1_pi07/`
- `ros2_ws/`
- `examples/unitree_g1/`
- `tests/test_core_pipeline.py`
- G1 processing scripts and `docs/reproduction/`

## Main modified OpenPI files

- `src/openpi/models/pi0_config.py`
- `src/openpi/models/tokenizer.py`
- `src/openpi/models/model.py`
- `src/openpi/models_pytorch/pi0_pytorch.py`
- `src/openpi/models_pytorch/preprocessing_pytorch.py`
- `src/openpi/policies/policy.py`
- `src/openpi/policies/unitree_g1_policy.py`
- `src/openpi/training/config.py`
- `src/openpi/training/data_loader.py`
- `src/openpi/training/g1_training.py`
- `src/openpi/transforms.py`
- `scripts/train_pytorch.py`
- `scripts/compute_norm_stats.py`

See `CODE_MAP.md` for feature mapping and `VALIDATION.md` for verification status.
