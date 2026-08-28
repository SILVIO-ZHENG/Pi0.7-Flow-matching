# Contributing

Changes must keep the G1 data contract, upstream compatibility, and physical-robot safety boundaries auditable.

Run the following checks before submitting a change:

```bash
uv run ruff check <changed-python-files>
uv run ruff format --check <changed-python-files>
uv run pytest tests -q
python -m compileall -q src scripts examples/unitree_g1 ros2_ws/src/g1_pi07_bringup
```

Changes to joint ordering, action normalization, temporal alignment, padding masks, gradient boundaries, or live command publication require unit tests. Do not commit datasets, weights, robot credentials, personal absolute paths, or unsanitized logs.

When code is migrated from OpenPI or another project, preserve its license, source revision, copyright, and attribution. Describe project-specific modifications separately from upstream code in each pull request.
