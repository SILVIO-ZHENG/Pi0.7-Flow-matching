#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate aligned G1 Parquet/MP4 episodes before LeRobot conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import polars as pl
import tyro

from g1_pi07.joints import ACTIVE_ACTION_DIM
from g1_pi07.joints import DEFAULT_LAYOUT
from g1_pi07.joints import FULL_DOF

REQUIRED_COLUMNS = {
    "episode_id",
    "step_index",
    "timestamp_ns",
    "state_timestamp_ns",
    "source_state_sequence",
    "state_delta_ns",
    "camera_timestamp_ns",
    "camera_delta_ns",
    "q",
    "dq",
    "tau_est",
    "imu",
    "imu_valid",
    "validity_mask",
    "observation_state",
    "action",
    "target_action",
    "command_applied",
    "action_validity_mask",
    "action_source",
    "compute_started_ns",
    "compute_finished_ns",
    "sent_ns",
    "task",
    "success",
    "failure_reason",
}


def _check_vector(rows: list[dict], key: str, size: int, errors: list[str]) -> None:
    for index, row in enumerate(rows):
        value = np.asarray(row[key])
        if value.shape != (size,):
            errors.append(f"row {index}: {key} shape={value.shape}, expected ({size},)")
            return
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            errors.append(f"row {index}: {key} contains NaN/Inf")
            return


def _video_metrics(path: Path) -> dict[str, Any]:
    count = 0
    shape: tuple[int, ...] | None = None
    for frame in iio.imiter(path, plugin="FFMPEG"):
        array = np.asarray(frame)
        if shape is None:
            shape = tuple(int(value) for value in array.shape)
        elif tuple(array.shape) != shape:
            raise ValueError(f"Video resolution changed: {array.shape} != {shape}")
        count += 1
    return {"frames": count, "shape": shape}


def validate_episode(
    episode_dir: Path,
    *,
    max_state_delta_ms: float,
    max_camera_delta_ms: float,
    require_all_joints: bool,
    require_imu: bool,
    require_applied_actions: bool,
    max_action_step: float | None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    try:
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return {"episode": episode_dir.name, "ok": False, "errors": [f"metadata: {exc}"], "warnings": []}

    if metadata.get("schema_version") != "g1-pi07-episode-v1":
        errors.append(f"Unknown metadata schema_version={metadata.get('schema_version')!r}")
    if str(metadata.get("episode_id", "")) != episode_dir.name:
        errors.append("metadata episode_id does not match the directory name")
    if tuple(metadata.get("full_joint_names", ())) != DEFAULT_LAYOUT.full_joint_names:
        errors.append("metadata full_joint_names does not match the canonical 43-dimensional order")
    if tuple(metadata.get("policy_joint_names", ())) != DEFAULT_LAYOUT.policy_joint_names:
        errors.append("metadata policy_joint_names does not match the canonical 28-dimensional order")
    if tuple(metadata.get("camera_names", ())) != ("head", "left_wrist", "right_wrist"):
        errors.append("metadata camera_names must use the fixed head/left_wrist/right_wrist order")
    if not str(metadata.get("task", "")).strip():
        errors.append("metadata task is empty")
    try:
        table = pl.read_parquet(episode_dir / "steps.parquet")
    except Exception as exc:
        return {"episode": episode_dir.name, "ok": False, "errors": [*errors, f"parquet: {exc}"], "warnings": []}
    rows = table.to_dicts()
    metrics["steps"] = len(rows)
    missing = sorted(REQUIRED_COLUMNS - set(table.columns))
    if missing:
        errors.append(f"Parquet is missing fields: {missing}")
        return {"episode": episode_dir.name, "ok": False, "errors": errors, "warnings": warnings, "metrics": metrics}
    if not rows:
        errors.append("Episode contains no steps")
        return {"episode": episode_dir.name, "ok": False, "errors": errors, "warnings": warnings, "metrics": metrics}

    for key, size in (
        ("q", FULL_DOF),
        ("dq", FULL_DOF),
        ("tau_est", FULL_DOF),
        ("imu", 10),
        ("validity_mask", FULL_DOF),
        ("observation_state", ACTIVE_ACTION_DIM),
        ("action", ACTIVE_ACTION_DIM),
        ("target_action", ACTIVE_ACTION_DIM),
        ("action_validity_mask", FULL_DOF),
    ):
        _check_vector(rows, key, size, errors)

    timestamps = np.asarray([row["timestamp_ns"] for row in rows], dtype=np.int64)
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        errors.append("Action timestamps must be strictly increasing")
    step_indices = np.asarray([row["step_index"] for row in rows], dtype=np.int64)
    if len(step_indices) > 1 and np.any(np.diff(step_indices) != 1):
        warnings.append("step_index contains gaps because actions were rejected during alignment")
    episode_values = {str(row["episode_id"]) for row in rows}
    if episode_values != {episode_dir.name}:
        errors.append(f"Parquet contains inconsistent episode_id values: {sorted(episode_values)}")
    if any(not str(row["task"]).strip() for row in rows):
        errors.append("Parquet contains an empty task")

    state_timestamps = np.asarray([row["state_timestamp_ns"] for row in rows], dtype=np.int64)
    stored_state_delta = np.asarray([row["state_delta_ns"] for row in rows], dtype=np.int64)
    if not np.array_equal(state_timestamps - timestamps, stored_state_delta):
        errors.append("state_timestamp_ns - timestamp_ns does not equal state_delta_ns")
    for row_index, row in enumerate(rows):
        camera_stamps = row["camera_timestamp_ns"]
        camera_deltas = row["camera_delta_ns"]
        if set(camera_stamps) != {"head", "left_wrist", "right_wrist"} or set(camera_deltas) != {
            "head",
            "left_wrist",
            "right_wrist",
        }:
            errors.append(f"row {row_index}: camera timestamp/delta keys are incomplete")
            break
        if any(
            int(camera_stamps[name]) - int(row["timestamp_ns"]) != int(camera_deltas[name]) for name in camera_stamps
        ):
            errors.append(f"row {row_index}: camera timestamp and delta are inconsistent")
            break

    for row_index, row in enumerate(rows):
        command_times = [row["compute_started_ns"], row["compute_finished_ns"], row["sent_ns"]]
        if any(value is None or int(value) <= 0 for value in command_times):
            errors.append(f"row {row_index}: command timestamps are missing or non-positive")
            break
        if command_times != sorted(command_times):
            errors.append(f"row {row_index}: command timestamps are out of order")
            break
    if any(not str(row["action_source"]).strip() for row in rows):
        warnings.append("Some action_source values are empty; XR and policy actions cannot be distinguished")

    fps = float(metadata.get("fps", 0.0))
    if fps <= 0:
        errors.append("metadata fps must be greater than zero")
    elif len(timestamps) > 1:
        median_period_ms = float(np.median(np.diff(timestamps)) / 1e6)
        metrics["median_action_period_ms"] = median_period_ms
        expected_period_ms = 1000.0 / fps
        if abs(median_period_ms - expected_period_ms) > max(5.0, expected_period_ms * 0.25):
            warnings.append(
                f"Median action period {median_period_ms:.2f} ms is inconsistent with metadata fps={fps:.2f}"
            )

    state_delta_ms = np.abs(np.asarray([row["state_delta_ns"] for row in rows], dtype=np.float64)) / 1e6
    camera_delta_ms = {
        camera: np.abs(np.asarray([row["camera_delta_ns"][camera] for row in rows], dtype=np.float64)) / 1e6
        for camera in ("head", "left_wrist", "right_wrist")
    }
    metrics["max_state_delta_ms"] = float(state_delta_ms.max())
    metrics["max_camera_delta_ms"] = {key: float(value.max()) for key, value in camera_delta_ms.items()}
    if np.any(state_delta_ms > max_state_delta_ms):
        errors.append(f"State alignment error exceeds {max_state_delta_ms} ms")
    for camera, values in camera_delta_ms.items():
        if np.any(values > max_camera_delta_ms):
            errors.append(f"{camera} alignment error exceeds {max_camera_delta_ms} ms")

    validity = np.asarray([row["validity_mask"] for row in rows], dtype=np.bool_)
    required_indices = np.arange(FULL_DOF) if require_all_joints else DEFAULT_LAYOUT.policy_indices
    invalid_rows = np.flatnonzero(~validity[:, required_indices].all(axis=1))
    if len(invalid_rows):
        errors.append(f"{len(invalid_rows)} frames lack required joint state; first frame={int(invalid_rows[0])}")
    action_validity = np.asarray([row["action_validity_mask"] for row in rows], dtype=np.bool_)
    invalid_actions = np.flatnonzero(~action_validity[:, DEFAULT_LAYOUT.policy_indices].all(axis=1))
    if len(invalid_actions):
        errors.append(
            f"{len(invalid_actions)} frames lack a valid 28-dimensional policy action; "
            f"first frame={int(invalid_actions[0])}"
        )
    command_applied = np.asarray([row["command_applied"] for row in rows], dtype=np.bool_)
    missing_applied = np.flatnonzero(~command_applied)
    metrics["applied_command_steps"] = int(command_applied.sum())
    if len(missing_applied):
        message = (
            f"{len(missing_applied)} frames do not match a safety-gate-approved command; "
            f"first frame={int(missing_applied[0])}"
        )
        (errors if require_applied_actions else warnings).append(message)
    if require_imu:
        invalid_imu = np.flatnonzero(~np.asarray([row["imu_valid"] for row in rows], dtype=np.bool_))
        if len(invalid_imu):
            errors.append(f"{len(invalid_imu)} frames have invalid IMU data; first frame={int(invalid_imu[0])}")
    actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
    states = np.asarray([row["observation_state"] for row in rows], dtype=np.float32)
    q_values = np.asarray([row["q"] for row in rows], dtype=np.float32)
    expected_states = DEFAULT_LAYOUT.full_to_policy(q_values)
    if not np.allclose(states, expected_states, atol=1e-6):
        errors.append("observation_state does not match the canonical 28-dimensional mapping from q[43]")
    if max_action_step is not None and len(actions) > 1:
        largest_step = float(np.abs(np.diff(actions, axis=0)).max())
        metrics["max_action_step"] = largest_step
        if largest_step > max_action_step:
            warnings.append(f"Largest adjacent action jump {largest_step:.4f} > {max_action_step:.4f}")

    videos: dict[str, Any] = {}
    for camera in ("head", "left_wrist", "right_wrist"):
        path = episode_dir / "videos" / f"{camera}.mp4"
        try:
            videos[camera] = _video_metrics(path)
            if videos[camera]["frames"] != len(rows):
                errors.append(f"{camera} video frames {videos[camera]['frames']} != Parquet rows {len(rows)}")
        except Exception as exc:
            errors.append(f"Failed to read {camera} video: {exc}")
    metrics["videos"] = videos
    if int(metadata.get("num_steps", -1)) != len(rows):
        errors.append("metadata num_steps does not match the Parquet row count")
    if int(metadata.get("applied_command_steps", -1)) != int(command_applied.sum()):
        errors.append("metadata applied_command_steps does not match Parquet")
    success_values = {row["success"] for row in rows if row["success"] is not None}
    if len(success_values) > 1:
        errors.append("Success labels are inconsistent within the episode")
    row_success = next(iter(success_values)) if len(success_values) == 1 else None
    if metadata.get("success") != row_success:
        errors.append("metadata success does not match Parquet success")
    if row_success is None:
        warnings.append("Episode has not been labelled as success or failure")
    elif row_success is False and not any(str(row["failure_reason"]).strip() for row in rows):
        warnings.append("Failed episode has no failure_reason")
    return {
        "episode": episode_dir.name,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
    }


def main(
    raw_root: Path = Path("./data/raw"),
    report: Path | None = Path("./outputs/g1_data_quality_report.json"),
    max_state_delta_ms: float = 15.0,
    max_camera_delta_ms: float = 40.0,
    max_action_step: float | None = None,
    *,
    require_all_joints: bool = True,
    require_imu: bool = True,
    require_applied_actions: bool = True,
) -> None:
    episode_dirs = sorted(path for path in raw_root.iterdir() if path.is_dir() and (path / "metadata.json").exists())
    if not episode_dirs:
        raise ValueError(f"No episodes found in {raw_root}")
    results = [
        validate_episode(
            path,
            max_state_delta_ms=max_state_delta_ms,
            max_camera_delta_ms=max_camera_delta_ms,
            require_all_joints=require_all_joints,
            require_imu=require_imu,
            require_applied_actions=require_applied_actions,
            max_action_step=max_action_step,
        )
        for path in episode_dirs
    ]
    payload = {
        "schema_version": "g1-data-quality-v1",
        "summary": {
            "episodes": len(results),
            "passed": sum(result["ok"] for result in results),
            "failed": sum(not result["ok"] for result in results),
        },
        "results": results,
    }
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote validation report -> {report}")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    if payload["summary"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    tyro.cli(main)
