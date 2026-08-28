#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert aligned G1 episode folders into a LeRobot video dataset."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import imageio.v3 as iio
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import polars as pl
import tqdm
import tyro

from g1_pi07.joints import ACTIVE_ACTION_DIM
from g1_pi07.joints import DEFAULT_LAYOUT
from g1_pi07.joints import FULL_DOF


DEFAULT_TASK = "Use both hands to pick up the box and place it in the target area."


def _episode_directories(raw_root: Path, split_file: Path | None, split: str) -> list[Path]:
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_root}")
    directories = sorted(path for path in raw_root.iterdir() if (path / "metadata.json").exists())
    if split_file is None:
        return directories
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    missing_splits = {"train", "validation", "test"} - set(payload)
    if missing_splits:
        raise ValueError(f"Split file is missing fields: {sorted(missing_splits)}")
    split_sets = {}
    for name in ("train", "validation", "test"):
        values = payload[name]
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"split.{name} must be a JSON list of non-empty strings")
        if len(set(values)) != len(values):
            raise ValueError(f"split.{name} contains duplicate episodes")
        split_sets[name] = set(values)
    split_pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    if any(split_sets[left] & split_sets[right] for left, right in split_pairs):
        raise ValueError("Split file assigns the same episode to multiple splits")
    allowed = split_sets[split]
    available = {path.name for path in directories}
    missing = sorted(allowed - available)
    if missing:
        raise ValueError(f"split={split} references missing episodes: {missing}")
    return [path for path in directories if path.name in allowed]


def _read_video(path: Path) -> list[np.ndarray]:
    return [np.asarray(frame, dtype=np.uint8) for frame in iio.imiter(path, plugin="FFMPEG")]


def _probe_image_shapes(episode_dir: Path) -> dict[str, tuple[int, int, int]]:
    shapes: dict[str, tuple[int, int, int]] = {}
    for name in ("head", "left_wrist", "right_wrist"):
        iterator = iio.imiter(episode_dir / "videos" / f"{name}.mp4", plugin="FFMPEG")
        try:
            frame = np.asarray(next(iterator), dtype=np.uint8)
        except StopIteration as exc:
            raise ValueError(f"{episode_dir.name}/{name}.mp4 contains no video frames") from exc
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"{episode_dir.name}/{name}.mp4 is not an RGB video: {frame.shape}")
        shapes[name] = tuple(int(value) for value in frame.shape)
    return shapes


def _create_dataset(
    repo_id: str,
    root: Path,
    *,
    fps: int,
    image_shapes: dict[str, tuple[int, int, int]],
    overwrite: bool,
) -> LeRobotDataset:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {root}")
        shutil.rmtree(root)
    names_43 = list(DEFAULT_LAYOUT.full_joint_names)
    names_28 = list(DEFAULT_LAYOUT.policy_joint_names)
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="unitree_g1_43dof_dual_dex3",
        fps=fps,
        features={
            "head_image": {
                "dtype": "image",
                "shape": image_shapes["head"],
                "names": ["height", "width", "channel"],
            },
            "left_wrist_image": {
                "dtype": "image",
                "shape": image_shapes["left_wrist"],
                "names": ["height", "width", "channel"],
            },
            "right_wrist_image": {
                "dtype": "image",
                "shape": image_shapes["right_wrist"],
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": (ACTIVE_ACTION_DIM,), "names": names_28},
            "actions": {"dtype": "float32", "shape": (ACTIVE_ACTION_DIM,), "names": names_28},
            "target_actions": {"dtype": "float32", "shape": (ACTIVE_ACTION_DIM,), "names": names_28},
            "command_applied": {"dtype": "bool", "shape": (1,), "names": ["command_applied"]},
            "q": {"dtype": "float32", "shape": (FULL_DOF,), "names": names_43},
            "dq": {"dtype": "float32", "shape": (FULL_DOF,), "names": names_43},
            "tau_est": {"dtype": "float32", "shape": (FULL_DOF,), "names": names_43},
            "imu": {
                "dtype": "float32",
                "shape": (10,),
                "names": ["qx", "qy", "qz", "qw", "gx", "gy", "gz", "ax", "ay", "az"],
            },
            "state_validity_mask": {"dtype": "bool", "shape": (FULL_DOF,), "names": names_43},
            "action_validity_mask": {"dtype": "bool", "shape": (FULL_DOF,), "names": names_43},
            "imu_valid": {"dtype": "bool", "shape": (1,), "names": ["imu_valid"]},
            "step_index": {"dtype": "int64", "shape": (1,), "names": ["step_index"]},
            "source_state_sequence": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["source_state_sequence"],
            },
            "timestamp_ns": {"dtype": "int64", "shape": (1,), "names": ["timestamp_ns"]},
            "command_timestamps_ns": {
                "dtype": "int64",
                "shape": (3,),
                "names": ["compute_started", "compute_finished", "sent"],
            },
            "camera_timestamps_ns": {
                "dtype": "int64",
                "shape": (3,),
                "names": ["head", "left_wrist", "right_wrist"],
            },
            "sync_delta_ns": {
                "dtype": "int64",
                "shape": (4,),
                "names": ["state", "head", "left_wrist", "right_wrist"],
            },
        },
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=0,
    )


def convert(
    raw_root: Path = Path("./data/raw"),
    root: Path = Path("./data/lerobot/local/g1_43dof_teleop"),
    repo_id: str = "local/g1_43dof_teleop",
    split_file: Path | None = Path("./data/splits.json"),
    split: str = "train",
    default_task: str = DEFAULT_TASK,
    fps: int = 20,
    *,
    overwrite: bool = False,
    push_to_hub: bool = False,
    require_applied_actions: bool = True,
) -> None:
    """Convert one episode-level split; no frame can leak across splits."""

    raw_root = raw_root.expanduser().resolve()
    root = root.expanduser().resolve()
    filesystem_root = Path(root.anchor)
    if root == filesystem_root or root == Path.cwd().resolve():
        raise ValueError(f"Refusing to write a dataset to an overly broad directory: {root}")
    if root == raw_root or root in raw_root.parents or raw_root in root.parents:
        raise ValueError("Dataset output directory and raw_root must not contain each other")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    if not repo_id.strip() or not default_task.strip():
        raise ValueError("repo_id and default_task must not be empty")
    episode_dirs = _episode_directories(raw_root, split_file, split)
    if not episode_dirs:
        raise ValueError(f"split={split} contains no episodes")
    image_shapes = _probe_image_shapes(episode_dirs[0])
    dataset = _create_dataset(
        repo_id,
        root,
        fps=fps,
        image_shapes=image_shapes,
        overwrite=overwrite,
    )
    source_episode_map: list[dict] = []

    for episode_index, episode_dir in enumerate(tqdm.tqdm(episode_dirs, desc=f"convert {split}")):
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != "g1-pi07-episode-v1":
            raise ValueError(f"{episode_dir.name} has an unsupported schema_version")
        if tuple(metadata.get("full_joint_names", ())) != DEFAULT_LAYOUT.full_joint_names:
            raise ValueError(f"{episode_dir.name} has an inconsistent 43-dimensional joint order")
        if tuple(metadata.get("policy_joint_names", ())) != DEFAULT_LAYOUT.policy_joint_names:
            raise ValueError(f"{episode_dir.name} has an inconsistent 28-dimensional policy-joint order")
        if int(round(float(metadata.get("fps", fps)))) != fps:
            raise ValueError(f"{episode_dir.name} fps={metadata.get('fps')} does not match target fps={fps}")
        rows = pl.read_parquet(episode_dir / "steps.parquet").to_dicts()
        if not rows:
            raise ValueError(f"{episode_dir.name} contains no convertible steps")
        missing_applied = [index for index, row in enumerate(rows) if not bool(row.get("command_applied", False))]
        if require_applied_actions and missing_applied:
            raise ValueError(
                f"{episode_dir.name} has {len(missing_applied)} frames without an approved command; "
                "a safety-gate-rejected target cannot be used as an expert action"
            )
        videos = {
            name: _read_video(episode_dir / "videos" / f"{name}.mp4")
            for name in ("head", "left_wrist", "right_wrist")
        }
        counts = {name: len(frames) for name, frames in videos.items()}
        if any(count != len(rows) for count in counts.values()):
            raise ValueError(f"{episode_dir.name} video/step count mismatch: rows={len(rows)}, videos={counts}")
        for camera, frames in videos.items():
            mismatched = [
                index for index, frame in enumerate(frames) if tuple(frame.shape) != image_shapes[camera]
            ]
            if mismatched:
                raise ValueError(
                    f"{episode_dir.name}/{camera} resolution differs from the first dataset frame; "
                    f"first mismatched frame={mismatched[0]}"
                )
        task = str(metadata.get("task") or default_task)
        for index, row in enumerate(rows):
            camera_delta = row["camera_delta_ns"]
            dataset.add_frame(
                {
                    "head_image": videos["head"][index],
                    "left_wrist_image": videos["left_wrist"][index],
                    "right_wrist_image": videos["right_wrist"][index],
                    "state": np.asarray(row["observation_state"], dtype=np.float32),
                    "actions": np.asarray(row["action"], dtype=np.float32),
                    "target_actions": np.asarray(row["target_action"], dtype=np.float32),
                    "command_applied": np.asarray([row["command_applied"]], dtype=np.bool_),
                    "q": np.asarray(row["q"], dtype=np.float32),
                    "dq": np.asarray(row["dq"], dtype=np.float32),
                    "tau_est": np.asarray(row["tau_est"], dtype=np.float32),
                    "imu": np.asarray(row["imu"], dtype=np.float32),
                    "state_validity_mask": np.asarray(row["validity_mask"], dtype=np.bool_),
                    "action_validity_mask": np.asarray(row["action_validity_mask"], dtype=np.bool_),
                    "imu_valid": np.asarray([row["imu_valid"]], dtype=np.bool_),
                    "step_index": np.asarray([row["step_index"]], dtype=np.int64),
                    "source_state_sequence": np.asarray(
                        [-1 if row["source_state_sequence"] is None else row["source_state_sequence"]],
                        dtype=np.int64,
                    ),
                    "timestamp_ns": np.asarray([row["timestamp_ns"]], dtype=np.int64),
                    "command_timestamps_ns": np.asarray(
                        [
                            -1 if row["compute_started_ns"] is None else row["compute_started_ns"],
                            -1 if row["compute_finished_ns"] is None else row["compute_finished_ns"],
                            -1 if row["sent_ns"] is None else row["sent_ns"],
                        ],
                        dtype=np.int64,
                    ),
                    "camera_timestamps_ns": np.asarray(
                        [
                            row["camera_timestamp_ns"]["head"],
                            row["camera_timestamp_ns"]["left_wrist"],
                            row["camera_timestamp_ns"]["right_wrist"],
                        ],
                        dtype=np.int64,
                    ),
                    "sync_delta_ns": np.asarray(
                        [
                            row["state_delta_ns"],
                            camera_delta["head"],
                            camera_delta["left_wrist"],
                            camera_delta["right_wrist"],
                        ],
                        dtype=np.int64,
                    ),
                    "task": task,
                }
            )
        dataset.save_episode()
        success_values = {row.get("success") for row in rows if row.get("success") is not None}
        source_episode_map.append(
            {
                "episode_index": episode_index,
                "source_episode_id": str(metadata.get("episode_id", episode_dir.name)),
                "source_directory": episode_dir.name,
                "num_steps": len(rows),
                "task": task,
                "subtask": str(metadata.get("subtask", "")),
                "success": next(iter(success_values)) if len(success_values) == 1 else None,
            }
        )

    map_path = root / "meta" / "g1_episode_map.json"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(
        json.dumps(
            {
                "schema_version": "g1-lerobot-episode-map-v1",
                "split": split,
                "episodes": source_episode_map,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    if push_to_hub:
        dataset.push_to_hub(
            tags=["unitree-g1", "dex3", "pi0.7-inspired", split],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        version = str(info.get("codebase_version", "unknown"))
        if not version.startswith("v3") and not version.startswith("3"):
            print(f"WARNING: generated LeRobot codebase_version={version}; run the bundled V3 compatibility check.")
    print(f"converted {len(episode_dirs)} {split} episodes -> {root}")


if __name__ == "__main__":
    tyro.cli(convert)
