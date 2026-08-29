"""Episode writer for aligned Parquet rows and three MP4 streams.

Heavy I/O dependencies are imported only at finalization so collection nodes can
start and report a clear installation error instead of failing at module import.
Raw ROS topics should additionally be recorded with the MCAP launch file under
``ros2_ws``; the derived files here are deterministic products of that raw bag.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
from pathlib import Path
from typing import Any

import numpy as np

from g1_pi07.data.types import AlignedStep
from g1_pi07.joints import DEFAULT_LAYOUT


@dataclass
class EpisodeWriter:
    """Accumulate one aligned episode and persist Parquet, video, and metadata.

    Rows and camera frames stay in lockstep in memory. ``finalize`` refuses
    mismatched stream lengths so the stored frame index remains deterministic.
    """

    output_root: Path
    episode_id: str
    fps: float = 20.0
    task: str = ""
    subtask: str = ""
    # Each row is keyed by the action timestamp, the canonical episode clock.
    rows: list[dict[str, Any]] = field(default_factory=list)
    # Camera lists must contain exactly one frame for every row.
    videos: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {"head": [], "left_wrist": [], "right_wrist": []}
    )

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        if not self.episode_id or Path(self.episode_id).name != self.episode_id or self.episode_id in {".", ".."}:
            raise ValueError("episode_id must be a single safe directory name")
        if self.fps <= 0 or not np.isfinite(self.fps):
            raise ValueError("fps must be positive and finite")

    def add(
        self,
        step: AlignedStep,
        *,
        applied_policy_action: np.ndarray | None = None,
        success: bool | None = None,
        failure_reason: str = "",
    ) -> None:
        if step.action.episode_id != self.episode_id:
            raise ValueError(f"Action episode_id={step.action.episode_id!r} does not match writer={self.episode_id!r}")
        if set(step.cameras) != set(self.videos):
            raise ValueError(f"Camera set must be {sorted(self.videos)}; got {sorted(step.cameras)}")
        if self.rows and step.timestamp_ns <= self.rows[-1]["timestamp_ns"]:
            raise ValueError("Action timestamps must be strictly increasing")
        if not np.asarray(step.action.validity_mask)[DEFAULT_LAYOUT.policy_indices].all():
            raise ValueError("Action lacks a valid target for one or more of the 28 policy joints")
        for name, frame in step.cameras.items():
            if self.videos[name] and frame.rgb.shape != self.videos[name][0].shape:
                raise ValueError(f"{name} camera resolution changed within the episode")
        # Training consumes only the 28 arm/hand joints from the 43-DoF state.
        policy_state = DEFAULT_LAYOUT.full_to_policy(step.state.q)
        target_action = DEFAULT_LAYOUT.full_to_policy(step.action.position)
        if applied_policy_action is None:
            policy_action = target_action
            command_applied = False
        else:
            policy_action = np.asarray(applied_policy_action, dtype=np.float32)
            if policy_action.shape != (28,) or not np.isfinite(policy_action).all():
                raise ValueError("applied_policy_action must be a finite 28-dimensional array")
            command_applied = True
        self.rows.append(
            {
                "episode_id": self.episode_id,
                "step_index": step.action.step_index,
                "timestamp_ns": step.timestamp_ns,
                "state_timestamp_ns": step.state.timestamp_ns,
                "source_state_sequence": step.action.source_state_sequence,
                "state_delta_ns": step.state_delta_ns,
                "camera_timestamp_ns": {name: frame.timestamp_ns for name, frame in step.cameras.items()},
                "camera_delta_ns": dict(step.camera_delta_ns),
                "q": step.state.q.tolist(),
                "dq": step.state.dq.tolist(),
                "tau_est": step.state.tau_est.tolist(),
                "imu": step.state.imu.tolist(),
                "validity_mask": step.state.validity_mask.tolist(),
                "imu_valid": step.state.imu_valid,
                "observation_state": policy_state.tolist(),
                "action": policy_action.tolist(),
                "target_action": target_action.tolist(),
                "command_applied": command_applied,
                "action_validity_mask": step.action.validity_mask.tolist(),
                "action_source": step.action.source,
                "compute_started_ns": step.action.compute_started_ns,
                "compute_finished_ns": step.action.compute_finished_ns,
                "sent_ns": step.action.sent_ns,
                "task": self.task,
                "subtask": self.subtask,
                "success": success,
                "failure_reason": failure_reason,
            }
        )
        for name, frame in step.cameras.items():
            self.videos[name].append(frame.rgb.copy())

    def finalize(self, *, source_mcap: str | None = None, overwrite: bool = False) -> Path:
        if not self.rows:
            raise ValueError("Episode contains no aligned steps to write")
        episode_dir = self.output_root / self.episode_id
        if episode_dir.exists() and not overwrite:
            raise FileExistsError(f"Output directory already exists: {episode_dir}")
        episode_dir.mkdir(parents=True, exist_ok=True)
        video_dir = episode_dir / "videos"
        video_dir.mkdir(exist_ok=True)

        try:
            import imageio.v2 as imageio
            import polars as pl
        except ImportError as exc:
            raise RuntimeError("Writing MP4/Parquet requires the project's data dependencies") from exc

        pl.DataFrame(self.rows).write_parquet(episode_dir / "steps.parquet")
        for name, frames in self.videos.items():
            if len(frames) != len(self.rows):
                raise ValueError(f"{name} frame count {len(frames)} does not match step count {len(self.rows)}")
            with imageio.get_writer(video_dir / f"{name}.mp4", fps=self.fps, codec="libx264") as writer:
                for frame in frames:
                    writer.append_data(frame)

        metadata = {
            "schema_version": "g1-pi07-episode-v1",
            "episode_id": self.episode_id,
            "fps": self.fps,
            "num_steps": len(self.rows),
            "applied_command_steps": sum(bool(row["command_applied"]) for row in self.rows),
            "task": self.task,
            "subtask": self.subtask,
            "success": self._episode_success(),
            "failure_reasons": sorted({row["failure_reason"] for row in self.rows if row["failure_reason"]}),
            "source_mcap": source_mcap,
            "full_joint_names": list(DEFAULT_LAYOUT.full_joint_names),
            "policy_joint_names": list(DEFAULT_LAYOUT.policy_joint_names),
            "camera_names": list(self.videos),
        }
        (episode_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return episode_dir

    def _episode_success(self) -> bool | None:
        labels = {row["success"] for row in self.rows if row["success"] is not None}
        return next(iter(labels)) if len(labels) == 1 else None
