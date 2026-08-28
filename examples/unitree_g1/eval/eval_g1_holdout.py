#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline holdout evaluation for an aligned G1 Parquet/MP4 episode."""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
import polars as pl
import tyro

from g1_pi07.joints import ACTIVE_ACTION_DIM
from openpi.policies import policy_config
from openpi.training import config as training_config


def _read_video(path: Path) -> list[np.ndarray]:
    return [np.asarray(frame, dtype=np.uint8) for frame in iio.imiter(path, plugin="FFMPEG")]


def _future_actions(rows: list[dict], start: int, horizon: int) -> tuple[np.ndarray, int]:
    valid_steps = min(horizon, len(rows) - start)
    indices = np.minimum(np.arange(start, start + horizon), len(rows) - 1)
    actions = np.stack([np.asarray(rows[int(index)]["action"], dtype=np.float32) for index in indices])
    if actions.shape != (horizon, ACTIVE_ACTION_DIM):
        raise ValueError(f"Expert actions must have shape [{horizon},{ACTIVE_ACTION_DIM}]; got {actions.shape}")
    return actions, valid_steps


def _save_contact_sheet(output: Path, videos: dict[str, list[np.ndarray]], indices: list[int]) -> None:
    canvas = Image.new("RGB", (960, 240 * len(indices)))
    for row_index, frame_index in enumerate(indices):
        for column, camera in enumerate(("head", "left_wrist", "right_wrist")):
            image = Image.fromarray(videos[camera][frame_index]).resize((320, 240))
            canvas.paste(image, (column * 320, row_index * 240))
    canvas.save(output / "visual_inputs.jpg", quality=95)


def main(
    checkpoint: Path = Path("./checkpoints/pi07_g1_43dof_joint/run/step"),
    episode_dir: Path = Path("./data/raw/validation_episode"),
    output_dir: Path = Path("./outputs/g1_holdout"),
    config_name: str = "pi07_g1_43dof_joint",
    prompt: str | None = None,
    num_frames: int = 6,
    seed: int = 0,
    device: str = "cuda",
    *,
    plan_subtask: bool = True,
    use_recorded_subtask: bool = False,
    require_applied_actions: bool = True,
) -> None:
    """Predict 50-step chunks and compare only non-padded future targets."""

    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    rows = pl.read_parquet(episode_dir / "steps.parquet").to_dicts()
    if not rows:
        raise ValueError("Holdout episode contains no data frames")
    if num_frames <= 0:
        raise ValueError("num_frames must be greater than zero")
    if require_applied_actions and any(not bool(row.get("command_applied", False)) for row in rows):
        raise ValueError(
            "Holdout episode contains targets that were not approved by the safety gate and cannot be expert actions"
        )
    videos = {
        camera: _read_video(episode_dir / "videos" / f"{camera}.mp4")
        for camera in ("head", "left_wrist", "right_wrist")
    }
    if any(len(frames) != len(rows) for frames in videos.values()):
        raise ValueError("Holdout episode Parquet rows do not match video frame counts")

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = training_config.get_config(config_name)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    task_prompt = prompt or str(metadata.get("task") or rows[0].get("task") or "")
    if not task_prompt.strip():
        raise ValueError("Holdout evaluation requires a non-empty task prompt")
    policy = policy_config.create_trained_policy(
        cfg,
        checkpoint,
        default_prompt=task_prompt,
        sample_kwargs={"num_steps": 10},
        pytorch_device=device,
    )

    horizon = int(cfg.model.action_horizon)
    frame_indices = np.linspace(0, len(rows) - 1, num=min(num_frames, len(rows)), dtype=int).tolist()
    rng = np.random.default_rng(seed)
    metrics: list[dict[str, float | int | str]] = []
    predictions: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}

    for frame_index in frame_indices:
        target, valid_steps = _future_actions(rows, frame_index, horizon)
        observation = {
            "observation/head_image": videos["head"][frame_index],
            "observation/left_wrist_image": videos["left_wrist"][frame_index],
            "observation/right_wrist_image": videos["right_wrist"][frame_index],
            "observation/state": np.asarray(rows[frame_index]["observation_state"], dtype=np.float32),
            "prompt": task_prompt,
            "plan_subtask": plan_subtask,
        }
        recorded_subtask = str(metadata.get("subtask", ""))
        if use_recorded_subtask and recorded_subtask:
            observation["subtask"] = recorded_subtask
            observation["plan_subtask"] = False
        noise = rng.standard_normal((horizon, cfg.model.action_dim), dtype=np.float32)
        result = policy.infer(observation, noise=noise)
        prediction = np.asarray(result["actions"], dtype=np.float32)
        if prediction.shape != (horizon, ACTIVE_ACTION_DIM):
            raise ValueError(f"Policy output must have shape [{horizon},{ACTIVE_ACTION_DIM}]; got {prediction.shape}")
        absolute_error = np.abs(prediction[:valid_steps] - target[:valid_steps])
        metrics.append(
            {
                "frame": frame_index,
                "valid_steps": valid_steps,
                "infer_ms": float(result["policy_timing"]["infer_ms"]),
                "mae_all28": float(absolute_error.mean()),
                "mae_arms14": float(absolute_error[:, :14].mean()),
                "mae_hands14": float(absolute_error[:, 14:].mean()),
                "first_step_l2": float(np.linalg.norm(prediction[0] - target[0])),
                "subtask": str(result.get("subtask", recorded_subtask)),
            }
        )
        key = f"frame_{frame_index:06d}"
        predictions[key] = prediction
        targets[key] = target

    np.savez_compressed(output_dir / "predictions.npz", **predictions)
    np.savez_compressed(output_dir / "targets.npz", **targets)
    _save_contact_sheet(output_dir, videos, frame_indices)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    (output_dir / "summary.json").write_text(
        json.dumps({"frames": frame_indices, "metrics": metrics}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"evaluated {len(frame_indices)} frames -> {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
