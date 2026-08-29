# SPDX-License-Identifier: Apache-2.0
"""OpenPI policy transforms for the Unitree G1 and dual Dex3-1 setup."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from g1_pi07.joints import ACTIVE_ACTION_DIM
from g1_pi07.joints import MODEL_ACTION_DIM
from openpi import transforms
from openpi.models import model as _model


def make_g1_example() -> dict:
    """Create one correctly shaped observation for policy-server smoke tests."""

    return {
        "observation/state": np.zeros(ACTIVE_ACTION_DIM, dtype=np.float32),
        "observation/head_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "prompt": "Use both hands to pick up the box and place it in the target area.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Image must be a three-dimensional array; shape={image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"The final image dimension must be RGB=3; shape={image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError("Image contains NaN or Inf")
        image = image.astype(np.float32)
        if float(image.min()) < 0.0:
            image = image * 0.5 + 0.5
        image = np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class G1Inputs(transforms.DataTransformFn):
    """Map LeRobot G1 fields into the three fixed OpenPI camera slots."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (ACTIVE_ACTION_DIM,):
            raise ValueError(f"G1 policy state must have {ACTIVE_ACTION_DIM} dimensions; got {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("G1 policy state contains NaN or Inf")
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": _parse_image(data["observation/head_image"]),
                "left_wrist_0_rgb": _parse_image(data["observation/left_wrist_image"]),
                "right_wrist_0_rgb": _parse_image(data["observation/right_wrist_image"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != ACTIVE_ACTION_DIM:
                raise ValueError(f"G1 expert actions must have shape [H,{ACTIVE_ACTION_DIM}]; got {actions.shape}")
            if not np.isfinite(actions).all():
                raise ValueError("G1 expert actions contain NaN or Inf")
            inputs["actions"] = actions

        prompt = data.get("prompt")
        if prompt is not None:
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        if "g1_action_step_mask" in data:
            inputs["action_step_mask"] = np.asarray(data["g1_action_step_mask"], dtype=np.bool_)
        if "g1_action_dim_mask" in data:
            raw_mask = np.asarray(data["g1_action_dim_mask"], dtype=np.bool_)
            if raw_mask.shape != (ACTIVE_ACTION_DIM,):
                raise ValueError("g1_action_dim_mask must have 28 dimensions")
            inputs["action_dim_mask"] = raw_mask
        if "g1_subtask" in data:
            inputs["subtask"] = data["g1_subtask"]
        # Keep numeric RECAP/RL sidecar fields until DataLoaderImpl separates
        # them from the Observation. Text memory is consumed before this
        # transform; action masks are copied above under model-facing names.
        inputs.update(
            {
                key: value
                for key, value in data.items()
                if key.startswith("g1_") and key not in {"g1_subtask", "g1_memory", "g1_next_memory"}
            }
        )
        return inputs


@dataclasses.dataclass(frozen=True)
class G1Outputs(transforms.DataTransformFn):
    """Drop the four model-only padding dimensions before robot control."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] != MODEL_ACTION_DIM:
            raise ValueError(f"Model actions must have shape [H,{MODEL_ACTION_DIM}]; got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Model actions contain NaN or Inf")
        return {"actions": actions[:, :ACTIVE_ACTION_DIM]}


def padded_dimension_mask() -> np.ndarray:
    mask = np.zeros(MODEL_ACTION_DIM, dtype=np.bool_)
    mask[:ACTIVE_ACTION_DIM] = True
    return mask
