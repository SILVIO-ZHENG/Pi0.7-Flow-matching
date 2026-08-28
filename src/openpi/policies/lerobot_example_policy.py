import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_lerobot_example() -> dict:
    """Create a random input example for the generic LeRobot policy."""
    return {
        # Match LeRobot order: left arm 7 + right arm 7 + head 2 + two grippers = 18 dimensions.
        "observation/state": np.random.rand(18),
        "observation/base_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Move the plate to the center and put the yellow stick into it.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LeRobotExampleInputs(transforms.DataTransformFn):
    """Transform generic LeRobot data into the input format required by pi0/pi0.5."""

    # Distinguish pi0, pi0.5, and related model variants.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # LeRobot images are commonly float32 CHW; convert them to uint8 HWC.
        base_image = _parse_image(data["observation/base_image"])
        left_image = _parse_image(data["observation/left_image"])
        right_image = _parse_image(data["observation/right_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions exist only during training; the model transform pads them to the model dimension.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LeRobotExampleOutputs(transforms.DataTransformFn):
    """Trim model output back to the example robot's 18-dimensional action."""

    def __call__(self, data: dict) -> dict:
        # The model action has 32 dimensions; the example robot controls the first 18.
        return {"actions": np.asarray(data["actions"][:, :18])}
