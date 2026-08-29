"""Configure Pi0 model construction and input shapes."""

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class RTCTrainingConfig:
    """Configure training-time RTC action-prefix conditioning."""

    enabled: bool = False
    min_prefix_steps: int = 0
    max_prefix_steps: int | None = None
    execution_horizon: int | None = None
    prefix_probability: float = 1.0

    def __post_init__(self):
        if self.min_prefix_steps < 0:
            raise ValueError("min_prefix_steps must be non-negative.")
        if self.max_prefix_steps is not None and self.max_prefix_steps < self.min_prefix_steps:
            raise ValueError("max_prefix_steps must be greater than or equal to min_prefix_steps.")
        if self.execution_horizon is not None and self.execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive when set.")
        if not 0.0 <= self.prefix_probability <= 1.0:
            raise ValueError("prefix_probability must be in [0, 1].")


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    """Model, action-shape, tokenizer, and RTC settings for Pi0."""

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"
    rtc_training: RTCTrainingConfig = dataclasses.field(default_factory=RTCTrainingConfig)
    # Optional joint training of PaliGemma with hierarchical FAST-token CE
    # while the continuous action expert keeps the standard flow objective.
    joint_fast_objective: bool = False
    fast_max_token_len: int = 256
    fast_tokenizer_path: str = "physical-intelligence/fast"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if self.action_dim <= 0 or self.action_horizon <= 0 or self.max_token_len <= 0:
            raise ValueError("action_dim, action_horizon, and max_token_len must be greater than zero")
        if self.joint_fast_objective and not self.pi05:
            raise ValueError("The joint FAST-CE + Flow objective supports only the pi0.5 model path")
        if self.fast_max_token_len <= 1:
            raise ValueError("fast_max_token_len must be greater than one")
        if (
            self.rtc_training.execution_horizon is not None
            and self.rtc_training.execution_horizon > self.action_horizon
        ):
            raise ValueError("RTC execution_horizon must not exceed action_horizon")
        if self.rtc_training.max_prefix_steps is not None and self.rtc_training.max_prefix_steps > self.action_horizon:
            raise ValueError("RTC max_prefix_steps must not exceed action_horizon")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                fast_tokenized_prompt=(
                    jax.ShapeDtypeStruct([batch_size, self.fast_max_token_len], jnp.int32)
                    if self.joint_fast_objective
                    else None
                ),
                fast_tokenized_prompt_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.fast_max_token_len], bool)
                    if self.joint_fast_objective
                    else None
                ),
                fast_token_ar_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.fast_max_token_len], jnp.int32)
                    if self.joint_fast_objective
                    else None
                ),
                fast_token_loss_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.fast_max_token_len], bool)
                    if self.joint_fast_objective
                    else None
                ),
                action_step_mask=None,
                action_dim_mask=None,
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
