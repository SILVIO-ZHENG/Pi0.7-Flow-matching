from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = dict(sample_kwargs or {})
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            model_config = getattr(self._model, "config", None)
            if getattr(model_config, "joint_fast_objective", False):
                self._hierarchical_tokenizer = _tokenizer.FASTTokenizer(
                    getattr(model_config, "fast_max_token_len", 256),
                    fast_tokenizer_path=getattr(model_config, "fast_tokenizer_path", "physical-intelligence/fast"),
                )
            else:
                self._hierarchical_tokenizer = None
            self._subtask_max_new_tokens = int(self._sample_kwargs.pop("subtask_max_new_tokens", 32))
            if self._hierarchical_tokenizer is not None and self._subtask_max_new_tokens <= 0:
                raise ValueError("subtask_max_new_tokens must be positive")
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
            self._hierarchical_tokenizer = None
            self._subtask_max_new_tokens = 0

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        policy_start = time.monotonic()
        rtc_guidance = obs.get("rtc_guidance")
        rtc_prefix = obs.get("rtc_prefix")
        obs_without_guidance = dict(obs)
        obs_without_guidance.pop("rtc_guidance", None)
        obs_without_guidance.pop("rtc_prefix", None)

        explicit_subtask = obs_without_guidance.pop("subtask", None)
        if explicit_subtask is not None:
            obs_without_guidance["g1_subtask"] = explicit_subtask
        plan_subtask = bool(
            obs_without_guidance.pop("plan_subtask", self._hierarchical_tokenizer is not None)
        )
        planner_start = time.monotonic()
        planned_subtask = obs_without_guidance.get("g1_subtask")
        if plan_subtask and self._hierarchical_tokenizer is not None and planned_subtask is None:
            planned_subtask = self._plan_subtask(obs_without_guidance)
            if planned_subtask:
                obs_without_guidance["g1_subtask"] = planned_subtask
        planner_ms = (time.monotonic() - planner_start) * 1000

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs_without_guidance)
        transform_start = time.monotonic()
        inputs = self._input_transform(inputs)
        transform_ms = (time.monotonic() - transform_start) * 1000

        rtc_prepare_start = time.monotonic()
        rtc_sample_kwargs = self._prepare_rtc_guidance(obs_without_guidance, rtc_guidance)
        rtc_sample_kwargs.update(self._prepare_rtc_prefix(obs_without_guidance, rtc_prefix))
        rtc_prepare_ms = (time.monotonic() - rtc_prepare_start) * 1000

        tensorize_start = time.monotonic()
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device
        tensorize_ms = (time.monotonic() - tensorize_start) * 1000

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        sample_kwargs.update(rtc_sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation_start = time.monotonic()
        observation = _model.Observation.from_dict(inputs)
        observation_ms = (time.monotonic() - observation_start) * 1000
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        model_timing = getattr(self._model, "_last_sample_timing", {})
        numpy_start = time.monotonic()
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        numpy_ms = (time.monotonic() - numpy_start) * 1000

        output_transform_start = time.monotonic()
        outputs = self._output_transform(outputs)
        output_transform_ms = (time.monotonic() - output_transform_start) * 1000
        policy_total_ms = (time.monotonic() - policy_start) * 1000
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
            "observation_tokenize_ms": transform_ms,
            "rtc_prefix_transform_ms": rtc_prepare_ms,
            "tensorize_ms": tensorize_ms,
            "observation_pack_ms": observation_ms,
            "model_sample_ms": model_time * 1000,
            "to_numpy_ms": numpy_ms,
            "output_transform_ms": output_transform_ms,
            "action_ready_ms": policy_total_ms,
            "subtask_planner_ms": planner_ms,
        }
        if planned_subtask:
            outputs["subtask"] = planned_subtask
        if model_timing:
            outputs["model_timing"] = model_timing
        return outputs

    def _plan_subtask(self, raw_observation: dict) -> str | None:
        """Run the PaliGemma autoregressive branch before flow action sampling."""

        transformed = self._input_transform(jax.tree.map(lambda x: x, raw_observation))
        tensors = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
            transformed,
        )
        observation = _model.Observation.from_dict(tensors)
        sequence = self._model.generate_fast_tokens(
            self._pytorch_device,
            observation,
            max_new_tokens=self._subtask_max_new_tokens,
        )
        tokens = np.asarray(sequence[0].detach().cpu(), dtype=np.int32)
        subtask = self._hierarchical_tokenizer.extract_subtask(tokens)
        if subtask is None:
            logging.warning("Hierarchical planner did not emit a parseable `Subtask:` field; using the task prompt only.")
        return subtask

    def _prepare_rtc_prefix(self, obs: dict, rtc_prefix: dict | None) -> dict[str, Any]:
        """Transform an RTC hard-prefix action into model sampling space."""
        if rtc_prefix is None:
            return {}
        if not self._is_pytorch_model:
            raise ValueError("RTC hard-prefix inference is currently implemented only for the PyTorch model path")

        raw_action_prefix = rtc_prefix.get("action_prefix", rtc_prefix.get("target_actions"))
        if raw_action_prefix is None:
            raise ValueError("rtc_prefix must contain action_prefix or target_actions")
        action_prefix = np.asarray(raw_action_prefix, dtype=np.float32)
        if action_prefix.ndim != 2 or not np.isfinite(action_prefix).all():
            raise ValueError(f"rtc_prefix action_prefix must be a finite 2D array; shape={action_prefix.shape}")

        prefix_obs = jax.tree.map(lambda x: x, obs)
        prefix_obs["actions"] = action_prefix
        transformed = self._input_transform(prefix_obs)
        model_prefix = np.asarray(transformed["actions"], dtype=np.float32)

        model_config = getattr(self._model, "config", None)
        model_horizon = int(getattr(model_config, "action_horizon", model_prefix.shape[0]))
        model_action_dim = int(getattr(model_config, "action_dim", model_prefix.shape[-1]))
        if model_prefix.shape != (model_horizon, model_action_dim):
            raise ValueError(
                f"Transformed rtc_prefix must have shape [{model_horizon},{model_action_dim}]; "
                f"got {model_prefix.shape}"
            )

        raw_delay = rtc_prefix.get("delay", rtc_prefix.get("prefix_steps"))
        if isinstance(raw_delay, bool) or not isinstance(raw_delay, (int, np.integer)):
            raise ValueError("rtc_prefix delay/prefix_steps must be an integer")
        delay = int(raw_delay)
        if not 0 <= delay <= model_horizon:
            raise ValueError(f"rtc_prefix delay must be in 0..{model_horizon}")
        return {
            "rtc_prefix": {
                "action_prefix": torch.from_numpy(model_prefix[None, ...]).to(self._pytorch_device),
                "delay": delay,
            }
        }

    def _prepare_rtc_guidance(self, obs: dict, rtc_guidance: dict | None) -> dict[str, Any]:
        """Transform deployment-side RTC target actions into model sampling space."""
        if rtc_guidance is None:
            return {}
        if not self._is_pytorch_model:
            raise ValueError("RTC-guided inference is currently implemented only for the PyTorch model path")

        target_actions = np.asarray(rtc_guidance["target_actions"], dtype=np.float32)
        weights = np.asarray(rtc_guidance["weights"], dtype=np.float32)
        if target_actions.ndim != 2 or not np.isfinite(target_actions).all():
            raise ValueError(f"RTC target_actions must be a 2D array; shape={target_actions.shape}")
        if weights.ndim == 1:
            if weights.shape[0] != target_actions.shape[0]:
                raise ValueError("A one-dimensional RTC weights array must match the target action horizon")
            weights = np.repeat(weights[:, None], target_actions.shape[-1], axis=-1)
        if weights.shape != target_actions.shape:
            raise ValueError(f"RTC weights shape {weights.shape} must equal target_actions shape {target_actions.shape}")
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("RTC weights must be finite and non-negative")
        beta = float(rtc_guidance.get("beta", 0.0))
        eps = float(rtc_guidance.get("eps", 1e-4))
        if not np.isfinite(beta) or beta < 0 or not np.isfinite(eps) or not 0 < eps < 0.5:
            raise ValueError("RTC beta must be finite and non-negative, and eps must be in (0, 0.5)")

        guidance_obs = jax.tree.map(lambda x: x, obs)
        guidance_obs["actions"] = target_actions
        transformed = self._input_transform(guidance_obs)
        model_targets = np.asarray(transformed["actions"], dtype=np.float32)

        model_config = getattr(self._model, "config", None)
        model_horizon = int(getattr(model_config, "action_horizon", model_targets.shape[0]))
        model_action_dim = int(getattr(model_config, "action_dim", model_targets.shape[-1]))
        model_targets = self._resize_rtc_array(
            model_targets,
            target_horizon=model_horizon,
            target_dim=model_action_dim,
            fill_value=0.0,
        )
        weights = self._resize_rtc_array(
            weights,
            target_horizon=model_horizon,
            target_dim=model_action_dim,
            fill_value=0.0,
        )

        return {
            "rtc_guidance": {
                "target_actions": torch.from_numpy(model_targets[None, ...]).to(self._pytorch_device),
                "weights": torch.from_numpy(weights[None, ...]).to(self._pytorch_device),
                "beta": beta,
                "eps": eps,
            }
        }

    @staticmethod
    def _resize_rtc_array(
        value: np.ndarray,
        *,
        target_horizon: int,
        target_dim: int,
        fill_value: float,
    ) -> np.ndarray:
        """Pad or truncate an RTC guidance array to the model action shape."""
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError("RTC array must be finite with shape [H,D]")
        if target_horizon <= 0 or target_dim <= 0:
            raise ValueError("RTC target horizon and dimension must be greater than zero")
        resized = np.full((target_horizon, target_dim), fill_value, dtype=np.float32)
        copy_horizon = min(value.shape[0], target_horizon)
        copy_dim = min(value.shape[-1], target_dim)
        resized[:copy_horizon, :copy_dim] = value[:copy_horizon, :copy_dim]
        return resized

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
