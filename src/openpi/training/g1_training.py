"""G1 training helpers for RECAP/MEM/RL-token metadata and Knowledge Insulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # Allow data-only tools to run without the GPU training stack.
    torch = None


G1_PREFIX = "g1_"
ADVANTAGE_KEY = "g1_advantage_indicator"
USE_ADVANTAGE_KEY = "g1_use_advantage"
INTERVENTION_KEY = "g1_is_human_intervention"
RL_WEIGHT_KEY = "g1_rl_token_weight"
MEMORY_KEY = "g1_memory"
NEXT_MEMORY_KEY = "g1_next_memory"
SUBTASK_KEY = "g1_subtask"

NUMERIC_METADATA_KEYS = (
    ADVANTAGE_KEY,
    USE_ADVANTAGE_KEY,
    INTERVENTION_KEY,
    RL_WEIGHT_KEY,
)


@dataclasses.dataclass(frozen=True)
class SidecarConfig:
    """Configure offline sidecar labels; disabled by default for upstream compatibility."""

    path: str | None = None
    enabled: bool = False
    append_memory_to_prompt: bool = False

    def __post_init__(self) -> None:
        if self.enabled and not self.path:
            raise ValueError("A sidecar path is required when sidecars are enabled")


@dataclasses.dataclass(frozen=True)
class RLTokenConfig:
    """Apply per-sample loss weights from RECAP/RL-style labels."""

    enabled: bool = False
    positive_weight: float = 1.0
    negative_weight: float = 1.0
    intervention_weight: float = 1.0
    min_weight: float = 0.0
    max_weight: float = 10.0

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.positive_weight,
                self.negative_weight,
                self.intervention_weight,
                self.min_weight,
                self.max_weight,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values < 0) or self.max_weight < self.min_weight:
            raise ValueError("RL-token weights must be finite and non-negative")


@dataclasses.dataclass(frozen=True)
class KnowledgeInsulationConfig:
    """Configure lightweight Knowledge Insulation for PyTorch pi0/pi0.5 training."""

    enabled: bool = False
    freeze_vlm: bool = True
    train_action_expert: bool = True
    train_action_projections: bool = True


class ActionMaskDataset:
    """Add future-step and real-dimension masks around a LeRobot dataset."""

    def __init__(self, dataset, *, action_horizon: int, valid_action_dim: int):
        if action_horizon <= 0 or valid_action_dim <= 0:
            raise ValueError("action_horizon and valid_action_dim must be greater than zero")
        self._dataset = dataset
        self._action_horizon = action_horizon
        self._valid_action_dim = valid_action_dim

    def __getitem__(self, index):
        item = dict(self._dataset[index])
        explicit_pad = item.pop("action_is_pad", item.pop("actions_is_pad", None))
        if explicit_pad is not None:
            step_mask = ~np.asarray(explicit_pad, dtype=np.bool_).reshape(-1)[: self._action_horizon]
        else:
            remaining = self._remaining_steps(item)
            valid = self._action_horizon if remaining is None else min(self._action_horizon, remaining)
            step_mask = np.arange(self._action_horizon) < valid
        if step_mask.shape != (self._action_horizon,):
            raise ValueError(f"Invalid action padding-mask shape: {step_mask.shape}")
        item["g1_action_step_mask"] = step_mask
        item["g1_action_dim_mask"] = np.ones(self._valid_action_dim, dtype=np.bool_)
        return item

    def _remaining_steps(self, item: Mapping[str, Any]) -> int | None:
        frame_index = _optional_int(item.get("frame_index"))
        episode_index = _optional_int(item.get("episode_index"))
        if frame_index is None or episode_index is None:
            return None
        meta = getattr(self._dataset, "meta", None)
        episodes = getattr(meta, "episodes", None)
        if episodes is not None:
            try:
                record = episodes[episode_index]
                length = record.get("length") if isinstance(record, Mapping) else getattr(record, "length", None)
                if length is not None:
                    return max(0, int(length) - frame_index)
            except (IndexError, KeyError, TypeError):
                pass
        episode_data_index = getattr(self._dataset, "episode_data_index", None)
        if isinstance(episode_data_index, Mapping):
            try:
                start = int(np.asarray(episode_data_index["from"][episode_index]).item())
                end = int(np.asarray(episode_data_index["to"][episode_index]).item())
                global_index = _optional_int(item.get("index"))
                if global_index is not None:
                    return max(0, end - global_index)
                return max(0, end - start - frame_index)
            except (IndexError, KeyError, TypeError, ValueError):
                pass
        return None

    def __len__(self) -> int:
        return len(self._dataset)

    def __getattr__(self, name: str):
        return getattr(self._dataset, name)


def is_g1_key(key: str) -> bool:
    """Return whether a field belongs to the G1 extension namespace."""

    return key.startswith(G1_PREFIX)


def read_sidecar(path: str | pathlib.Path) -> dict[tuple[int | None, int | None], dict[str, Any]]:
    """Read a JSON/JSONL sidecar and index records by episode and frame."""

    sidecar_path = pathlib.Path(path)
    if not sidecar_path.exists():
        raise FileNotFoundError(f"Sidecar file does not exist: {sidecar_path}")

    if sidecar_path.suffix == ".jsonl":
        records = [json.loads(line) for line in sidecar_path.read_text().splitlines() if line.strip()]
    else:
        payload = json.loads(sidecar_path.read_text())
        records = payload["records"] if isinstance(payload, dict) and "records" in payload else payload
    if not isinstance(records, list):
        raise ValueError("Sidecar must be a record list or a JSON object containing a records list")

    index: dict[tuple[int | None, int | None], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        episode = _optional_int(record.get("episode_index"))
        frame = _optional_int(record.get("frame_index", record.get("index")))
        if episode is None:
            raise ValueError("Every sidecar record must contain episode_index")
        key = (episode, frame)
        if key in index:
            raise ValueError(f"Duplicate episode/frame key in sidecar: {key}")
        index[key] = dict(record)
    return index


class SidecarDataset:
    """Merge G1 sidecar fields without modifying the source LeRobot dataset."""

    def __init__(
        self,
        dataset,
        sidecar: Mapping[tuple[int | None, int | None], Mapping[str, Any]],
        *,
        include_text: bool = False,
    ):
        self._dataset = dataset
        self._sidecar = sidecar
        self._include_text = include_text

    def __getitem__(self, index):
        item = dict(self._dataset[index])
        episode = _optional_int(item.get("episode_index"))
        raw_frame = item.get("frame_index")
        if raw_frame is None:
            raw_frame = item.get("index", int(index))
        frame = _optional_int(raw_frame)
        record = self._sidecar.get((episode, frame)) or self._sidecar.get((episode, None)) or {}
        item.update(_standardize_record(record, include_text=self._include_text))
        return item

    def __len__(self) -> int:
        return len(self._dataset)

    def __getattr__(self, name: str):
        """Expose LeRobot metadata through the sidecar wrapper.

        ``ActionMaskDataset`` needs the underlying episode boundaries to mask
        repeat-last padding at the end of an episode.  Without this forwarding,
        enabling the optional sidecar would silently turn every padded step into
        a training target.
        """

        return getattr(self._dataset, name)


class AppendMemoryToPrompt:
    """Append optional memory text to the prompt for MEM/context smoke tests."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        memory = data.pop(MEMORY_KEY, "")
        if isinstance(memory, np.ndarray):
            memory = memory.item()
        if memory:
            prompt = data.get("prompt", "")
            if isinstance(prompt, np.ndarray):
                prompt = prompt.item()
            data["prompt"] = f"{prompt}\nMemory: {memory}"
        data.pop(NEXT_MEMORY_KEY, None)
        return data


def make_rl_token_weights(metadata: Mapping[str, torch.Tensor] | None, config: RLTokenConfig) -> torch.Tensor | None:
    """Build batch-level loss weights from sidecar metadata."""

    if not config.enabled or metadata is None:
        return None
    _require_torch()
    advantage = metadata.get(ADVANTAGE_KEY)
    use_advantage = metadata.get(USE_ADVANTAGE_KEY)
    intervention = metadata.get(INTERVENTION_KEY)
    explicit = metadata.get(RL_WEIGHT_KEY)
    if advantage is None and intervention is None and explicit is None:
        return None

    reference = next(value for value in (explicit, advantage, intervention) if value is not None)
    base = explicit.to(dtype=torch.float32) if explicit is not None else torch.ones_like(reference, dtype=torch.float32)
    if advantage is not None:
        advantage = advantage.to(dtype=torch.float32)
        weighted = torch.where(advantage > 0, base * config.positive_weight, base * config.negative_weight)
        if use_advantage is not None:
            base = torch.where(use_advantage.to(dtype=torch.bool), weighted, base)
        else:
            base = weighted
    if intervention is not None:
        base = torch.where(intervention.to(dtype=torch.bool), base * config.intervention_weight, base)
    return base.clamp(min=config.min_weight, max=config.max_weight)


def apply_loss_weights(losses: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    """Broadcast batch-level weights across a Flow Matching loss tensor."""

    _require_torch()
    if weights is None:
        return losses
    while weights.ndim < losses.ndim:
        weights = weights[..., None]
    return losses * weights.to(device=losses.device, dtype=losses.dtype)


def reduce_flow_losses(
    losses: torch.Tensor,
    observation,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average only valid action steps/dimensions, then apply sample weights."""

    _require_torch()
    if losses.ndim != 3 or not torch.isfinite(losses).all():
        raise ValueError("Flow losses must be a finite [B,H,D] tensor")
    mask = torch.ones_like(losses, dtype=torch.bool)
    if observation.action_step_mask is not None:
        step_mask = observation.action_step_mask.to(device=losses.device, dtype=torch.bool)
        if tuple(step_mask.shape) != tuple(losses.shape[:2]):
            raise ValueError("action_step_mask must have shape [B,H]")
        mask &= step_mask[:, :, None]
    if observation.action_dim_mask is not None:
        dim_mask = observation.action_dim_mask.to(device=losses.device, dtype=torch.bool)
        if tuple(dim_mask.shape) != (losses.shape[0], losses.shape[2]):
            raise ValueError("action_dim_mask must have shape [B,D]")
        mask &= dim_mask[:, None, :]
    count = mask.sum(dim=(1, 2)).clamp_min(1)
    per_sample = (losses * mask.to(dtype=losses.dtype)).sum(dim=(1, 2)) / count
    if sample_weights is None:
        return per_sample.mean()
    weights = sample_weights.to(device=losses.device, dtype=losses.dtype)
    if weights.ndim != 1 or weights.shape[0] != losses.shape[0]:
        raise ValueError("sample_weights must have shape [B]")
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("sample_weights must be finite and non-negative")
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)


def apply_knowledge_insulation(model: torch.nn.Module, config: KnowledgeInsulationConfig) -> dict[str, int]:
    """Freeze VLM or Action Expert parameters by module name and return counts."""

    _require_torch()
    if not config.enabled:
        return {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad), "frozen": 0}

    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        allow_train = False
        if config.train_action_expert and ("gemma_expert" in name or "time_mlp" in name):
            allow_train = True
        if config.train_action_projections and ("action_in_proj" in name or "action_out_proj" in name):
            allow_train = True
        if not config.freeze_vlm:
            allow_train = True
        param.requires_grad_(allow_train)
        if allow_train:
            trainable += param.numel()
        else:
            frozen += param.numel()
    return {"trainable": trainable, "frozen": frozen}


def _standardize_record(record: Mapping[str, Any], *, include_text: bool = True) -> dict[str, Any]:
    output: dict[str, Any] = {}
    output[ADVANTAGE_KEY] = float(record.get("advantage_indicator", record.get("advantage", 0.0)))
    output[USE_ADVANTAGE_KEY] = float(_as_bool(record.get("use_advantage", bool(output[ADVANTAGE_KEY]))))
    output[INTERVENTION_KEY] = float(
        _as_bool(record.get("is_human_intervention", record.get("human_intervention", False)))
    )
    output[RL_WEIGHT_KEY] = float(record.get("rl_token_weight", 1.0))
    if not all(np.isfinite(output[key]) for key in NUMERIC_METADATA_KEYS):
        raise ValueError("Numeric sidecar fields must not contain NaN or Inf")
    if output[RL_WEIGHT_KEY] < 0:
        raise ValueError("sidecar rl_token_weight must not be negative")
    if include_text and "memory" in record:
        output[MEMORY_KEY] = str(record["memory"])
    if include_text and "next_memory" in record:
        output[NEXT_MEMORY_KEY] = str(record["next_memory"])
    if "subtask" in record:
        output[SUBTASK_KEY] = str(record["subtask"])
    return output


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", "", "none", "null"}:
            return False
        raise ValueError(f"Cannot parse boolean value: {value!r}")
    return bool(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(np.asarray(value).item())
    except (TypeError, ValueError):
        return None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("This training operation requires PyTorch; run `uv sync` to install full dependencies")


def write_jsonl(records: Sequence[Mapping[str, Any]], path: str | pathlib.Path) -> None:
    """Write a JSONL sidecar for later manual annotation or scripted updates."""

    sidecar_path = pathlib.Path(path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) for record in records)
    sidecar_path.write_text(text + ("\n" if text else ""))
