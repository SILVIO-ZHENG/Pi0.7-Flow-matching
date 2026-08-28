"""Loss composition with an explicit Knowledge Insulation gradient boundary."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable


@dataclass(frozen=True)
class JointObjectiveConfig:
    enabled: bool = False
    fast_ce_weight: float = 1.0
    flow_weight: float = 1.0
    stop_flow_gradient_to_vlm: bool = True

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.fast_ce_weight)
            or not math.isfinite(self.flow_weight)
            or self.fast_ce_weight < 0
            or self.flow_weight < 0
        ):
            raise ValueError("Loss weights must be finite and non-negative")
        if self.enabled and self.fast_ce_weight == 0 and self.flow_weight == 0:
            raise ValueError("FAST-CE and Flow weights cannot both be zero when joint training is enabled")


def knowledge_insulated_backward(
    *,
    fast_loss_fn: Callable,
    flow_loss_fn: Callable,
    vlm_parameters: Iterable,
    config: JointObjectiveConfig,
):
    """Build a flow graph with frozen VLM leaves, then backpropagate once.

    Temporarily disabling gradients on VLM parameters while constructing the
    flow branch is the explicit Stop-Gradient boundary: the Action Expert still
    trains, while VLM leaves receive only FAST-CE gradients. One combined
    backward avoids cloning the very large PaliGemma gradient set.

    The joint trainer currently rejects DDP because two model forwards feeding
    one backward require a dedicated reducer integration. The baseline flow
    trainer remains DDP-capable.
    """

    if not config.enabled:
        flow_loss = flow_loss_fn()
        (config.flow_weight * flow_loss).backward()
        return {"flow_loss": flow_loss.detach(), "fast_ce_loss": None}

    fast_loss = fast_loss_fn()
    vlm_parameters = list(vlm_parameters)
    if config.stop_flow_gradient_to_vlm:
        requires_grad = [parameter.requires_grad for parameter in vlm_parameters]
        try:
            for parameter in vlm_parameters:
                parameter.requires_grad_(False)
            flow_loss = flow_loss_fn()
        finally:
            for parameter, enabled in zip(vlm_parameters, requires_grad, strict=True):
                parameter.requires_grad_(enabled)
    else:
        flow_loss = flow_loss_fn()
    (config.fast_ce_weight * fast_loss + config.flow_weight * flow_loss).backward()
    return {"flow_loss": flow_loss.detach(), "fast_ce_loss": fast_loss.detach()}
