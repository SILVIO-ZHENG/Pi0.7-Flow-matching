"""Deterministic episode-level train/validation/test splitting."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable


def split_episode_ids(
    episode_ids: Iterable[str],
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    raw_ids = [str(item) for item in episode_ids]
    if any(not item or item in {".", ".."} or "/" in item or "\\" in item for item in raw_ids):
        raise ValueError("episode_id must be a non-empty single safe name")
    ids = sorted(set(raw_ids))
    if len(ids) < 3:
        raise ValueError("At least three episodes are required for train/validation/test splits")
    if (
        not math.isfinite(validation_fraction)
        or not math.isfinite(test_fraction)
        or validation_fraction < 0
        or test_fraction < 0
        or validation_fraction + test_fraction >= 1
    ):
        raise ValueError("Validation and test ratios must be non-negative and sum to less than one")

    def rank(episode_id: str) -> bytes:
        return hashlib.blake2b(f"{seed}:{episode_id}".encode(), digest_size=16).digest()

    ranked = sorted(ids, key=rank)
    validation_count = max(1, round(len(ids) * validation_fraction)) if validation_fraction else 0
    test_count = max(1, round(len(ids) * test_fraction)) if test_fraction else 0
    if validation_count + test_count >= len(ids):
        raise ValueError("Too few episodes remain to preserve a training split at the requested ratios")
    return {
        "validation": ranked[:validation_count],
        "test": ranked[validation_count : validation_count + test_count],
        "train": ranked[validation_count + test_count :],
    }
