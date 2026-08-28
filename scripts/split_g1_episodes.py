#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create a deterministic episode-level train/validation/test split."""

from __future__ import annotations

import json
from pathlib import Path

import tyro

from g1_pi07.data.split import split_episode_ids


def main(
    raw_root: Path = Path("./data/raw"),
    output: Path = Path("./data/splits.json"),
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    *,
    include_failed: bool = False,
    include_unlabeled: bool = False,
) -> None:
    episode_ids: list[str] = []
    excluded_failed: list[str] = []
    excluded_unlabeled: list[str] = []
    for path in sorted(raw_root.iterdir()):
        metadata_path = path / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        success = metadata.get("success")
        if success is False and not include_failed:
            excluded_failed.append(path.name)
            continue
        if success is not True and success is not False and not include_unlabeled:
            excluded_unlabeled.append(path.name)
            continue
        episode_ids.append(path.name)
    splits = split_episode_ids(
        episode_ids,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    payload = {
        "schema_version": "g1-episode-split-v1",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "include_failed": include_failed,
        "include_unlabeled": include_unlabeled,
        "excluded_failed": excluded_failed,
        "excluded_unlabeled": excluded_unlabeled,
        **splits,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(", ".join(f"{name}={len(ids)}" for name, ids in splits.items()))
    if excluded_failed:
        print(f"excluded_failed={len(excluded_failed)}")
    if excluded_unlabeled:
        print(f"excluded_unlabeled={len(excluded_unlabeled)}")
    print(f"wrote episode split -> {output}")


if __name__ == "__main__":
    tyro.cli(main)
