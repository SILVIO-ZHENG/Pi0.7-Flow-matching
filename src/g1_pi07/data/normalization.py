"""Per-dimension q01/q99 statistics used by training and deployment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class QuantileStats:
    """Per-dimension robust bounds fitted only from the training split.

    Values are mapped from ``[q01, q99]`` to ``[-1, 1]``. Dimensions whose
    quantile range is smaller than ``eps`` are treated as constant.
    """

    q01: np.ndarray
    q99: np.ndarray
    count: int
    eps: float = 1e-6

    def __post_init__(self) -> None:
        q01 = np.asarray(self.q01, dtype=np.float32)
        q99 = np.asarray(self.q99, dtype=np.float32)
        if q01.ndim != 1 or q01.shape != q99.shape:
            raise ValueError("q01 and q99 must be one-dimensional arrays with identical shapes")
        if self.count <= 0 or not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("count must be greater than zero and eps must be positive and finite")
        if not np.all(np.isfinite(q01)) or not np.all(np.isfinite(q99)):
            raise ValueError("q01/q99 must not contain NaN or Inf")
        if np.any(q99 < q01):
            raise ValueError("q99 must not be smaller than q01")
        object.__setattr__(self, "q01", q01)
        object.__setattr__(self, "q99", q99)

    @classmethod
    def fit(cls, values: np.ndarray, *, valid_rows: np.ndarray | None = None) -> QuantileStats:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or len(array) == 0:
            raise ValueError("values must be a non-empty [N,D] array")
        if valid_rows is not None:
            row_mask = np.asarray(valid_rows, dtype=np.bool_)
            if row_mask.shape != (len(array),):
                raise ValueError("valid_rows must have shape [N]")
            array = array[row_mask]
        if len(array) == 0 or not np.all(np.isfinite(array)):
            raise ValueError("Statistics input is empty or contains NaN/Inf")
        return cls(
            q01=np.quantile(array, 0.01, axis=0).astype(np.float32),
            q99=np.quantile(array, 0.99, axis=0).astype(np.float32),
            count=len(array),
        )

    @property
    def scale(self) -> np.ndarray:
        return np.maximum(self.q99 - self.q01, self.eps)

    def normalize(
        self,
        values: np.ndarray,
        *,
        clip: bool = True,
        dim_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 0 or array.shape[-1] != len(self.q01):
            raise ValueError("The final values dimension does not match the statistics dimension")
        if not np.isfinite(array).all():
            raise ValueError("values must not contain NaN or Inf")
        # Constant dimensions map to zero instead of amplifying numerical noise.
        dynamic = (self.q99 - self.q01) > self.eps
        output = np.where(dynamic, 2.0 * (array - self.q01) / self.scale - 1.0, 0.0)
        if clip:
            output = np.clip(output, -1.0, 1.0)
        if dim_mask is not None:
            mask = np.asarray(dim_mask, dtype=np.bool_)
            if mask.shape != self.q01.shape:
                raise ValueError("dim_mask shape does not match")
            output = np.where(mask, output, 0.0)
        return output.astype(np.float32)

    def denormalize(self, values: np.ndarray, *, dim_mask: np.ndarray | None = None) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 0 or array.shape[-1] != len(self.q01):
            raise ValueError("The final values dimension does not match the statistics dimension")
        if not np.isfinite(array).all():
            raise ValueError("values must not contain NaN or Inf")
        # Use q01 as the stable inverse value for constant dimensions.
        dynamic = (self.q99 - self.q01) > self.eps
        output = np.where(dynamic, (array + 1.0) * 0.5 * self.scale + self.q01, self.q01)
        if dim_mask is not None:
            mask = np.asarray(dim_mask, dtype=np.bool_)
            if mask.shape != self.q01.shape:
                raise ValueError("dim_mask shape does not match")
            output = np.where(mask, output, 0.0)
        return output.astype(np.float32)

    def to_dict(self) -> dict:
        return {"q01": self.q01.tolist(), "q99": self.q99.tolist(), "count": self.count, "eps": self.eps}

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> QuantileStats:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            np.asarray(payload["q01"]),
            np.asarray(payload["q99"]),
            int(payload["count"]),
            float(payload.get("eps", 1e-6)),
        )
