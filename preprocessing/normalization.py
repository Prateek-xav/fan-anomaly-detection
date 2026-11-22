"""Fit reusable per-Mel-band normalization from training chunks."""
import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Normalization:
    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, batches):
        sums = None
        squares = None
        count = 0
        for batch in batches:
            features = np.asarray(batch, dtype=np.float64)
            batch_sum = features.sum(axis=(0, 1, 3))
            batch_sq = np.square(features).sum(axis=(0, 1, 3))
            sums = batch_sum if sums is None else sums + batch_sum
            squares = batch_sq if squares is None else squares + batch_sq
            count += features.shape[0] * features.shape[-1]
        if count == 0:
            raise ValueError("Cannot normalize an empty training set.")
        mean = sums / count
        variance = np.maximum(squares / count - np.square(mean), 0)
        return cls(mean.tolist(), np.maximum(np.sqrt(variance), 1e-4).tolist())

    def apply(self, features):
        mean = np.asarray(self.mean, dtype=np.float32).reshape(1, 1, -1, 1)
        std = np.asarray(self.std, dtype=np.float32).reshape(1, 1, -1, 1)
        return (features - mean) / std

    def save(self, path):
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(path.read_text(encoding="utf-8")))
