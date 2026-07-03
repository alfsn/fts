import hashlib
import json
from typing import Any, List, Sequence, Tuple

import numpy as np

from .schemas import BarData


def calculate_dataset_hash(bars: Sequence[Any]) -> str:
    """
    Computes a deterministic SHA-256 hash for a sequence of bar objects.
    """
    sorted_bars = sorted(bars, key=lambda b: b.timestamp)
    data_list = [
        (
            (
                b.timestamp.isoformat()
                if hasattr(b.timestamp, "isoformat")
                else str(b.timestamp)
            ),
            float(b.close),
            float(b.volume),
        )
        for b in sorted_bars
    ]
    serialized = json.dumps(data_list, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class DatasetBuilder:
    """
    Utility for converting structural data (List[BarData]) into
    feature matrices (X) and target vectors (y).
    """

    @staticmethod
    def to_matrix(
        bars: Sequence[BarData], feature_cols: List[str] = None
    ) -> np.ndarray:
        """
        Converts a list of bars into a numpy matrix.
        Default feature_cols is ['close'].
        """
        if feature_cols is None:
            feature_cols = ["close"]

        data = []
        for bar in bars:
            row = [getattr(bar, col) for col in feature_cols]
            data.append(row)

        return np.array(data, dtype=np.float32)

    @staticmethod
    def create_sliding_windows(
        data: np.ndarray, lookback: int, horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates sliding windows for time-series forecasting.
        X shape: (samples, lookback, features)
        y shape: (samples, horizon, features) or (samples,) if flattened.
        """
        X, y = [], []
        for i in range(len(data) - lookback - horizon + 1):
            X.append(data[i : i + lookback])
            y.append(data[i + lookback : i + lookback + horizon])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    @staticmethod
    def split_train_val_purged_embargoed(
        X: np.ndarray,
        y: np.ndarray,
        val_ratio: float = 0.2,
        horizon: int = 1,
        embargo_pct: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits X and y into train and validation sets with purging and embargoing.
        Assumes chronological ordering where validation is at the end.
        """
        n_samples = len(X)
        if n_samples <= 2:
            return (
                X,
                y,
                np.empty((0,) + X.shape[1:], dtype=X.dtype),
                np.empty((0,) + y.shape[1:], dtype=y.dtype),
            )

        val_size = int(n_samples * val_ratio)
        if val_size == 0:
            val_size = min(1, n_samples // 5)
            if val_size == 0:
                val_size = 1

        train_size = n_samples - val_size

        # Validation indices (chronologically at the end)
        val_indices = np.arange(train_size, n_samples)

        # Initial train indices (before validation)
        train_indices = np.arange(0, train_size)

        # Purging: remove training samples whose labels overlap with validation start
        # Training sample i has label window [i, i + horizon].
        # First validation sample is at index `train_size` (val_start).
        # We must purge training samples where index + horizon >= train_size
        # which means index >= train_size - horizon
        purged_train_indices = train_indices[train_indices < train_size - horizon]

        # Embargoing: since validation is after training in this chronological split,
        # we don't have training data after the validation set.
        # But if we did (or for future proofing), we exclude train indices in [val_end + 1, val_end + 1 + embargo_size]
        embargo_size = int(np.ceil(n_samples * embargo_pct)) if embargo_pct > 0 else 0

        final_train_indices = []
        if len(val_indices) > 0:
            val_start = val_indices[0]
            val_end = val_indices[-1]
            for idx in purged_train_indices:
                # 1. Purge: if train index is before val and its label overlaps with val
                is_purged = (idx < val_start) and (idx + horizon >= val_start)
                # 2. Embargo: if train index is after val and falls within the embargo window
                is_embargoed = (idx > val_end) and (idx <= val_end + embargo_size)
                if not (is_purged or is_embargoed):
                    final_train_indices.append(idx)
        else:
            final_train_indices = list(purged_train_indices)

        if len(final_train_indices) == 0:
            final_train_indices = (
                list(purged_train_indices) if len(purged_train_indices) > 0 else [0]
            )

        final_train_indices = np.array(final_train_indices, dtype=np.int32)

        return (
            X[final_train_indices],
            y[final_train_indices],
            X[val_indices],
            y[val_indices],
        )
