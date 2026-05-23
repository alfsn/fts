# src/trading_bot/core/dataset.py

from typing import List, Sequence, Tuple

import numpy as np

from .schemas import BarData


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
