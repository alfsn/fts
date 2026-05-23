# src/trading_bot/core/transforms.py

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransform(ABC, BaseEstimator, TransformerMixin):
    """Abstract base for data transformations, compatible with sklearn."""

    def fit(self, X: Any, y: Any = None) -> "BaseTransform":
        return self

    @abstractmethod
    def transform(self, X: Any) -> Any:
        pass


class LogReturnTransform(BaseTransform):
    """Calculates logarithmic returns: ln(p_t / p_{t-1})."""

    def transform(self, X: Any) -> np.ndarray:
        X_np = np.array(X, dtype=np.float32)
        if len(X_np) < 2:
            return np.array([], dtype=np.float32)

        # Calculate returns: log(X[t] / X[t-1])
        # Handling zeros by adding a small epsilon or just masking
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = np.diff(np.log(X_np), axis=0)
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        return returns
