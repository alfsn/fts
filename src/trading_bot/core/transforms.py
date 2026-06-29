# src/trading_bot/core/transforms.py

import importlib
from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def get_class(class_path: str) -> Any:
    """Dynamically imports a class from a module path string."""
    try:
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, AttributeError, ImportError) as e:
        raise ImportError(f"Failed to dynamically load class {class_path}: {e}") from e


class BaseTransform(ABC, BaseEstimator, TransformerMixin):
    """Abstract base for data transformations, compatible with sklearn."""

    def fit(self, X: Any, y: Any = None) -> "BaseTransform":
        return self

    @abstractmethod
    def transform(self, X: Any) -> Any:
        pass

    def to_dict(self) -> dict:
        """Serializes the transform configuration to a dictionary."""
        return {
            "class_path": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "params": {},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BaseTransform":
        """Reconstructs a transform subclass from its serialized dictionary representation."""
        class_path = data.get("class_path")
        if not class_path:
            raise ValueError("Missing class_path in transformation dictionary.")

        target_cls = get_class(class_path)
        if hasattr(target_cls, "_from_dict_impl"):
            return target_cls._from_dict_impl(data)

        return target_cls(**data.get("params", {}))


class LogReturnTransform(BaseTransform):
    """Calculates logarithmic returns: ln(p_t / p_{t-1}) for a targeted column or all columns."""

    def __init__(self, col_idx: Any = None) -> None:
        self.col_idx = col_idx

    def transform(self, X: Any) -> np.ndarray:
        X_np = np.array(X, dtype=np.float32)

        # 1. Extract the column if targeted, otherwise keep the full matrix
        if self.col_idx is not None and X_np.ndim == 2:
            if self.col_idx >= X_np.shape[1]:
                raise ValueError(
                    f"Column index {self.col_idx} out of bounds for shape {X_np.shape}"
                )
            col = X_np[:, self.col_idx]
        else:
            col = X_np

        # 2. Return an empty array of correct shape if we don't have enough rows to differentiate
        if len(col) < 2:
            if X_np.ndim == 2:
                n_cols = X_np.shape[1] if self.col_idx is None else 1
                return np.empty((0, n_cols), dtype=np.float32)
            return np.empty((0,), dtype=np.float32)

        # 3. Calculate log returns: ln(x_t / x_{t-1})
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = np.diff(np.log(col), axis=0)
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. Ensure shape matching (maintain column vector for 2D inputs)
        if X_np.ndim == 2 and self.col_idx is not None:
            return returns.reshape(-1, 1)
        return returns

    def to_dict(self) -> dict:
        return {
            "class_path": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "params": {"col_idx": self.col_idx},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LogReturnTransform":
        return cls(**data.get("params", {}))


class RatioTransform(BaseTransform):
    """Calculates log ratio between two columns: ln(X[num_idx] / X[den_idx])."""

    def __init__(self, num_idx: int, den_idx: int) -> None:
        self.num_idx = num_idx
        self.den_idx = den_idx

    def transform(self, X: Any) -> np.ndarray:
        X_np = np.array(X, dtype=np.float32)
        if X_np.ndim != 2:
            raise ValueError("RatioTransform requires a 2D input matrix.")

        if X_np.shape[1] <= self.num_idx or X_np.shape[1] <= self.den_idx:
            raise ValueError(
                f"Indices {self.num_idx}, {self.den_idx} out of bounds for shape {X_np.shape}"
            )

        num = X_np[:, self.num_idx]
        den = X_np[:, self.den_idx]

        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.log(num / den)
            ratios = np.nan_to_num(ratios, nan=0.0, posinf=0.0, neginf=0.0)

        return ratios.reshape(-1, 1)

    def to_dict(self) -> dict:
        return {
            "class_path": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "params": {"num_idx": self.num_idx, "den_idx": self.den_idx},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RatioTransform":
        return cls(**data.get("params", {}))


class FeaturePipeline(BaseTransform):
    """
    Orchestrates multiple individual BaseTransform instances.
    Runs them in parallel, aligns their lengths, and stacks them horizontally.
    """

    def __init__(self, transforms: List[BaseTransform]) -> None:
        self.transforms = list(transforms)

    def fit(self, X: Any, y: Any = None) -> "FeaturePipeline":
        for transform in self.transforms:
            transform.fit(X, y)
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not self.transforms:
            raise ValueError("FeaturePipeline has no configured transformations.")

        X_np = np.array(X, dtype=np.float32)

        # Apply each transform
        outputs = [t.transform(X_np) for t in self.transforms]

        # Find minimum sequence length
        min_len = min(len(out) for out in outputs)

        # Align all outputs to the same end (chronological order)
        aligned_outputs = []
        for out in outputs:
            aligned_outputs.append(out[-min_len:])

        # Horizontally stack aligned arrays
        return np.hstack(aligned_outputs)

    def to_dict(self) -> dict:
        return {
            "class_path": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "params": {"transforms": [t.to_dict() for t in self.transforms]},
        }

    @classmethod
    def _from_dict_impl(cls, data: dict) -> "FeaturePipeline":
        params = data.get("params", {})
        transforms_data = params.get("transforms", [])
        transforms = [BaseTransform.from_dict(t) for t in transforms_data]
        return cls(transforms=transforms)
