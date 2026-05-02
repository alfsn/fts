# src/trading_bot/core/transforms.py

import math
from abc import ABC, abstractmethod
from typing import List, Sequence


class BaseTransform(ABC):
    """Abstract base for data transformations."""

    @abstractmethod
    def transform(self, data: Sequence[float]) -> List[float]:
        pass


class LogReturnTransform(BaseTransform):
    """Calculates logarithmic returns: ln(p_t / p_{t-1})."""

    def transform(self, data: Sequence[float]) -> List[float]:
        if len(data) < 2:
            return []

        returns = []
        for i in range(1, len(data)):
            if data[i - 1] == 0:
                returns.append(0.0)
            else:
                returns.append(math.log(data[i] / data[i - 1]))
        return returns
