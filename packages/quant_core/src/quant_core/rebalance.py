from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional

from .models import FXContext, Portfolio


class RebalanceStrategy(ABC):
    @abstractmethod
    def should_rebalance(
        self, portfolio: Portfolio, current_time: datetime, fx_context: FXContext
    ) -> bool:
        pass

    @abstractmethod
    def generate_target_weights(self, portfolio: Portfolio) -> Dict[str, float]:
        pass

    @abstractmethod
    def update_state(self, current_time: datetime) -> None:
        pass


class PeriodicRebalanceStrategy(RebalanceStrategy):
    def __init__(self, target_weights: Dict[str, float], interval_days: int):
        self.target_weights = target_weights
        self.interval_days = interval_days
        self.last_rebalance_time: Optional[datetime] = None

    def should_rebalance(
        self, portfolio: Portfolio, current_time: datetime, fx_context: FXContext
    ) -> bool:
        if self.last_rebalance_time is None:
            return True
        return (current_time - self.last_rebalance_time).days >= self.interval_days

    def generate_target_weights(self, portfolio: Portfolio) -> Dict[str, float]:
        return self.target_weights

    def update_state(self, current_time: datetime) -> None:
        self.last_rebalance_time = current_time


class DeviationRebalanceStrategy(RebalanceStrategy):
    def __init__(self, target_weights: Dict[str, float], max_deviation: float):
        self.target_weights = target_weights
        self.max_deviation = max_deviation

    def should_rebalance(
        self, portfolio: Portfolio, current_time: datetime, fx_context: FXContext
    ) -> bool:
        total_value = portfolio.total_value(fx_context)
        if total_value == 0:
            return False

        for position in portfolio.positions:
            rate = fx_context.get_rate(position.currency)
            current_weight = (
                (position.current_price or 0.0) * position.quantity * rate
            ) / total_value
            target = self.target_weights.get(position.instrument_id, 0.0)
            if abs(current_weight - target) > self.max_deviation:
                return True
        return False

    def generate_target_weights(self, portfolio: Portfolio) -> Dict[str, float]:
        return self.target_weights

    def update_state(self, current_time: datetime) -> None:
        pass
