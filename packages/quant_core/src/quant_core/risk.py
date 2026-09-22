from abc import ABC, abstractmethod
from typing import List, Optional

from .models import FXContext, Portfolio


class RuleViolation(Exception):
    def __init__(self, rule_name: str, message: str):
        super().__init__(f"[{rule_name}] {message}")
        self.rule_name = rule_name
        self.message = message


class PortfolioRule(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def evaluate(
        self, portfolio: Portfolio, fx_context: FXContext
    ) -> Optional[RuleViolation]:
        pass


class MaxAssetWeight(PortfolioRule):
    def __init__(self, instrument_id: str, max_weight: float):
        self.instrument_id = instrument_id
        self.max_weight = max_weight

    @property
    def name(self) -> str:
        return f"MaxAssetWeight({self.instrument_id}, {self.max_weight})"

    def evaluate(
        self, portfolio: Portfolio, fx_context: FXContext
    ) -> Optional[RuleViolation]:
        total_val = portfolio.total_value(fx_context)
        if total_val == 0:
            return None

        for p in portfolio.positions:
            if p.instrument_id == self.instrument_id:
                rate = fx_context.get_rate(p.currency)
                weight = ((p.current_price or 0.0) * p.quantity * rate) / total_val
                if weight > self.max_weight:
                    return RuleViolation(
                        self.name,
                        f"Weight {weight:.2%} exceeds max {self.max_weight:.2%}",
                    )

        return None


class RiskEngine:
    def __init__(self, rules: List[PortfolioRule]):
        self.rules = rules

    def evaluate_portfolio(
        self, portfolio: Portfolio, fx_context: FXContext
    ) -> List[RuleViolation]:
        violations = []
        for rule in self.rules:
            violation = rule.evaluate(portfolio, fx_context)
            if violation:
                violations.append(violation)
        return violations
