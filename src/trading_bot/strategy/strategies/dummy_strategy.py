# src/trading_bot/strategy/strategies/dummy_strategy.py

from typing import Sequence

from ...core.schemas import IngestionEngineOutput, SignalType, TradeSignal
from ..abc import BaseStrategy


class DummyStrategy(BaseStrategy):
    """
    A simple strategy that always generates a BUY signal if data is present.
    Used for testing the StrategyEngine and BacktestSimulator.
    """

    @property
    def name(self) -> str:
        return "dummy_strategy"

    def evaluate(self, data: IngestionEngineOutput) -> Sequence[TradeSignal]:
        signals = []
        for market_id, market_data in data.market_data.items():
            # Simply check if we have any price data
            if market_data.recent_bars:
                signals.append(
                    TradeSignal(
                        market_id=market_id,
                        strategy_name=self.name,
                        signal_type=SignalType.BUY,
                        confidence=1.0,
                    )
                )
        return signals
