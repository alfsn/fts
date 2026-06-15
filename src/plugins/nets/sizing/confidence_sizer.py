# src/plugins/nets/sizing/confidence_sizer.py

from trading_bot.core.enums import SignalType, SizingStrategyType
from trading_bot.core.schemas import SizingInput, SizingOutput
from trading_bot.risk_management.abc import BaseSizingStrategy


class ConfidenceSizer(BaseSizingStrategy):
    """
    Sizes positions based on the confidence of the signal.
    Size = base_amount * confidence
    """

    def __init__(self, base_amount_quote: float) -> None:
        self.base_amount_quote = base_amount_quote

    @property
    def strategy_type(self) -> SizingStrategyType:
        # We'll use FIXED_AMOUNT as the base type for this custom sizer
        return SizingStrategyType.FIXED_AMOUNT

    def calculate_size(self, input_data: SizingInput) -> SizingOutput:
        if input_data.signal.signal_type not in (SignalType.BUY, SignalType.SELL):
            return SizingOutput(amount_quote=0.0, size_shares=0.0)

        confidence = input_data.signal.confidence

        # Linear scaling: 0% confidence -> 0 size, 100% confidence -> base_amount
        target_amount = self.base_amount_quote * confidence

        # Calculate shares based on current price
        # Using midpoint or last close as estimate
        price = (
            input_data.market_data.recent_bars[-1].close
            if input_data.market_data.recent_bars
            else 0.0
        )

        if price == 0:
            return SizingOutput(amount_quote=0.0, size_shares=0.0)

        size_shares = target_amount / price

        return SizingOutput(amount_quote=target_amount, size_shares=size_shares)
