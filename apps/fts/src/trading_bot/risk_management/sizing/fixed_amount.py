# src/trading_bot/risk_management/sizing/fixed_amount.py

import logging

from ...core.enums import SignalType, SizingStrategyType
from ...core.schemas import SizingInput, SizingOutput
from ..abc import BaseSizingStrategy

logger = logging.getLogger(__name__)


class FixedAmountSizer(BaseSizingStrategy):
    """
    Implements a simple fixed-amount sizing strategy.

    This sizer allocates a constant, predefined amount of quote currency to
    every trade, regardless of portfolio size or confidence.
    """

    def __init__(self, default_amount_quote: float = 10.0) -> None:
        """
        :param default_amount_quote: The fixed amount of quote currency to
                                    allocate per trade.
        """
        if default_amount_quote <= 0:
            raise ValueError("default_amount_quote must be positive.")
        self.default_amount_quote = default_amount_quote
        logger.info(
            f"FixedAmountSizer initialized with amount: ${default_amount_quote}"
        )

    @property
    def strategy_type(self) -> SizingStrategyType:
        """Returns the type of this sizing strategy."""
        return SizingStrategyType.FIXED_AMOUNT

    def calculate_size(self, input_data: SizingInput) -> SizingOutput:
        """
        Calculates the trade size based on a fixed amount.

        :param input_data: The SizingInput data packet.
        :return: A SizingOutput object.
        """
        if input_data.signal.signal_type == SignalType.HOLD:
            return SizingOutput(amount_quote=0, size_shares=0)

        price = self._get_execution_price(input_data)

        if price <= 1e-6:  # Avoid division by zero
            logger.warning(
                f"Invalid price ({price}) for {input_data.signal.market_id}. "
                "Returning zero size."
            )
            return SizingOutput(amount_quote=0, size_shares=0)

        size_shares = self.default_amount_quote / price
        return SizingOutput(
            amount_quote=self.default_amount_quote, size_shares=size_shares
        )
