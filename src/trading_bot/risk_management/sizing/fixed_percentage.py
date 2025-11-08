# src/trading_bot/risk_management/sizing/fixed_percentage.py

import logging

from ...core.enums import SignalType, SizingStrategyType
from ...core.schemas import SizingInput, SizingOutput
from ..abc import BaseSizingStrategy

logger = logging.getLogger(__name__)


class FixedPercentageSizer(BaseSizingStrategy):
    """
    Implements a fixed-percentage sizing strategy.

    This sizer allocates a percentage of the total portfolio
    equity to each trade.
    """

    def __init__(self, default_percentage: float = 0.01):
        """
        :param default_percentage: The fixed percentage of total equity
                                   to allocate (e.g., 0.01 for 1%).
        """
        if not 0 < default_percentage <= 1.0:
            raise ValueError("default_percentage must be between 0 and 1.")
        self.default_percentage = default_percentage
        logger.info(
            f"FixedPercentageSizer initialized with percentage: "
            f"{default_percentage * 100:.2f}%"
        )

    @property
    def strategy_type(self) -> SizingStrategyType:
        """Returns the type of this sizing strategy."""
        return SizingStrategyType.FIXED_PERCENTAGE

    def calculate_size(self, input_data: SizingInput) -> SizingOutput:
        """
        Calculates the trade size based on a fixed portfolio percentage.

        :param input_data: The SizingInput data packet.
        :return: A SizingOutput object.
        """
        if input_data.signal.signal_type == SignalType.HOLD:
            return SizingOutput(amount_usdc=0, size_shares=0)

        book = input_data.market_data.order_book
        price = 0.0

        try:
            if input_data.signal.signal_type == SignalType.BUY:
                if not book.asks:
                    logger.warning(
                        f"No asks available for {input_data.signal.market_id}, "
                        "cannot calculate buy size."
                    )
                    return SizingOutput(amount_usdc=0, size_shares=0)
                price = book.asks[0].price  # Best ask price
            else:  # SignalType.SELL
                if not book.bids:
                    logger.warning(
                        f"No bids available for {input_data.signal.market_id}, "
                        "cannot calculate sell size."
                    )
                    return SizingOutput(amount_usdc=0, size_shares=0)
                price = book.bids[0].price  # Best bid price
        except IndexError:
            logger.error(
                f"Order book list was empty for {input_data.signal.market_id} "
                "despite check. Cannot size."
            )
            return SizingOutput(amount_usdc=0, size_shares=0)

        if price <= 1e-6:  # Avoid division by zero
            logger.warning(
                f"Invalid price ({price}) for {input_data.signal.market_id}. "
                "Returning zero size."
            )
            return SizingOutput(amount_usdc=0, size_shares=0)

        # Calculate the USDC amount based on total portfolio equity
        total_equity = input_data.portfolio_state.total_balance_usdc
        amount_usdc = total_equity * self.default_percentage

        size_shares = amount_usdc / price
        return SizingOutput(amount_usdc=amount_usdc, size_shares=size_shares)
