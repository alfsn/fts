# src/trading_bot/risk_management/sizing/kelly_criterion.py

import logging

from ...core.enums import SignalType, SizingStrategyType
from ...core.schemas import SizingInput, SizingOutput
from ..abc import BaseSizingStrategy

logger = logging.getLogger(__name__)


class KellyCriterionSizer(BaseSizingStrategy):
    """
    Implements the Kelly Criterion for optimal position sizing.

    The formula is: f = (bp - q) / b
    where:
    - f = fraction of bankroll to wager
    - p = probability of winning (from signal.confidence)
    - q = probability of losing (1 - p)
    - b = net odds (payout / risk)

    For a BUY:
    - We bet `price` to win `(1 - price)`. So, `b = (1 - price) / price`.
    - `p` is our confidence the outcome WILL happen.
    - `f = (p * b - q) / b`

    For a SELL:
    - We bet `(1 - price)` to win `price`. So, `b = price / (1 - price)`.
    - Our "win" is the outcome NOT happening, so we use `q` as `p_win`.
    - `f = (q * b - p) / b`
    """

    def __init__(self, kelly_fraction: float = 0.5):
        """
        :param kelly_fraction: A safety multiplier (e.g., 0.5 for
                             "half-Kelly") to reduce risk.
        """
        if not 0 < kelly_fraction <= 1.0:
            raise ValueError("kelly_fraction must be between 0 and 1.")
        self.kelly_fraction = kelly_fraction
        logger.info(f"KellyCriterionSizer initialized with fraction: {kelly_fraction}")

    @property
    def strategy_type(self) -> SizingStrategyType:
        """Returns the type of this sizing strategy."""
        return SizingStrategyType.KELLY_CRITERION

    def calculate_size(self, input_data: SizingInput) -> SizingOutput:
        """
        Calculates the trade size using the Kelly Criterion.

        :param input_data: The SizingInput data packet.
        :return: A SizingOutput object.
        """
        if input_data.signal.signal_type == SignalType.HOLD:
            return SizingOutput(amount_quote=0, size_shares=0)

        book = input_data.market_data.order_book
        price = 0.0
        p_true = input_data.signal.confidence
        q_true = 1.0 - p_true
        f_kelly = 0.0

        try:
            if input_data.signal.signal_type == SignalType.BUY:
                if not book.asks:
                    logger.warning(
                        f"No asks available for {input_data.signal.market_id}, "
                        "cannot calculate Kelly size."
                    )
                    return SizingOutput(amount_quote=0, size_shares=0)
                price = book.asks[0].price
                if price >= 0.9999:  # No potential profit
                    return SizingOutput(amount_quote=0, size_shares=0)

                b_odds = (1.0 - price) / price
                f_kelly = (p_true * b_odds - q_true) / b_odds

            else:  # SignalType.SELL
                if not book.bids:
                    logger.warning(
                        f"No bids available for {input_data.signal.market_id}, "
                        "cannot calculate Kelly size."
                    )
                    return SizingOutput(amount_quote=0, size_shares=0)
                price = book.bids[0].price
                if price <= 0.0001:  # No potential profit
                    return SizingOutput(amount_quote=0, size_shares=0)

                b_odds = price / (1.0 - price)
                f_kelly = (q_true * b_odds - p_true) / b_odds

        except (IndexError, ZeroDivisionError) as e:
            logger.error(
                f"Error calculating Kelly odds for {input_data.signal.market_id}: {e}"
            )
            return SizingOutput(amount_quote=0, size_shares=0)

        # If Kelly fraction is negative or zero, there is no edge.
        if f_kelly <= 0:
            logger.info(
                f"No edge found for {input_data.signal.market_id}. "
                f"(p={p_true:.2f}, price={price:.2f}). No order."
            )
            return SizingOutput(amount_quote=0, size_shares=0)

        # Apply safety fraction
        f_safe = f_kelly * self.kelly_fraction

        # Calculate final size
        total_equity = input_data.portfolio_state.total_balance_quote
        amount_quote = total_equity * f_safe
        size_shares = amount_quote / price

        return SizingOutput(amount_quote=amount_quote, size_shares=size_shares)
