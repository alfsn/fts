# src/trading_bot/risk_management/manager.py

import logging
from typing import Dict, Optional

from ..core.enums import OrderSide, SignalType
from ..core.schemas import (
    MarketData,
    OrderRequest,
    PortfolioState,
    SizingInput,
    SizingOutput,
    TradeSignal,
)
from .abc import BaseSizingStrategy
from .portfolio import Portfolio

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Coordinates sizing and risk management for trade signals.

    This class consumes signals from the Strategy Engine, uses an
    injected sizing strategy (e.g., Kelly Criterion) to calculate
    a potential order size, and then applies a set of portfolio-level
    risk checks (e.g., available balance, max allocation) before
    issuing a final OrderRequest.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        sizer: BaseSizingStrategy,
        max_allocation_per_market: float = 0.25,
        max_total_positions: int = 10,
    ):
        """
        Initializes the RiskManager.

        :param portfolio: An instance of the Portfolio class to get state.
        :param sizer: A concrete implementation of BaseSizingStrategy.
        :param max_allocation_per_market: The max % of total equity to
                                          allocate to a single market.
        :param max_total_positions: The max number of concurrent
                                    open positions.
        """
        self.portfolio = portfolio
        self.sizer = sizer
        self.max_allocation_per_market = max_allocation_per_market
        self.max_total_positions = max_total_positions
        logger.info(
            f"RiskManager initialized with sizer: {sizer.strategy_type.value}, "
            f"max_allocation: {max_allocation_per_market*100}%, "
            f"max_positions: {max_total_positions}"
        )

    def process_signal(
        self, signal: TradeSignal, market_data_map: Dict[str, MarketData]
    ) -> Optional[OrderRequest]:
        """
        Processes a single TradeSignal and returns a valid OrderRequest
        if all risk checks pass, otherwise returns None.

        :param signal: The TradeSignal from the Strategy Engine.
        :param market_data_map: The latest map of all market data.
        :return: An OrderRequest if the trade is approved, else None.
        """
        # 1. Ignore HOLD signals
        if signal.signal_type == SignalType.HOLD:
            return None

        # 2. Get current portfolio and market state
        portfolio_state = self.portfolio.get_state(market_data_map)
        market_data = market_data_map.get(signal.market_id)

        if not market_data:
            logger.warning(
                f"No market data for {signal.market_id}. " "Cannot process signal."
            )
            return None

        # 3. Delegate to the Sizing Strategy
        sizing_input = SizingInput(
            signal=signal,
            market_data=market_data,
            portfolio_state=portfolio_state,
        )
        sizing_output = self.sizer.calculate_size(sizing_input)

        # 4. Run Risk Checks
        if not self._passes_risk_checks(
            sizing_output, signal, portfolio_state, market_data_map
        ):
            return None  # Risk checks failed, (logs inside)

        # 5. All checks passed. Create the final OrderRequest.
        # We use the price calculated by the sizer (amount / shares)
        # as the limit price for the order.
        order_amount_usdc = sizing_output.amount_usdc
        order_shares = sizing_output.size_shares
        limit_price = order_amount_usdc / order_shares
        order_side = (
            OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL
        )

        final_order = OrderRequest(
            market_id=signal.market_id,
            side=order_side,
            size=order_shares,
            price=limit_price,
        )

        logger.info(
            f"RiskManager approved order for {signal.market_id}: "
            f"{order_side.value} {order_shares:.2f} shares "
            f"@ ${limit_price:.4f}"
        )
        return final_order

    def _passes_risk_checks(
        self,
        sizing_output: SizingOutput,
        signal: TradeSignal,
        portfolio_state: PortfolioState,
        market_data_map: Dict[str, MarketData],
    ) -> bool:
        """
        A helper method to run a chain of risk validation checks.
        """
        order_amount_usdc = sizing_output.amount_usdc
        order_shares = sizing_output.size_shares

        # Check 1: Sizer returned non-zero size
        if order_amount_usdc <= 1e-6 or order_shares <= 1e-6:
            logger.info(
                f"Sizer returned zero size for {signal.market_id}. " "No order."
            )
            return False

        # Check 2: Available Balance (for BUYs)
        if (
            signal.signal_type == SignalType.BUY
            and order_amount_usdc > portfolio_state.available_balance_usdc
        ):
            logger.warning(
                f"Order for {signal.market_id} rejected. "
                f"Cost ${order_amount_usdc:.2f} exceeds available "
                f"balance ${portfolio_state.available_balance_usdc:.2f}."
            )
            return False

        # (Note: A SELL check would verify we have shares to sell,
        # but our Portfolio logic handles net positions, so this
        # balance check is the most critical pre-trade check.)

        # Check 3: Max Total Positions (if opening a new position)
        is_new_position = True
        for pos in portfolio_state.positions:
            if pos.market_id == signal.market_id:
                is_new_position = False
                break

        if (
            is_new_position
            and len(portfolio_state.positions) >= self.max_total_positions
        ):
            logger.warning(
                f"Order for {signal.market_id} rejected. "
                f"Would exceed max positions ({self.max_total_positions})."
            )
            return False

        # Check 4: Max Allocation per Market
        total_equity = portfolio_state.total_balance_usdc
        if total_equity <= 0:
            logger.error("Total equity is zero or negative. Cannot trade.")
            return False

        # Find current value of all positions in this market
        current_market_value = 0.0
        pnl_map = self.portfolio.calculate_unrealized_pnl(market_data_map)
        for pos in portfolio_state.positions:
            if pos.market_id == signal.market_id:
                pnl = pnl_map.get(pos.market_id, 0.0)
                current_market_value += (pos.size * pos.entry_price) + pnl

        new_total_allocation = current_market_value + order_amount_usdc
        allocation_pct = new_total_allocation / total_equity

        if allocation_pct > self.max_allocation_per_market:
            logger.warning(
                f"Order for {signal.market_id} rejected. "
                f"New allocation ({allocation_pct*100:.1f}%) would "
                f"exceed max ({self.max_allocation_per_market*100:.1f}%)."
            )
            return False

        return True
