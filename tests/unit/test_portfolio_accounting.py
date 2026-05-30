# tests/unit/test_portfolio_accounting.py

import time
from unittest.mock import MagicMock

import pytest

from trading_bot.core.enums import OrderSide, OrderStatus
from trading_bot.core.loop import RealTimePollingLoop
from trading_bot.core.pipeline import TradingPipeline
from trading_bot.core.schemas import ExecutionResult, OrderRequest
from trading_bot.risk_management.portfolio import Portfolio


def test_portfolio_accounting_long_reduce():
    """
    Verifies that reducing a Long position correctly updates the cash balance
    by transaction flows without double-counting realized P&L.
    """
    portfolio = Portfolio(initial_balance=1000.0, quote_currency="USD")

    # 1. Buy 10 shares at $100
    buy_order = OrderRequest(
        market_id="BTC-USD",
        side=OrderSide.BUY,
        size=10.0,
        price=100.0,
    )
    buy_result = ExecutionResult(
        order_id="order-1",
        status=OrderStatus.FILLED,
        filled_size=10.0,
        avg_price=100.0,
        timestamp=MagicMock(),
    )
    portfolio.add_open_order("order-1", buy_order)
    portfolio.update_order_status(db=None, result=buy_result)

    assert portfolio._cash_balance == 0.0
    assert "BTC-USD" in portfolio._positions
    assert portfolio._positions["BTC-USD"].size == 10.0
    assert portfolio._positions["BTC-USD"].entry_price == 100.0

    # 2. Sell 5 shares at $120 (realized P&L should be $100, cash should be $600)
    sell_order = OrderRequest(
        market_id="BTC-USD",
        side=OrderSide.SELL,
        size=5.0,
        price=120.0,
    )
    sell_result = ExecutionResult(
        order_id="order-2",
        status=OrderStatus.FILLED,
        filled_size=5.0,
        avg_price=120.0,
        timestamp=MagicMock(),
    )
    portfolio.add_open_order("order-2", sell_order)
    portfolio.update_order_status(db=None, result=sell_result)

    # Cash balance MUST be $600 (not $700, which would happen if P&L was double counted)
    assert portfolio._cash_balance == 600.0
    assert portfolio._positions["BTC-USD"].size == 5.0
    assert portfolio._positions["BTC-USD"].entry_price == 100.0


def test_portfolio_accounting_short_reduce():
    """
    Verifies that buying to cover a Short position correctly updates the cash balance
    by transaction flows without double-counting realized P&L.
    """
    portfolio = Portfolio(initial_balance=1000.0, quote_currency="USD")

    # 1. Short 10 shares at $100 (SELL order)
    short_order = OrderRequest(
        market_id="BTC-USD",
        side=OrderSide.SELL,
        size=10.0,
        price=100.0,
    )
    short_result = ExecutionResult(
        order_id="order-1",
        status=OrderStatus.FILLED,
        filled_size=10.0,
        avg_price=100.0,
        timestamp=MagicMock(),
    )
    portfolio.add_open_order("order-1", short_order)
    portfolio.update_order_status(db=None, result=short_result)

    assert portfolio._cash_balance == 2000.0
    assert "BTC-USD" in portfolio._positions
    assert portfolio._positions["BTC-USD"].size == -10.0
    assert portfolio._positions["BTC-USD"].entry_price == 100.0

    # 2. Cover 5 shares at $80 (BUY order, realized P&L should be $100, cash should be $1600)
    cover_order = OrderRequest(
        market_id="BTC-USD",
        side=OrderSide.BUY,
        size=5.0,
        price=80.0,
    )
    cover_result = ExecutionResult(
        order_id="order-2",
        status=OrderStatus.FILLED,
        filled_size=5.0,
        avg_price=80.0,
        timestamp=MagicMock(),
    )
    portfolio.add_open_order("order-2", cover_order)
    portfolio.update_order_status(db=None, result=cover_result)

    # Cash balance MUST be $1600 (not $1700, which would happen if P&L was double counted)
    assert portfolio._cash_balance == 1600.0
    assert portfolio._positions["BTC-USD"].size == -5.0
    assert portfolio._positions["BTC-USD"].entry_price == 100.0


def test_portfolio_accounting_long_flip():
    """
    Verifies that flipping a position from Long to Short correctly updates the cash
    balance by transaction flows without double-counting realized P&L.
    """
    portfolio = Portfolio(initial_balance=1000.0, quote_currency="USD")

    # 1. Buy 10 shares at $100
    buy_order = OrderRequest(
        market_id="BTC-USD",
        side=OrderSide.BUY,
        size=10.0,
        price=100.0,
    )
    buy_result = ExecutionResult(
        order_id="order-1",
        status=OrderStatus.FILLED,
        filled_size=10.0,
        avg_price=100.0,
        timestamp=MagicMock(),
    )
    portfolio.add_open_order("order-1", buy_order)
    portfolio.update_order_status(db=None, result=buy_result)

    # 2. Sell 15 shares at $120 (flips to Short of -5, cash should be $1800)
    flip_order = OrderRequest(
        market_id="BTC-USD",
        side=OrderSide.SELL,
        size=15.0,
        price=120.0,
    )
    flip_result = ExecutionResult(
        order_id="order-2",
        status=OrderStatus.FILLED,
        filled_size=15.0,
        avg_price=120.0,
        timestamp=MagicMock(),
    )
    portfolio.add_open_order("order-2", flip_order)
    portfolio.update_order_status(db=None, result=flip_result)

    # Cash balance MUST be $1800 (not $2000, which would happen if P&L was double counted)
    assert portfolio._cash_balance == 1800.0
    assert portfolio._positions["BTC-USD"].size == -5.0
    assert portfolio._positions["BTC-USD"].entry_price == 120.0


def test_real_time_polling_loop_drift_correction():
    """
    Verifies that the RealTimePollingLoop timing timing corrects for clock drift.
    """
    mock_pipeline = MagicMock(spec=TradingPipeline)

    # We define a function for execute_single_tick that simulates execution latency
    def simulate_tick_latency(*args, **kwargs):
        time.sleep(0.04)

    mock_pipeline.execute_single_tick = simulate_tick_latency

    loop_driver = RealTimePollingLoop(interval_seconds=0.1, max_ticks=2)

    start_time = time.time()
    loop_driver.start(pipeline=mock_pipeline)
    total_elapsed = time.time() - start_time

    # Since ticks execute in 0.04s, the remaining sleep time per tick should be 0.06s.
    # Total loop time for 2 ticks should be:
    # Tick 1 start -> execute (0.04s) -> sleep remaining (0.06s) -> Tick 2 start -> execute (0.04s) -> sleep remaining (0.06s) -> Stop.
    # Total expected elapsed time is exactly 0.2s.
    # If there was drift (sleeping standard 0.1s on top of tick), total time would be ~0.28s.
    # We assert that the total elapsed time is close to 0.20s (tolerating small OS scheduling differences).
    assert total_elapsed < 0.26
