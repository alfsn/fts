# tests/unit/test_simulated_handler.py

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from trading_bot.core.enums import BarType, OrderSide, OrderStatus, OrderType
from trading_bot.core.schemas import (
    BarData,
    ExecutionResult,
    IngestionEngineOutput,
    MarketData,
    MarketDetails,
    OrderRequest,
)
from trading_bot.execution.delay import KBarExecuteDelay
from trading_bot.execution.engine import ExecutionEngine
from trading_bot.execution.handlers.simulated_handler import SimulatedExecutionHandler
from trading_bot.execution.slippage import FlatPriceSlip
from trading_bot.risk_management.portfolio import Portfolio


def test_validation():
    # KBar delay must be >= 1
    with pytest.raises(ValueError, match="Execution delay shift k must be >= 1"):
        KBarExecuteDelay(k=0)

    with pytest.raises(ValueError, match="Execution delay shift k must be >= 1"):
        KBarExecuteDelay(k=-1)

    # SimulatedExecutionHandler validation for execution_price_source
    delay = KBarExecuteDelay(k=1)
    slip = FlatPriceSlip(slippage_pct=0.0)
    with pytest.raises(
        ValueError, match="execution_price_source must be 'open' or 'close'"
    ):
        SimulatedExecutionHandler(
            delay_model=delay,
            slippage_model=slip,
            execution_price_source="invalid",
        )


def test_simulated_handler_delay_and_price_source():
    delay = KBarExecuteDelay(k=2)
    slip = FlatPriceSlip(slippage_pct=0.0)  # No slippage
    handler = SimulatedExecutionHandler(
        delay_model=delay,
        slippage_model=slip,
        execution_price_source="close",
        initial_balances={"USD": 1000.0},
    )

    # Place order
    order = OrderRequest(
        market_id="GGAL",
        side=OrderSide.BUY,
        size=10.0,
        price=100.0,
        order_type=OrderType.MARKET,
    )

    # At start, GGAL tick count is 0 because no ticks have arrived yet
    res = handler.execute_order(order)
    assert res.status == OrderStatus.OPEN
    order_id = res.order_id

    # 1. First tick arrives
    details = MarketDetails(
        market_id="GGAL",
        name="GGAL",
        end_date=datetime.now(timezone.utc),
        resolution_source="test",
    )
    bar1 = BarData(
        timestamp=datetime(2026, 6, 21, 10, 0, tzinfo=timezone.utc),
        open=95.0,
        high=98.0,
        low=94.0,
        close=97.0,
        volume=100.0,
        bar_type=BarType.TIME,
        ticks_count=5,
        dollar_volume=9700.0,
    )
    mdata1 = MarketData(
        market_id="GGAL",
        details=details,
        recent_bars=[bar1],
    )
    tick1 = IngestionEngineOutput(
        timestamp=bar1.timestamp,
        market_data={"GGAL": mdata1},
        external_data=[],
        bars={"GGAL": [bar1]},
    )

    handler.on_tick(tick1)

    # Order is scheduled for signal_tick (0) + k (2) = 2.
    # Current tick is 1. Order should still be open (not filled)
    status_res = handler.get_order_status(order_id)
    assert status_res.status == OrderStatus.OPEN

    # 2. Second tick arrives (tick count = 2)
    bar2 = BarData(
        timestamp=datetime(2026, 6, 21, 10, 5, tzinfo=timezone.utc),
        open=97.0,
        high=102.0,
        low=96.0,
        close=101.0,
        volume=120.0,
        bar_type=BarType.TIME,
        ticks_count=5,
        dollar_volume=12120.0,
    )
    mdata2 = MarketData(
        market_id="GGAL",
        details=details,
        recent_bars=[bar2],
    )
    tick2 = IngestionEngineOutput(
        timestamp=bar2.timestamp,
        market_data={"GGAL": mdata2},
        external_data=[],
        bars={"GGAL": [bar2]},
    )

    handler.on_tick(tick2)

    # Current tick is 2. Target tick 2 reached. Order must be filled at close of bar2 (101.0)
    status_res = handler.get_order_status(order_id)
    assert status_res.status == OrderStatus.FILLED
    assert status_res.avg_price == 101.0
    assert status_res.filled_size == 10.0
    assert handler.get_account_balances()["USD"] == 1000.0 - (10.0 * 101.0)


def test_simulated_handler_slippage():
    # 1. BUY order slippage
    delay = KBarExecuteDelay(k=1)
    slip = FlatPriceSlip(slippage_pct=0.01)  # 1% slippage
    handler = SimulatedExecutionHandler(
        delay_model=delay,
        slippage_model=slip,
        execution_price_source="open",
        initial_balances={"USD": 1000.0},
    )

    # Feed initial tick so GGAL tick index = 1
    details = MarketDetails(
        market_id="GGAL",
        name="GGAL",
        end_date=datetime.now(timezone.utc),
        resolution_source="test",
    )
    bar1 = BarData(
        timestamp=datetime(2026, 6, 21, 10, 0, tzinfo=timezone.utc),
        open=95.0,
        high=98.0,
        low=94.0,
        close=97.0,
        volume=100.0,
        bar_type=BarType.TIME,
        ticks_count=5,
        dollar_volume=9700.0,
    )
    mdata1 = MarketData(
        market_id="GGAL",
        details=details,
        recent_bars=[bar1],
    )
    tick1 = IngestionEngineOutput(
        timestamp=bar1.timestamp,
        market_data={"GGAL": mdata1},
        external_data=[],
        bars={"GGAL": [bar1]},
    )
    handler.on_tick(tick1)

    # Place BUY order at tick index 1, execution at 1 + 1 = 2
    order_buy = OrderRequest(
        market_id="GGAL",
        side=OrderSide.BUY,
        size=5.0,
        price=100.0,
        order_type=OrderType.MARKET,
    )
    res_buy = handler.execute_order(order_buy)
    buy_id = res_buy.order_id

    # 2. Tick 2 arrives
    bar2 = BarData(
        timestamp=datetime(2026, 6, 21, 10, 5, tzinfo=timezone.utc),
        open=100.0,
        high=102.0,
        low=96.0,
        close=101.0,
        volume=120.0,
        bar_type=BarType.TIME,
        ticks_count=5,
        dollar_volume=12120.0,
    )
    mdata2 = MarketData(
        market_id="GGAL",
        details=details,
        recent_bars=[bar2],
    )
    tick2 = IngestionEngineOutput(
        timestamp=bar2.timestamp,
        market_data={"GGAL": mdata2},
        external_data=[],
        bars={"GGAL": [bar2]},
    )
    handler.on_tick(tick2)

    # Filled at open of bar 2 (100.0) + 1% slippage = 101.0
    status_buy = handler.get_order_status(buy_id)
    assert status_buy.status == OrderStatus.FILLED
    assert status_buy.avg_price == 101.0

    # 3. SELL order slippage
    # Place SELL order at tick index 2, execution at 2 + 1 = 3
    order_sell = OrderRequest(
        market_id="GGAL",
        side=OrderSide.SELL,
        size=5.0,
        price=100.0,
        order_type=OrderType.MARKET,
    )
    res_sell = handler.execute_order(order_sell)
    sell_id = res_sell.order_id

    # Tick 3 arrives
    bar3 = BarData(
        timestamp=datetime(2026, 6, 21, 10, 10, tzinfo=timezone.utc),
        open=100.0,
        high=102.0,
        low=96.0,
        close=99.0,
        volume=120.0,
        bar_type=BarType.TIME,
        ticks_count=5,
        dollar_volume=12120.0,
    )
    mdata3 = MarketData(
        market_id="GGAL",
        details=details,
        recent_bars=[bar3],
    )
    tick3 = IngestionEngineOutput(
        timestamp=bar3.timestamp,
        market_data={"GGAL": mdata3},
        external_data=[],
        bars={"GGAL": [bar3]},
    )
    handler.on_tick(tick3)

    # Filled at open of bar 3 (100.0) - 1% slippage = 99.0
    status_sell = handler.get_order_status(sell_id)
    assert status_sell.status == OrderStatus.FILLED
    assert status_sell.avg_price == 99.0


def test_simulated_handler_cancel():
    delay = KBarExecuteDelay(k=2)
    slip = FlatPriceSlip(slippage_pct=0.0)
    handler = SimulatedExecutionHandler(delay_model=delay, slippage_model=slip)

    order = OrderRequest(
        market_id="GGAL",
        side=OrderSide.BUY,
        size=10.0,
        price=100.0,
        order_type=OrderType.MARKET,
    )

    res = handler.execute_order(order)
    order_id = res.order_id

    # Cancel before execution
    cancel_res = handler.cancel_order(order_id)
    assert cancel_res.status == OrderStatus.CANCELLED

    # Check status
    assert handler.get_order_status(order_id).status == OrderStatus.CANCELLED


def test_execution_engine_integration():
    delay = KBarExecuteDelay(k=1)
    slip = FlatPriceSlip(slippage_pct=0.0)
    handler = SimulatedExecutionHandler(delay_model=delay, slippage_model=slip)
    portfolio = MagicMock(spec=Portfolio)
    portfolio._open_orders = {}

    engine = ExecutionEngine(
        execution_handler=handler,
        portfolio=portfolio,
    )

    # Place order
    order = OrderRequest(
        market_id="GGAL",
        side=OrderSide.BUY,
        size=10.0,
        price=100.0,
        order_type=OrderType.MARKET,
    )

    db_mock = MagicMock(spec=Session)

    res = engine.execute_order(order, db=db_mock, strategy_name="test_strat")
    order_id = res.order_id

    # The engine execute_order should add order to portfolio open orders dict
    portfolio._open_orders[order_id] = order

    # Mock check_order_status
    engine.check_order_status = MagicMock(return_value=res)

    # Create tick data
    details = MarketDetails(
        market_id="GGAL",
        name="GGAL",
        end_date=datetime.now(timezone.utc),
        resolution_source="test",
    )
    bar1 = BarData(
        timestamp=datetime(2026, 6, 21, 10, 0, tzinfo=timezone.utc),
        open=95.0,
        high=98.0,
        low=94.0,
        close=97.0,
        volume=100.0,
        bar_type=BarType.TIME,
        ticks_count=5,
        dollar_volume=9700.0,
    )
    mdata1 = MarketData(
        market_id="GGAL",
        details=details,
        recent_bars=[bar1],
    )
    tick1 = IngestionEngineOutput(
        timestamp=bar1.timestamp,
        market_data={"GGAL": mdata1},
        external_data=[],
        bars={"GGAL": [bar1]},
    )

    # Call on_tick on engine
    engine.on_tick(tick1, db=db_mock)

    # check_order_status should be called for the open order
    engine.check_order_status.assert_called_once_with(order_id, db_mock)
