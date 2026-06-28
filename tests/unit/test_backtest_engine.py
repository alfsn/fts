# tests/unit/test_backtest_engine.py

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from trading_bot.backtesting.abc import BaseBacktestDataReader
from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.backtesting.results import BacktestResult
from trading_bot.core.database import Base
from trading_bot.core.enums import OrderStatus
from trading_bot.core.models import (
    BacktestEquityLog,
    BacktestPredictionLog,
    OrderLog,
    Position,
    TradeLog,
)
from trading_bot.core.pipeline import TradingPipeline
from trading_bot.core.schemas import (
    BarData,
    IngestionEngineOutput,
    MarketData,
    MarketDetails,
)


@pytest.fixture(scope="function")
def sql_db_session() -> Session:
    """
    Creates an in-memory SQLite database session for unit testing persistence.
    """
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(engine)


class DummyDataReader(BaseBacktestDataReader):
    """
    Simulated data reader yielding pre-defined ticks.
    """

    def __init__(self, market_id: str, ticks: list) -> None:
        self.market_id = market_id
        self.ticks = ticks

    def read_data(self):
        for tick in self.ticks:
            yield tick


def create_mock_tick(
    timestamp: datetime, market_id: str, close_price: float
) -> IngestionEngineOutput:
    """
    Helper to build IngestionEngineOutput packets.
    """
    bar = BarData(
        timestamp=timestamp,
        open=close_price,
        high=close_price,
        low=close_price,
        close=close_price,
        volume=10.0,
        bar_type="time",
        ticks_count=100,
        dollar_volume=close_price * 10.0,
    )
    details = MarketDetails(
        market_id=market_id,
        name="Test Market",
        end_date=timestamp,
        resolution_source="test",
    )
    market_data = MarketData(
        market_id=market_id,
        details=details,
        recent_bars=[bar],
        order_book=None,
        recent_trades=[],
    )
    return IngestionEngineOutput(
        timestamp=timestamp,
        market_data={market_id: market_data},
        external_data=[],
        bars={market_id: [bar]},
    )


def test_backtest_result_metrics_calculation(sql_db_session: Session):
    """
    Verifies that BacktestResult computes cumulative return, drawdown, and Sharpe ratio correctly.
    """
    run_id = "test_run_123"
    market_id = "BTC/USDT"
    strategy_name = "test_strat"

    # Seed some filled trades to the database to check trade count calculation
    order1 = OrderLog(
        order_id="ord_1",
        run_id=run_id,
        market_id=market_id,
        strategy_name=strategy_name,
        side="BUY",
        status=OrderStatus.FILLED,
        requested_price=50000.0,
        requested_size=0.1,
        filled_size=0.1,
    )
    order2 = OrderLog(
        order_id="ord_2",
        run_id=run_id,
        market_id=market_id,
        strategy_name=strategy_name,
        side="SELL",
        status=OrderStatus.FILLED,
        requested_price=55000.0,
        requested_size=0.1,
        filled_size=0.1,
    )
    sql_db_session.add_all([order1, order2])
    sql_db_session.commit()

    # Seed BacktestEquityLog records to the database representing the simulated equity curve
    base_time = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    log1 = BacktestEquityLog(
        run_id=run_id,
        timestamp=base_time,
        cash=10000.0,
        position=0.0,
        close=50000.0,
        equity=10000.0,
    )
    log2 = BacktestEquityLog(
        run_id=run_id,
        timestamp=base_time + timedelta(days=1),
        cash=10200.0,
        position=0.0,
        close=51000.0,
        equity=10200.0,
    )
    log3 = BacktestEquityLog(
        run_id=run_id,
        timestamp=base_time + timedelta(days=2),
        cash=10100.0,
        position=0.0,
        close=50500.0,
        equity=10100.0,
    )
    log4 = BacktestEquityLog(
        run_id=run_id,
        timestamp=base_time + timedelta(days=3),
        cash=10500.0,
        position=0.0,
        close=52500.0,
        equity=10500.0,
    )

    sql_db_session.add_all([log1, log2, log3, log4])
    sql_db_session.commit()

    result = BacktestResult(
        run_id=run_id,
        market_id=market_id,
        strategy_name=strategy_name,
        db_session=sql_db_session,
    )

    summary = result.to_dict()

    assert summary["run_id"] == run_id
    assert summary["market_id"] == market_id
    assert summary["strategy_name"] == strategy_name
    assert summary["initial_equity"] == 10000.0
    assert summary["final_equity"] == 10500.0
    assert summary["total_return_pct"] == pytest.approx(
        5.0
    )  # (10500-10000)/10000 * 100
    assert summary["max_drawdown_pct"] == pytest.approx(
        -0.9803921
    )  # (10100-10200)/10200 * 100
    assert summary["total_trades"] == 2
    # Ensure Sharpe ratio can be calculated and is a float
    assert isinstance(summary["sharpe_ratio"], float)


def test_backtest_engine_run_flow(sql_db_session: Session):
    """
    Verifies that BacktestEngine drives the pipeline, tracks equity, cleans up logs,
    and returns a clean BacktestResult.
    """
    run_id = "test_run_engine"
    market_id = "BTC/USDT"
    strategy_name = "test_strat"

    # Seed mock database logs to test clear_previous_run functionality
    old_log = BacktestPredictionLog(
        run_id=run_id,
        market_id=market_id,
        strategy_name=strategy_name,
        timestamp=datetime.now(timezone.utc),
        predicted_signal="buy",
        confidence=0.8,
    )
    old_equity_log = BacktestEquityLog(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc),
        cash=10000.0,
        position=0.0,
        close=50000.0,
        equity=10000.0,
    )
    sql_db_session.add_all([old_log, old_equity_log])
    sql_db_session.commit()

    assert (
        sql_db_session.query(BacktestPredictionLog).filter_by(run_id=run_id).count()
        == 1
    )
    assert sql_db_session.query(BacktestEquityLog).filter_by(run_id=run_id).count() == 1

    # Mock Pipeline and subcomponents
    mock_portfolio = MagicMock()
    # Mock portfolio cash and positions state
    mock_portfolio._cash_balance = 10000.0
    mock_portfolio._positions = {}
    type(mock_portfolio).cash_balance = PropertyMock(
        side_effect=lambda: mock_portfolio._cash_balance
    )
    type(mock_portfolio).positions = PropertyMock(
        side_effect=lambda: mock_portfolio._positions
    )

    mock_strategy = MagicMock()
    mock_strategy_instance = MagicMock()
    mock_strategy_instance.name = strategy_name
    mock_strategy.strategies = [mock_strategy_instance]

    mock_prediction_logger = MagicMock()

    mock_execution = MagicMock()

    pipeline = MagicMock(spec=TradingPipeline)
    pipeline.portfolio = mock_portfolio
    pipeline.strategy = mock_strategy
    pipeline.prediction_logger = mock_prediction_logger
    pipeline.execution = mock_execution

    # Define historical ticks
    base_time = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    ticks = [
        create_mock_tick(base_time, market_id, 50000.0),
        create_mock_tick(base_time + timedelta(days=1), market_id, 51000.0),
    ]

    data_reader = DummyDataReader(market_id=market_id, ticks=ticks)

    engine = BacktestEngine(
        pipeline=pipeline,
        data_reader=data_reader,
        db=sql_db_session,
        market_id=market_id,
    )

    # We will simulate portfolio state updates when execute_single_tick is run
    def simulate_pipeline_execution(db, ingestion_output):
        # Tick 1: Buy 0.1 BTC (cash = 5000.0, position = 0.1 BTC)
        # Tick 2: Cash remains 5000.0, price rises to 51000.0
        if ingestion_output.timestamp == base_time:
            mock_portfolio._cash_balance = 5000.0
            mock_portfolio._positions[market_id] = MagicMock(size=0.1)
        elif ingestion_output.timestamp == base_time + timedelta(days=1):
            mock_portfolio._cash_balance = 5000.0
            mock_portfolio._positions[market_id] = MagicMock(size=0.1)

    pipeline.execute_single_tick.side_effect = simulate_pipeline_execution

    # Run the backtest and enable DB cleanup
    result = engine.run(run_id=run_id, clear_previous_run=True)

    # Assertions
    # 1. Old DB logs matching run_id were cleared
    assert (
        sql_db_session.query(BacktestPredictionLog).filter_by(run_id=run_id).count()
        == 0
    )
    # 1b. Old equity logs matching run_id were cleared, and 2 new logs were inserted
    equity_db_logs = (
        sql_db_session.query(BacktestEquityLog).filter_by(run_id=run_id).all()
    )
    assert len(equity_db_logs) == 2
    assert equity_db_logs[0].equity == 10000.0
    assert equity_db_logs[1].equity == 10100.0

    # 2. Pipeline components were configured with correct run_id
    assert mock_prediction_logger.run_id == run_id
    assert mock_prediction_logger.commit is False
    assert mock_execution.run_id == run_id

    # 3. Pipeline tick was executed twice
    assert pipeline.execute_single_tick.call_count == 2

    # 4. Check compiled equity curve details
    # Tick 1: cash=5000, pos=0.1, close=50000, equity = 5000 + 0.1*50000 = 10000.0
    # Tick 2: cash=5000, pos=0.1, close=51000, equity = 5000 + 0.1*51000 = 10100.0
    curve = result.equity_curve
    assert len(curve) == 2
    assert curve[0]["equity"] == 10000.0
    assert curve[1]["equity"] == 10100.0

    # 5. Check result calculations
    assert result.initial_equity == 10000.0
    assert result.final_equity == 10100.0
    assert result.total_return_pct == pytest.approx(1.0)
