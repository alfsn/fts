from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from quant_core.enums import AssetType, Geography, OrderSide, OrderStatus
from quant_core.models import ExecutionResult, OrderRequest, TradFiDetails
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session, sessionmaker
from trading_bot.backtesting.abc import BaseBacktestDataReader
from trading_bot.backtesting.engine import BacktestEngine
from trading_bot.backtesting.results import BacktestResult
from trading_bot.core.database import ResearchBase
from trading_bot.core.pipeline import TradingPipeline
from trading_bot.core.schemas import (
    BarData,
    IngestionEngineOutput,
    Instrument,
    MarketData,
)
from trading_bot.risk_management.portfolio import Portfolio
from trading_bot.runs import (
    EquityPoint,
    Fill,
    RunExistsError,
    RunHeader,
    RunIntegrityError,
    RunRecord,
    RunStore,
    SqlRunRecorder,
    delete_run,
)
from trading_bot.runs.tables import (
    RUN_TABLES,
    BacktestEquityLog,
    BacktestPredictionLog,
    OrderLog,
    RunLog,
    TradeLog,
)

# tests/unit/test_backtest_engine.py


@pytest.fixture(scope="function")
def sql_db_session() -> Session:
    """
    Creates an in-memory SQLite database session for unit testing persistence.
    """
    engine = create_engine("sqlite+pysqlite:///:memory:")
    ResearchBase.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        ResearchBase.metadata.drop_all(engine)


class DummyDataReader(BaseBacktestDataReader):
    """
    Simulated data reader yielding pre-defined ticks.
    """

    def __init__(self, instrument_id: str, ticks: list) -> None:
        self.instrument_id = instrument_id
        self.ticks = ticks

    def read_data(self):
        for tick in self.ticks:
            yield tick


def create_mock_tick(
    timestamp: datetime,
    instrument_id: str,
    close_price: float,
    with_bar: bool = True,
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
    details = Instrument(
        instrument_id=instrument_id,
        name="Test Market",
        details=TradFiDetails(
            asset_type=AssetType.STOCK,
            geography=Geography.US,
            industry="Tech",
            currency="USD",
        ),
    )
    market_data = MarketData(
        instrument_id=instrument_id,
        details=details,
        recent_bars=[bar] if with_bar else [],
        order_book=None,
        recent_trades=[],
    )
    return IngestionEngineOutput(
        timestamp=timestamp,
        market_data=[market_data],
        external_data=[],
        bars={instrument_id: [bar]} if with_bar else {},
    )


def _fill(pipeline, db, ts, side, quantity, price, order_id):
    """Trade the way ExecutionEngine does: the Portfolio applies the fill and
    the run's recorder saves exactly that fill, at the bar time."""
    order = OrderRequest(
        instrument_id=pipeline.instrument_id,
        side=side,
        quantity=quantity,
        price=price,
    )
    result = ExecutionResult(
        order_id=order_id,
        status=OrderStatus.FILLED,
        filled_quantity=quantity,
        avg_price=price,
        timestamp=ts,
    )
    pipeline.portfolio.add_open_order(order_id, order)
    applied = pipeline.portfolio.update_order_status(result)
    pipeline.execution.recorder.record_fill(db, applied, result)


def _pipeline(portfolio, instrument_id, strategy_name="test_strat"):
    strategy = MagicMock()
    strategy.name = strategy_name
    pipeline = MagicMock(spec=TradingPipeline)
    pipeline.portfolio = portfolio
    pipeline.strategy = MagicMock(strategies=[strategy])
    pipeline.prediction_logger = MagicMock()
    pipeline.execution = MagicMock()
    pipeline.instrument_id = instrument_id
    return pipeline


def test_backtest_result_metrics_calculation():
    """
    Verifies that BacktestResult computes cumulative return, drawdown, and Sharpe ratio correctly.
    """
    run_id = "test_run_123"
    instrument_id = "BTC/USDT"
    strategy_name = "test_strat"

    base_time = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    record = RunRecord(
        header=RunHeader(run_id=run_id, kind="backtest", initial_cash=10000.0),
        equity=[
            EquityPoint(
                timestamp=base_time + timedelta(days=i),
                cash=e,
                position=0.0,
                close=c,
                equity=e,
            )
            for i, (e, c) in enumerate(
                [
                    (10000.0, 50000.0),
                    (10200.0, 51000.0),
                    (10100.0, 50500.0),
                    (10500.0, 52500.0),
                ]
            )
        ],
        fills=[
            Fill(
                instrument_id=instrument_id,
                side=side,
                quantity=0.1,
                fill_price=price,
                fill_timestamp=base_time + timedelta(days=day),
            )
            for side, price, day in [
                (OrderSide.BUY, 50000.0, 0),
                (OrderSide.SELL, 55000.0, 3),
            ]
        ],
    )

    result = BacktestResult(
        run_id=run_id,
        instrument_id=instrument_id,
        strategy_name=strategy_name,
        record=record,
    )

    summary = result.to_dict()

    assert summary["run_id"] == run_id
    assert summary["instrument_id"] == instrument_id
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
    # One closed round trip, a winner: the win rate is a number (it was NaN
    # before M1, when no fill rows were ever written)
    assert result.metrics.win_rate == 1.0


def test_backtest_engine_run_flow(sql_db_session: Session):
    """
    Verifies that BacktestEngine drives the pipeline, tracks equity, cleans up logs,
    and returns a clean BacktestResult.
    """
    run_id = "test_run_engine"
    instrument_id = "BTC/USDT"
    strategy_name = "test_strat"

    # Seed mock database logs to test clear_previous_run functionality
    old_log = BacktestPredictionLog(
        run_id=run_id,
        instrument_id=instrument_id,
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

    # A real Portfolio: the engine logs its single valuation (PR 7b)
    mock_portfolio = Portfolio(initial_balance=10000.0)
    pipeline = _pipeline(mock_portfolio, instrument_id, strategy_name)
    mock_prediction_logger = pipeline.prediction_logger
    mock_execution = pipeline.execution

    # Define historical ticks
    base_time = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    ticks = [
        create_mock_tick(base_time, instrument_id, 50000.0),
        create_mock_tick(base_time + timedelta(days=1), instrument_id, 51000.0),
    ]

    data_reader = DummyDataReader(instrument_id=instrument_id, ticks=ticks)

    engine = BacktestEngine(
        pipeline=pipeline,
        data_reader=data_reader,
        db=sql_db_session,
        instrument_id=instrument_id,
    )

    # We will simulate portfolio state updates when execute_single_tick is run
    def simulate_pipeline_execution(db, ingestion_output):
        # Tick 1: Buy 0.1 BTC (cash = 5000.0, position = 0.1 BTC)
        # Tick 2: Cash remains 5000.0, price rises to 51000.0
        if ingestion_output.timestamp == base_time:
            _fill(pipeline, db, base_time, OrderSide.BUY, 0.1, 50000.0, "o1")

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

    # 2. Pipeline components write through the run's recorder and session
    assert mock_prediction_logger.recorder.run_id == run_id
    assert mock_prediction_logger.db is sql_db_session
    assert mock_prediction_logger.commit is False
    assert mock_execution.recorder is mock_prediction_logger.recorder

    # 2b. The fill row carries the bar time, not the wall clock (M1)
    fill = sql_db_session.query(TradeLog).filter_by(run_id=run_id).one()
    assert fill.fill_timestamp.replace(tzinfo=timezone.utc) == base_time

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


def _run_engine(sql_db_session, ticks, instrument_id, buy_at):
    """Engine over ``ticks`` with a real Portfolio that buys 0.1 at ``buy_at``."""
    pipeline = _pipeline(Portfolio(initial_balance=10000.0), instrument_id)

    def buy_on_first_tick(db, ingestion_output):
        if ingestion_output.timestamp == ticks[0].timestamp:
            _fill(pipeline, db, ticks[0].timestamp, OrderSide.BUY, 0.1, buy_at, "o1")

    pipeline.execute_single_tick.side_effect = buy_on_first_tick
    engine = BacktestEngine(
        pipeline=pipeline,
        data_reader=DummyDataReader(instrument_id=instrument_id, ticks=ticks),
        db=sql_db_session,
        instrument_id=instrument_id,
    )
    return engine.run(run_id="pr7b", clear_previous_run=True)


def test_a_tick_without_a_bar_keeps_the_last_close(sql_db_session: Session):
    """PR 7b: a missing bar used to value the position at 0 (equity = cash)."""
    instrument_id = "BTC/USDT"
    t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    ticks = [
        create_mock_tick(t0, instrument_id, 50000.0),
        create_mock_tick(t0 + timedelta(days=1), instrument_id, 51000.0),
        create_mock_tick(t0 + timedelta(days=2), instrument_id, 1.0, with_bar=False),
    ]

    result = _run_engine(sql_db_session, ticks, instrument_id, buy_at=50000.0)

    assert [p["equity"] for p in result.equity_curve] == [10000.0, 10100.0, 10100.0]
    assert [p["close"] for p in result.equity_curve] == [50000.0, 51000.0, 51000.0]
    assert result.final_equity == 10100.0  # not 5000 (cash only)


def test_ticks_before_the_first_bar_are_not_logged(sql_db_session: Session):
    instrument_id = "BTC/USDT"
    t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    ticks = [
        create_mock_tick(t0, instrument_id, 1.0, with_bar=False),
        create_mock_tick(t0 + timedelta(days=1), instrument_id, 50000.0),
    ]

    result = _run_engine(sql_db_session, ticks, instrument_id, buy_at=50000.0)

    assert len(result.equity_curve) == 1
    assert result.equity_curve[0]["close"] == 50000.0


# --- M1: one store per run --------------------------------------------------


def test_run_tables_cover_every_table_with_a_run_id():
    """A new table with a run_id must join RUN_TABLES, or delete and reset
    would silently leave its rows behind. (``model_registry.run_id`` is the
    training run, not a backtest run.)"""
    with_run_id = {
        t.name
        for t in ResearchBase.metadata.sorted_tables
        if "run_id" in t.c and t.name != "model_registry"
    }
    assert with_run_id == {t.name for t in RUN_TABLES}


def test_a_run_lands_in_one_db_and_delete_removes_every_row(sql_db_session):
    instrument_id = "BTC/USDT"
    t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    ticks = [
        create_mock_tick(t0, instrument_id, 50000.0),
        create_mock_tick(t0 + timedelta(days=1), instrument_id, 51000.0),
    ]
    _run_engine(sql_db_session, ticks, instrument_id, buy_at=50000.0)

    store = RunStore(sessionmaker(bind=sql_db_session.bind))
    record = store.load("pr7b")
    assert record.header.strategy_name is not None
    assert len(record.segments) == 1
    assert len(record.equity) == 2
    assert len(record.fills) == 1

    store.delete("pr7b")
    for table in RUN_TABLES:
        n = sql_db_session.scalar(
            select(func.count()).select_from(table).where(table.c.run_id == "pr7b")
        )
        assert n == 0, table.name


def test_a_missing_fill_row_fails_the_run(sql_db_session):
    """Fills are the truth (D19): a fill the Portfolio applied but the run
    didn't save makes the end-of-run replay disagree."""
    instrument_id = "BTC/USDT"
    t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    pipeline = _pipeline(Portfolio(initial_balance=10000.0), instrument_id)

    def buy_without_saving(db, ingestion_output):
        if ingestion_output.timestamp == t0:
            order = OrderRequest(
                instrument_id=instrument_id,
                side=OrderSide.BUY,
                quantity=0.1,
                price=50000.0,
            )
            pipeline.portfolio.add_open_order("o1", order)
            pipeline.portfolio.update_order_status(
                ExecutionResult(
                    order_id="o1",
                    status=OrderStatus.FILLED,
                    filled_quantity=0.1,
                    avg_price=50000.0,
                    timestamp=t0,
                )
            )

    pipeline.execute_single_tick.side_effect = buy_without_saving
    engine = BacktestEngine(
        pipeline=pipeline,
        data_reader=DummyDataReader(
            instrument_id, [create_mock_tick(t0, instrument_id, 50000.0)]
        ),
        db=sql_db_session,
        instrument_id=instrument_id,
    )
    with pytest.raises(RunIntegrityError):
        engine.run(run_id="missing_fill")
    # The failed run is deleted, not left behind as a wrong record
    assert RunStore(sessionmaker(bind=sql_db_session.bind)).load("missing_fill") is None


def test_a_run_that_never_prices_or_trades_returns_an_empty_result(sql_db_session):
    """No ticks: no equity, no fills. The check passes (flat, cash untouched)
    instead of crashing on a missing first timestamp."""
    pipeline = _pipeline(Portfolio(initial_balance=10000.0), "BTC/USDT")
    engine = BacktestEngine(
        pipeline=pipeline,
        data_reader=DummyDataReader("BTC/USDT", []),
        db=sql_db_session,
        instrument_id="BTC/USDT",
    )
    result = engine.run(run_id="empty")
    assert result.equity_curve == []
    assert result.record.fills == []


def test_a_run_id_is_never_reused_without_clearing(sql_db_session):
    """Re-running under the same run_id would add a second run's fills and
    equity to the first one's."""
    instrument_id = "BTC/USDT"
    t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    ticks = [create_mock_tick(t0, instrument_id, 50000.0)]
    _run_engine(sql_db_session, ticks, instrument_id, buy_at=50000.0)

    pipeline = _pipeline(Portfolio(initial_balance=10000.0), instrument_id)
    engine = BacktestEngine(
        pipeline=pipeline,
        data_reader=DummyDataReader(instrument_id, ticks),
        db=sql_db_session,
        instrument_id=instrument_id,
    )
    with pytest.raises(RunExistsError):
        engine.run(run_id="pr7b")


def test_delete_run_respects_foreign_keys():
    """Postgres enforces trade_logs -> order_logs; SQLite only with the pragma."""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    event.listen(
        engine, "connect", lambda conn, _: conn.execute("PRAGMA foreign_keys=ON")
    )
    ResearchBase.metadata.create_all(engine)
    t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    with Session(engine) as db:
        db.add(RunLog(run_id="r1", kind="backtest"))
        db.add(
            OrderLog(
                order_id="o1",
                run_id="r1",
                instrument_id="BTC/USDT",
                strategy_name="s",
                side=OrderSide.BUY,
                requested_quantity=1.0,
                requested_price=1.0,
                status=OrderStatus.FILLED,
            )
        )
        db.add(
            TradeLog(
                run_id="r1",
                order_id="o1",
                instrument_id="BTC/USDT",
                side=OrderSide.BUY,
                quantity=1.0,
                fill_price=1.0,
                fill_timestamp=t0,
            )
        )
        db.commit()
        delete_run(db, "r1")
        db.commit()
        assert db.scalar(select(func.count()).select_from(OrderLog)) == 0
