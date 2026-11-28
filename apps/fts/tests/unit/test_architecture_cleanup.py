# tests/unit/test_architecture_cleanup.py

import os
import re
from pathlib import Path

import pytest
from quant_core.enums import OrderSide, OrderStatus
from quant_core.models import ExecutionResult, OrderRequest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from trading_bot.core.database import Base, create_db_session, init_db
from trading_bot.risk_management.portfolio import Portfolio
from trading_bot.runs import RunHeader, SqlRunRecorder
from trading_bot.runs.tables import (
    BacktestPredictionLog,
    ModelPredictionLog,
    PredictionLog,
    RunLog,
    TradeLog,
)


def test_dynamic_db_session_and_init():
    """Verifies that create_db_session connects dynamically and init_db runs on custom bind."""
    test_db_file = "./temp_test_isolation.db"
    if os.path.exists(test_db_file):
        os.remove(test_db_file)

    try:
        db_url = f"sqlite+pysqlite:///{test_db_file}"
        session = create_db_session(db_url)

        # Verify custom engine and session binding
        assert session.bind.url.database == test_db_file

        # Initialize tables on custom bind
        init_db(bind_engine=session.bind)

        # Check table creation by running simple query
        assert session.query(RunLog).count() == 0
        session.close()
    finally:
        if os.path.exists(test_db_file):
            os.remove(test_db_file)


def test_prediction_log_single_table_inheritance():
    """Verifies that ModelPredictionLog and BacktestPredictionLog share the same table via STI."""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        from datetime import datetime, timezone

        from quant_data.db import Market

        # Add a dummy market first for foreign key constraint
        market = Market(
            instrument_id="AAPL",
            name="Apple Stock",
            end_date=datetime.now(timezone.utc),
            resolution_source="test",
        )
        db.add(market)
        db.commit()

        # Create a live prediction log
        live_log = ModelPredictionLog(
            run_id="live_run_1",
            timestamp=datetime.now(timezone.utc),
            strategy_name="nets_strategy_cnn",
            instrument_id="AAPL",
            predicted_signal="buy",
            confidence=0.9,
        )

        # Create a backtest prediction log
        backtest_log = BacktestPredictionLog(
            run_id="backtest_run_1",
            timestamp=datetime.now(timezone.utc),
            strategy_name="nets_strategy_cnn",
            instrument_id="AAPL",
            predicted_signal="sell",
            confidence=0.75,
        )

        db.add(live_log)
        db.add(backtest_log)
        db.commit()

        # Both classes query the SAME physical table "prediction_logs"
        assert live_log.__tablename__ == "prediction_logs"
        assert backtest_log.__tablename__ == "prediction_logs"

        # Check polymorphic behavior
        # ModelPredictionLog query should only return live logs (identity = live)
        live_records = db.query(ModelPredictionLog).all()
        assert len(live_records) == 1
        assert live_records[0].run_id == "live_run_1"
        assert live_records[0].log_type == "live"

        # BacktestPredictionLog query should only return backtest logs (identity = backtest)
        backtest_records = db.query(BacktestPredictionLog).all()
        assert len(backtest_records) == 1
        assert backtest_records[0].run_id == "backtest_run_1"
        assert backtest_records[0].log_type == "backtest"

        # Querying the base PredictionLog should return both
        all_records = db.query(PredictionLog).all()
        assert len(all_records) == 2
    finally:
        db.close()
        Base.metadata.drop_all(engine)


def test_portfolio_public_property_accessors():
    """Verifies that Portfolio exposes public properties instead of private attributes."""
    portfolio = Portfolio(initial_balance=5000.0, quote_currency="USD")

    # Verify public cash balance property
    assert portfolio.cash_balance == 5000.0

    # Verify public positions and open_orders properties
    assert isinstance(portfolio.positions, dict)
    assert isinstance(portfolio.open_orders, dict)


def test_the_run_recorder_never_commits():
    """The recorder writes into the caller's transaction; the loop or the
    execution engine decides when to commit."""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        from datetime import datetime, timezone

        from quant_data.db import Market

        db.add(
            Market(
                instrument_id="AAPL",
                name="Apple Stock",
                end_date=datetime.now(timezone.utc),
                resolution_source="test",
            )
        )
        db.commit()

        recorder = SqlRunRecorder(run_id="test_run", kind="backtest")
        order = OrderRequest(
            instrument_id="AAPL", side=OrderSide.BUY, quantity=10.0, price=150.0
        )
        result = ExecutionResult(
            order_id="o1",
            status=OrderStatus.FILLED,
            filled_quantity=10.0,
            avg_price=150.0,
            timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )

        recorder.start(db, RunHeader(run_id="test_run", kind="backtest"))
        recorder.record_fill(db, order, result)
        assert db.query(TradeLog).count() == 1

        db.rollback()
        assert db.query(TradeLog).count() == 0
        assert db.query(RunLog).count() == 0
    finally:
        db.close()
        Base.metadata.drop_all(engine)


# --- M1 guards (text tests; the import rule is an import-linter contract) ---

SRC = Path(__file__).resolve().parents[2] / "src" / "trading_bot"


def _source_hits(pattern: str):
    rx = re.compile(pattern)
    return [
        f"{p.relative_to(SRC)}:{i}"
        for p in SRC.rglob("*.py")
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if rx.search(line)
    ]


def test_one_database_no_backtest_url_or_test_switch():
    """D16: no second backtest database and no "test"-in-URL switch."""
    assert _source_hits(r"BACKTEST_DATABASE_URL") == []
    assert _source_hits(r"""["']test["']\s+in\s+\w*url""") == []


def test_positions_are_not_stored():
    """D19: positions are derived from fills; nothing persists them."""
    assert _source_hits(r"\bPositionRepository\b|__tablename__ = \"positions\"") == []


def test_run_reset_drops_only_run_tables(tmp_path):
    """The reset drops the run tables and the retired ``positions`` table,
    never what they point at (``markets``) or the model registry."""
    from sqlalchemy import inspect, text
    from trading_bot.runs.reset import reset_runs
    from trading_bot.runs.tables import RUN_TABLES

    engine = create_engine(f"sqlite:///{tmp_path / 'research.db'}")
    init_db(bind_engine=engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE positions (id INTEGER PRIMARY KEY, instrument_id "
                "VARCHAR REFERENCES markets (instrument_id))"
            )
        )

    reset_runs(engine)

    tables = set(inspect(engine).get_table_names())
    assert "positions" not in tables
    assert {"markets", "model_registry", "bar_data_logs"} <= tables
    assert {t.name for t in RUN_TABLES} <= tables
