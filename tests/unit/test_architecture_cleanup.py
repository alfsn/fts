# tests/unit/test_architecture_cleanup.py

import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trading_bot.core.database import Base, create_db_session, init_db
from trading_bot.core.enums import OrderSide, OrderStatus, PositionStatus
from trading_bot.core.models import (
    BacktestPredictionLog,
    ModelPredictionLog,
)
from trading_bot.core.models import Position as PositionModel
from trading_bot.core.models import (
    PredictionLog,
)
from trading_bot.core.repository import PositionRepository
from trading_bot.core.schemas import Position as PositionSchema
from trading_bot.risk_management.portfolio import Portfolio


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
        assert session.query(PositionModel).count() == 0
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

        from trading_bot.core.models import Market

        # Add a dummy market first for foreign key constraint
        market = Market(
            market_id="AAPL",
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
            market_id="AAPL",
            predicted_signal="buy",
            confidence=0.9,
        )

        # Create a backtest prediction log
        backtest_log = BacktestPredictionLog(
            run_id="backtest_run_1",
            timestamp=datetime.now(timezone.utc),
            strategy_name="nets_strategy_cnn",
            market_id="AAPL",
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


def test_decoupled_repository_transactions():
    """Verifies that repositories no longer auto-commit or rollback transactions."""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        from datetime import datetime, timezone

        from trading_bot.core.models import Market

        # Add market
        market = Market(
            market_id="AAPL",
            name="Apple Stock",
            end_date=datetime.now(timezone.utc),
            resolution_source="test",
        )
        db.add(market)
        db.commit()

        repo = PositionRepository(db)

        pos_schema = PositionSchema(
            market_id="AAPL",
            size=10.0,
            entry_price=150.0,
            run_id="test_run",
        )

        # Call save_position (which does not call commit internally anymore)
        repo.save_position(pos_schema)

        # In-transaction query should see the position
        assert db.query(PositionModel).count() == 1

        # Roll back transaction - should remove it (proving it wasn't committed)
        db.rollback()
        assert db.query(PositionModel).count() == 0

        # Now save and commit explicitly from caller
        repo.save_position(pos_schema)
        db.commit()

        # Verify it is fully persisted now after rollback
        db.rollback()
        assert db.query(PositionModel).count() == 1

    finally:
        db.close()
        Base.metadata.drop_all(engine)
