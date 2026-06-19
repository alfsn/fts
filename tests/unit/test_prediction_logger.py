# tests/unit/test_prediction_logger.py
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trading_bot.core.database import Base
from trading_bot.core.enums import SignalType
from trading_bot.core.models import BacktestPredictionLog, ModelPredictionLog
from trading_bot.core.schemas import TradeSignal
from trading_bot.monitoring.prediction_logger import DatabasePredictionLogger


@pytest.fixture
def db_session():
    # Setup in-memory SQLite DB
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=test_engine)

    SessionLocal = sessionmaker(bind=test_engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_database_prediction_logger(db_session):
    # Given: A database logger observer
    logger = DatabasePredictionLogger(db_session, commit=True, run_id="test_run")

    # And: A sample TradeSignal with serialized prediction_output
    signal = TradeSignal(
        market_id="BTC/USDT",
        strategy_name="nets_strategy_rnn",
        signal_type=SignalType.BUY,
        confidence=0.85,
        timestamp=datetime(2026, 6, 16, 10, 0, tzinfo=timezone.utc),
        prediction_output="[0.10, 0.05, 0.85]",
    )

    # When: The observer callback is triggered
    logger.on_prediction(signal)

    # Then: It should have saved the prediction log to the database
    log = db_session.query(ModelPredictionLog).first()
    assert log is not None
    assert log.run_id == "test_run"
    assert log.market_id == "BTC/USDT"
    assert log.strategy_name == "nets_strategy_rnn"
    assert log.predicted_signal == "buy"
    assert log.confidence == 0.85
    assert log.prediction_output == "[0.10, 0.05, 0.85]"
    # Compare timestamps (handling timezone-aware mapping)
    log_ts = (
        log.timestamp.replace(tzinfo=timezone.utc)
        if log.timestamp.tzinfo is None
        else log.timestamp
    )
    assert log_ts == signal.timestamp


def test_timezone_normalization_prevents_duplicates(db_session):
    # Given: Logger instanced with run_id
    logger = DatabasePredictionLogger(db_session, commit=True, run_id="norm_run")

    # And: Two signals at the exact same instant but with different timezone representation
    ts_utc = datetime(2026, 6, 16, 10, 0, tzinfo=timezone.utc)
    ts_local = ts_utc.astimezone(timezone(timedelta(hours=-3)))  # UTC-3

    signal1 = TradeSignal(
        market_id="ETH/USDT",
        strategy_name="nets_strategy_cnn",
        signal_type=SignalType.BUY,
        confidence=0.80,
        timestamp=ts_utc,
        prediction_output="[0.2, 0.8]",
    )

    signal2 = TradeSignal(
        market_id="ETH/USDT",
        strategy_name="nets_strategy_cnn",
        signal_type=SignalType.SELL,  # updating to sell
        confidence=0.90,
        timestamp=ts_local,
        prediction_output="[0.9, 0.1]",
    )

    # When: Logging both
    logger.on_prediction(signal1)
    logger.on_prediction(signal2)

    # Then: Only one record should exist in the database (since they represent the same timestamp)
    records = db_session.query(ModelPredictionLog).all()
    assert len(records) == 1

    # And: The values should be updated to the second signal
    log = records[0]
    assert log.predicted_signal == "sell"
    assert log.confidence == 0.90
    assert log.prediction_output == "[0.9, 0.1]"


def test_unique_constraint_enforces_run_isolation(db_session):
    # Given: Two loggers for different runs
    logger1 = DatabasePredictionLogger(db_session, commit=True, run_id="run_a")
    logger2 = DatabasePredictionLogger(db_session, commit=True, run_id="run_b")

    ts = datetime(2026, 6, 16, 10, 0, tzinfo=timezone.utc)
    signal = TradeSignal(
        market_id="BTC/USDT",
        strategy_name="nets_strategy_rnn",
        signal_type=SignalType.BUY,
        confidence=0.85,
        timestamp=ts,
        prediction_output="[0.15, 0.85]",
    )

    # When: Logging under separate runs
    logger1.on_prediction(signal)
    logger2.on_prediction(signal)

    # Then: Both predictions exist since they have different run_ids
    records = db_session.query(ModelPredictionLog).all()
    assert len(records) == 2
    assert {r.run_id for r in records} == {"run_a", "run_b"}


def test_backtest_prediction_logger(db_session):
    # Given: A logger configured to log to BacktestPredictionLog
    logger = DatabasePredictionLogger(
        db_session,
        commit=True,
        model_class=BacktestPredictionLog,
        run_id="hash_bt_123",
    )

    signal = TradeSignal(
        market_id="GGAL",
        strategy_name="nets_strategy_cnn",
        signal_type=SignalType.SELL,
        confidence=0.75,
        timestamp=datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc),
        prediction_output="[0.75, 0.25]",
    )

    # When: Logging the backtest prediction
    logger.on_prediction(signal)

    # Then: It should be saved in the backtest_prediction_logs table
    log = db_session.query(BacktestPredictionLog).first()
    assert log is not None
    assert log.run_id == "hash_bt_123"
    assert log.market_id == "GGAL"
    assert log.strategy_name == "nets_strategy_cnn"
    assert log.predicted_signal == "sell"
    assert log.confidence == 0.75

    # And: Nothing should be written to model_prediction_logs
    assert db_session.query(ModelPredictionLog).count() == 0
