from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trading_bot.core.database import Base
from trading_bot.core.enums import SignalType
from trading_bot.core.models import ModelPredictionLog
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
    logger = DatabasePredictionLogger(db_session, commit=True)

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
