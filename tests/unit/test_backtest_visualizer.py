# tests/unit/test_backtest_visualizer.py

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from trading_bot.backtesting.visualizer import BacktestVisualizer
from trading_bot.core.database import Base, SessionLocal
from trading_bot.core.database import engine as dev_engine
from trading_bot.core.enums import BarType
from trading_bot.core.models import (
    BacktestEquityLog,
    BacktestPredictionLog,
    BarDataLog,
    Market,
    OrderLog,
    OrderSide,
    OrderStatus,
)


@pytest.fixture
def db_session():
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=test_engine)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("trading_bot.core.database.engine", test_engine)
        SessionLocal.configure(bind=test_engine)
        db = SessionLocal()

        market = Market(
            market_id="BTC/USDT",
            name="Bitcoin / Tether",
            end_date=datetime(2025, 12, 31, tzinfo=timezone.utc),
            resolution_source="Binance",
        )
        db.add(market)

        base_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        for i in range(5):
            bar = BarDataLog(
                market_id="BTC/USDT",
                timestamp=base_time + timedelta(minutes=i * 5),
                open=40000.0 + i * 10,
                high=40050.0 + i * 10,
                low=39980.0 + i * 10,
                close=40010.0 + i * 10,
                volume=1.5,
                bar_type=BarType.TIME,
                ticks_count=50,
                dollar_volume=60000.0,
            )
            db.add(bar)

            pred = BacktestPredictionLog(
                market_id="BTC/USDT",
                timestamp=base_time + timedelta(minutes=i * 5),
                strategy_name="cnn",
                prediction_output="[0.1, 0.2, 0.7]",
                predicted_signal="BUY" if i % 2 == 0 else "HOLD",
                confidence=0.8,
                actual_future_return=0.01,
                run_id="run_123",
            )
            db.add(pred)

            equity = BacktestEquityLog(
                run_id="run_123",
                timestamp=base_time + timedelta(minutes=i * 5),
                cash=10000.0,
                position=1.0,
                close=40010.0 + i * 10,
                equity=10000.0 + i * 100.0,
            )
            db.add(equity)

        db.commit()
        yield db
        db.close()
        SessionLocal.configure(bind=dev_engine)


def test_backtest_visualizer_load_data_with_none_run_id(db_session: Session):
    viz = BacktestVisualizer(db_url="sqlite:///:memory:")
    viz.SessionLocal = lambda: db_session

    # When run_id is None, load_data should still load bar logs and join predictions/equity
    df = viz.load_data(market_id="BTC/USDT", strategy_name="cnn", run_id=None)
    assert not df.empty
    assert len(df) == 5
    assert "close" in df.columns
    assert "predicted_signal" in df.columns


def test_backtest_visualizer_load_data_with_specific_run_id(db_session: Session):
    viz = BacktestVisualizer(db_url="sqlite:///:memory:")
    viz.SessionLocal = lambda: db_session

    df = viz.load_data(market_id="BTC/USDT", strategy_name="cnn", run_id="run_123")
    assert not df.empty
    assert len(df) == 5
    assert "actual_equity" in df.columns


def test_backtest_visualizer_get_available_runs(db_session: Session):
    viz = BacktestVisualizer(db_url="sqlite:///:memory:")
    viz.SessionLocal = lambda: db_session

    runs = viz.get_available_runs(market_id="BTC/USDT", strategy_name="None")
    assert runs == ["run_123"]
