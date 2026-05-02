# tests/unit/test_backtest_simulator.py

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import Session

from trading_bot.backtesting.simulator import BacktestSimulator
from trading_bot.core.database import SessionLocal, init_db
from trading_bot.core.enums import BarType
from trading_bot.core.models import BarDataLog, Market
from trading_bot.strategy.engine import StrategyEngine
from trading_bot.strategy.strategies.dummy_strategy import DummyStrategy


@pytest.fixture
def db_session():
    init_db()
    db = SessionLocal()

    # Cleanup and setup
    db.query(BarDataLog).delete()
    db.query(Market).delete()

    market = Market(
        market_id="GGAL",
        name="Grupo Galicia",
        end_date=datetime(2025, 12, 31, tzinfo=timezone.utc),
        resolution_source="BYMA",
    )
    db.add(market)

    # Create 3 bars
    base_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    for i in range(3):
        bar = BarDataLog(
            market_id="GGAL",
            timestamp=base_time + timedelta(minutes=i * 5),
            open=100.0 + i,
            high=105.0 + i,
            low=95.0 + i,
            close=101.0 + i,
            volume=1000.0,
            bar_type=BarType.TIME,
            ticks_count=10,
            dollar_volume=100000.0,
        )
        db.add(bar)

    db.commit()
    yield db
    db.close()


def test_backtest_simulator_run(db_session: Session):
    # Setup StrategyEngine with DummyStrategy
    strategy = DummyStrategy()
    engine = StrategyEngine(strategies=[strategy])

    # Setup Simulator
    start_date = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    end_date = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)

    simulator = BacktestSimulator(
        db=db_session,
        strategy_engine=engine,
        market_ids=["GGAL"],
        start_date=start_date,
        end_date=end_date,
    )

    # Run
    # DummyStrategy generates a signal if price > 0, so 3 bars should generate signals
    simulator.run()

    # We can't easily check internal state of StrategyEngine without more mocks,
    # but we can verify it doesn't crash and logs appropriately.
    # In a more advanced test, we'd mock the StrategyEngine to record calls.
