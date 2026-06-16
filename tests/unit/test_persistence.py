# tests/unit/test_persistence.py

import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from trading_bot.core.database import SessionLocal, init_db
from trading_bot.core.enums import BarType
from trading_bot.core.models import BarDataLog, Market


@pytest.fixture(scope="module")
def setup_db():
    # Use a separate test database file
    test_db = "tests/test_persistence.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    # We need to monkeypatch the engine or just use it as is if it's configurable
    # For this test, we'll just run against whatever is in settings,
    # but ideally we should use an in-memory db or a dedicated test db.

    init_db()
    yield

    if os.path.exists(test_db):
        os.remove(test_db)


def test_bar_data_log_persistence():
    # Ensure tables are created
    init_db()

    db: Session = SessionLocal()
    try:
        # 1. Create a Market (or get it if it exists)
        market_id = "GGAL"
        market = db.execute(
            select(Market).where(Market.market_id == market_id)
        ).scalar_one_or_none()

        if not market:
            market = Market(
                market_id=market_id,
                name="Grupo Galicia",
                end_date=datetime(2025, 12, 31, tzinfo=timezone.utc),
                resolution_source="BYMA",
            )
            db.add(market)
            db.commit()

        # 2. Create a BarDataLog
        now = datetime.now(timezone.utc)
        bar = BarDataLog(
            market_id=market_id,
            timestamp=now,
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            volume=1000.0,
            bar_type=BarType.DOLLAR,
            ticks_count=50,
            dollar_volume=100000.0,
        )
        db.add(bar)
        db.commit()

        # 3. Retrieve and verify
        stmt = select(BarDataLog).where(BarDataLog.market_id == "GGAL")
        retrieved_bars = db.execute(stmt).scalars().all()

        assert len(retrieved_bars) >= 1
        # Find the one we just added or just check the last one
        latest_bar = retrieved_bars[-1]
        assert latest_bar.open == 100.0
        assert latest_bar.bar_type == BarType.DOLLAR
        assert latest_bar.market.name == "Grupo Galicia"
    finally:
        db.close()


def test_init_db_with_extra_models():
    # This test checks if init_db can handle a list of modules
    # We'll use a built-in module as a dummy, or just verify it doesn't crash
    # since we don't have a real plugin module yet.
    try:
        init_db(extra_models=["os"])  # os is not a model but it should import fine
    except Exception as e:
        pytest.fail(f"init_db with extra_models failed: {e}")
