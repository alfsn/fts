# tests/unit/test_repository.py

from datetime import datetime

import pytest
from quant_core.enums import OrderSide, OrderStatus, PositionStatus
from quant_core.models import Position as PositionSchema
from quant_data.db import Market as MarketModel
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from trading_bot.core.database import Base


@pytest.fixture
def in_memory_db():
    """Sets up an in-memory SQLite database and creates the schema."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionClass = sessionmaker(bind=engine)
    session = SessionClass()

    # Pre-populate a market record since foreign key constraints exist
    market = MarketModel(
        instrument_id="AAPL",
        name="Apple Inc. Stock",
        end_date=datetime.now(),
    )
    session.add(market)
    session.commit()

    try:
        yield session
    finally:
        session.close()


def test_market_data_repository(in_memory_db):
    """Tests ensuring markets and bulk saving unique bars via MarketDataRepository."""
    from quant_core.enums import AssetType, BarType, Geography
    from quant_core.models import Instrument, TradFiDetails
    from quant_data.db.repositories import MarketDataRepository
    from trading_bot.core.schemas import BarData

    repo = MarketDataRepository(in_memory_db)

    # 1. Test ensure_market
    details = Instrument(
        instrument_id="MSFT",
        name="Microsoft Corp",
        details=TradFiDetails(
            asset_type=AssetType.STOCK,
            geography=Geography.US,
            industry="Tech",
            currency="USD",
        ),
    )
    market = repo.ensure_market(details)
    assert market.instrument_id == "MSFT"
    assert market.name == "Microsoft Corp"

    # Verify update to details
    details.name = "Microsoft Corporation"
    market = repo.ensure_market(details)
    assert market.name == "Microsoft Corporation"

    # 2. Test save_bars
    bars = [
        BarData(
            timestamp=datetime(2026, 6, 15, 12, 0),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=500.0,
            bar_type=BarType.TIME,
            interval="1m",
            ticks_count=1,
            dollar_volume=50250.0,
        ),
        BarData(
            timestamp=datetime(2026, 6, 15, 12, 1),
            open=100.5,
            high=102.0,
            low=100.0,
            close=101.5,
            volume=600.0,
            bar_type=BarType.TIME,
            interval="1m",
            ticks_count=1,
            dollar_volume=60900.0,
        ),
    ]

    # Save the bars
    saved_count = repo.save_bars("MSFT", bars, source="test")
    assert saved_count == 2

    # Verify records in DB
    from quant_data.db import BarDataLog as BarDataLogModel

    db_bars = in_memory_db.query(BarDataLogModel).filter_by(instrument_id="MSFT").all()
    assert len(db_bars) == 2

    # Save again with same bars (should prevent duplicates and return 0)
    saved_count2 = repo.save_bars("MSFT", bars, source="test")
    assert saved_count2 == 0

    # 3. Test load_bars with filters
    loaded_bars = repo.load_bars("MSFT", "1m", bar_type=BarType.TIME)
    assert len(loaded_bars) == 2

    # Test filtering by a non-existent bar type
    loaded_dollar_bars = repo.load_bars("MSFT", "1m", bar_type=BarType.DOLLAR)
    assert len(loaded_dollar_bars) == 0

    # Save bars with an explicit interval
    bars_with_interval = [
        BarData(
            timestamp=datetime(2026, 6, 15, 13, 0),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=500.0,
            bar_type=BarType.TIME,
            interval="1h",
            ticks_count=1,
            dollar_volume=50250.0,
        )
    ]
    repo.save_bars("MSFT", bars_with_interval, source="test")

    # load_bars filtering by interval
    hourly_bars = repo.load_bars("MSFT", "1h")
    assert len(hourly_bars) == 1
    assert hourly_bars[0].interval == "1h"
