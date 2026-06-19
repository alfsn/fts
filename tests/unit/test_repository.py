# tests/unit/test_repository.py

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from trading_bot.core.database import Base
from trading_bot.core.enums import OrderSide, OrderStatus, PositionStatus
from trading_bot.core.models import Market as MarketModel
from trading_bot.core.repository import OrderRepository, PositionRepository
from trading_bot.core.schemas import Position as PositionSchema


@pytest.fixture
def in_memory_db():
    """Sets up an in-memory SQLite database and creates the schema."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionClass = sessionmaker(bind=engine)
    session = SessionClass()

    # Pre-populate a market record since foreign key constraints exist
    market = MarketModel(
        market_id="AAPL",
        name="Apple Inc. Stock",
        end_date=datetime.now(),
    )
    session.add(market)
    session.commit()

    try:
        yield session
    finally:
        session.close()


def test_position_repository_create_update_delete(in_memory_db):
    """Tests saving, updating, loading, and closing positions via PositionRepository."""
    pos_repo = PositionRepository(in_memory_db)

    # 1. Create a Position
    pos = PositionSchema(
        market_id="AAPL",
        outcome=None,
        size=10.0,
        entry_price=150.0,
    )
    pos_repo.save_position(pos)

    # Verify saved state
    open_positions = pos_repo.get_open_positions()
    assert len(open_positions) == 1
    assert open_positions[0].market_id == "AAPL"
    assert open_positions[0].outcome is None
    assert open_positions[0].size == 10.0
    assert open_positions[0].entry_price == 150.0

    # 2. Update Position Size and Entry Price
    pos.size = 15.0
    pos.entry_price = 155.0
    pos_repo.save_position(pos)

    open_positions = pos_repo.get_open_positions()
    assert len(open_positions) == 1
    assert open_positions[0].size == 15.0
    assert open_positions[0].entry_price == 155.0

    # 3. Close (Delete) Position
    pos_repo.save_position(pos, is_delete=True)

    open_positions = pos_repo.get_open_positions()
    assert len(open_positions) == 0


def test_order_repository_flow(in_memory_db):
    """Tests logging and updating orders in OrderRepository."""
    order_repo = OrderRepository(in_memory_db)
    order_id = "test-order-123"

    # 1. Log a new Order request
    order_repo.create_order(
        order_id=order_id,
        market_id="AAPL",
        strategy_name="momentum_v1",
        side=OrderSide.BUY,
        outcome=None,
        requested_size=100.0,
        requested_price=150.0,
        status=OrderStatus.PENDING,
    )

    # Query DB directly to verify pending order log
    from trading_bot.core.models import OrderLog as OrderLogModel

    db_order = in_memory_db.query(OrderLogModel).filter_by(order_id=order_id).first()
    assert db_order is not None
    assert db_order.market_id == "AAPL"
    assert db_order.side == OrderSide.BUY
    assert db_order.outcome is None
    assert db_order.requested_size == 100.0
    assert db_order.requested_price == 150.0
    assert db_order.status == OrderStatus.PENDING

    # 2. Update the Order status to FILLED
    order_repo.update_order(
        order_id=order_id,
        status=OrderStatus.FILLED,
        filled_size=100.0,
        avg_fill_price=149.8,
    )

    in_memory_db.refresh(db_order)
    assert db_order.status == OrderStatus.FILLED
    assert db_order.filled_size == 100.0
    assert db_order.avg_fill_price == 149.8


def test_market_data_repository(in_memory_db):
    """Tests ensuring markets and bulk saving unique bars via MarketDataRepository."""
    from trading_bot.core.enums import BarType
    from trading_bot.core.repository import MarketDataRepository
    from trading_bot.core.schemas import BarData, MarketDetails

    repo = MarketDataRepository(in_memory_db)

    # 1. Test ensure_market
    details = MarketDetails(
        market_id="MSFT",
        name="Microsoft Corp",
        end_date=datetime(2026, 12, 31),
        resolution_source="test_source",
    )
    market = repo.ensure_market(details)
    assert market.market_id == "MSFT"
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
            ticks_count=1,
            dollar_volume=60900.0,
        ),
    ]

    # Save the bars
    saved_count = repo.save_bars("MSFT", bars)
    assert saved_count == 2

    # Verify records in DB
    from trading_bot.core.models import BarDataLog as BarDataLogModel

    db_bars = in_memory_db.query(BarDataLogModel).filter_by(market_id="MSFT").all()
    assert len(db_bars) == 2

    # Save again with same bars (should prevent duplicates and return 0)
    saved_count2 = repo.save_bars("MSFT", bars)
    assert saved_count2 == 0

    # 3. Test get_bars with filters
    loaded_bars = repo.get_bars("MSFT", bar_type=BarType.TIME)
    assert len(loaded_bars) == 2

    # Test filtering by a non-existent bar type
    loaded_dollar_bars = repo.get_bars("MSFT", bar_type=BarType.DOLLAR)
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
    repo.save_bars("MSFT", bars_with_interval)

    # get_bars filtering by interval
    hourly_bars = repo.get_bars("MSFT", bar_type=BarType.TIME, interval="1h")
    assert len(hourly_bars) == 1
    assert hourly_bars[0].interval == "1h"
