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
