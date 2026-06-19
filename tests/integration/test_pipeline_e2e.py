# tests/integration/test_pipeline_e2e.py

from datetime import datetime, timezone
from typing import Dict, List, Sequence

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from trading_bot.core.database import Base
from trading_bot.core.enums import (
    BarType,
    OrderSide,
    OrderStatus,
    PositionStatus,
    SignalType,
)
from trading_bot.core.models import OrderLog as OrderLogModel
from trading_bot.core.models import Position as PositionModel
from trading_bot.core.pipeline import TradingPipeline
from trading_bot.core.repository import OrderRepository, PositionRepository
from trading_bot.core.schemas import (
    BarData,
    IngestionEngineOutput,
    MarketData,
    MarketDetails,
    OrderBook,
    PriceLevel,
    TradeSignal,
)
from trading_bot.data_ingestion.abc import BaseMarketDataProvider
from trading_bot.data_ingestion.engine import DataIngestionEngine
from trading_bot.execution.engine import ExecutionEngine
from trading_bot.execution.handlers.polymarket_handler import PolymarketHandler
from trading_bot.risk_management.manager import RiskManager
from trading_bot.risk_management.portfolio import Portfolio
from trading_bot.risk_management.sizing.fixed_amount import FixedAmountSizer
from trading_bot.strategy.abc import BaseStrategy
from trading_bot.strategy.engine import StrategyEngine


class FakeMarketProvider(BaseMarketDataProvider):
    """
    A fake implementation of the market data provider that returns
    pre-configured data for testing the ingestion stage.
    """

    def __init__(self, data: Dict[str, MarketData]):
        self.data_to_return = data

    def get_market_data(self, market_id: str) -> MarketData | None:
        return self.data_to_return.get(market_id)

    def list_tradable_markets(self) -> Sequence[MarketDetails]:
        return [md.details for md in self.data_to_return.values()]

    def get_market_details(self, market_id: str) -> MarketDetails:
        return self.data_to_return.get(market_id).details

    def get_order_book(self, market_id: str) -> OrderBook:
        return self.data_to_return.get(market_id).order_book

    def get_trade_history(self, market_id: str) -> List:
        return []

    def get_bars(self, market_id: str, count: int = 100) -> List[BarData]:
        return getattr(self.data_to_return.get(market_id), "recent_bars", [])


class FakeStrategy(BaseStrategy):
    """
    A fake strategy to generate predictable signals for testing.
    """

    def __init__(self, name: str = "fake_strategy") -> None:
        self._name = name
        self.signal_to_generate = SignalType.BUY
        self.confidence = 0.8
        self.market_id = "BTC/USDT"

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, data: IngestionEngineOutput) -> Sequence[TradeSignal]:
        if self.signal_to_generate == SignalType.FLAT:
            return []
        return [
            TradeSignal(
                market_id=self.market_id,
                strategy_name=self.name,
                signal_type=self.signal_to_generate,
                confidence=self.confidence,
            )
        ]


@pytest.fixture(scope="function")
def db_session() -> Session:
    """
    Creates an in-memory SQLite database session for testing database logs
    during pipeline integration ticks.
    """
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(engine)


def test_pipeline_e2e_flow_buy_and_sell_with_db(db_session: Session):
    """
    Tests the full TradingPipeline end-to-end flow with actual and mock components:
    1. Runs ingestion flow querying FakeMarketProvider.
    2. Runs strategy flow generating predictable BUY/SELL signals.
    3. Runs RiskManager to size the approved OrderRequests using fixed sizing.
    4. Runs ExecutionEngine with PolymarketHandler to fill orders.
    5. Verifies portfolio balance, DB OrderLog, and DB Position logs after each step.
    """
    market_id = "BTC/USDT"

    # --- 1. SETUP DATA AND REAL/MOCK PLUGINS ---

    # Setup mock bar data
    bar = BarData(
        timestamp=datetime.now(timezone.utc),
        open=50000.0,
        high=50500.0,
        low=49500.0,
        close=50000.0,
        volume=10.0,
        bar_type=BarType.TIME,
        ticks_count=100,
        dollar_volume=500000.0,
    )

    # Setup mock order book (best ask = 50200.0, best bid = 49800.0)
    order_book = OrderBook(
        bids=[PriceLevel(price=49800.0, size=2.0)],
        asks=[PriceLevel(price=50200.0, size=2.0)],
        timestamp=datetime.now(timezone.utc),
    )

    # Assemble market data
    market_data = MarketData(
        market_id=market_id,
        details=MarketDetails(
            market_id=market_id,
            name="Bitcoin/Tether",
            end_date=datetime.max.replace(tzinfo=timezone.utc),
            resolution_source="test",
        ),
        recent_bars=[bar],
        order_book=order_book,
        recent_trades=[],
    )

    # Initialize mock components
    market_provider = FakeMarketProvider(data={market_id: market_data})
    fake_strategy = FakeStrategy()

    # Initialize real database repositories
    pos_repo = PositionRepository(db_session)
    order_repo = OrderRepository(db_session)

    # Initialize real Portfolio
    portfolio = Portfolio(
        initial_balance=10000.0,
        quote_currency="USD",
        pos_repo=pos_repo,
        order_repo=order_repo,
    )
    portfolio.load_positions()

    # Initialize real orchestrations
    ingestion_engine = DataIngestionEngine(
        market_provider=market_provider,
        external_providers=[],
        market_ids=[market_id],
    )

    strategy_engine = StrategyEngine(strategies=[fake_strategy])

    sizer = FixedAmountSizer(default_amount_quote=5020.0)  # 5020 USD

    risk_manager = RiskManager(
        portfolio=portfolio,
        sizer=sizer,
        max_allocation_per_market=2.0,
    )

    # PolymarketHandler simulates instant fills (buying at best ask, selling at best bid)
    execution_handler = PolymarketHandler()

    execution_engine = ExecutionEngine(
        execution_handler=execution_handler,
        portfolio=portfolio,
    )

    # Bind unified TradingPipeline
    pipeline = TradingPipeline(
        ingestion=ingestion_engine,
        strategy=strategy_engine,
        risk=risk_manager,
        execution=execution_engine,
        portfolio=portfolio,
    )

    # Verify initial states
    assert portfolio._cash_balance == 10000.0
    initial_state = portfolio.get_state({market_id: market_data})
    assert initial_state.available_balance_quote == 10000.0
    assert len(initial_state.positions) == 0

    # --- 2. EXECUTE FLOW 1: BUY TICK ---

    fake_strategy.signal_to_generate = SignalType.BUY

    # Execute a single pipeline tick (will ingest data, run strategy, run risk, and place filled order)
    pipeline.execute_single_tick(db=db_session)

    # --- 3. ASSERTIONS FOR FLOW 1 (BUY) ---

    # Cash should be: 10000.0 - 5020.0 = 4980.0
    assert portfolio._cash_balance == pytest.approx(4980.0)

    # Purchased size should be: 5020.0 / 50200.0 (best ask) = 0.1 shares/contracts
    post_buy_state = portfolio.get_state({market_id: market_data})
    assert len(post_buy_state.positions) == 1
    buy_pos = post_buy_state.positions[0]
    assert buy_pos.market_id == market_id
    assert buy_pos.size == pytest.approx(0.1)
    assert buy_pos.entry_price == 50200.0

    # Database checks: order log should be persisted as FILLED
    db_orders = db_session.query(OrderLogModel).all()
    assert len(db_orders) == 1
    db_order = db_orders[0]
    assert db_order.market_id == market_id
    assert db_order.side == OrderSide.BUY
    assert db_order.status == OrderStatus.FILLED
    assert db_order.requested_price == 50200.0
    assert db_order.filled_size == pytest.approx(0.1)

    # Database checks: position log should be persisted as OPEN
    db_positions = db_session.query(PositionModel).all()
    assert len(db_positions) == 1
    db_pos = db_positions[0]
    assert db_pos.market_id == market_id
    assert db_pos.size == pytest.approx(0.1)
    assert db_pos.entry_price == 50200.0
    assert db_pos.status == PositionStatus.OPEN

    # --- 4. EXECUTE FLOW 2: SELL TICK ---

    # Update strategy signal to SELL
    fake_strategy.signal_to_generate = SignalType.SELL

    # Simulate price going up: update best bid in order book to 52000.0
    order_book.bids = [PriceLevel(price=52000.0, size=2.0)]
    order_book.asks = [PriceLevel(price=52500.0, size=2.0)]

    # Sizer for SELL signal (sells 5200 USD worth of position at best bid of 52000.0)
    # Size in shares to sell = 5200 / 52000.0 = 0.1 shares (which is our entire position)
    sizer.default_amount_quote = 5200.0

    # Run pipeline tick to execute sell
    pipeline.execute_single_tick(db=db_session)

    # --- 5. ASSERTIONS FOR FLOW 2 (SELL) ---

    # Cash should be: 4980.0 (prev) + (0.1 shares * 52000.0 bid price) = 4980.0 + 5200.0 = 10180.0
    # Profit realized: 180.0 USD
    assert portfolio._cash_balance == pytest.approx(10180.0)

    # Position should be closed (size = 0.0 in DB or deleted in portfolio state representation)
    post_sell_state = portfolio.get_state({market_id: market_data})
    assert len(post_sell_state.positions) == 0

    # Database checks: second order log (SELL) should be registered in DB as FILLED
    db_orders_final = (
        db_session.query(OrderLogModel).order_by(OrderLogModel.created_at.asc()).all()
    )
    assert len(db_orders_final) == 2
    sell_order = db_orders_final[1]
    assert sell_order.side == OrderSide.SELL
    assert sell_order.status == OrderStatus.FILLED
    assert sell_order.requested_price == 52000.0
    assert sell_order.filled_size == pytest.approx(0.1)

    # Database checks: position status should be CLOSED in DB (or size updated to 0.0 depending on save_position implementation)
    db_pos_final = (
        db_session.query(PositionModel).filter_by(market_id=market_id).first()
    )
    assert db_pos_final is not None
    assert db_pos_final.status == PositionStatus.CLOSED
    assert db_pos_final.size == 0.0
