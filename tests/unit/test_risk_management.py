# tests/unit/test_risk_management.py

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

# Import Enums from your project
from trading_bot.core.enums import (
    OrderSide,
    OrderStatus,
    PositionStatus,
    SignalType,
)

# Import ORM Models from your project
from trading_bot.core.models import Position as PositionModel

# Import Schemas from your project
from trading_bot.core.schemas import (
    ExecutionResult,
    MarketData,
    MarketDetails,
    OrderBook,
    OrderRequest,
    PortfolioState,
    Position,
    PriceLevel,
    SizingOutput,
    TradeSignal,
)

# Import Classes to test
from trading_bot.risk_management.abc import BaseSizingStrategy
from trading_bot.risk_management.manager import RiskManager
from trading_bot.risk_management.portfolio import Portfolio
from trading_bot.risk_management.sizing.fixed_amount import FixedAmountSizer

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Fixtures ---


@pytest.fixture
def mock_db_session():
    """Mocks the SQLAlchemy Session."""
    db = MagicMock(spec=Session)
    db.query.return_value.filter_by.return_value.first.return_value = None
    return db


@pytest.fixture
def base_portfolio():
    """Returns a new Portfolio with 10,000 quote currency."""
    return Portfolio(initial_balance=10000.0)


@pytest.fixture
def fixed_sizer():
    """Returns a FixedAmountSizer of 100 quote currency."""
    return FixedAmountSizer(default_amount_quote=100.0)


@pytest.fixture
def mock_market_data():
    """Returns a mock MarketData object."""
    return MarketData(
        market_id="MARKET_01",
        order_book=OrderBook(
            bids=[PriceLevel(price=0.49, size=100)],
            asks=[PriceLevel(price=0.51, size=100)],
        ),
        recent_trades=[],
        details=MarketDetails(
            market_id="MARKET_01",
            name="Test Market",
            end_date=datetime.now(timezone.utc),
            resolution_source="test",
        ),
    )


@pytest.fixture
def market_data_map(mock_market_data):
    """Returns a map containing the mock market data."""
    return {"MARKET_01": mock_market_data}


@pytest.fixture
def mock_buy_signal():
    """Returns a simple BUY signal."""
    return TradeSignal(
        market_id="MARKET_01",
        strategy_name="test_strat",
        signal_type=SignalType.BUY,
        confidence=0.6,
    )


@pytest.fixture
def mock_sell_signal():
    """Returns a simple SELL signal."""
    return TradeSignal(
        market_id="MARKET_01",
        strategy_name="test_strat",
        signal_type=SignalType.SELL,
        confidence=0.4,
    )


# --- TestPortfolio Class ---


class TestPortfolio:
    """Tests the Portfolio class logic."""

    def test_initialization(self):
        portfolio = Portfolio(initial_balance=5000.0)
        assert portfolio._cash_balance == 5000.0
        assert portfolio._positions == {}
        assert portfolio._open_orders == {}

    def test_add_open_order(self, base_portfolio: Portfolio):
        order = OrderRequest(
            market_id="MKT1",
            side=OrderSide.BUY,
            size=100,
            price=0.5,
        )
        base_portfolio.add_open_order("ORDER_123", order)
        assert "ORDER_123" in base_portfolio._open_orders
        assert base_portfolio._open_orders["ORDER_123"] == order

    def test_get_state_initial(self, base_portfolio: Portfolio):
        state = base_portfolio.get_state({})
        assert state.total_balance_quote == 10000.0
        assert state.available_balance_quote == 10000.0
        assert state.positions == []
        assert state.open_orders == []

    def test_get_state_with_open_buy_order(self, base_portfolio: Portfolio):
        order = OrderRequest(
            market_id="MKT1",
            side=OrderSide.BUY,
            size=100,
            price=0.5,
        )
        base_portfolio.add_open_order("ORDER_123", order)
        state = base_portfolio.get_state({})

        # Total balance is unchanged, available is reduced
        assert state.total_balance_quote == 10000.0
        assert state.available_balance_quote == 9950.0  # 10000 - (100 * 0.5)
        assert len(state.open_orders) == 1

    def test_update_order_status_buy_new_long(
        self, base_portfolio: Portfolio, mock_db_session
    ):
        order = OrderRequest(
            market_id="MKT1",
            side=OrderSide.BUY,
            size=200,
            price=0.5,
        )
        base_portfolio.add_open_order("ORDER_123", order)

        fill = ExecutionResult(
            order_id="ORDER_123",
            status=OrderStatus.FILLED,
            filled_size=200,
            avg_price=0.5,
            timestamp=datetime.now(timezone.utc),
        )

        base_portfolio.update_order_status(mock_db_session, fill)

        assert base_portfolio._cash_balance == 9900.0  # 10000 - (200 * 0.5)
        assert len(base_portfolio._open_orders) == 0

        pos_key = "MKT1"
        assert pos_key in base_portfolio._positions
        assert base_portfolio._positions[pos_key].size == 200
        assert base_portfolio._positions[pos_key].entry_price == 0.5
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

    def test_update_order_status_sell_new_short(
        self, base_portfolio: Portfolio, mock_db_session
    ):
        order = OrderRequest(
            market_id="MKT1",
            side=OrderSide.SELL,
            size=100,
            price=0.6,
        )
        base_portfolio.add_open_order("ORDER_123", order)

        fill = ExecutionResult(
            order_id="ORDER_123",
            status=OrderStatus.FILLED,
            filled_size=100,
            avg_price=0.6,
            timestamp=datetime.now(timezone.utc),
        )

        base_portfolio.update_order_status(mock_db_session, fill)

        assert base_portfolio._cash_balance == 10060.0  # 10000 + (100 * 0.6)

        pos_key = "MKT1"
        assert pos_key in base_portfolio._positions
        assert base_portfolio._positions[pos_key].size == -100
        assert base_portfolio._positions[pos_key].entry_price == 0.6
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

    def test_update_order_status_buy_add_to_long(
        self, base_portfolio: Portfolio, mock_db_session
    ):
        # 1. Create initial position
        pos = Position(market_id="MKT1", size=100, entry_price=0.4)
        base_portfolio._positions["MKT1"] = pos
        base_portfolio._cash_balance = 9960.0  # 10000 - (100 * 0.4)

        # 2. Add new order
        order = OrderRequest(
            market_id="MKT1",
            side=OrderSide.BUY,
            size=100,
            price=0.6,
        )
        base_portfolio.add_open_order("ORDER_123", order)

        # 3. Process fill
        fill = ExecutionResult(
            order_id="ORDER_123",
            status=OrderStatus.FILLED,
            filled_size=100,
            avg_price=0.6,
            timestamp=datetime.now(timezone.utc),
        )
        base_portfolio.update_order_status(mock_db_session, fill)

        # Cash = 9960 - (100 * 0.6) = 9900
        assert base_portfolio._cash_balance == 9900.0

        pos_key = "MKT1"
        assert pos_key in base_portfolio._positions

        # Size = 100 + 100 = 200
        assert base_portfolio._positions[pos_key].size == 200

        # Entry = (100 * 0.4 + 100 * 0.6) / 200 = 0.5
        assert base_portfolio._positions[pos_key].entry_price == 0.5
        mock_db_session.commit.assert_called_once()

    def test_update_order_status_sell_close_long(
        self, base_portfolio: Portfolio, mock_db_session
    ):
        # 1. Create initial position
        pos = Position(market_id="MKT1", size=100, entry_price=0.4)
        base_portfolio._positions["MKT1"] = pos
        base_portfolio._cash_balance = 9960.0  # 10000 - (100 * 0.4)

        # 2. Add sell order
        order = OrderRequest(
            market_id="MKT1",
            side=OrderSide.SELL,
            size=100,
            price=0.7,
        )
        base_portfolio.add_open_order("ORDER_123", order)

        # 3. Process fill
        fill = ExecutionResult(
            order_id="ORDER_123",
            status=OrderStatus.FILLED,
            filled_size=100,
            avg_price=0.7,
            timestamp=datetime.now(timezone.utc),
        )
        base_portfolio.update_order_status(mock_db_session, fill)

        # Realized P&L = 100 * (0.7 - 0.4) = 30
        # Cash = 9960 (start) + 70 (fill value) + 30 (pnl) = 10060
        assert base_portfolio._cash_balance == 10060.0

        # Position should be closed
        pos_key = "MKT1"
        assert pos_key not in base_portfolio._positions
        mock_db_session.commit.assert_called_once()

    def test_update_order_status_non_fill(self, base_portfolio: Portfolio):
        order = OrderRequest(
            market_id="MKT1",
            side=OrderSide.BUY,
            size=100,
            price=0.5,
        )
        base_portfolio.add_open_order("ORDER_123", order)

        result = ExecutionResult(
            order_id="ORDER_123",
            status=OrderStatus.CANCELLED,
            filled_size=0,
            avg_price=0,
            timestamp=datetime.now(timezone.utc),
        )

        base_portfolio.update_order_status(MagicMock(), result)

        # No change to cash, order removed from open list
        assert base_portfolio._cash_balance == 10000.0
        assert len(base_portfolio._open_orders) == 0
        assert len(base_portfolio._positions) == 0

    def test_calculate_unrealized_pnl(
        self, base_portfolio: Portfolio, mock_market_data
    ):
        # Long position
        pos_long = Position(market_id="MARKET_01", size=100, entry_price=0.4)
        base_portfolio._positions["MARKET_01"] = pos_long

        # Short position
        pos_short = Position(market_id="MARKET_01", size=-100, entry_price=0.6)
        base_portfolio._positions[("MARKET_01", "no")] = pos_short

        pnl_map = base_portfolio.calculate_unrealized_pnl(
            {"MARKET_01": mock_market_data}
        )

        # Long P&L = size * (current_price - entry_price) = 100 * (0.49 - 0.40) = 9.0
        assert pnl_map["MARKET_01"] == pytest.approx(9.0)

        pos_short_yes = Position(market_id="MARKET_01", size=-100, entry_price=0.6)
        base_portfolio._positions["MARKET_01"] = pos_short_yes

        pnl_map = base_portfolio.calculate_unrealized_pnl(
            {"MARKET_01": mock_market_data}
        )

        # Short 'YES' P&L = size * (current_price - entry_price)
        #                 = -100 * (0.51 - 0.60) = 9.0
        assert pnl_map["MARKET_01"] == pytest.approx(9.0)

    def test_load_positions(self, base_portfolio: Portfolio, mock_db_session):
        mock_pos_orm = PositionModel(
            id=1,
            market_id="MKT_DB",
            size=50,
            entry_price=0.2,
            status=PositionStatus.OPEN,
        )
        mock_db_session.query.return_value.filter_by.return_value.all.return_value = [
            mock_pos_orm
        ]

        base_portfolio.load_positions(mock_db_session)

        pos_key = "MKT_DB"
        assert pos_key in base_portfolio._positions
        assert base_portfolio._positions[pos_key].size == 50
        assert base_portfolio._positions[pos_key].entry_price == 0.2

    def test_persist_position_create(self, mock_db_session):
        # Test _persist_position when no position exists
        pos_schema = Position(market_id="MKT_NEW", size=-10, entry_price=0.3)
        portfolio = Portfolio(100)

        # Mock query to find no existing position
        mock_db_session.query.return_value.filter_by.return_value.first.return_value = (
            None
        )

        portfolio._persist_position(mock_db_session, pos_schema, create=True)

        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

        # Check that the object added was a PositionModel
        added_obj = mock_db_session.add.call_args[0][0]
        assert isinstance(added_obj, PositionModel)
        assert added_obj.market_id == "MKT_NEW"
        assert added_obj.size == -10
        assert added_obj.status == PositionStatus.OPEN


# --- TestRiskManager Class ---


class TestRiskManager:
    """Tests the RiskManager class logic."""

    @pytest.fixture
    def mock_portfolio(self):
        """Mocks the Portfolio object."""
        portfolio = MagicMock(spec=Portfolio)
        # Default state
        portfolio.get_state.return_value = PortfolioState(
            total_balance_quote=10000.0,
            available_balance_quote=10000.0,
            positions=[],
            open_orders=[],
        )
        return portfolio

    def test_initialization(self, mock_portfolio, fixed_sizer):
        rm = RiskManager(
            portfolio=mock_portfolio,
            sizer=fixed_sizer,
            max_allocation_per_market=0.5,
            max_total_positions=5,
        )
        assert rm.portfolio == mock_portfolio
        assert rm.sizer == fixed_sizer
        assert rm.max_allocation_per_market == 0.5
        assert rm.max_total_positions == 5

    def test_process_signal_hold(self, mock_portfolio, fixed_sizer, market_data_map):
        rm = RiskManager(portfolio=mock_portfolio, sizer=fixed_sizer)
        signal = TradeSignal(
            market_id="MARKET_01",
            strategy_name="test",
            signal_type=SignalType.HOLD,
            confidence=0.5,
        )
        order = rm.process_signal(signal, market_data_map)
        assert order is None

    def test_process_signal_no_market_data(
        self, mock_portfolio, fixed_sizer, mock_buy_signal, caplog
    ):
        rm = RiskManager(portfolio=mock_portfolio, sizer=fixed_sizer)
        # Pass an empty map
        order = rm.process_signal(mock_buy_signal, {})
        assert order is None
        assert "No market data for MARKET_01" in caplog.text

    def test_process_signal_sizer_returns_zero(
        self, mock_portfolio, mock_buy_signal, market_data_map, caplog
    ):
        # Use a sizer that will return 0
        zero_sizer = MagicMock(spec=BaseSizingStrategy)
        zero_sizer.calculate_size.return_value = SizingOutput(
            amount_quote=0, size_shares=0
        )

        rm = RiskManager(portfolio=mock_portfolio, sizer=zero_sizer)

        # Set caplog level to capture INFO messages
        with caplog.at_level(logging.INFO):
            order = rm.process_signal(mock_buy_signal, market_data_map)

        assert order is None
        assert "Sizer returned zero size" in caplog.text

    def test_process_signal_risk_check_fail(
        self, mock_portfolio, fixed_sizer, mock_buy_signal, market_data_map, caplog
    ):
        # Make available balance too low
        low_balance_state = PortfolioState(
            total_balance_quote=50.0,
            available_balance_quote=50.0,
            positions=[],
            open_orders=[],
        )
        mock_portfolio.get_state.return_value = low_balance_state

        # Fixed sizer is 100 quote currency, which is > 50
        rm = RiskManager(portfolio=mock_portfolio, sizer=fixed_sizer)
        order = rm.process_signal(mock_buy_signal, market_data_map)

        assert order is None
        assert "exceeds available balance" in caplog.text

    def test_process_signal_success_buy(
        self, mock_portfolio, fixed_sizer, mock_buy_signal, market_data_map
    ):
        rm = RiskManager(portfolio=mock_portfolio, sizer=fixed_sizer)
        order = rm.process_signal(mock_buy_signal, market_data_map)

        assert isinstance(order, OrderRequest)
        assert order.market_id == "MARKET_01"
        assert order.side == OrderSide.BUY

        # Sizer is 100 quote currency. Price is 0.51 (best ask).
        # Size = 100 / 0.51 = 196.078...
        assert order.size == pytest.approx(100.0 / 0.51)
        assert order.price == 0.51

    def test_process_signal_success_sell(
        self, mock_portfolio, fixed_sizer, mock_sell_signal, market_data_map
    ):
        rm = RiskManager(portfolio=mock_portfolio, sizer=fixed_sizer)
        order = rm.process_signal(mock_sell_signal, market_data_map)

        assert isinstance(order, OrderRequest)
        assert order.side == OrderSide.SELL

        # Sizer is 100 quote currency. Price is 0.49 (best bid).
        # Size = 100 / 0.49 = 204.08...
        assert order.size == pytest.approx(100.0 / 0.49)
        assert order.price == 0.49

    # --- Direct tests for _passes_risk_checks ---

    @pytest.fixture
    def risk_check_deps(self, mock_market_data):
        """Dependencies for testing _passes_risk_checks directly."""
        rm = RiskManager(MagicMock(spec=Portfolio), MagicMock())  # Mock portfolio
        sizing_output = SizingOutput(amount_quote=1000, size_shares=2000)
        signal = TradeSignal(
            market_id="MKT1",
            strategy_name="test",
            signal_type=SignalType.BUY,
            confidence=0.7,
        )
        portfolio_state = PortfolioState(
            total_balance_quote=10000,
            available_balance_quote=10000,
            positions=[],
            open_orders=[],
        )
        market_map = {"MKT1": mock_market_data}
        return rm, sizing_output, signal, portfolio_state, market_map

    def test_passes_risk_checks_insufficient_balance(self, risk_check_deps, caplog):
        rm, sizing_output, signal, portfolio_state, market_map = risk_check_deps

        portfolio_state.available_balance_quote = 500  # Less than 1000

        passes = rm._passes_risk_checks(
            sizing_output, signal, portfolio_state, market_map
        )
        assert not passes
        assert "exceeds available balance" in caplog.text

    def test_passes_risk_checks_max_positions(self, risk_check_deps, caplog):
        rm, sizing_output, signal, portfolio_state, market_map = risk_check_deps

        rm.max_total_positions = 1
        portfolio_state.positions = [
            Position(
                market_id="OTHER_MKT",
                size=10,
                entry_price=0.1,
            )
        ]

        passes = rm._passes_risk_checks(
            sizing_output, signal, portfolio_state, market_map
        )
        assert not passes
        assert "Would exceed max positions (1)" in caplog.text

    def test_passes_risk_checks_max_allocation(self, risk_check_deps, caplog):
        rm, sizing_output, signal, portfolio_state, market_map = risk_check_deps

        rm.max_allocation_per_market = 0.1  # 10%
        # total_equity is 10000. max_alloc is 1000.
        # sizing_output.amount_quote is 1000.

        portfolio_state.positions = [
            Position(market_id="MKT1", size=10, entry_price=0.1)
        ]
        # Mock PnL calculation on the RiskManager's portfolio instance
        rm.portfolio.calculate_unrealized_pnl.return_value = {"MKT1": 0.0}

        passes = rm._passes_risk_checks(
            sizing_output, signal, portfolio_state, market_map
        )

        assert not passes
        # (10 * 0.1) + 1000 = 1001. 1001 / 10000 = 10.01%
        assert "New allocation (10.0%) would exceed max (10.0%)" in caplog.text

    def test_passes_risk_checks_all_pass(self, risk_check_deps):
        rm, sizing_output, signal, portfolio_state, market_map = risk_check_deps

        # Use default risk params
        rm.max_total_positions = 10
        rm.max_allocation_per_market = 0.25

        # Mock PnL calculation
        rm.portfolio.calculate_unrealized_pnl.return_value = {}

        passes = rm._passes_risk_checks(
            sizing_output, signal, portfolio_state, market_map
        )
        assert passes
