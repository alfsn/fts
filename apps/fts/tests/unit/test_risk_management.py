import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

# Import Enums from your project
from quant_core.enums import (
    AssetType,
    Geography,
    OrderSide,
    OrderStatus,
    PositionStatus,
    SignalType,
)
from quant_core.models import CashBalance, Instrument
from quant_core.models import Portfolio as PortfolioSchema
from quant_core.models import TradFiDetails
from sqlalchemy.orm import Session

# Import Schemas from your project
from trading_bot.core.schemas import (
    ExecutionResult,
    Instrument,
    MarketData,
    OrderBook,
    OrderRequest,
    Portfolio,
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

# Import ORM Models from your project


# tests/unit/test_risk_management.py


# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Fixtures ---


@pytest.fixture
def mock_db_session():
    """Mocks the SQLAlchemy Session."""
    db = MagicMock(spec=Session)
    db.query.return_value.filter_by.return_value.first.return_value = None
    db.query.return_value.filter_by.return_value.filter.return_value.first.return_value = (
        None
    )
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
        instrument_id="MARKET_01",
        order_book=OrderBook(
            bids=[PriceLevel(price=0.49, quantity=100)],
            asks=[PriceLevel(price=0.51, quantity=100)],
        ),
        recent_trades=[],
        details=Instrument(
            instrument_id="MARKET_01",
            name="Test Market",
            details=TradFiDetails(
                asset_type=AssetType.STOCK,
                geography=Geography.US,
                industry="Tech",
                currency="USD",
            ),
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
        instrument_id="MARKET_01",
        strategy_name="test_strat",
        signal_type=SignalType.BUY,
        confidence=0.6,
    )


@pytest.fixture
def mock_sell_signal():
    """Returns a simple SELL signal."""
    return TradeSignal(
        instrument_id="MARKET_01",
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
            instrument_id="MKT1",
            side=OrderSide.BUY,
            quantity=100,
            price=0.5,
        )
        base_portfolio.add_open_order("ORDER_123", order)
        assert "ORDER_123" in base_portfolio._open_orders
        assert base_portfolio._open_orders["ORDER_123"] == order

    def test_get_state_initial(self, base_portfolio: Portfolio):
        state = base_portfolio.get_state({})
        assert state.cash_balances[0].total == 10000.0
        assert state.cash_balances[0].available == 10000.0
        assert state.positions == []
        assert state.open_orders == []

    def test_get_state_with_open_buy_order(self, base_portfolio: Portfolio):
        order = OrderRequest(
            instrument_id="MKT1",
            side=OrderSide.BUY,
            quantity=100,
            price=0.5,
        )
        base_portfolio.add_open_order("ORDER_123", order)
        state = base_portfolio.get_state({})

        # Total balance is unchanged, available is reduced
        assert state.cash_balances[0].total == 10000.0
        assert state.cash_balances[0].available == 9950.0  # 10000 - (100 * 0.5)
        assert len(state.open_orders) == 1

    def test_update_order_status_buy_new_long(
        self, base_portfolio: Portfolio, mock_db_session
    ):
        order = OrderRequest(
            instrument_id="MKT1",
            side=OrderSide.BUY,
            quantity=200,
            price=0.5,
        )
        base_portfolio.add_open_order("ORDER_123", order)

        fill = ExecutionResult(
            order_id="ORDER_123",
            status=OrderStatus.FILLED,
            filled_quantity=200,
            avg_price=0.5,
            timestamp=datetime.now(timezone.utc),
        )

        base_portfolio.update_order_status(fill)

        assert base_portfolio._cash_balance == 9900.0  # 10000 - (200 * 0.5)
        assert len(base_portfolio._open_orders) == 0

        pos_key = "MKT1"
        assert pos_key in base_portfolio._positions
        assert base_portfolio._positions[pos_key].quantity == 200
        assert base_portfolio._positions[pos_key].cost_basis == 0.5

    def test_update_order_status_sell_new_short(
        self, base_portfolio: Portfolio, mock_db_session
    ):
        order = OrderRequest(
            instrument_id="MKT1",
            side=OrderSide.SELL,
            quantity=100,
            price=0.6,
        )
        base_portfolio.add_open_order("ORDER_123", order)

        fill = ExecutionResult(
            order_id="ORDER_123",
            status=OrderStatus.FILLED,
            filled_quantity=100,
            avg_price=0.6,
            timestamp=datetime.now(timezone.utc),
        )

        base_portfolio.update_order_status(fill)

        assert base_portfolio._cash_balance == 10060.0  # 10000 + (100 * 0.6)

        pos_key = "MKT1"
        assert pos_key in base_portfolio._positions
        assert base_portfolio._positions[pos_key].quantity == -100
        assert base_portfolio._positions[pos_key].cost_basis == 0.6

    def test_update_order_status_buy_add_to_long(
        self, base_portfolio: Portfolio, mock_db_session
    ):
        # 1. Create initial position
        pos = Position(instrument_id="MKT1", quantity=100, cost_basis=0.4)
        base_portfolio._positions["MKT1"] = pos
        base_portfolio._cash_balance = 9960.0  # 10000 - (100 * 0.4)

        # 2. Add new order
        order = OrderRequest(
            instrument_id="MKT1",
            side=OrderSide.BUY,
            quantity=100,
            price=0.6,
        )
        base_portfolio.add_open_order("ORDER_123", order)

        # 3. Process fill
        fill = ExecutionResult(
            order_id="ORDER_123",
            status=OrderStatus.FILLED,
            filled_quantity=100,
            avg_price=0.6,
            timestamp=datetime.now(timezone.utc),
        )
        base_portfolio.update_order_status(fill)

        # Cash = 9960 - (100 * 0.6) = 9900
        assert base_portfolio._cash_balance == 9900.0

        pos_key = "MKT1"
        assert pos_key in base_portfolio._positions

        # Size = 100 + 100 = 200
        assert base_portfolio._positions[pos_key].quantity == 200

        # Entry = (100 * 0.4 + 100 * 0.6) / 200 = 0.5
        assert base_portfolio._positions[pos_key].cost_basis == 0.5

    def test_update_order_status_sell_close_long(
        self, base_portfolio: Portfolio, mock_db_session
    ):
        # 1. Create initial position
        pos = Position(instrument_id="MKT1", quantity=100, cost_basis=0.4)
        base_portfolio._positions["MKT1"] = pos
        base_portfolio._cash_balance = 9960.0  # 10000 - (100 * 0.4)

        # 2. Add sell order
        order = OrderRequest(
            instrument_id="MKT1",
            side=OrderSide.SELL,
            quantity=100,
            price=0.7,
        )
        base_portfolio.add_open_order("ORDER_123", order)

        # 3. Process fill
        fill = ExecutionResult(
            order_id="ORDER_123",
            status=OrderStatus.FILLED,
            filled_quantity=100,
            avg_price=0.7,
            timestamp=datetime.now(timezone.utc),
        )
        base_portfolio.update_order_status(fill)

        # Realized P&L = 100 * (0.7 - 0.4) = 30
        # Cash = 9960 (start) + 70 (fill value) = 10030
        assert base_portfolio._cash_balance == 10030.0

        # Position should be closed
        pos_key = "MKT1"
        assert pos_key not in base_portfolio._positions

    def test_update_order_status_non_fill(self, base_portfolio: Portfolio):
        order = OrderRequest(
            instrument_id="MKT1",
            side=OrderSide.BUY,
            quantity=100,
            price=0.5,
        )
        base_portfolio.add_open_order("ORDER_123", order)

        result = ExecutionResult(
            order_id="ORDER_123",
            status=OrderStatus.CANCELLED,
            filled_quantity=0,
            avg_price=0,
            timestamp=datetime.now(timezone.utc),
        )

        base_portfolio.update_order_status(result)

        # No change to cash, order removed from open list
        assert base_portfolio._cash_balance == 10000.0
        assert len(base_portfolio._open_orders) == 0
        assert len(base_portfolio._positions) == 0

    def test_calculate_unrealized_pnl(
        self, base_portfolio: Portfolio, mock_market_data
    ):
        # Long position
        pos_long = Position(instrument_id="MARKET_01", quantity=100, cost_basis=0.4)
        base_portfolio._positions["MARKET_01"] = pos_long

        # Short position
        pos_short = Position(instrument_id="MARKET_01", quantity=-100, cost_basis=0.6)
        base_portfolio._positions[("MARKET_01", "no")] = pos_short

        pnl_map = base_portfolio.calculate_unrealized_pnl(
            {"MARKET_01": mock_market_data}
        )

        # Long P&L = size * (current_price - entry_price) = 100 * (0.49 - 0.40) = 9.0
        assert pnl_map["MARKET_01"] == pytest.approx(9.0)

        pos_short_yes = Position(
            instrument_id="MARKET_01", quantity=-100, cost_basis=0.6
        )
        base_portfolio._positions["MARKET_01"] = pos_short_yes

        pnl_map = base_portfolio.calculate_unrealized_pnl(
            {"MARKET_01": mock_market_data}
        )

        # Short 'YES' P&L = size * (current_price - entry_price)
        #                 = -100 * (0.51 - 0.60) = 9.0
        assert pnl_map["MARKET_01"] == pytest.approx(9.0)


# --- TestRiskManager Class ---


class TestRiskManager:
    """Tests the RiskManager class logic."""

    @pytest.fixture
    def mock_portfolio(self):
        """Mocks the Portfolio object."""
        portfolio = MagicMock(spec=Portfolio)
        # Default state
        portfolio.get_state.return_value = PortfolioSchema(
            cash_balances=[
                CashBalance(currency="USD", total=10000.0, available=10000.0)
            ],
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
        assert rm.quantityr == fixed_sizer
        assert rm.max_allocation_per_market == 0.5
        assert rm.max_total_positions == 5

    def test_process_signal_hold(self, mock_portfolio, fixed_sizer, market_data_map):
        rm = RiskManager(portfolio=mock_portfolio, sizer=fixed_sizer)
        signal = TradeSignal(
            instrument_id="MARKET_01",
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
            amount_quote=0, quantity_shares=0
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
        low_balance_state = PortfolioSchema(
            cash_balances=[CashBalance(currency="USD", total=50.0, available=50.0)],
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
        assert order.instrument_id == "MARKET_01"
        assert order.side == OrderSide.BUY

        # Sizer is 100 quote currency. Price is 0.51 (best ask).
        # Size = 100 / 0.51 = 196.078...
        assert order.quantity == pytest.approx(100.0 / 0.51)
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
        assert order.quantity == pytest.approx(100.0 / 0.49)
        assert order.price == 0.49

    def test_process_signal_flat(self, mock_portfolio, fixed_sizer, market_data_map):
        rm = RiskManager(portfolio=mock_portfolio, sizer=fixed_sizer)

        # Setup a position to flatten (Long 100 shares)
        pos = Position(instrument_id="MARKET_01", quantity=100, cost_basis=0.4)
        mock_portfolio.get_state.return_value = PortfolioSchema(
            cash_balances=[
                CashBalance(currency="USD", total=10000.0, available=10000.0)
            ],
            available_balance_quote=9960.0,
            positions=[pos],
            open_orders=[],
        )

        signal = TradeSignal(
            instrument_id="MARKET_01",
            strategy_name="test",
            signal_type=SignalType.FLAT,
            confidence=1.0,
        )

        order = rm.process_signal(signal, market_data_map)

        assert order is not None
        assert order.side == OrderSide.SELL
        assert order.quantity == 100
        assert order.price == 0.49  # Best bid from market_data_map fixture

    # --- Direct tests for _passes_risk_checks ---

    @pytest.fixture
    def risk_check_deps(self, mock_market_data):
        """Dependencies for testing _passes_risk_checks directly."""
        rm = RiskManager(MagicMock(spec=Portfolio), MagicMock())  # Mock portfolio
        sizing_output = SizingOutput(amount_quote=1000, quantity_shares=2000)
        signal = TradeSignal(
            instrument_id="MKT1",
            strategy_name="test",
            signal_type=SignalType.BUY,
            confidence=0.7,
        )
        portfolio_state = PortfolioSchema(
            cash_balances=[
                CashBalance(currency="USD", total=10000.0, available=10000.0)
            ],
            positions=[],
            open_orders=[],
        )
        market_map = {"MKT1": mock_market_data}
        return rm, sizing_output, signal, portfolio_state, market_map

    def test_passes_risk_checks_insufficient_balance(self, risk_check_deps, caplog):
        rm, sizing_output, signal, portfolio_state, market_map = risk_check_deps

        portfolio_state.cash_balances[0].available = 500  # Less than 1000

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
                instrument_id="OTHER_MKT",
                quantity=10,
                cost_basis=0.1,
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
            Position(instrument_id="MKT1", quantity=10, cost_basis=0.1)
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


# --- PR 7b: one equity number -------------------------------------------------


def _with_bar(market_data, close):
    from trading_bot.core.schemas import BarData

    bar = BarData(
        timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
        bar_type="time",
        ticks_count=1,
        dollar_volume=close,
    )
    return market_data.model_copy(update={"recent_bars": [bar]})


class TestSingleValuation:
    def test_equity_uses_the_last_close_not_the_bid(
        self, base_portfolio: Portfolio, mock_market_data
    ):
        base_portfolio._positions["MARKET_01"] = Position(
            instrument_id="MARKET_01", quantity=100, cost_basis=0.40
        )
        data = {"MARKET_01": _with_bar(mock_market_data, 0.50)}

        assert base_portfolio.equity(data) == pytest.approx(10000 + 100 * 0.50)
        assert base_portfolio.get_state(data).cash_balances[0].total == pytest.approx(
            10050.0
        )
        # the allocation-limit check still prices the position at the bid
        assert base_portfolio.calculate_unrealized_pnl(data)[
            "MARKET_01"
        ] == pytest.approx(100 * (0.49 - 0.40))

    def test_a_tick_without_a_bar_keeps_the_last_close(
        self, base_portfolio: Portfolio, mock_market_data
    ):
        base_portfolio._positions["MARKET_01"] = Position(
            instrument_id="MARKET_01", quantity=100, cost_basis=0.40
        )
        base_portfolio.equity({"MARKET_01": _with_bar(mock_market_data, 0.50)})

        # no bar, and an empty map: still 0.50, never 0
        assert base_portfolio.equity({"MARKET_01": mock_market_data}) == 10050.0
        assert base_portfolio.equity({}) == 10050.0
        assert base_portfolio.last_close("MARKET_01") == 0.50

    def test_limit_check_without_a_book_uses_the_last_close(
        self, base_portfolio: Portfolio, mock_market_data
    ):
        base_portfolio._positions["MARKET_01"] = Position(
            instrument_id="MARKET_01", quantity=100, cost_basis=0.40
        )
        no_book = _with_bar(mock_market_data, 0.45).model_copy(
            update={"order_book": None}
        )
        pnl = base_portfolio.calculate_unrealized_pnl({"MARKET_01": no_book})
        assert pnl["MARKET_01"] == pytest.approx(100 * (0.45 - 0.40))  # not 0

    def test_before_any_close_a_position_is_at_cost_basis(
        self, base_portfolio: Portfolio
    ):
        base_portfolio._positions["MARKET_01"] = Position(
            instrument_id="MARKET_01", quantity=100, cost_basis=0.40
        )
        assert base_portfolio.equity({}) == pytest.approx(10040.0)

    def test_outcome_positions_are_at_cost_basis(
        self, base_portfolio: Portfolio, mock_market_data
    ):
        """Known limit (backlog PR 7b): the bars are the instrument's, not the
        YES/NO side's, so an outcome position is not marked to them."""
        base_portfolio._positions["MARKET_01:YES"] = Position(
            instrument_id="MARKET_01", outcome="YES", quantity=100, cost_basis=0.40
        )
        data = {"MARKET_01": _with_bar(mock_market_data, 0.90)}
        assert base_portfolio.equity(data) == pytest.approx(10040.0)


class TestSizersUseEquity:
    """Both sizers read an attribute Portfolio never had (total_balance_usd)."""

    def _input(self, market_data, signal, total):
        from trading_bot.core.schemas import SizingInput

        state = PortfolioSchema(
            cash_balances=[CashBalance(currency="USD", total=total, available=total)],
            positions=[],
        )
        return SizingInput(
            signal=signal, market_data=market_data, portfolio_state=state
        )

    def test_fixed_percentage(self, mock_market_data, mock_buy_signal):
        from trading_bot.risk_management.sizing.fixed_percentage import (
            FixedPercentageSizer,
        )

        out = FixedPercentageSizer(0.02).calculate_size(
            self._input(mock_market_data, mock_buy_signal, 10000.0)
        )
        assert out.amount_quote == pytest.approx(200.0)
        assert out.quantity_shares == pytest.approx(200.0 / 0.51)

    def test_kelly(self, mock_market_data, mock_buy_signal):
        from trading_bot.risk_management.sizing.kelly_criterion import (
            KellyCriterionSizer,
        )

        b = 0.49 / 0.51
        f = (0.6 * b - 0.4) / b
        out = KellyCriterionSizer(0.5).calculate_size(
            self._input(mock_market_data, mock_buy_signal, 10000.0)
        )
        assert out.amount_quote == pytest.approx(10000.0 * f * 0.5)
