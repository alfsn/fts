from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from quant_core.enums import (
    AlertSeverity,
    AssetType,
    Geography,
    OrderSide,
    OrderStatus,
    SignalType,
)
from quant_core.models import CashBalance, Instrument, TradFiDetails
from trading_bot.core.schemas import (
    Alert,
    ExecutionResult,
    ExternalData,
    IngestionEngineOutput,
    Instrument,
    MarketData,
    OrderBook,
    OrderRequest,
    Portfolio,
    Position,
    PriceLevel,
    SizingInput,
    SizingOutput,
    Trade,
    TradeSignal,
)

# tests/unit/test_core_schemas.py


# --- Fixtures for Reusable Test Data ------------------------------------------


@pytest.fixture
def sample_datetime() -> datetime:
    """Provides a consistent datetime object for testing."""
    return datetime.now(timezone.utc)


@pytest.fixture
def valid_price_level() -> PriceLevel:
    """Provides a valid PriceLevel object."""
    return PriceLevel(price=0.55, quantity=100.0)


@pytest.fixture
def valid_order_book(valid_price_level: PriceLevel) -> OrderBook:
    """Provides a valid OrderBook object."""
    bids = [PriceLevel(price=0.49, quantity=50)]
    asks = [PriceLevel(price=0.51, quantity=70)]
    return OrderBook(bids=bids, asks=asks)


@pytest.fixture
def valid_trade(sample_datetime: datetime) -> Trade:
    """Provides a valid Trade object."""
    return Trade(
        price=0.50, quantity=10.0, timestamp=sample_datetime, side=OrderSide.BUY
    )


@pytest.fixture
def valid_market_details(sample_datetime: datetime) -> Instrument:
    """Provides a valid Instrument object."""
    return Instrument(
        instrument_id="market-123",
        name="Will test market pass?",
        details=TradFiDetails(
            asset_type=AssetType.STOCK,
            geography=Geography.US,
            industry="Tech",
            currency="USD",
        ),
    )


@pytest.fixture
def valid_market_data(
    valid_order_book: OrderBook,
    valid_trade: Trade,
    valid_market_details: Instrument,
) -> MarketData:
    """Provides a valid, composite MarketData object."""
    return MarketData(
        instrument_id="market-123",
        order_book=valid_order_book,
        recent_trades=[valid_trade],
        details=valid_market_details,
    )


@pytest.fixture
def valid_trade_signal() -> TradeSignal:
    """Provides a valid TradeSignal object."""
    return TradeSignal(
        instrument_id="market-123",
        strategy_name="test_strategy_v1",
        signal_type=SignalType.BUY,
        confidence=0.85,
    )


@pytest.fixture
def valid_position() -> Position:
    """Provides a valid Position object."""
    return Position(
        instrument_id="market-123",
        quantity=100.0,
        cost_basis=0.45,
    )


@pytest.fixture
def valid_order_request() -> OrderRequest:
    """Provides a valid OrderRequest object."""
    return OrderRequest(
        instrument_id="market-456",
        side=OrderSide.SELL,
        quantity=25.0,
        price=0.75,
    )


@pytest.fixture
def valid_portfolio_state(
    valid_position: Position, valid_order_request: OrderRequest
) -> Portfolio:
    """Provides a valid, composite Portfolio object."""
    return Portfolio(
        cash_balances=[CashBalance(currency="USD", total=10000.0, available=7500.0)],
        positions=[valid_position],
        open_orders=[valid_order_request],
    )


# --- Schema Test Functions ----------------------------------------------------


def test_price_level_valid():
    """Tests successful creation of a PriceLevel."""
    level = PriceLevel(price=0.5, quantity=100.0)
    assert level.price == 0.5
    assert level.quantity == 100.0

    # Size can be zero
    level_zero_size = PriceLevel(price=0.5, quantity=0.0)
    assert level_zero_size.quantity == 0.0


def test_price_level_invalid():
    """Tests validation errors for PriceLevel."""
    # Price must be positive (gt=0)
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        PriceLevel(price=0.0, quantity=100.0)
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        PriceLevel(price=-1.0, quantity=100.0)

    # Size must be non-negative (ge=0)
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        PriceLevel(price=0.5, quantity=-1.0)

    # Test missing required fields
    with pytest.raises(ValidationError, match="price"):
        PriceLevel(quantity=100.0)
    with pytest.raises(ValidationError, match="quantity"):
        PriceLevel(price=0.5)


def test_order_book_valid(valid_price_level: PriceLevel):
    """Tests successful creation of an OrderBook."""
    ob = OrderBook(bids=[valid_price_level], asks=[valid_price_level])
    assert len(ob.bids) == 1
    assert ob.bids[0] == valid_price_level

    # Empty lists are also valid
    ob_empty = OrderBook(bids=[], asks=[])
    assert len(ob_empty.bids) == 0


def test_order_book_invalid():
    """Tests validation errors for OrderBook."""
    # Test missing required fields
    with pytest.raises(ValidationError, match="bids"):
        OrderBook(asks=[])

    # Test incorrect inner type
    # vvv FIX: Match the actual Pydantic error message vvv
    with pytest.raises(ValidationError, match="float_parsing"):
        OrderBook(bids=[{"price": 0.5, "quantity": "not-a-float"}], asks=[])


def test_trade_valid(sample_datetime: datetime):
    """Tests successful creation of a Trade."""
    trade = Trade(
        price=0.55, quantity=10.0, timestamp=sample_datetime, side=OrderSide.SELL
    )
    assert trade.price == 0.55
    assert trade.side == OrderSide.SELL


def test_trade_invalid(sample_datetime: datetime):
    """Tests validation errors for Trade."""
    # Test invalid enum value for side
    with pytest.raises(ValidationError, match="side"):
        Trade(price=0.5, quantity=10, timestamp=sample_datetime, side="INVALID_SIDE")

    # Test price constraint (gt=0)
    with pytest.raises(ValidationError, match="price"):
        Trade(price=0, quantity=10, timestamp=sample_datetime, side=OrderSide.BUY)

    # Test size constraint (gt=0)
    with pytest.raises(ValidationError, match="quantity"):
        Trade(price=0.5, quantity=0, timestamp=sample_datetime, side=OrderSide.BUY)


def test_market_details_valid(sample_datetime: datetime):
    """Tests successful creation of Instrument."""
    details = Instrument(
        instrument_id="market-abc",
        name="Test Market",
        details=TradFiDetails(
            asset_type=AssetType.STOCK,
            geography=Geography.US,
            industry="Tech",
            currency="USD",
        ),
    )
    assert details.instrument_id == "market-abc"


def test_market_details_invalid(sample_datetime: datetime):
    """Tests validation errors for Instrument."""
    # Test missing required instrument_id
    with pytest.raises(ValidationError, match="instrument_id"):
        Instrument(
            name="Test",
            details=TradFiDetails(
                asset_type=AssetType.STOCK,
                geography=Geography.US,
                industry="Tech",
                currency="USD",
            ),
        )


def test_market_data_valid(
    valid_order_book: OrderBook,
    valid_trade: Trade,
    valid_market_details: Instrument,
):
    """Tests successful creation of a composite MarketData."""
    market_data = MarketData(
        instrument_id="market-123",
        order_book=valid_order_book,
        recent_trades=[valid_trade],
        details=valid_market_details,
    )
    assert market_data.instrument_id == "market-123"
    assert market_data.order_book == valid_order_book
    assert market_data.details.name == "Will test market pass?"


def test_external_data_valid(sample_datetime: datetime):
    """Tests successful creation of ExternalData."""
    ext_data = ExternalData(
        source="twitter_api",
        timestamp=sample_datetime,
        content={"sentiment": 0.9, "tweet_id": "12345"},
    )
    assert ext_data.source == "twitter_api"
    assert ext_data.content["sentiment"] == 0.9


def test_external_data_invalid(sample_datetime: datetime):
    """Tests validation errors for ExternalData."""
    # Test missing required source
    with pytest.raises(ValidationError, match="source"):
        ExternalData(timestamp=sample_datetime, content={})


def test_ingestion_engine_output_valid(
    sample_datetime: datetime, valid_market_data: MarketData
):
    """Tests successful creation of IngestionEngineOutput."""
    ext_data = ExternalData(
        source="test", timestamp=sample_datetime, content={"value": 1}
    )
    output = IngestionEngineOutput(
        timestamp=sample_datetime,
        market_data={"market-123": valid_market_data},
        external_data=[ext_data],
    )
    assert output.market_data["market-123"] == valid_market_data
    assert output.external_data[0] == ext_data


def test_trade_signal_valid():
    """Tests successful creation of a TradeSignal."""
    signal = TradeSignal(
        instrument_id="market-123",
        strategy_name="test_strat",
        signal_type=SignalType.BUY,
        confidence=1.0,
    )
    assert signal.confidence == 1.0
    assert signal.signal_type == SignalType.BUY

    signal_hold = TradeSignal(
        instrument_id="market-123",
        strategy_name="test_strat",
        signal_type=SignalType.HOLD,
        confidence=0.0,
    )
    assert signal_hold.confidence == 0.0
    assert signal_hold.signal_type == SignalType.HOLD


def test_trade_signal_invalid():
    """Tests validation errors for TradeSignal."""
    # Test confidence constraints (ge=0.0, le=1.0)
    with pytest.raises(
        ValidationError, match="Input should be less than or equal to 1"
    ):
        TradeSignal(
            instrument_id="m1",
            strategy_name="s1",
            signal_type=SignalType.BUY,
            confidence=1.01,
        )
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        TradeSignal(
            instrument_id="m1",
            strategy_name="s1",
            signal_type=SignalType.SELL,
            confidence=-0.1,
        )

    # Test invalid enum
    with pytest.raises(ValidationError, match="signal_type"):
        TradeSignal(
            instrument_id="m1",
            strategy_name="s1",
            signal_type="INVALID",
            confidence=0.5,
        )


def test_position_valid():
    """Tests successful creation of a Position."""
    pos = Position(
        instrument_id="market-123",
        quantity=50.0,
        cost_basis=0.25,
    )
    assert pos.quantity == 50.0


def test_position_invalid():
    """Tests validation errors for Position."""
    # Test entry_price constraint (ge=0)
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        Position(instrument_id="m1", quantity=10, cost_basis=-0.01)


def test_portfolio_state_valid(
    valid_position: Position, valid_order_request: OrderRequest
):
    """Tests successful creation of a composite Portfolio."""
    state = Portfolio(
        cash_balances=[CashBalance(currency="USD", total=1000.0, available=500.0)],
        positions=[valid_position],
        open_orders=[valid_order_request],
    )
    assert state.cash_balances[0].available == 500.0
    assert len(state.positions) == 1
    assert state.positions[0] == valid_position
    assert len(state.open_orders) == 1
    assert state.open_orders[0] == valid_order_request


def test_sizing_input_valid(
    valid_trade_signal: TradeSignal,
    valid_market_data: MarketData,
    valid_portfolio_state: Portfolio,
):
    """Tests successful creation of a composite SizingInput."""
    sizing_input = SizingInput(
        signal=valid_trade_signal,
        market_data=valid_market_data,
        portfolio_state=valid_portfolio_state,
    )
    assert sizing_input.signal == valid_trade_signal
    assert sizing_input.market_data == valid_market_data
    assert sizing_input.portfolio_state == valid_portfolio_state


def test_sizing_output_valid():
    """Tests successful creation of a SizingOutput."""
    output = SizingOutput(amount_quote=100.0, quantity_shares=150.0)
    assert output.amount_quote == 100.0
    assert output.quantity_shares == 150.0

    # A "no trade" output is also valid
    output_zero = SizingOutput(amount_quote=0.0, quantity_shares=0.0)
    assert output_zero.amount_quote == 0.0


def test_sizing_output_invalid():
    """Tests validation errors for SizingOutput."""
    # Test amount_quote constraint (ge=0)
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        SizingOutput(amount_quote=-1.0, quantity_shares=100.0)

    # Test size_shares constraint (ge=0)
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        SizingOutput(amount_quote=100.0, quantity_shares=-1.0)


def test_order_request_valid():
    """Tests successful creation of an OrderRequest."""
    req = OrderRequest(
        instrument_id="market-123",
        side=OrderSide.BUY,
        quantity=100.0,
        price=0.65,
    )
    assert req.quantity == 100.0
    assert req.price == 0.65


def test_order_request_invalid():
    """Tests validation errors for OrderRequest."""
    # Test size constraint (gt=0)
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        OrderRequest(
            instrument_id="m1",
            side=OrderSide.BUY,
            quantity=0,
            price=0.5,
        )

    # Test price constraint (gt=0)
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        OrderRequest(
            instrument_id="m1",
            side=OrderSide.BUY,
            quantity=10,
            price=0,
        )


def test_execution_result_valid(sample_datetime: datetime):
    """Tests successful creation of an ExecutionResult."""
    result = ExecutionResult(
        order_id="order-xyz-123",
        status=OrderStatus.FILLED,
        filled_quantity=100.0,
        avg_price=0.65,
        timestamp=sample_datetime,
    )
    assert result.status == OrderStatus.FILLED
    assert result.avg_price == 0.65

    # Test a partially filled order
    result_partial = ExecutionResult(
        order_id="order-xyz-456",
        status=OrderStatus.PARTIALLY_FILLED,
        filled_quantity=50.0,
        avg_price=0.64,
        timestamp=sample_datetime,
    )
    assert result_partial.status == OrderStatus.PARTIALLY_FILLED
    assert result_partial.filled_quantity == 50.0

    # Test a failed order
    result_failed = ExecutionResult(
        order_id="order-xyz-789",
        status=OrderStatus.FAILED,
        filled_quantity=0.0,
        avg_price=0.0,
        timestamp=sample_datetime,
    )
    assert result_failed.status == OrderStatus.FAILED
    assert result_failed.filled_quantity == 0.0
    assert result_failed.avg_price == 0.0


def test_execution_result_invalid(sample_datetime: datetime):
    """Tests validation errors for ExecutionResult."""
    # Test invalid enum for status
    with pytest.raises(ValidationError, match="status"):
        ExecutionResult(
            order_id="o1",
            status="COMPLETED",
            filled_quantity=10,
            avg_price=0.5,
            timestamp=sample_datetime,
        )

    # Test filled_size constraint (ge=0)
    with pytest.raises(ValidationError, match="filled_quantity"):
        ExecutionResult(
            order_id="o1",
            status=OrderStatus.FILLED,
            filled_quantity=-1.0,
            avg_price=0.5,
            timestamp=sample_datetime,
        )

    # Test avg_price constraint (ge=0)
    with pytest.raises(ValidationError, match="avg_price"):
        ExecutionResult(
            order_id="o1",
            status=OrderStatus.FILLED,
            filled_quantity=10.0,
            avg_price=-0.1,
            timestamp=sample_datetime,
        )


def test_alert_valid():
    """Tests successful creation of an Alert."""
    alert = Alert(message="This is a test alert.", severity=AlertSeverity.WARNING)
    assert alert.message == "This is a test alert."
    assert alert.severity == AlertSeverity.WARNING

    # Test that the timestamp is auto-generated
    assert isinstance(alert.timestamp, datetime)


def test_alert_invalid():
    """Tests validation errors for Alert."""
    # Test invalid enum for severity
    with pytest.raises(ValidationError, match="severity"):
        Alert(message="Test", severity="HIGH")

    # Test missing required message
    with pytest.raises(ValidationError, match="message"):
        Alert(severity=AlertSeverity.INFO)
