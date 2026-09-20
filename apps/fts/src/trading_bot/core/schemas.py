# src/trading_bot/core/schemas.py

"""
Pydantic Schemas for Core Data Contracts (Data Contracts).

This file defines the core data structures used throughout the application,
acting as the "contracts" between different modules. Using Pydantic
ensures that all data is validated, typed, and well-documented.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# Refactored: MarketOutcome removed
from .enums import AlertSeverity, BarType, OrderSide, OrderStatus, OrderType, SignalType


def utc_now() -> datetime:
    """Returns the current UTC datetime."""
    return datetime.now(timezone.utc)


# --- Module 1: Data Ingestion Engine Schemas ---


class PriceLevel(BaseModel):
    """
    Represents a single price level in an order book (either a bid or an ask).
    """

    price: float = Field(
        ..., description="The price of the orders at this level.", gt=0
    )
    size: float = Field(
        ...,
        description="The total volume (number of shares) of orders at price level.",
        ge=0,
    )


class OrderBook(BaseModel):
    """
    Represents the current order book for a specific market,
    containing lists of buy (bids) and sell (asks) levels.
    """

    bids: List[PriceLevel] = Field(
        ...,
        description="""A list of price levels for buy orders (bids),
        typically sorted highest to lowest.""",
    )
    asks: List[PriceLevel] = Field(
        ...,
        description="""A list of price levels for sell orders (asks),
                        typically sorted lowest to highest.""",
    )


class Trade(BaseModel):
    """
    Represents a single executed trade that occurred in a market.
    """

    price: float = Field(
        ..., description="The price at which the trade was executed.", gt=0
    )
    size: float = Field(
        ..., description="The volume (number of shares) of the trade.", gt=0
    )
    timestamp: datetime = Field(
        ..., description="The timestamp when the trade was executed."
    )
    side: OrderSide = Field(
        ...,
        description="The side of the trade (buy or sell) from the taker's perspective.",
    )
    outcome: Optional[str] = Field(
        None,
        description="""
        The specific outcome traded,
        if this is a prediction market (e.g. 'yes' or 'no').
        """,
    )


class MarketDetails(BaseModel):
    """
    Contains static, descriptive information about a market (e.g., its
    question, resolution criteria, and end date).
    """

    market_id: str = Field(..., description="The unique identifier for the market.")
    name: str = Field(
        ...,
        description="""The human-readable name or question of the market
        (e.g., 'Will X happen by Y?' or 'AAPL Stock').""",
    )
    end_date: datetime = Field(
        ..., description="The timestamp when the market is scheduled to resolve."
    )
    resolution_source: str = Field(
        ...,
        description="""The official source that will be used to
        determine the market's outcome.""",
    )


class BarData(BaseModel):
    """
    Represents a single aggregated data bar (OHLCV).
    """

    timestamp: datetime = Field(..., description="The end time of the bar.")
    open: float = Field(..., description="The opening price.", gt=0)
    high: float = Field(..., description="The highest price during the interval.", gt=0)
    low: float = Field(..., description="The lowest price during the interval.", gt=0)
    close: float = Field(..., description="The closing price.", gt=0)
    volume: float = Field(..., description="The total volume traded.", ge=0)
    bar_type: BarType = Field(
        ..., description="The type of bar (Time, Volume, Dollar)."
    )
    interval: Optional[str] = Field(
        None, description="The timeframe/interval (e.g. 1m, 5m, 1h, 1d)."
    )
    ticks_count: int = Field(..., description="Number of ticks in this bar.", ge=1)
    dollar_volume: float = Field(
        ..., description="Total currency units traded in this bar.", ge=0
    )


class MarketData(BaseModel):
    """
    A composite snapshot of a single market's current state. This is the
    primary object produced by a BaseMarketDataProvider.
    """

    market_id: str = Field(
        ..., description="The unique identifier for the market this data pertains to."
    )
    order_book: Optional[OrderBook] = Field(
        None, description="The current order book state, if available."
    )
    recent_trades: Optional[List[Trade]] = Field(
        None,
        description="A list of recently executed trades for this market, if available.",
    )
    details: MarketDetails = Field(..., description="The static details of the market.")
    recent_bars: List[BarData] = Field(
        default_factory=list,
        description="A list of recent aggregated bars for this market.",
    )


# --- Module 2: Strategy Engine Schemas ---


class ExternalData(BaseModel):
    """
    Represents a data point from an external (non-exchange) source,
    such as a news API, social media sentiment, or other predictive data.
    """

    source: str = Field(
        ...,
        description="""A unique name for the data source
        (e.g., 'twitter_sentiment', 'weather_api').""",
    )
    timestamp: datetime = Field(
        ..., description="The time the data was fetched or generated."
    )
    content: Dict = Field(
        ..., description="The actual data content, as a flexible dictionary."
    )


class IngestionEngineOutput(BaseModel):
    """
    A container for all data gathered by the Ingestion Engine during one
    'tick' or update cycle. This is the primary input for the Strategy Engine.
    """

    timestamp: datetime = Field(
        ..., description="The time this data packet was generated."
    )
    market_data: Dict[str, MarketData] = Field(
        ..., description="A dictionary mapping market_id to its latest MarketData."
    )
    external_data: List[ExternalData] = Field(
        ...,
        description="A list of all new external data points fetched in this cycle.",
    )
    bars: Dict[str, List[BarData]] = Field(
        default_factory=dict,
        description="A dictionary mapping market_id to its latest aggregated bars.",
    )


class TradeSignal(BaseModel):
    """
    The primary output of the Strategy Engine. This represents a
    *recommendation* to trade, which is then passed to the
    Risk Manager for sizing and approval.
    """

    market_id: str = Field(
        ..., description="The unique identifier of the market to trade in."
    )
    strategy_name: str = Field(
        ..., description="The name of the strategy that generated this signal."
    )
    signal_type: SignalType = Field(
        ..., description="The type of signal (BUY, SELL, or HOLD)."
    )
    outcome: Optional[str] = Field(
        None,
        description="""
        The specific outcome to trade,
        if this is a prediction market (e.g. 'yes' or 'no').
        """,
    )
    confidence: float = Field(
        ...,
        description="""The strategy's confidence in this signal,
        typically scaled from 0.0 to 1.0.""",
        ge=0.0,
        le=1.0,
    )
    timestamp: datetime = Field(
        default_factory=utc_now,
        description="The timestamp when this signal was generated.",
    )
    prediction_output: Optional[str] = Field(
        None,
        description="JSON-serialized raw prediction output or probabilities from the model.",
    )


# --- Module 3: Risk & Position Management Schemas ---


class Position(BaseModel):
    """
    Represents a single held position (an asset) in the portfolio.
    """

    market_id: str = Field(..., description="The market this position is in.")
    outcome: Optional[str] = Field(
        None,
        description="""
        The prediction market outcome, if applicable (e.g. 'yes' or 'no').
        """,
    )
    size: float = Field(
        ...,
        description="The number of shares held. Positive for long, negative for short.",
    )
    entry_price: float = Field(
        ..., description="The average price at which the position was entered.", ge=0
    )
    run_id: Optional[str] = Field(
        None, description="The backtest or session run ID associated with the position."
    )


class PortfolioState(BaseModel):
    """
    A snapshot of the portfolio's current state, used for risk calculations.
    """

    total_balance_quote: float = Field(
        ..., description="Total account value in the quote currency (e.g., USD)."
    )
    available_balance_quote: float = Field(
        ..., description="Quote currency not tied up in orders or positions."
    )

    positions: List[Position] = Field(
        ..., description="List of all currently held positions."
    )
    open_orders: List["OrderRequest"] = Field(
        ..., description="List of all orders active on the exchange."
    )


class SizingInput(BaseModel):
    """
    The data packet required by a BaseSizingStrategy to calculate an order size.
    It contains the signal, the current market state, and the portfolio state.
    """

    signal: TradeSignal = Field(..., description="The signal from the Strategy Engine.")
    market_data: MarketData = Field(
        ..., description="The current market data for the signaled market."
    )
    portfolio_state: PortfolioState = Field(
        ..., description="The current state of the portfolio."
    )


class SizingOutput(BaseModel):
    """
    The output of a BaseSizingStrategy, specifying the exact size of the
    order to be placed.
    """

    amount_quote: float = Field(
        ...,
        description="The amount of quote currency to allocate. 0 means no trade.",
        ge=0,
    )
    size_shares: float = Field(
        ..., description="The number of shares to trade. 0 means no trade.", ge=0
    )


# --- Module 4: Execution Engine Schemas ---


class OrderRequest(BaseModel):
    """
    The primary output of the Risk Manager. This is a concrete,
    sized, and risk-checked order that is sent to the
    Execution Engine to be placed on the exchange.
    """

    market_id: str = Field(..., description="The market to place the order in.")
    side: OrderSide = Field(..., description="The side of the order (BUY or SELL).")
    outcome: Optional[str] = Field(
        None,
        description="""
        The specific outcome to trade,
        if this is a prediction market (e.g. 'yes' or 'no').
        """,
    )
    size: float = Field(..., description="The exact number of shares to trade.", gt=0)
    price: float = Field(
        ...,
        description="""The limit price for the order.
        The order should not be filled at a worse price.""",
        gt=0,
    )
    order_type: OrderType = Field(
        OrderType.LIMIT,
        description="The type of order (LIMIT, MARKET, STOP).",
    )


class ExecutionResult(BaseModel):
    """
    The primary output of the Execution Engine. This object provides
    feedback on the status of a placed (or failed) order.
    """

    order_id: str = Field(
        ...,
        description="""The unique identifier for the order
        (can be assigned by us or the exchange).""",
    )
    status: OrderStatus = Field(
        ..., description="The current status of the order (e.g., OPEN, FILLED, FAILED)."
    )
    filled_size: float = Field(
        ...,
        description="The total number of shares that have been filled for this order.",
        ge=0,
    )
    avg_price: float = Field(
        ..., description="The average price at which the shares were filled.", ge=0
    )
    timestamp: datetime = Field(
        ..., description="The timestamp of this execution status update."
    )
    order_type: OrderType = Field(
        OrderType.LIMIT,
        description="The type of order (LIMIT, MARKET, STOP).",
    )


# --- Module 5: Monitoring Schemas (Add to schemas.py) ---


class Alert(BaseModel):
    """
    A data contract for sending an alert to the monitoring system
    (e.g., to Telegram or a dashboard).
    """

    message: str = Field(..., description="The content of the alert message.")
    severity: AlertSeverity = Field(..., description="The severity level of the alert.")
    timestamp: datetime = Field(
        default_factory=utc_now,
        description="The time the alert was generated.",
    )


# --- Module 6: Data Catalog & Backtest Visibility Schemas ---


class ModelCatalogItem(BaseModel):
    """
    Lightweight summary contract for model registry listing in data catalog.
    """

    model_id: str = Field(..., description="Unique model identifier.")
    run_id: Optional[str] = Field(None, description="Linked training run ID.")
    model_type: str = Field(..., description="Algorithm/architecture type.")
    market_id: str = Field(..., description="Market identifier.")
    interval: str = Field(..., description="Bar time resolution.")
    horizon: int = Field(..., description="Prediction horizon step size.")
    dataset_id: Optional[str] = Field(None, description="Source dataset ID.")
    status: str = Field(
        "candidate", description="Lifecycle status (candidate, production, archived)."
    )
    onnx_path: str = Field(..., description="Path to ONNX serialized artifact.")
    hyperparameters: Dict = Field(
        default_factory=dict, description="Training hyperparameters."
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict, description="Evaluation metric summary."
    )
    created_at: Optional[datetime] = Field(None, description="Creation timestamp.")


class ModelDetailDTO(BaseModel):
    """
    Detailed contract for single model deep-dive and inspection.
    """

    model_id: str = Field(..., description="Unique model identifier.")
    run_id: Optional[str] = Field(None, description="Linked training run ID.")
    model_type: str = Field(..., description="Algorithm/architecture type.")
    market_id: str = Field(..., description="Market identifier.")
    interval: str = Field(..., description="Bar time resolution.")
    horizon: int = Field(..., description="Prediction horizon step size.")
    dataset_id: Optional[str] = Field(None, description="Source dataset ID.")
    status: str = Field("candidate", description="Lifecycle status.")
    onnx_path: str = Field(..., description="Path to ONNX file.")
    hyperparameters: Dict = Field(
        default_factory=dict, description="Training hyperparameters."
    )
    metrics: Dict = Field(
        default_factory=dict, description="Comprehensive evaluation metrics."
    )
    created_at: Optional[datetime] = Field(None, description="Creation timestamp.")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp.")
    onnx_exists: bool = Field(
        False, description="Whether ONNX artifact file exists on disk."
    )


class BacktestRunCatalogItem(BaseModel):
    """
    Lightweight summary contract for backtest run catalog listing.
    """

    run_id: str = Field(..., description="Unique backtest simulation run ID.")
    strategy_name: str = Field("unknown", description="Strategy algorithm name.")
    market_id: str = Field(
        "all", description="Market identifier or multi-market scope."
    )
    model_id: Optional[str] = Field(None, description="Linked model ID.")
    hyperparameters: Dict = Field(
        default_factory=dict, description="Linked model hyperparameters."
    )
    start_time: Optional[datetime] = Field(
        None, description="Simulation start timestamp."
    )
    end_time: Optional[datetime] = Field(None, description="Simulation end timestamp.")
    total_return: float = Field(0.0, description="Total cumulative percentage return.")
    sharpe_ratio: float = Field(0.0, description="Annualized Sharpe ratio.")
    max_drawdown: float = Field(0.0, description="Maximum peak-to-trough drawdown.")
    win_rate: float = Field(0.0, description="Percentage of profitable trades.")
    total_trades: int = Field(0, description="Total filled trade executions count.")


class BacktestDetailDTO(BaseModel):
    """
    Detailed contract for backtest run analysis including time-series equity and trades.
    """

    run_id: str = Field(..., description="Unique backtest simulation run ID.")
    strategy_name: str = Field("unknown", description="Strategy algorithm name.")
    market_id: str = Field("all", description="Market identifier.")
    start_time: Optional[datetime] = Field(
        None, description="Simulation start timestamp."
    )
    end_time: Optional[datetime] = Field(None, description="Simulation end timestamp.")
    total_return: float = Field(0.0, description="Total percentage return.")
    sharpe_ratio: float = Field(0.0, description="Annualized Sharpe ratio.")
    max_drawdown: float = Field(0.0, description="Maximum peak-to-trough drawdown.")
    win_rate: float = Field(0.0, description="Percentage of winning trades.")
    total_trades: int = Field(0, description="Total filled trade count.")
    equity_curve: List[Dict] = Field(
        default_factory=list, description="Time-series tick equity logs."
    )
    trades: List[Dict] = Field(
        default_factory=list, description="Filled trade execution records."
    )
    predictions: List[Dict] = Field(
        default_factory=list, description="ML prediction decision log entries."
    )
