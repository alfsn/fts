# src/trading_bot/core/schemas.py

"""
Pydantic Schemas for Core Data Contracts (Data Contracts).

This file defines the core data structures used throughout the application,
acting as the "contracts" between different modules. Using Pydantic
ensures that all data is validated, typed, and well-documented.
"""

from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel, Field

from .enums import AlertSeverity, MarketOutcome, OrderSide, OrderStatus, SignalType
from .schemas import OrderRequest

# --- Module 1: Data Ingestion Engine Schemas ---


class PriceLevel(BaseModel):
    """
    Represents a single price level in an order book (either a bid or an ask).
    """

    price: float = Field(
        ..., description="The price of the orders at this level.", gt=0
    )  # Price must be positive
    size: float = Field(
        ...,
        description="The total volume (number of shares) of orders at this price level",
        ge=0,  # Size can be zero if a level is cleared
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


class MarketDetails(BaseModel):
    """
    Contains static, descriptive information about a market (e.g., its
    question, resolution criteria, and end date).
    """

    market_id: str = Field(..., description="The unique identifier for the market.")
    name: str = Field(
        ...,
        description="""The human-readable name or question of the market
        (e.g., 'Will X happen by Y?').""",
    )
    end_date: datetime = Field(
        ..., description="The timestamp when the market is scheduled to resolve."
    )
    resolution_source: str = Field(
        ...,
        description="The official source used to determine the market's outcome.",
    )


class MarketData(BaseModel):
    """
    A composite snapshot of a single market's current state. This is the
    primary object produced by a BaseMarketDataProvider.
    """

    market_id: str = Field(
        ..., description="The unique identifier for the market this data pertains to."
    )
    order_book: OrderBook = Field(..., description="The current order book state.")
    recent_trades: List[Trade] = Field(
        ..., description="A list of recently executed trades for this market."
    )
    details: MarketDetails = Field(..., description="The static details of the market.")


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
        ..., description="A list of all new external data points fetched in this cycle."
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
    outcome: MarketOutcome = Field(
        ..., description="The specific outcome to trade (e.g., YES or NO)."
    )
    confidence: float = Field(
        ...,
        description="""The strategy's confidence in this signal,
        typically scaled from 0.0 to 1.0.""",
        ge=0.0,
        le=1.0,
    )


# --- Module 3: Risk & Position Management Schemas ---


class Position(BaseModel):
    """
    Represents a single held position (an asset) in the portfolio.
    """

    market_id: str = Field(..., description="The market this position is in.")
    outcome: MarketOutcome = Field(..., description="The outcome held (YES or NO).")
    size: float = Field(
        ...,
        description="""The number of shares held.
        Can be positive (long) or negative (short).""",
    )
    entry_price: float = Field(
        ..., description="The average price at which the position was entered.", ge=0
    )


class PortfolioState(BaseModel):
    """
    A snapshot of the portfolio's current state, used for risk calculations.
    """

    total_balance_usdc: float = Field(..., description="Total account value in USDC.")
    available_balance_usdc: float = Field(
        ..., description="USDC not tied up in orders or positions."
    )
    positions: List[Position] = Field(
        ..., description="List of all currently held positions."
    )
    open_orders: List[OrderRequest] = Field(
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

    amount_usdc: float = Field(
        ..., description="The amount of USDC to allocate. 0 means no trade.", ge=0
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
    outcome: MarketOutcome = Field(
        ..., description="The specific outcome to trade (YES or NO)."
    )
    size: float = Field(..., description="The exact number of shares to trade.", gt=0)
    price: float = Field(
        ...,
        description="The limit price for the order. "
        "The order should not be filled at a worse price.",
        gt=0,
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


# --- Module 5: Monitoring Schemas (Add to schemas.py) ---


class Alert(BaseModel):
    """
    A data contract for sending an alert to the monitoring system
    (e.g., to Telegram or a dashboard).
    """

    message: str = Field(..., description="The content of the alert message.")
    severity: AlertSeverity = Field(..., description="The severity level of the alert.")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="The time the alert was generated."
    )
