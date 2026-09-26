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
from quant_core.models import (
    BarData,
    CashBalance,
    ExecutionResult,
    Instrument,
    MarketData,
    OrderBook,
    OrderRequest,
    Portfolio,
    Position,
    PriceLevel,
    Trade,
)

# Refactored: MarketOutcome removed
from .enums import AlertSeverity, BarType, OrderSide, OrderStatus, OrderType, SignalType


def utc_now() -> datetime:
    """Returns the current UTC datetime."""
    return datetime.now(timezone.utc)


# --- Module 1: Data Ingestion Engine Schemas ---


# Models moved to quant_core
MarketDetails = Instrument

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
    market_data: List[MarketData] = Field(
        ...,
        description="A list containing the latest MarketData for all requested subscriptions.",
    )
    external_data: List[ExternalData] = Field(
        ...,
        description="A list of all new external data points fetched in this cycle.",
    )
    bars: Dict[str, List[BarData]] = Field(
        default_factory=dict,
        description="A dictionary mapping instrument_id to its latest aggregated bars.",
    )


class TradeSignal(BaseModel):
    """
    The primary output of the Strategy Engine. This represents a
    *recommendation* to trade, which is then passed to the
    Risk Manager for sizing and approval.
    """

    instrument_id: str = Field(
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


class SizingInput(BaseModel):
    """
    The data packet required by a BaseSizingStrategy to calculate an order size.
    It contains the signal, the current market state, and the portfolio state.
    """

    signal: TradeSignal = Field(..., description="The signal from the Strategy Engine.")
    market_data: MarketData = Field(
        ..., description="The current market data for the signaled market."
    )
    portfolio_state: Portfolio = Field(
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
    quantity_shares: float = Field(
        ..., description="The number of shares to trade. 0 means no trade.", ge=0
    )


# --- Module 4: Execution Engine Schemas ---


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
    instrument_id: str = Field(..., description="Market identifier.")
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
    instrument_id: str = Field(..., description="Market identifier.")
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
    instrument_id: str = Field(
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
    instrument_id: str = Field("all", description="Market identifier.")
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
