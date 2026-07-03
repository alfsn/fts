# src/trading_bot/core/spec_models.py

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class MarketSpec(BaseModel):
    """Specification model for target market pair and aggregation timeframe."""

    market_id: str = Field(default="BTC/USDT", description="Target market pair")
    interval: str = Field(default="30m", description="Bar aggregation timeframe")


class DateRangeSpec(BaseModel):
    """Specification model for historical data training or backtesting date range boundaries."""

    start_date: Optional[datetime] = Field(
        default=None, description="Start date timestamp for data range"
    )
    end_date: Optional[datetime] = Field(
        default=None, description="End date timestamp for data range"
    )


class FeatureSetSpec(BaseModel):
    """Specification model for lookback window, raw attributes, and feature pipeline configuration."""

    lookback_period: int = Field(
        default=20, ge=1, description="Lookback window size in bars"
    )
    feature_cols: List[str] = Field(
        default_factory=lambda: ["close"],
        description="Raw OHLCV attributes used for feature engineering",
    )
    feature_pipeline: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Serialized BaseTransform/FeaturePipeline dict specification",
    )


class ExecutionRiskSpec(BaseModel):
    """Specification model for order execution, latency delay, slippage, and portfolio sizing."""

    execution_delay_k: int = Field(
        default=1, ge=0, description="Bar execution delay k for orders"
    )
    slippage_pct: float = Field(
        default=0.001, ge=0.0, description="Flat price slippage percentage per fill"
    )
    initial_balance: float = Field(
        default=10000.0, gt=0.0, description="Starting cash balance in quote currency"
    )
    quote_currency: str = Field(
        default="USD", description="Base portfolio currency denomination"
    )
    position_size_pct: float = Field(
        default=0.10, gt=0.0, le=1.0, description="Position sizing fraction of equity"
    )


class ClassifierSpec(BaseModel):
    """Specification model for dynamic threshold classifier strategy parameters."""

    classifier_k: float = Field(
        default=0.03, gt=0.0, description="Dynamic threshold sensitivity factor"
    )
    confidence_multiplier: float = Field(
        default=20.0, gt=0.0, description="Confidence scaling multiplier"
    )
    period: int = Field(
        default=10, ge=1, description="Rolling window period for classifier volatility"
    )
    allow_in_sample: bool = Field(
        default=False, description="Whether to allow in-sample backtest timestamps"
    )


class OptunaStudySpec(BaseModel):
    """Specification model for Optuna hyperparameter study parameters."""

    study_name: str = Field(
        default="optuna_study", description="Unique name identifier for Optuna study"
    )
    direction: Literal["minimize", "maximize"] = Field(
        default="minimize",
        description="Optimization direction (minimize validation loss)",
    )
    n_trials: int = Field(
        default=10, ge=1, description="Number of Optuna search trials"
    )
    model_type: str = Field(
        default="lstm", description="Target model architecture name"
    )
