# src/trading_bot/backtesting/spec.py

from typing import Optional

from pydantic import Field

from trading_bot.core.spec_base import BaseComposableSpec
from trading_bot.core.spec_models import (
    ClassifierSpec,
    DateRangeSpec,
    ExecutionRiskSpec,
    MarketSpec,
)


class BacktestSpec(BaseComposableSpec):
    """
    Canonical specification model for historical backtest simulations.
    Composes market criteria, replay window boundaries, execution models,
    risk management settings, and classifier parameters.
    """

    market: MarketSpec = Field(default_factory=MarketSpec)
    dates: DateRangeSpec = Field(default_factory=DateRangeSpec)
    execution: ExecutionRiskSpec = Field(default_factory=ExecutionRiskSpec)
    classifier: ClassifierSpec = Field(default_factory=ClassifierSpec)

    # Model Selection & Simulation Specifics
    model_type: str = Field(default="lstm", description="ML model architecture name")
    horizon: int = Field(default=1, ge=1, description="Model forecasting horizon")
    warmup_bars: int = Field(
        default=100, ge=0, description="Warmup bars for strategy indicators"
    )
    lookback_limit: int = Field(
        default=1000, ge=10, description="Maximum historical bars retained in memory"
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for database logs",
    )

    # Output & Reporting
    output_dir: str = Field(
        default="../runs/reports",
        description="Path to save backtest HTML/JSON summaries",
    )
