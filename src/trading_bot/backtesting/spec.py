# src/trading_bot/backtesting/spec.py

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, Field


class BacktestSpec(BaseModel):
    """
    Canonical specification model for historical backtest simulations.
    Defines model selection criteria, replay window boundaries, execution models,
    risk management settings, and classifier parameters.
    """

    # Model Selection Criteria
    market_id: str = Field(default="BTC/USDT", description="Target market pair")
    model_type: str = Field(default="lstm", description="ML model architecture name")
    interval: str = Field(default="30m", description="Bar aggregation timeframe")
    horizon: int = Field(default=1, ge=1, description="Model forecasting horizon")

    # Simulation Window & Replay Loop
    start_date: Optional[datetime] = Field(
        default=None, description="Start date of out-of-sample backtest simulation"
    )
    end_date: Optional[datetime] = Field(
        default=None, description="End date of out-of-sample backtest simulation"
    )
    warmup_bars: int = Field(
        default=100, ge=0, description="Warmup bars for strategy indicators"
    )
    lookback_limit: int = Field(
        default=1000, ge=10, description="Maximum historical bars retained in memory"
    )
    run_id: str = Field(
        default="backtest_run", description="Unique identifier for database logs"
    )

    # Execution & Portfolio
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

    # Classifier Strategy Parameters
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

    # Output & Reporting
    output_dir: str = Field(
        default="../runs/reports",
        description="Path to save backtest HTML/JSON summaries",
    )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "BacktestSpec":
        """Loads and validates a BacktestSpec from a YAML file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Backtest spec file not found at: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**(data or {}))
