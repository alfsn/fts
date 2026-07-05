# src/trading_bot/backtesting/sweep_results.py

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SweepTrialResult(BaseModel):
    """
    Encapsulates metrics and identification details for an individual parameter sweep trial.
    """

    trial_index: int = Field(
        ..., description="0-indexed sequence position of the trial within the sweep"
    )
    param_value: Any = Field(
        ..., description="Tested hyperparameter value for this trial"
    )
    model_id: str = Field(
        ..., description="Unique registered candidate model identifier"
    )
    run_id: str = Field(
        ..., description="Backtest run identifier used in DB execution logs"
    )
    val_ic: float = Field(default=0.0, description="Validation Information Coefficient")
    val_loss: float = Field(default=0.0, description="Validation loss metric")
    oos_pnl: float = Field(default=0.0, description="Out-of-sample total realized P&L")
    oos_sharpe: float = Field(default=0.0, description="Out-of-sample Sharpe ratio")
    oos_max_dd: float = Field(
        default=0.0, description="Out-of-sample maximum drawdown percentage"
    )
    win_rate: float = Field(
        default=0.0, description="Percentage of winning trades (0.0 to 1.0)"
    )
    total_trades: int = Field(
        default=0,
        description="Total filled trade executions during out-of-sample backtest",
    )
    final_equity: float = Field(default=0.0, description="Ending balance/equity value")


class SweepResult:
    """
    Encapsulates the aggregated results of a parameter sweep experiment across multiple trials.

    Follows the architectural pattern of `BacktestResult`, allowing JSON summary serialization,
    trial aggregation, and seamless visualization/exporting.
    """

    def __init__(
        self,
        sweep_name: str,
        sweep_param: str,
        sweep_values: List[Any],
        market_id: str = "",
        trials: Optional[List[SweepTrialResult]] = None,
        created_at: Optional[str] = None,
    ) -> None:
        self.sweep_name = sweep_name
        self.sweep_param = sweep_param
        self.sweep_values = sweep_values
        self.market_id = market_id
        self.trials: List[SweepTrialResult] = trials or []
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()

    def add_trial(self, trial: Union[SweepTrialResult, Dict[str, Any]]) -> None:
        """
        Appends a trial result to the sweep evaluation set.
        """
        if isinstance(trial, dict):
            trial = SweepTrialResult(**trial)
        self.trials.append(trial)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the complete sweep result metadata and trial metrics to a dictionary representation.
        """
        return {
            "sweep_name": self.sweep_name,
            "sweep_param": self.sweep_param,
            "sweep_values": self.sweep_values,
            "market_id": self.market_id,
            "created_at": self.created_at,
            "total_trials": len(self.trials),
            "trials": [t.model_dump() for t in self.trials],
        }

    def save_summary(self, file_path: str) -> None:
        """
        Saves the summary parameter sweep performance metrics and metadata to a JSON file.
        """
        try:
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=4)
            logger.info(f"Sweep summary saved successfully to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save sweep summary to {file_path}: {e}")
            raise e

    @classmethod
    def load_summary(cls, file_path: str) -> "SweepResult":
        """
        Loads a `SweepResult` instance from a previously saved JSON summary file.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            trials = [SweepTrialResult(**t) for t in data.get("trials", [])]
            return cls(
                sweep_name=data.get("sweep_name", ""),
                sweep_param=data.get("sweep_param", ""),
                sweep_values=data.get("sweep_values", []),
                market_id=data.get("market_id", ""),
                trials=trials,
                created_at=data.get("created_at"),
            )
        except Exception as e:
            logger.error(f"Failed to load sweep summary from {file_path}: {e}")
            raise e
