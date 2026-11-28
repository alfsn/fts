# src/trading_bot/backtesting/results.py

import json
import logging
import os
from typing import Any, Dict

from quant_core.models import PerformanceMetrics

from ..core.performance import build_performance_metrics
from ..runs import RunRecord

logger = logging.getLogger(__name__)


class BacktestResult:
    """
    Encapsulates the results of a backtesting simulation run.

    Summarises a saved run; KPI math is delegated to quant_core (through
    ``core.performance``) so backtests and the catalog report identical numbers.
    """

    def __init__(
        self,
        run_id: str,
        instrument_id: str,
        strategy_name: str,
        record: RunRecord,
    ) -> None:
        """
        Initializes the backtest results and computes metrics.

        :param run_id: Unique identifier for this backtest execution run.
        :param instrument_id: Market identifier (e.g., 'BTC/USDT').
        :param strategy_name: Name of the strategy evaluated.
        :param record: The saved run (``RunStore.load`` / ``load_record``).
        """
        self.run_id = run_id
        self.instrument_id = instrument_id
        self.strategy_name = strategy_name
        self.record = record

        self.initial_equity: float = 0.0
        self.final_equity: float = 0.0
        self.total_return_pct: float = 0.0
        self.max_drawdown_pct: float = 0.0
        self.sharpe_ratio: float = 0.0
        self.total_trades: int = len(record.fills)
        self.metrics = PerformanceMetrics()
        self.equity_curve = [p.model_dump() for p in record.equity]

        if not record.equity:
            logger.warning(
                "Empty equity curve for this run. All metrics will default to 0.0."
            )
            return

        self.initial_equity = float(record.equity[0].equity)
        self.final_equity = float(record.equity[-1].equity)
        self.metrics = build_performance_metrics(record.equity, record.fills)
        self.total_return_pct = self.metrics.total_return * 100.0
        self.max_drawdown_pct = self.metrics.max_drawdown * 100.0
        self.sharpe_ratio = self.metrics.sharpe_ratio

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the performance metrics to a dictionary representation.
        """
        return {
            "run_id": self.run_id,
            "instrument_id": self.instrument_id,
            "strategy_name": self.strategy_name,
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "total_return_pct": self.total_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "total_trades": self.total_trades,
        }

    def save_summary(self, file_path: str) -> None:
        """
        Saves the summary performance metrics to a JSON file.
        """
        try:
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=4)
            logger.info(f"Backtest summary saved successfully to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save backtest summary to {file_path}: {e}")
            raise e
