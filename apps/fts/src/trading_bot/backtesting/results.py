# src/trading_bot/backtesting/results.py

import json
import logging
import os
from typing import Any, Dict

from quant_core.enums import OrderStatus
from quant_core.metrics import compute_performance_metrics
from quant_core.models import PerformanceMetrics
from sqlalchemy.orm import Session

from ..core.models import OrderLog as OrderLogModel
from ..core.performance import equity_series

logger = logging.getLogger(__name__)


class BacktestResult:
    """
    Encapsulates the results of a backtesting simulation run.

    Loads the simulated equity curve and order logs; KPI math is delegated to
    quant_core so backtests and the catalog report identical numbers.
    """

    def __init__(
        self,
        run_id: str,
        instrument_id: str,
        strategy_name: str,
        db_session: Session,
    ) -> None:
        """
        Initializes the backtest results and computes metrics.

        :param run_id: Unique identifier for this backtest execution run.
        :param instrument_id: Market identifier (e.g., 'BTC/USDT').
        :param strategy_name: Name of the strategy evaluated.
        :param db_session: Database session to query actual order execution metrics.
        """
        self.run_id = run_id
        self.instrument_id = instrument_id
        self.strategy_name = strategy_name

        self.initial_equity: float = 0.0
        self.final_equity: float = 0.0
        self.total_return_pct: float = 0.0
        self.max_drawdown_pct: float = 0.0
        self.sharpe_ratio: float = 0.0
        self.total_trades: int = 0
        self.metrics = PerformanceMetrics()

        self._calculate_metrics(db_session)

    def _calculate_metrics(self, db_session: Session) -> None:
        try:
            from ..core.models import BacktestEquityLog

            equity_logs = (
                db_session.query(BacktestEquityLog)
                .filter(BacktestEquityLog.run_id == self.run_id)
                .order_by(BacktestEquityLog.timestamp.asc())
                .all()
            )
            self.equity_curve = [
                {
                    "timestamp": log.timestamp,
                    "cash": log.cash,
                    "position": log.position,
                    "close": log.close,
                    "equity": log.equity,
                }
                for log in equity_logs
            ]
        except Exception as e:
            logger.error(f"Failed to query equity curve from database: {e}")
            self.equity_curve = []

        if not self.equity_curve:
            logger.warning(
                "Empty equity curve loaded from database. All metrics will default to 0.0."
            )
            return

        self.initial_equity = float(equity_logs[0].equity)
        self.final_equity = float(equity_logs[-1].equity)

        self.metrics = compute_performance_metrics(equity_series(equity_logs))
        self.total_return_pct = self.metrics.total_return * 100.0
        self.max_drawdown_pct = self.metrics.max_drawdown * 100.0
        self.sharpe_ratio = self.metrics.sharpe_ratio

        # Total Trades (query database for actual FILLED orders)
        try:
            trades = (
                db_session.query(OrderLogModel)
                .filter(
                    OrderLogModel.run_id == self.run_id,
                    OrderLogModel.status == OrderStatus.FILLED,
                )
                .all()
            )
            self.total_trades = len(trades)
        except Exception as e:
            logger.error(f"Failed to query trade count from database: {e}")
            self.total_trades = 0

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
