# src/trading_bot/backtesting/results.py

import json
import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..core.enums import OrderStatus
from ..core.models import OrderLog as OrderLogModel

logger = logging.getLogger(__name__)


class BacktestResult:
    """
    Encapsulates the results of a backtesting simulation run.

    Responsible for evaluating key performance indicators (KPIs) such as
    cumulative returns, maximum drawdown, total trades, and annualized Sharpe ratio
    from the simulated equity curve and database logs.
    """

    def __init__(
        self,
        run_id: str,
        market_id: str,
        strategy_name: str,
        db_session: Session,
    ) -> None:
        """
        Initializes the backtest results and computes metrics.

        :param run_id: Unique identifier for this backtest execution run.
        :param market_id: Market identifier (e.g., 'BTC/USDT').
        :param strategy_name: Name of the strategy evaluated.
        :param db_session: Database session to query actual order execution metrics.
        """
        self.run_id = run_id
        self.market_id = market_id
        self.strategy_name = strategy_name

        self.initial_equity: float = 0.0
        self.final_equity: float = 0.0
        self.total_return_pct: float = 0.0
        self.max_drawdown_pct: float = 0.0
        self.sharpe_ratio: float = 0.0
        self.total_trades: int = 0

        self._calculate_metrics(db_session)

    def _calculate_metrics(self, db_session: Session) -> None:
        """
        Calculates return, drawdown, Sharpe ratio, and trade statistics from the equity curve.
        """
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

        df_eq = pd.DataFrame(self.equity_curve)
        self.initial_equity = float(df_eq["equity"].iloc[0])
        self.final_equity = float(df_eq["equity"].iloc[-1])

        # 1. Total Cumulative Return
        if self.initial_equity > 0:
            self.total_return_pct = float(
                (self.final_equity - self.initial_equity) / self.initial_equity * 100.0
            )
        else:
            self.total_return_pct = 0.0

        # 2. Maximum Drawdown
        df_eq["peak"] = df_eq["equity"].cummax()
        df_eq["drawdown"] = (df_eq["equity"] - df_eq["peak"]) / df_eq["peak"]
        max_dd = df_eq["drawdown"].min()
        self.max_drawdown_pct = float(max_dd * 100.0) if not pd.isna(max_dd) else 0.0

        # 3. Sharpe Ratio (using daily log returns)
        self.sharpe_ratio = self._calculate_sharpe_ratio(df_eq)

        # 4. Total Trades (query database for actual FILLED orders)
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

    def _calculate_sharpe_ratio(self, df_eq: pd.DataFrame) -> float:
        """
        Calculates the annualized Sharpe ratio from the daily ending equity.
        Assuming daily frequency (365 days/year) and 0% risk-free rate.
        """
        try:
            df = df_eq.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["date"] = df["timestamp"].dt.date

            # Extract last recorded equity for each calendar day
            df_daily = df.groupby("date").last().reset_index()

            if len(df_daily) < 2:
                return 0.0

            # Calculate daily log returns
            df_daily["log_return"] = np.log(
                df_daily["equity"] / df_daily["equity"].shift(1)
            )
            returns = df_daily["log_return"].dropna()

            if len(returns) < 2:
                return 0.0

            mean_return = returns.mean()
            std_return = returns.std(ddof=1)

            if std_return == 0.0:
                return 0.0

            # Annualize assuming crypto 365-day trading year
            sharpe = float((mean_return / std_return) * np.sqrt(365))
            return sharpe if not np.isnan(sharpe) and not np.isinf(sharpe) else 0.0
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the performance metrics to a dictionary representation.
        """
        return {
            "run_id": self.run_id,
            "market_id": self.market_id,
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
