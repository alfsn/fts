# src/trading_bot/backtesting/sweep_visualizer.py

import logging
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from trading_bot.backtesting.sweep_results import SweepResult, SweepTrialResult
from trading_bot.config import get_settings
from trading_bot.core.database import create_db_engine
from trading_bot.core.models import BacktestEquityLog

logger = logging.getLogger(__name__)


class SweepVisualizer:
    """
    Provides Plotly charts and visual layouts for parameter sweep runs.

    Renders parameter sensitivity curves (Val IC vs. OOS Sharpe), drawdown & P&L profiles,
    overlaid trial equity curves, and interactive trial metrics tables.
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        """
        Initializes the visualizer with a database URL.
        Defaults to BACKTEST_DATABASE_URL where backtest equity logs are persisted.
        """
        st = get_settings()
        resolved_db_url = db_url or getattr(
            st, "BACKTEST_DATABASE_URL", st.DATABASE_URL
        )
        self.engine = create_db_engine(resolved_db_url)
        self.SessionLocal = sessionmaker(
            bind=self.engine, autocommit=False, autoflush=False
        )

    def load_trial_equity_curves(
        self, trial_run_ids: List[str], db_session: Optional[Session] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Queries the database for historical equity logs for a set of trial run IDs.
        Returns a dictionary mapping run_id -> DataFrame of equity curve over time.
        """
        session_created = False
        if db_session is None:
            db_session = self.SessionLocal()
            session_created = True

        curves: Dict[str, pd.DataFrame] = {}
        try:
            for run_id in trial_run_ids:
                logs = (
                    db_session.query(BacktestEquityLog)
                    .filter(BacktestEquityLog.run_id == run_id)
                    .order_by(BacktestEquityLog.timestamp.asc())
                    .all()
                )
                if not logs and not session_created:
                    fallback_session = self.SessionLocal()
                    try:
                        logs = (
                            fallback_session.query(BacktestEquityLog)
                            .filter(BacktestEquityLog.run_id == run_id)
                            .order_by(BacktestEquityLog.timestamp.asc())
                            .all()
                        )
                    finally:
                        fallback_session.close()

                if logs:
                    data = [
                        {
                            "timestamp": pd.to_datetime(log.timestamp),
                            "cash": log.cash,
                            "position": log.position,
                            "close": log.close,
                            "equity": log.equity,
                        }
                        for log in logs
                    ]
                    df_eq = pd.DataFrame(data)
                    curves[run_id] = df_eq
                else:
                    curves[run_id] = pd.DataFrame()
            return curves
        except Exception as e:
            logger.warning(f"Error querying trial equity curves from DB: {e}")
            return curves
        finally:
            if session_created:
                db_session.close()

    def render_charts(
        self,
        sweep_result: SweepResult,
        db_session: Optional[Session] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Renders a comprehensive multi-subplot Plotly figure summarizing the parameter sweep.
        """
        from quant_ui.plots.sweeps import plot_parameter_sensitivity

        run_ids = [t.run_id for t in sweep_result.trials]
        equity_curves = self.load_trial_equity_curves(run_ids, db_session=db_session)

        trials_df = pd.DataFrame([t.model_dump() for t in sweep_result.trials])

        main_title = (
            title
            or f"Parameter Sweep Evaluation: {sweep_result.sweep_name} ({sweep_result.instrument_id})"
        )

        return plot_parameter_sensitivity(
            trials_df=trials_df,
            sweep_param=sweep_result.sweep_param,
            equity_curves=equity_curves,
            title=main_title,
        )
