# src/trading_bot/backtesting/sweep_visualizer.py

import logging
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from sqlalchemy.orm import sessionmaker
from trading_bot.backtesting.sweep_results import SweepResult
from trading_bot.config import get_settings
from trading_bot.core.database import create_db_engine
from trading_bot.runs import RunStore

logger = logging.getLogger(__name__)


class SweepVisualizer:
    """
    Provides Plotly charts and visual layouts for parameter sweep runs.

    Renders parameter sensitivity curves (Val IC vs. OOS Sharpe), drawdown & P&L profiles,
    overlaid trial equity curves, and interactive trial metrics tables.
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        """
        Initializes the visualizer on the research database (D16), where the
        trial runs are saved.
        """
        self.engine = create_db_engine(db_url or get_settings().DATABASE_URL)
        self.SessionLocal = sessionmaker(
            bind=self.engine, autocommit=False, autoflush=False
        )
        self.store = RunStore(self.SessionLocal)

    def load_trial_equity_curves(
        self, trial_run_ids: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Equity curve per trial run_id (empty when the run isn't saved)."""
        curves: Dict[str, pd.DataFrame] = {}
        for run_id in trial_run_ids:
            record = self.store.load(run_id, predictions=False)
            points = [p.model_dump() for p in record.equity] if record else []
            df_eq = pd.DataFrame(points)
            if not df_eq.empty:
                df_eq["timestamp"] = pd.to_datetime(df_eq["timestamp"])
            curves[run_id] = df_eq
        return curves

    def render_charts(
        self,
        sweep_result: SweepResult,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Renders a comprehensive multi-subplot Plotly figure summarizing the parameter sweep.
        """
        from quant_ui.plots.sweeps import plot_parameter_sensitivity

        run_ids = [t.run_id for t in sweep_result.trials]
        equity_curves = self.load_trial_equity_curves(run_ids)

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
