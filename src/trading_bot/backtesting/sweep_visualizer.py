# src/trading_bot/backtesting/sweep_visualizer.py

import logging
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from trading_bot.backtesting.sweep_results import SweepResult, SweepTrialResult
from trading_bot.core.models import BacktestEquityLog

logger = logging.getLogger(__name__)


class SweepVisualizer:
    """
    Provides Plotly charts and visual layouts for parameter sweep runs.

    Renders parameter sensitivity curves (Val IC vs. OOS Sharpe), drawdown & P&L profiles,
    overlaid trial equity curves, and interactive trial metrics tables.
    """

    def __init__(self, db_url: str = "sqlite:///./dev.db") -> None:
        """
        Initializes the visualizer with a database URL.
        """
        self.engine = create_engine(db_url)
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
        if not sweep_result.trials:
            fig = go.Figure()
            fig.update_layout(
                title=title
                or f"Parameter Sweep: {sweep_result.sweep_name} (No Trials Found)"
            )
            return fig

        sweep_param = sweep_result.sweep_param
        trials = sweep_result.trials

        # Extract X and Y arrays
        x_vals = [str(t.param_value) for t in trials]
        val_ics = [t.val_ic for t in trials]
        val_losses = [t.val_loss for t in trials]
        oos_pnls = [t.oos_pnl for t in trials]
        oos_sharpes = [t.oos_sharpe for t in trials]
        max_dds = [t.oos_max_dd for t in trials]
        win_rates = [t.win_rate * 100.0 for t in trials]
        total_trades = [t.total_trades for t in trials]

        # 4 Subplot Rows
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.08,
            subplot_titles=(
                f"1. Parameter Sensitivity: Val IC vs. Out-of-Sample Sharpe ({sweep_param})",
                "2. Out-of-Sample Performance Profile: Realized P&L vs. Max Drawdown",
                "3. Overlaid Out-of-Sample Equity Curves across Sweep Trials",
                "4. Sweep Trials Metric Comparison Table",
            ),
            specs=[
                [{"secondary_y": True}],
                [{"secondary_y": True}],
                [{"secondary_y": False}],
                [{"type": "table"}],
            ],
            row_heights=[0.25, 0.25, 0.30, 0.20],
        )

        # Row 1: Val IC vs OOS Sharpe Ratio
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=val_ics,
                mode="lines+markers",
                name="Validation IC",
                line=dict(color="#1f77b4", width=2.5),
                marker=dict(size=8),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=oos_sharpes,
                mode="lines+markers",
                name="OOS Sharpe Ratio",
                line=dict(color="#2ca02c", width=2.5, dash="dash"),
                marker=dict(size=8),
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

        # Row 2: OOS Realized PnL vs Max Drawdown
        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=oos_pnls,
                name="OOS P&L ($)",
                marker_color="#ff7f0e",
                opacity=0.75,
            ),
            row=2,
            col=1,
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=max_dds,
                mode="lines+markers",
                name="Max Drawdown (%)",
                line=dict(color="#d62728", width=2),
                marker=dict(size=8),
            ),
            row=2,
            col=1,
            secondary_y=True,
        )

        # Row 3: Overlaid Equity Curves
        run_ids = [t.run_id for t in trials]
        equity_curves = self.load_trial_equity_curves(run_ids, db_session=db_session)

        has_equity_data = False
        colors = [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ]

        for i, t in enumerate(trials):
            df_eq = equity_curves.get(t.run_id)
            if df_eq is not None and not df_eq.empty:
                has_equity_data = True
                color = colors[i % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=df_eq["timestamp"],
                        y=df_eq["equity"],
                        mode="lines",
                        name=f"{sweep_param}={t.param_value}",
                        line=dict(color=color, width=1.8),
                    ),
                    row=3,
                    col=1,
                )

        if not has_equity_data:
            fig.add_annotation(
                text="No DB equity curve logs found for trials",
                xref="x3",
                yref="y3",
                showarrow=False,
                row=3,
                col=1,
            )

        # Row 4: Summary Table
        table_headers = [
            f"Param ({sweep_param})",
            "Model ID",
            "Val IC",
            "Val Loss",
            "OOS P&L",
            "OOS Sharpe",
            "Max DD (%)",
            "Win Rate (%)",
            "Trades",
        ]

        table_cells = [
            x_vals,
            [t.model_id[:12] for t in trials],
            [f"{t.val_ic:.4f}" for t in trials],
            [f"{t.val_loss:.4f}" for t in trials],
            [f"${t.oos_pnl:,.2f}" for t in trials],
            [f"{t.oos_sharpe:.2f}" for t in trials],
            [f"{t.oos_max_dd:.2f}%" for t in trials],
            [f"{wr:.1f}%" for wr in win_rates],
            total_trades,
        ]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=table_headers,
                    fill_color="#2a3f5f",
                    align="center",
                    font=dict(color="white", size=11),
                ),
                cells=dict(
                    values=table_cells,
                    fill_color="#f5f7fa",
                    align="center",
                    font=dict(color="black", size=11),
                ),
            ),
            row=4,
            col=1,
        )

        # Axes titles & styling
        fig.update_xaxes(title_text=f"{sweep_param} Value", row=1, col=1)
        fig.update_yaxes(title_text="Validation IC", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="OOS Sharpe Ratio", row=1, col=1, secondary_y=True)

        fig.update_xaxes(title_text=f"{sweep_param} Value", row=2, col=1)
        fig.update_yaxes(title_text="Realized P&L ($)", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Max Drawdown (%)", row=2, col=1, secondary_y=True)

        fig.update_xaxes(title_text="Date / Time", row=3, col=1)
        fig.update_yaxes(title_text="Portfolio Equity ($)", row=3, col=1)

        main_title = (
            title
            or f"Parameter Sweep Evaluation: {sweep_result.sweep_name} ({sweep_result.market_id})"
        )
        fig.update_layout(
            title=dict(text=main_title, font=dict(size=18)),
            height=1200,
            showlegend=True,
            template="plotly_white",
        )

        return fig
