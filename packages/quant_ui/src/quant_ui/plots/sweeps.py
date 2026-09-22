from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from quant_ui.plots.theme import setup_bloomberg_theme

setup_bloomberg_theme()


def plot_parameter_sensitivity(
    trials_df: pd.DataFrame,
    sweep_param: str,
    equity_curves: Optional[Dict[str, pd.DataFrame]] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Pure UI function: Takes sweep trials data and equity curves,
    returns a Plotly Figure summarizing parameter sensitivity.
    """
    if trials_df is None or trials_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title or "Parameter Sweep (No Trials Found)")
        return fig

    # Extract X and Y arrays
    x_vals = trials_df["param_value"].astype(str).tolist()
    val_ics = trials_df.get("val_ic", pd.Series([0.0] * len(trials_df))).tolist()
    val_losses = trials_df.get("val_loss", pd.Series([0.0] * len(trials_df))).tolist()
    oos_pnls = trials_df.get("oos_pnl", pd.Series([0.0] * len(trials_df))).tolist()
    oos_sharpes = trials_df.get(
        "oos_sharpe", pd.Series([0.0] * len(trials_df))
    ).tolist()
    max_dds = trials_df.get("oos_max_dd", pd.Series([0.0] * len(trials_df))).tolist()
    win_rates = (
        trials_df.get("win_rate", pd.Series([0.0] * len(trials_df))) * 100.0
    ).tolist()
    total_trades = trials_df.get(
        "total_trades", pd.Series([0] * len(trials_df))
    ).tolist()
    run_ids = trials_df.get("run_id", pd.Series([""] * len(trials_df))).tolist()
    model_ids = trials_df.get("model_id", pd.Series([""] * len(trials_df))).tolist()

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

    if equity_curves:
        for i, (idx, row) in enumerate(trials_df.iterrows()):
            run_id = row.get("run_id")
            param_val = row.get("param_value")
            df_eq = equity_curves.get(run_id)
            if df_eq is not None and not df_eq.empty:
                has_equity_data = True
                color = colors[i % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=df_eq["timestamp"],
                        y=df_eq["equity"],
                        mode="lines",
                        name=f"{sweep_param}={param_val}",
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
        [mid[:12] for mid in model_ids],
        [f"{v:.4f}" for v in val_ics],
        [f"{v:.4f}" for v in val_losses],
        [f"${v:,.2f}" for v in oos_pnls],
        [f"{v:.2f}" for v in oos_sharpes],
        [f"{v:.2f}%" for v in max_dds],
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
                fill_color="#1e1e1e",  # Dark background for Bloomberg
                align="center",
                font=dict(color="#FFB900", size=11),  # Amber text
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

    main_title = title or "Parameter Sweep Evaluation"
    fig.update_layout(
        title=dict(text=main_title, font=dict(size=18)),
        height=1200,
        showlegend=True,
    )

    return fig
