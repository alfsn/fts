"""Backtests: fts runs with their KPIs, one run's charts, and a comparison
(ported from fts's dashboard). Every series comes computed from the catalog
(``run_curves``, ``compare_curves``); this page only draws."""

import pandas as pd
import streamlit as st
from quant_ui.plots.analytics import (
    plot_benchmark_kpi_table,
    plot_cumulative_returns,
    plot_risk_kpi_table,
    plot_underwater,
    plot_vol_forecast_kpi_table,
)
from quant_ui.plots.backtest import (
    plot_cumulative_comparison,
    plot_prediction_scatter,
    plot_trade_overlay,
)
from workbench import state
from workbench.filters import keep, parameter_filter

catalog = state.catalog()
st.title("Backtest runs")

a, b = st.columns(2)
instrument = a.text_input("Instrument").strip()
min_sharpe_text = b.text_input(
    "Minimum Sharpe (blank: no minimum; a run too short for a Sharpe is "
    "hidden by any minimum)"
).strip()
try:
    min_sharpe = float(min_sharpe_text) if min_sharpe_text else None
except ValueError:
    st.error(f"Minimum Sharpe must be a number, not {min_sharpe_text!r}.")
    st.stop()

runs = catalog.backtest_repo.list_runs(
    instrument_id=instrument or None, min_sharpe=min_sharpe
)
if not runs:
    st.info("No runs match. Runs appear here after a backtest is recorded.")
    st.stop()

ranges, text = parameter_filter([r.hyperparameters for r in runs], key="runs")
shown = [
    r
    for r in runs
    if keep(r.hyperparameters, ranges, text, haystack=f"{r.run_id} {r.model_id}")
]
if not shown:
    st.warning("No runs match the parameter filters.")
    st.stop()

percent = st.column_config.NumberColumn(format="percent")
st.dataframe(
    pd.DataFrame(
        [
            {
                "Run ID": r.run_id,
                "Model": r.model_id,
                "Strategy": r.strategy_name,
                "Instrument": r.instrument_id,
                "Total return": r.metrics.total_return,
                "Sharpe": r.metrics.sharpe_ratio,
                "Sortino": r.metrics.sortino_ratio,
                "Max drawdown": r.metrics.max_drawdown,
                "Win rate (round trips)": r.metrics.win_rate,
                "Fills": r.metrics.total_trades,
                "Start": r.start_time,
                "End": r.end_time,
            }
            for r in shown
        ]
    ),
    use_container_width=True,
    column_config={
        "Total return": percent,
        "Max drawdown": percent,
        "Win rate (round trips)": percent,
        "Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "Sortino": st.column_config.NumberColumn(format="%.2f"),
    },
)

st.divider()
st.subheader("One run")
run_id = st.selectbox("Run", [r.run_id for r in shown])
detail = catalog.backtest_repo.get_run_details(run_id)
curves = catalog.backtest_repo.run_curves(run_id)
if detail is None or curves is None:
    st.stop()

left, right = st.columns(2)
left.plotly_chart(plot_risk_kpi_table(detail.metrics), use_container_width=True)
right.plotly_chart(
    plot_benchmark_kpi_table(
        detail.benchmark_metrics, f"Buy & hold {detail.instrument_id} (raw close)"
    ),
    use_container_width=True,
)
st.plotly_chart(plot_cumulative_returns(curves.cumulative), use_container_width=True)
st.plotly_chart(
    plot_underwater(curves.drawdown, curves.episodes), use_container_width=True
)
left, right = st.columns(2)
left.plotly_chart(
    plot_trade_overlay(curves.close, curves.fills), use_container_width=True
)
right.plotly_chart(
    plot_prediction_scatter(curves.predictions), use_container_width=True
)
st.plotly_chart(
    plot_vol_forecast_kpi_table(detail.vol_forecast), use_container_width=True
)

st.divider()
st.subheader("Compare runs")
chosen = st.multiselect(
    "Runs", [r.run_id for r in shown], default=[r.run_id for r in shown[:3]]
)
align = st.radio(
    "Line them up by",
    ["calendar", "elapsed"],
    format_func={"calendar": "Calendar date", "elapsed": "Days since start"}.get,
    horizontal=True,
)
st.plotly_chart(
    plot_cumulative_comparison(
        catalog.backtest_repo.compare_curves(chosen, align=align),
        x_title="Days since start" if align == "elapsed" else "Date",
    ),
    use_container_width=True,
)
st.caption(
    "Each curve starts at its own first bar (growth of 1). Picking the best of "
    "many runs by eye overfits: the multiple-testing caveat applies."
)
