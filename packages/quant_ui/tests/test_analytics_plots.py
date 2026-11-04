import numpy as np
import pandas as pd
import plotly.graph_objects as go
from quant_core.analytics import (
    compute_rolling_vol,
    drawdown_series,
    period_returns,
    top_drawdowns,
    wealth_index,
)
from quant_core.metrics import compute_performance_metrics
from quant_ui.plots.analytics import (
    plot_cumulative_returns,
    plot_risk_kpi_table,
    plot_rolling_volatility,
    plot_underwater,
)

VALUES = pd.Series(
    100 * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, 300)),
    index=pd.bdate_range("2024-01-01", periods=300),
)


def test_plot_cumulative_returns_with_baseline():
    cum = wealth_index(VALUES) - 1
    fig = plot_cumulative_returns(cum, baseline=cum * 0.5, baseline_name="SPY")
    assert isinstance(fig, go.Figure)
    assert [t.name for t in fig.data] == ["Portfolio", "SPY"]


def test_plot_rolling_volatility_one_subchart_per_window():
    r = period_returns(VALUES)
    vols = {f"{w}d": compute_rolling_vol(r, w, 252) for w in (30, 90, 252)}
    assert len(plot_rolling_volatility(vols).data) == 3


def test_plot_underwater_highlights_top_episodes():
    episodes = top_drawdowns(VALUES, n=5)
    fig = plot_underwater(drawdown_series(VALUES), episodes)
    assert len(fig.layout.shapes) == len(episodes) <= 5


def test_plot_risk_kpi_table_renders_nan_as_na():
    fig = plot_risk_kpi_table(compute_performance_metrics(VALUES.iloc[:1]))
    assert "n/a" in fig.data[0].cells.values[1]
