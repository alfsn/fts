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


def _benchmark_analysis(n):
    from quant_core.metrics import compute_benchmark_metrics

    rng = np.random.default_rng(5)
    spy = VALUES.iloc[:n] * (1 + rng.normal(0, 0.003, n))  # correlated benchmark
    return compute_benchmark_metrics(VALUES.iloc[:n], None, spy)


def test_benchmark_scatter_draws_the_fitted_line():
    from quant_ui.plots.analytics import plot_benchmark_scatter

    analysis = _benchmark_analysis(300)
    assert np.isfinite(analysis.metrics.beta)
    fig = plot_benchmark_scatter(analysis.excess, analysis.metrics, "SPY")
    assert [t.name for t in fig.data] == ["Daily excess returns", "OLS fit"]
    assert "n = 299" in fig.layout.title.text


def test_benchmark_views_show_na_below_the_minimum_sample():
    from quant_ui.plots.analytics import (
        plot_benchmark_kpi_table,
        plot_benchmark_scatter,
    )

    analysis = _benchmark_analysis(50)
    fig = plot_benchmark_scatter(analysis.excess, analysis.metrics, "SPY")
    assert len(fig.data) == 1 and "n/a" in fig.layout.title.text
    cells = plot_benchmark_kpi_table(analysis.metrics, "SPY", 252).data[0].cells.values
    assert cells[1][0] == "n/a" and cells[1][7] == "49 (min 252)"
