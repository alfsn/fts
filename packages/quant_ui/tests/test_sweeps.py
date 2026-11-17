import pandas as pd
import plotly.graph_objects as go
from quant_ui.plots.sweeps import plot_parameter_sensitivity


def test_plot_parameter_sensitivity():
    # Mock trials_df
    trials_df = pd.DataFrame(
        {
            "param_value": [10, 20, 30],
            "val_ic": [0.05, 0.08, 0.04],
            "val_loss": [1.2, 1.1, 1.3],
            "oos_pnl": [500.0, 1000.0, -200.0],
            "oos_sharpe": [1.5, 2.1, -0.5],
            "oos_max_dd": [5.0, 8.0, 15.0],
            "win_rate": [0.55, 0.60, 0.45],
            "total_trades": [100, 150, 120],
            "run_id": ["run_10", "run_20", "run_30"],
            "model_id": [
                "model_10_abcdefgh1234",
                "model_20_abcdefgh1234",
                "model_30_abcdefgh1234",
            ],
        }
    )

    # Mock equity curves
    dates = pd.date_range("2023-01-01", periods=10)
    equity_curves = {
        "run_10": pd.DataFrame(
            {"timestamp": dates, "equity": [1000 + i * 10 for i in range(10)]}
        ),
        "run_20": pd.DataFrame(
            {"timestamp": dates, "equity": [1000 + i * 20 for i in range(10)]}
        ),
    }

    fig = plot_parameter_sensitivity(
        trials_df=trials_df,
        sweep_param="moving_average_window",
        equity_curves=equity_curves,
        title="Test Sweep",
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 5  # Multiple traces expected
    assert fig.layout.title.text == "Test Sweep"


def test_plot_parameter_sensitivity_empty():
    fig = plot_parameter_sensitivity(pd.DataFrame(), sweep_param="test")
    assert isinstance(fig, go.Figure)
    assert "No Trials Found" in fig.layout.title.text


def test_plot_parameter_sensitivity_no_equity():
    trials_df = pd.DataFrame(
        {
            "param_value": [10],
            "run_id": ["run_10"],
        }
    )
    fig = plot_parameter_sensitivity(trials_df, sweep_param="test", equity_curves=None)
    assert isinstance(fig, go.Figure)
