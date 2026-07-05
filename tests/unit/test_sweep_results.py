# tests/unit/test_sweep_results.py

import os
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine

from trading_bot.backtesting.sweep_exporter import HTMLSweepExporter
from trading_bot.backtesting.sweep_results import SweepResult, SweepTrialResult
from trading_bot.backtesting.sweep_visualizer import SweepVisualizer
from trading_bot.core.database import Base, SessionLocal
from trading_bot.core.database import engine as dev_engine


@pytest.fixture
def sample_sweep_result():
    trials = [
        SweepTrialResult(
            trial_index=0,
            param_value=1,
            model_id="model_layer_1",
            run_id="run_layer_1",
            val_ic=0.045,
            val_loss=0.012,
            oos_pnl=150.50,
            oos_sharpe=1.85,
            oos_max_dd=-5.2,
            win_rate=0.58,
            total_trades=12,
            final_equity=10150.50,
        ),
        SweepTrialResult(
            trial_index=1,
            param_value=2,
            model_id="model_layer_2",
            run_id="run_layer_2",
            val_ic=0.062,
            val_loss=0.009,
            oos_pnl=320.00,
            oos_sharpe=2.40,
            oos_max_dd=-3.1,
            win_rate=0.65,
            total_trades=15,
            final_equity=10320.00,
        ),
    ]
    return SweepResult(
        sweep_name="lstm_num_layers_sweep",
        sweep_param="num_layers",
        sweep_values=[1, 2],
        market_id="BTC/USDT",
        trials=trials,
    )


def test_sweep_result_serialization(sample_sweep_result, tmp_path):
    d = sample_sweep_result.to_dict()
    assert d["sweep_name"] == "lstm_num_layers_sweep"
    assert d["sweep_param"] == "num_layers"
    assert d["total_trials"] == 2
    assert len(d["trials"]) == 2
    assert d["trials"][0]["param_value"] == 1
    assert d["trials"][1]["param_value"] == 2

    # Save summary JSON
    json_file = str(tmp_path / "sweep_summary.json")
    sample_sweep_result.save_summary(json_file)
    assert os.path.exists(json_file)

    # Load summary JSON
    loaded_sweep = SweepResult.load_summary(json_file)
    assert loaded_sweep.sweep_name == sample_sweep_result.sweep_name
    assert loaded_sweep.sweep_param == sample_sweep_result.sweep_param
    assert len(loaded_sweep.trials) == 2
    assert loaded_sweep.trials[0].val_ic == 0.045
    assert loaded_sweep.trials[1].oos_sharpe == 2.40


def test_sweep_visualizer_render(sample_sweep_result):
    visualizer = SweepVisualizer(db_url="sqlite:///:memory:")
    fig = visualizer.render_charts(sample_sweep_result)
    assert fig is not None
    # Check Plotly figure data & titles
    assert "lstm_num_layers_sweep" in fig.layout.title.text
    # 4 subplot rows traces
    assert len(fig.data) >= 4


def test_html_sweep_exporter(sample_sweep_result, tmp_path):
    exporter = HTMLSweepExporter(db_url="sqlite:///:memory:")
    output_file = tmp_path / "sweep_report.html"

    report_path = exporter.export(sample_sweep_result, output_path=str(output_file))

    assert report_path == str(output_file)
    assert os.path.exists(report_path)

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "Plotly" in content or "plotly" in content.lower()
        assert "lstm_num_layers_sweep" in content
        assert "num_layers" in content


def test_html_sweep_exporter_empty_raises():
    empty_sweep = SweepResult(
        sweep_name="empty_sweep",
        sweep_param="hidden_dim",
        sweep_values=[],
        market_id="BTC/USDT",
    )
    exporter = HTMLSweepExporter(db_url="sqlite:///:memory:")
    with pytest.raises(ValueError, match="Cannot export empty SweepResult"):
        exporter.export(empty_sweep)
