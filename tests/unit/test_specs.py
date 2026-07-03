# tests/unit/test_specs.py

from datetime import datetime

import pytest
from nets.spec import HParamStudySpec

from trading_bot.backtesting.spec import BacktestSpec


def test_backtest_spec_loading(tmp_path):
    yaml_content = """
market_id: "BTC/USDT"
model_type: "lstm"
interval: "30m"
horizon: 1

start_date: "2026-06-01T00:00:00+00:00"
end_date: "2026-06-21T00:00:00+00:00"
warmup_bars: 100
lookback_limit: 1000
run_id: "test_backtest_lstm"

execution_delay_k: 1
slippage_pct: 0.001
initial_balance: 10000.0
position_size_pct: 0.10

classifier_k: 0.03
confidence_multiplier: 20.0
period: 10
allow_in_sample: false

output_dir: "../runs/reports"
"""
    spec_file = tmp_path / "test_backtest.yaml"
    spec_file.write_text(yaml_content, encoding="utf-8")

    spec = BacktestSpec.from_yaml(spec_file)

    assert spec.market_id == "BTC/USDT"
    assert spec.model_type == "lstm"
    assert spec.interval == "30m"
    assert spec.execution_delay_k == 1
    assert spec.slippage_pct == 0.001
    assert spec.initial_balance == 10000.0
    assert spec.position_size_pct == 0.10
    assert spec.classifier_k == 0.03
    assert spec.run_id == "test_backtest_lstm"
    assert isinstance(spec.start_date, datetime)
    assert spec.start_date.tzinfo is not None


def test_hparam_study_spec_loading(tmp_path):
    yaml_content = """
study_name: "lstm_btc_optimization"
direction: "minimize"
n_trials: 5
model_type: "lstm"
market_id: "BTC/USDT"
interval: "30m"
start_date: "2025-11-04T19:30:00+00:00"
end_date: "2026-05-30T19:30:00+00:00"

lookback_period: 20
feature_cols: ["open", "high", "low", "close", "volume"]
feature_pipeline:
  class_path: "trading_bot.core.transforms.FeaturePipeline"
  params:
    transforms:
      - class_path: "trading_bot.core.transforms.LogReturnTransform"
        params:
          col_idx: 3

search_space:
  learning_rate:
    type: "float"
    low: 0.0001
    high: 0.01
"""
    spec_file = tmp_path / "test_hparam.yaml"
    spec_file.write_text(yaml_content, encoding="utf-8")

    spec = HParamStudySpec.from_yaml(spec_file)

    assert spec.study_name == "lstm_btc_optimization"
    assert spec.n_trials == 5
    assert spec.model_type == "lstm"
    assert len(spec.feature_cols) == 5
    assert spec.feature_pipeline is not None
    assert (
        spec.feature_pipeline["class_path"]
        == "trading_bot.core.transforms.FeaturePipeline"
    )
    assert "learning_rate" in spec.search_space


def test_spec_file_not_found():
    with pytest.raises(FileNotFoundError):
        BacktestSpec.from_yaml("non_existent_file.yaml")

    with pytest.raises(FileNotFoundError):
        HParamStudySpec.from_yaml("non_existent_file.yaml")
