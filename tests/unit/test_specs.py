# tests/unit/test_specs.py

from datetime import datetime

import pytest
from nets.spec import HParamStudySpec

from trading_bot.backtesting.spec import BacktestSpec
from trading_bot.core.spec_base import PROJECT_ROOT


def test_backtest_spec_loading(tmp_path):
    yaml_content = """
market:
  market_id: "BTC/USDT"
  interval: "30m"

dates:
  start_date: "2026-06-01T00:00:00+00:00"
  end_date: "2026-06-21T00:00:00+00:00"

execution:
  execution_delay_k: 1
  slippage_pct: 0.001
  initial_balance: 10000.0
  position_size_pct: 0.10

classifier:
  classifier_k: 0.03
  confidence_multiplier: 20.0
  period: 10
  allow_in_sample: false

model_type: "lstm"
horizon: 1
warmup_bars: 100
lookback_limit: 1000
run_id: "test_backtest_lstm"
output_dir: "../runs/reports"
"""
    spec_file = tmp_path / "test_backtest.yaml"
    spec_file.write_text(yaml_content, encoding="utf-8")

    spec = BacktestSpec.from_yaml(spec_file)

    assert spec.market.market_id == "BTC/USDT"
    assert spec.model_type == "lstm"
    assert spec.market.interval == "30m"
    assert spec.execution.execution_delay_k == 1
    assert spec.execution.slippage_pct == 0.001
    assert spec.execution.initial_balance == 10000.0
    assert spec.execution.position_size_pct == 0.10
    assert spec.classifier.classifier_k == 0.03
    assert spec.run_id == "test_backtest_lstm"
    assert isinstance(spec.dates.start_date, datetime)
    assert spec.dates.start_date.tzinfo is not None


def test_hparam_study_spec_loading(tmp_path):
    yaml_content = """
study:
  study_name: "lstm_btc_optimization"
  direction: "minimize"
  n_trials: 5
  model_type: "lstm"

market:
  market_id: "BTC/USDT"
  interval: "30m"

dates:
  start_date: "2025-11-04T19:30:00+00:00"
  end_date: "2026-05-30T19:30:00+00:00"

features:
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

    assert spec.study.study_name == "lstm_btc_optimization"
    assert spec.study.n_trials == 5
    assert spec.study.model_type == "lstm"
    assert len(spec.features.feature_cols) == 5
    assert spec.features.feature_pipeline is not None
    assert (
        spec.features.feature_pipeline["class_path"]
        == "trading_bot.core.transforms.FeaturePipeline"
    )
    assert "learning_rate" in spec.search_space


def test_modular_spec_imports_and_overrides(tmp_path):
    # Component 1: Market
    market_comp = tmp_path / "market_comp.yaml"
    market_comp.write_text(
        "market:\n  market_id: 'ETH/USDT'\n  interval: '1h'\n", encoding="utf-8"
    )

    # Top level spec overriding market_id but inheriting interval
    main_spec = tmp_path / "main_spec.yaml"
    main_spec.write_text(
        f"imports:\n  - '{market_comp}'\nmarket:\n  market_id: 'SOL/USDT'\nmodel_type: 'cnn'\n",
        encoding="utf-8",
    )

    spec = BacktestSpec.from_yaml(main_spec)

    assert spec.market.market_id == "SOL/USDT"  # Root override applied
    assert spec.market.interval == "1h"  # Inherited from component
    assert spec.model_type == "cnn"


def test_circular_imports_detection(tmp_path):
    file_a = tmp_path / "a.yaml"
    file_b = tmp_path / "b.yaml"

    file_a.write_text(f"imports:\n  - '{file_b}'\n", encoding="utf-8")
    file_b.write_text(f"imports:\n  - '{file_a}'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Circular dependency detected"):
        BacktestSpec.from_yaml(file_a)


def test_spec_file_not_found():
    with pytest.raises(FileNotFoundError):
        BacktestSpec.from_yaml("non_existent_file.yaml")

    with pytest.raises(FileNotFoundError):
        HParamStudySpec.from_yaml("non_existent_file.yaml")


def test_repo_specs_loading():
    # Verify canonical backtest spec load
    backtest_path = PROJECT_ROOT / "specs/backtests/BTCUSDT/lstm_close_backtest.yaml"
    bt_spec = BacktestSpec.from_yaml(backtest_path)
    assert bt_spec.market.market_id == "BTC/USDT"
    assert bt_spec.market.interval == "30m"
    assert bt_spec.model_type == "lstm"
    assert bt_spec.execution.execution_delay_k == 1

    # Verify canonical training spec load
    train_path = PROJECT_ROOT / "specs/train/BTCUSDT/lstm_hparam_ohlcv.yaml"
    tr_spec = HParamStudySpec.from_yaml(train_path)
    assert tr_spec.market.market_id == "BTC/USDT"
    assert tr_spec.market.interval == "30m"
    assert tr_spec.study.model_type == "lstm"
    assert len(tr_spec.features.feature_cols) == 5


def test_backtest_spec_run_id_field():
    spec1 = BacktestSpec()
    assert spec1.run_id is None

    spec_override = BacktestSpec(run_id="custom_id_123")
    assert spec_override.run_id == "custom_id_123"
