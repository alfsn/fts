import os
import re
import runpy
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
from unittest.mock import patch

import pytest
import yaml
from quant_core.enums import BarType
from quant_core.models import BarData
from quant_data.db.models import BarDataLog
from trading_bot.config import settings
from trading_bot.core.database import SessionLocal, create_engine, init_db
from trading_bot.core.models import ModelRegistryLog


@pytest.fixture
def isolated_cli_env(tmp_path):
    # Setup temporary environment variables
    db_path = tmp_path / "test_cli.db"
    db_url = f"sqlite+pysqlite:///{db_path}"

    runs_dir = tmp_path / "runs"
    models_dir = tmp_path / "models"

    original_env = dict(os.environ)
    os.environ["FTS_DATABASE_URL"] = db_url
    os.environ["FTS_RUNS_DIR"] = str(runs_dir)
    os.environ["FTS_MODELS_DIR"] = str(models_dir)

    # We must patch settings directly since the module might have already been imported
    original_db_url = settings.DATABASE_URL
    original_runs_dir = settings.RUNS_DIR
    original_models_dir = settings.MODELS_DIR

    settings.DATABASE_URL = db_url
    settings.RUNS_DIR = str(runs_dir)
    settings.MODELS_DIR = str(models_dir)

    # Initialize the database schema
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal.configure(bind=engine)
    init_db(extra_models=["trading_bot.core.models"], bind_engine=engine)

    yield tmp_path

    # Teardown
    os.environ.clear()
    os.environ.update(original_env)
    settings.DATABASE_URL = original_db_url
    settings.RUNS_DIR = original_runs_dir
    settings.MODELS_DIR = original_models_dir


def test_full_cli_workflow(isolated_cli_env):
    tmp_path = isolated_cli_env

    # ---------------------------------------------------------
    # 1. Ingest Data
    # ---------------------------------------------------------
    def mock_get_bars(self, request):
        bars = []
        current = request.resolve_bounds()[0] or (
            datetime.now(timezone.utc) - timedelta(days=1)
        )
        for i in range(request.count or 1440):
            bars.append(
                BarData(
                    timestamp=current,
                    open=50000.0,
                    high=50100.0,
                    low=49900.0,
                    close=50050.0,
                    volume=1.5,
                    bar_type=BarType.TIME,
                    interval=request.interval or "1m",
                    ticks_count=10,
                    dollar_volume=75000.0,
                )
            )
            current += timedelta(minutes=1)
        return bars

    with (
        patch(
            "quant_data.providers.ccxt_provider.CCXTMarketDataProvider.get_bars",
            mock_get_bars,
        ),
        patch(
            "sys.argv",
            [
                "ingest_historical",
                "-p",
                "ccxt",
                "-t",
                "BTC/USDT",
                "-f",
                "30m",
                "-d",
                "1d",
            ],
        ),
    ):

        runpy.run_module("trading_bot.utils.ingest_historical", run_name="__main__")

    db = SessionLocal()
    bars_count = db.query(BarDataLog).filter_by(instrument_id="BTC/USDT").count()
    assert bars_count > 0, "Database should contain ingested bars"
    db.close()

    # ---------------------------------------------------------
    # 2. Train Model
    # ---------------------------------------------------------
    spec_path = tmp_path / "fast_spec.yaml"
    spec_data = {
        "study": {
            "study_name": "test_study",
            "direction": "minimize",
            "n_trials": 1,
            "model_type": "lstm",
        },
        "market": {"instrument_id": "BTC/USDT", "interval": "30m"},
        "features": {"lookback_period": 20, "feature_cols": ["close"]},
        "search_space": {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.01},
            "hidden_dim": {"type": "int", "low": 8, "high": 8},
            "num_layers": {"type": "int", "low": 1, "high": 1},
            "dropout": {"type": "float", "low": 0.0, "high": 0.0},
            "epochs": {"type": "int", "low": 1, "high": 1},
        },
    }
    with open(spec_path, "w") as f:
        yaml.safe_dump(spec_data, f)

    f_out = StringIO()
    with (
        patch("sys.argv", ["hparam_search", "--spec", str(spec_path)]),
        redirect_stdout(f_out),
    ):
        runpy.run_module("nets.training.hparam_search", run_name="__main__")

    stdout_val = f_out.getvalue()
    match = re.search(r"Generated Model ID: ([a-f0-9]+)", stdout_val)
    assert match is not None, f"Could not find model ID in stdout: {stdout_val}"
    model_id = match.group(1)

    db = SessionLocal()
    model_log = db.query(ModelRegistryLog).filter_by(model_id=model_id).first()
    assert model_log is not None
    assert model_log.status == "candidate"
    assert os.path.exists(model_log.onnx_path)
    db.close()

    # ---------------------------------------------------------
    # 3. Promote Model
    # ---------------------------------------------------------
    with patch("sys.argv", ["promote", model_id, "-y"]):
        runpy.run_module("trading_bot.utils.promote", run_name="__main__")

    db = SessionLocal()
    model_log = db.query(ModelRegistryLog).filter_by(model_id=model_id).first()
    assert model_log.status == "production"
    db.close()

    # ---------------------------------------------------------
    # 4. Backtest (Bot Execution)
    # ---------------------------------------------------------
    task_config = {
        "name": "CLI E2E Backtest",
        "market_ids": ["BTC/USDT"],
        "strategies": [
            {
                "class_path": "nets.strategies.nets_strategy.NetsStrategy",
                "params": {
                    "lookback_period": 20,
                    "allow_in_sample": True,
                    "predictor": {
                        "class_path": "nets.inference.inference.ONNXPredictor",
                        "params": {"model_path": model_log.onnx_path},
                    },
                    "output_selector": {
                        "class_path": "nets.output_selectors.SimpleThresholdClassifier",
                        "params": {"threshold": 0.001},
                    },
                },
            }
        ],
        "sizing_strategy": {
            "class_path": "trading_bot.risk_management.sizing.fixed_amount.FixedAmountSizer",
            "params": {"default_amount_quote": 1000.0},
        },
        "execution_handler": {
            "class_path": "trading_bot.execution.handlers.simulated_handler.SimulatedExecutionHandler",
            "params": {
                "delay_model": {
                    "class_path": "trading_bot.execution.delay.KBarExecuteDelay",
                    "params": {"k": 1},
                },
                "slippage_model": {
                    "class_path": "trading_bot.execution.slippage.FlatPriceSlip",
                    "params": {"slippage_pct": 0.0},
                },
            },
        },
        "loop_driver": {
            "class_path": "trading_bot.core.loop.HistoricalReplayLoop",
            "params": {
                "save_backtest_report": True,
                "backtest_report_dir": str(tmp_path / "reports"),
                "data_reader": {
                    "class_path": "trading_bot.backtesting.readers.SQLBacktestDataReader",
                    "params": {
                        "instrument_id": "BTC/USDT",
                        "warmup_bars": 10,
                        "lookback_limit": 100,
                    },
                },
            },
        },
    }

    task_path = tmp_path / "cli_task.yaml"
    with open(task_path, "w") as f:
        yaml.safe_dump(task_config, f)

    with patch("sys.argv", ["trading_bot", "--config", str(task_path)]):
        runpy.run_module("trading_bot", run_name="__main__")

    reports_dir = tmp_path / "reports"
    report_files = list(reports_dir.glob("*.html"))
    assert len(report_files) == 1, "HTML Backtest report should have been generated"
