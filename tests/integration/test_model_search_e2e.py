import os
from datetime import datetime, timedelta, timezone

import pytest
import yaml
from nets.training.hparam_search import run_hparam_search

from trading_bot.config import settings
from trading_bot.core.database import Base, create_db_session, init_db
from trading_bot.core.enums import BarType
from trading_bot.core.repository import MarketDataRepository, ModelRepository
from trading_bot.core.schemas import BarData, MarketDetails


@pytest.fixture
def temp_db_and_config(tmp_path):
    # 1. Override settings DATABASE_URL to a temporary SQLite file
    db_file = tmp_path / "test_dev.db"
    original_db_url = settings.DATABASE_URL
    settings.DATABASE_URL = f"sqlite+pysqlite:///{db_file}"

    # Initialize the database schema on the temporary database
    from sqlalchemy import create_engine

    engine = create_engine(settings.DATABASE_URL)
    init_db(extra_models=["trading_bot.core.models"], bind_engine=engine)

    # 2. Add mock bar data so training can run
    db = create_db_session(settings.DATABASE_URL)
    market_repo = MarketDataRepository(db)

    # Ensure market exists
    market_repo.ensure_market(
        MarketDetails(
            market_id="BTC_USD",
            name="Bitcoin/USD Test",
            end_date=datetime.now(timezone.utc) + timedelta(days=10),
            resolution_source="test",
        )
    )

    bars = []
    base_time = datetime.now(timezone.utc) - timedelta(days=5)
    for i in range(100):
        bars.append(
            BarData(
                timestamp=base_time + timedelta(hours=i),
                open=100.0 + i * 0.5,
                high=101.0 + i * 0.5,
                low=99.0 + i * 0.5,
                close=100.2 + i * 0.5,
                volume=10.0,
                bar_type=BarType.TIME,
                interval="1h",
                ticks_count=10,
                dollar_volume=1000.0,
            )
        )
    market_repo.save_bars("BTC_USD", bars)
    db.commit()
    db.close()

    # 3. Create a small search config
    config_dict = {
        "study_name": "test_search",
        "direction": "minimize",
        "n_trials": 2,  # Keep it small for fast tests
        "model_type": "lstm",
        "market_id": "BTC_USD",
        "interval": "1h",
        "lookback_period": 10,
        "feature_cols": ["close"],
        "search_space": {
            "learning_rate": {"type": "float", "low": 0.001, "high": 0.005},
            "hidden_dim": {"type": "int", "low": 8, "high": 16},
            "num_layers": {"type": "int", "low": 1, "high": 1},
            "dropout": {"type": "float", "low": 0.0, "high": 0.0},
            "epochs": {"type": "int", "low": 1, "high": 1},
        },
    }

    config_file = tmp_path / "search_config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config_dict, f)

    try:
        yield config_file
    finally:
        # Restore settings
        settings.DATABASE_URL = original_db_url


def test_hparam_search_and_promotion_integration(temp_db_and_config, tmp_path):
    # Run the search
    run_hparam_search(str(temp_db_and_config))

    # Connect to db and verify models were registered
    db = create_db_session(settings.DATABASE_URL)
    model_repo = ModelRepository(db)

    # Verify all registry rows
    from trading_bot.core.models import ModelRegistryLog

    models = db.query(ModelRegistryLog).all()
    assert len(models) == 2  # n_trials is 2

    # Check candidates exist
    for m in models:
        assert m.status == "candidate"
        assert m.model_type == "lstm"
        assert m.market_id == "BTC_USD"
        assert m.interval == "1h"
        assert m.horizon == 1
        assert os.path.exists(m.onnx_path)

    # Let's promote the best candidate
    best_candidate = models[0]
    model_repo.promote_to_production(best_candidate.model_id)
    db.commit()

    # Retrieve production model
    prod = model_repo.get_production_model(
        model_type="lstm", market_id="BTC_USD", interval="1h", horizon=1
    )
    assert prod is not None
    assert prod.model_id == best_candidate.model_id
    assert prod.status == "production"

    db.close()


def test_all_config_files_integration(temp_db_and_config, tmp_path):
    configs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "configs",
    )

    model_types = ["lstm", "rnn", "cnn", "linear_regression", "xgboost"]

    for model_type in model_types:
        config_path = os.path.join(
            configs_dir, "train", "BTCUSDT", f"{model_type}_hparam_search.yaml"
        )
        assert os.path.exists(config_path), f"Config file not found: {config_path}"

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Override configurations to run extremely fast in tests
        config_dict["n_trials"] = 1
        config_dict["study_name"] = f"test_study_{model_type}"
        config_dict["market_id"] = "BTC_USD"
        config_dict["interval"] = "1h"

        # neural network specific scaling-down for testing speed
        if "search_space" in config_dict:
            if "epochs" in config_dict["search_space"]:
                config_dict["search_space"]["epochs"]["low"] = 1
                config_dict["search_space"]["epochs"]["high"] = 1
            if "hidden_dim" in config_dict["search_space"]:
                config_dict["search_space"]["hidden_dim"]["low"] = 8
                config_dict["search_space"]["hidden_dim"]["high"] = 8
            if "dense_units" in config_dict["search_space"]:
                config_dict["search_space"]["dense_units"]["low"] = 8
                config_dict["search_space"]["dense_units"]["high"] = 8

        temp_config_file = tmp_path / f"{model_type}_test_config.yaml"
        with open(temp_config_file, "w") as f:
            yaml.safe_dump(config_dict, f)

        # Run the search
        run_hparam_search(str(temp_config_file))

        # Verify candidate was registered in DB
        db = create_db_session(settings.DATABASE_URL)
        from trading_bot.core.models import ModelRegistryLog

        models = db.query(ModelRegistryLog).filter_by(model_type=model_type).all()
        assert len(models) == 1, f"Failed to find registered model for {model_type}"
        m = models[0]
        assert m.status == "candidate"
        assert m.market_id == "BTC_USD"
        assert m.interval == "1h"
        assert os.path.exists(m.onnx_path)

        db.close()
