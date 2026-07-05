from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from trading_bot.core.database import Base
from trading_bot.core.models import ModelRegistryLog, TimeSeriesDataset
from trading_bot.core.repository import ModelRepository


@pytest.fixture
def test_db():
    """Sets up an in-memory SQLite database and creates the schema."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionClass = sessionmaker(bind=engine)
    session = SessionClass()
    try:
        yield session
    finally:
        session.close()


def test_model_registration_and_retrieval(test_db):
    repo = ModelRepository(test_db)

    # 1. Create a TimeSeriesDataset entry
    dataset_hash = "abc123xyz7890000000000000000000000000000000000000000000000000000"
    dataset = repo.get_or_create_dataset(
        market_id="BTC_USD",
        interval="1h",
        start_time=datetime(2026, 6, 27, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 6, 27, 12, 0, tzinfo=timezone.utc),
        hash_val=dataset_hash,
    )
    test_db.commit()

    # Assert dataset_id is ds_<12_char_hash>
    assert dataset.dataset_id == "ds_abc123xyz789"
    assert dataset.hash == dataset_hash

    # 2. Register candidate model linked to dataset
    hparams = {"learning_rate": 0.001, "hidden_dim": 32}
    metrics = {"loss": 0.05, "ic": 0.12}
    model_id = "a1b2c3d4e5f6"

    repo.register_model(
        model_id=model_id,
        model_type="lstm",
        market_id="BTC_USD",
        interval="1h",
        horizon=1,
        onnx_path="models/registry/trials/test.onnx",
        hyperparameters=hparams,
        metrics=metrics,
        run_id="run_123",
        status="candidate",
        dataset_id=dataset.dataset_id,
    )
    test_db.commit()

    # 3. Retrieve model by ID and verify fields & relationship
    model = repo.get_model(model_id)
    assert model is not None
    assert model.model_id == model_id
    assert model.model_type == "lstm"
    assert model.market_id == "BTC_USD"
    assert model.interval == "1h"
    assert model.horizon == 1
    assert model.onnx_path == "models/registry/trials/test.onnx"
    assert model.hyperparameters == hparams
    assert model.metrics == metrics
    assert model.status == "candidate"
    assert model.dataset_id == "ds_abc123xyz789"
    assert model.dataset.market_id == "BTC_USD"
    assert model.dataset.hash == dataset_hash


def test_model_promotion_flow(test_db, tmp_path):
    repo = ModelRepository(test_db)

    hparams = {"learning_rate": 0.001}
    metrics = {"loss": 0.05}

    # Setup temporary trials and registry directories
    trials_dir = tmp_path / "registry" / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    model_id_1 = "111111111111"
    model_id_2 = "222222222222"

    onnx_path_1 = trials_dir / f"{model_id_1}.onnx"
    onnx_path_2 = trials_dir / f"{model_id_2}.onnx"

    # Write dummy bytes
    onnx_path_1.write_bytes(b"onnx_model_1_bytes")
    onnx_path_2.write_bytes(b"onnx_model_2_bytes")

    # Register two models under the same target signature
    repo.register_model(
        model_id=model_id_1,
        model_type="lstm",
        market_id="BTC_USD",
        interval="1h",
        horizon=1,
        onnx_path=str(onnx_path_1),
        hyperparameters=hparams,
        metrics=metrics,
        status="candidate",
    )
    repo.register_model(
        model_id=model_id_2,
        model_type="lstm",
        market_id="BTC_USD",
        interval="1h",
        horizon=1,
        onnx_path=str(onnx_path_2),
        hyperparameters=hparams,
        metrics=metrics,
        status="candidate",
    )
    test_db.commit()

    # Verify no production model exists initially
    prod_model = repo.get_production_model(
        model_type="lstm", market_id="BTC_USD", interval="1h", horizon=1
    )
    assert prod_model is None

    # Promote first model
    repo.promote_to_production(model_id_1)
    test_db.commit()

    # Verify model_1 is now production, status updated, path promoted
    prod_model = repo.get_production_model(
        model_type="lstm", market_id="BTC_USD", interval="1h", horizon=1
    )
    assert prod_model is not None
    assert prod_model.model_id == model_id_1
    assert prod_model.status == "production"

    expected_perm_path = tmp_path / "registry" / f"{model_id_1}.onnx"
    assert prod_model.onnx_path == str(expected_perm_path)
    assert expected_perm_path.exists()
    assert expected_perm_path.read_bytes() == b"onnx_model_1_bytes"

    # Verify model_2 is cleaned up from DB (since it was candidate under same signature)
    m2 = repo.get_model(model_id_2)
    assert m2 is None

    # Verify model_2's ONNX file is removed from disk
    assert not onnx_path_2.exists()


def test_partial_unique_index_constraint(test_db):
    # Try to manually insert two models marked as production for the same signature
    # This should fail due to the partial unique index uq_production_model_signature
    m1 = ModelRegistryLog(
        model_id="model_1",
        model_type="lstm",
        market_id="BTC_USD",
        interval="1h",
        horizon=1,
        onnx_path="test.onnx",
        hyperparameters={},
        metrics={},
        status="production",
    )
    m2 = ModelRegistryLog(
        model_id="model_2",
        model_type="lstm",
        market_id="BTC_USD",
        interval="1h",
        horizon=1,
        onnx_path="test2.onnx",
        hyperparameters={},
        metrics={},
        status="production",
    )

    test_db.add(m1)
    test_db.add(m2)

    with pytest.raises(IntegrityError):
        test_db.commit()


def test_register_duplicate_model_id_idempotent(test_db):
    repo = ModelRepository(test_db)
    model_id = "dup_model_123"

    m1 = repo.register_model(
        model_id=model_id,
        model_type="linear_regression",
        market_id="BTC/USDT",
        interval="30m",
        horizon=1,
        onnx_path="test_path_1.onnx",
        hyperparameters={},
        metrics={"loss": 0.01},
        run_id="run_1",
    )
    test_db.commit()

    # Attempt duplicate registration with same model_id
    m2 = repo.register_model(
        model_id=model_id,
        model_type="linear_regression",
        market_id="BTC/USDT",
        interval="30m",
        horizon=1,
        onnx_path="test_path_2.onnx",
        hyperparameters={},
        metrics={"loss": 0.02},
        run_id="run_2",
    )
    test_db.commit()

    assert m2.model_id == model_id
    assert m2.onnx_path == "test_path_1.onnx"  # Preserved original entry


def test_model_retrieval_by_feature_cols(test_db):
    repo = ModelRepository(test_db)

    # Register candidate model with close-only features
    repo.register_model(
        model_id="model_close_123",
        model_type="lstm",
        market_id="BTC/USDT",
        interval="30m",
        horizon=1,
        onnx_path="models/close.onnx",
        hyperparameters={"feature_cols": ["close"]},
        metrics={"loss": 0.1},
        status="candidate",
    )

    # Register candidate model with OHLCV features
    repo.register_model(
        model_id="model_ohlcv_456",
        model_type="lstm",
        market_id="BTC/USDT",
        interval="30m",
        horizon=1,
        onnx_path="models/ohlcv.onnx",
        hyperparameters={"feature_cols": ["open", "high", "low", "close", "volume"]},
        metrics={"loss": 0.05},
        status="candidate",
    )
    test_db.commit()

    # Query for candidate matching OHLCV features
    ohlcv_model = repo.get_candidate_model(
        model_type="lstm",
        market_id="BTC/USDT",
        interval="30m",
        horizon=1,
        feature_cols=["open", "high", "low", "close", "volume"],
    )
    assert ohlcv_model is not None
    assert ohlcv_model.model_id == "model_ohlcv_456"

    # Query for candidate matching close features
    close_model = repo.get_candidate_model(
        model_type="lstm",
        market_id="BTC/USDT",
        interval="30m",
        horizon=1,
        feature_cols=["close"],
    )
    assert close_model is not None
    assert close_model.model_id == "model_close_123"

    # Query for non-existent feature set returns None
    missing_model = repo.get_candidate_model(
        model_type="lstm",
        market_id="BTC/USDT",
        interval="30m",
        horizon=1,
        feature_cols=["open", "close"],
    )
    assert missing_model is None
