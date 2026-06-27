import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from trading_bot.core.database import Base
from trading_bot.core.models import ModelRegistryLog
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

    # 1. Register candidate model
    hparams = {"learning_rate": 0.001, "hidden_dim": 32}
    metrics = {"loss": 0.05, "ic": 0.12}
    model_id = "model_lstm_btcusd_1h_20260627_120000_abc123"

    repo.register_model(
        model_id=model_id,
        model_type="lstm",
        market_id="BTC_USD",
        interval="1h",
        horizon=1,
        onnx_path="models/registry/test.onnx",
        hyperparameters=hparams,
        metrics=metrics,
        run_id="run_123",
        status="candidate",
    )
    test_db.commit()

    # 2. Retrieve model by ID
    model = repo.get_model(model_id)
    assert model is not None
    assert model.model_id == model_id
    assert model.model_type == "lstm"
    assert model.market_id == "BTC_USD"
    assert model.interval == "1h"
    assert model.horizon == 1
    assert model.onnx_path == "models/registry/test.onnx"
    assert model.hyperparameters == hparams
    assert model.metrics == metrics
    assert model.status == "candidate"


def test_model_promotion_flow(test_db):
    repo = ModelRepository(test_db)

    hparams = {"learning_rate": 0.001}
    metrics = {"loss": 0.05}

    # Register two models under the same target signature
    model_id_1 = "model_lstm_btcusd_1h_20260627_120000_1"
    model_id_2 = "model_lstm_btcusd_1h_20260627_120000_2"

    repo.register_model(
        model_id=model_id_1,
        model_type="lstm",
        market_id="BTC_USD",
        interval="1h",
        horizon=1,
        onnx_path="models/registry/test1.onnx",
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
        onnx_path="models/registry/test2.onnx",
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

    prod_model = repo.get_production_model(
        model_type="lstm", market_id="BTC_USD", interval="1h", horizon=1
    )
    assert prod_model is not None
    assert prod_model.model_id == model_id_1
    assert prod_model.status == "production"

    # Promote second model (should archive the first)
    repo.promote_to_production(model_id_2)
    test_db.commit()

    # Verify model_2 is now production
    prod_model = repo.get_production_model(
        model_type="lstm", market_id="BTC_USD", interval="1h", horizon=1
    )
    assert prod_model is not None
    assert prod_model.model_id == model_id_2
    assert prod_model.status == "production"

    # Verify model_1 is archived
    m1 = repo.get_model(model_id_1)
    assert m1.status == "archived"


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
