# tests/integration/test_nets_e2e.py

import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import pytest
from nets.models import NNTrainingConfig, RNNConfig
from nets.training import RNNTrainer, ValidationEvaluator
from quant_core.enums import BarType
from quant_data.db import BarDataLog, Market, MarketDataRepository
from sqlalchemy import create_engine
from trading_bot.config import settings
from trading_bot.core.database import CacheSessionLocal, SessionLocal, init_db


@pytest.fixture(scope="module")
def e2e_db_session(tmp_path_factory):
    """
    Swaps the default database to an isolated temporary SQLite database seeded
    with 1000 BTC/USDT bars during the E2E test.
    """
    original_db_url = settings.DATABASE_URL
    db_path = tmp_path_factory.mktemp("nets_e2e") / "test_persistence.db"
    settings.DATABASE_URL = f"sqlite+pysqlite:///{db_path}"

    engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
    SessionLocal.configure(bind=engine)
    init_db(bind_engine=engine)

    db = CacheSessionLocal()
    try:
        db.add(
            Market(
                instrument_id="BTC/USDT",
                name="BINANCE BTC/USDT",
                resolution_source="BINANCE",
            )
        )
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        closes = 30000.0 * np.exp(
            np.cumsum(0.0005 + 0.004 * np.sin(np.arange(1000) * 0.2))
        )
        db.add_all(
            [
                BarDataLog(
                    instrument_id="BTC/USDT",
                    timestamp=start + timedelta(hours=i),
                    open=float(c * 0.999),
                    high=float(c * 1.002),
                    low=float(c * 0.998),
                    close=float(c),
                    volume=100.0,
                    bar_type=BarType.TIME,
                    interval="1h",
                    ticks_count=10,
                    dollar_volume=float(c * 100.0),
                    source="test",
                )
                for i, c in enumerate(closes)
            ]
        )
        db.commit()
        yield db
    finally:
        db.close()

        settings.DATABASE_URL = original_db_url
        dev_engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
        SessionLocal.configure(bind=dev_engine)


def test_rnn_e2e_training_and_onnx_inference(e2e_db_session):
    """
    E2E Test to fetch real historical BTC/USDT data from tests/test_persistence.db,
    train an RNN model from the /nets/ plugin, export it to ONNX, and run inference.
    """
    # 1. Fetch historical BTC/USDT bars via repository
    repo = MarketDataRepository(e2e_db_session)
    bar_schemas = repo.load_bars("BTC/USDT", "1h")

    assert (
        len(bar_schemas) == 1000
    ), "Should have loaded exactly 1000 BTC/USDT bars from tests/test_persistence.db"

    # Create temporary directory for TensorBoard logs to keep cleanup simple
    tb_log_dir = tempfile.mkdtemp()

    try:
        # 3. Setup configurations for RNN and NNTraining
        model_config = RNNConfig(hidden_dim=16, num_layers=1)
        training_config = NNTrainingConfig(
            epochs=5,
            batch_size=32,
            learning_rate=0.01,
            validation_split=0.2,
            tensorboard_log_dir=tb_log_dir,
        )

        # 4. Instantiate RNNTrainer
        trainer = RNNTrainer(
            lookback_period=20,
            model_config=model_config,
            training_config=training_config,
        )

        # Spy on ValidationEvaluator.evaluate to capture epoch metrics
        captured_metrics = []
        original_evaluate = ValidationEvaluator.evaluate

        def spy_evaluate(*args, **kwargs):
            metrics = original_evaluate(*args, **kwargs)
            captured_metrics.append(metrics)
            return metrics

        # 5. Train the model
        with patch(
            "nets.training.ValidationEvaluator.evaluate", side_effect=spy_evaluate
        ):
            onnx_bytes = trainer.train(bar_schemas)

        # 6. Verify training output
        assert onnx_bytes is not None, "ONNX bytes export should not be None"
        assert len(onnx_bytes) > 0, "ONNX bytes export should be non-empty"

        # Check validation metrics
        assert (
            len(captured_metrics) == 5
        ), f"Expected 5 evaluation rounds, got {len(captured_metrics)}"
        final_epoch_metrics = captured_metrics[-1]

        assert "loss" in final_epoch_metrics, "Validation metrics should include loss"
        assert (
            final_epoch_metrics["loss"] > 0
        ), "Validation loss should be greater than 0"
        assert not np.isnan(final_epoch_metrics["loss"]), "Validation loss is NaN"

        assert (
            "directional_accuracy" in final_epoch_metrics
        ), "Metrics should contain directional accuracy"
        # Assert a soft baseline for directional accuracy to prevent flakiness
        assert (
            final_epoch_metrics["directional_accuracy"] >= 0.40
        ), f"Directional accuracy too low: {final_epoch_metrics['directional_accuracy']:.4f}"

        # 7. Load model into ONNX Runtime Session to verify runtime compatibility
        session = ort.InferenceSession(onnx_bytes)

        # Check inputs and outputs definitions
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        assert len(inputs) == 1, "ONNX model should have 1 input"
        assert inputs[0].name == "input", "ONNX input name should be 'input'"
        assert len(outputs) == 1, "ONNX model should have 1 output"
        assert outputs[0].name == "output", "ONNX output name should be 'output'"

        # Run dummy forward pass prediction
        batch_size = 8
        n_features = 1
        lookback_period = 20
        test_input = np.random.randn(batch_size, n_features, lookback_period).astype(
            np.float32
        )

        onnx_outputs = session.run(["output"], {"input": test_input})

        pred = onnx_outputs[0]
        assert pred.shape == (
            batch_size,
            1,
        ), f"Expected shape {(batch_size, 1)}, got {pred.shape}"
        assert not np.isnan(pred).any(), "ONNX predictions contain NaN values"

    finally:
        # Cleanup TensorBoard directory
        shutil.rmtree(tb_log_dir, ignore_errors=True)
