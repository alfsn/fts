# tests/unit/test_nets_plugin.py

from datetime import datetime

import numpy as np
import pytest
from nets.enums import PredictionSignal
from nets.output_selectors import DynamicThresholdClassifier, SimpleThresholdClassifier

from trading_bot.core.enums import BarType
from trading_bot.core.schemas import BarData
from trading_bot.core.transforms import LogReturnTransform


def test_log_return_transform():
    transform = LogReturnTransform()
    prices = np.array([100.0, 110.0, 105.0]).reshape(-1, 1)
    returns = transform.transform(prices)

    # ln(110/100) = 0.0953
    # ln(105/110) = -0.0465
    assert len(returns) == 2
    assert pytest.approx(float(returns[0, 0]), 0.001) == 0.0953
    assert pytest.approx(float(returns[1, 0]), 0.001) == -0.0465


def test_threshold_selectors():
    dummy = SimpleThresholdClassifier(threshold=0.01)
    assert dummy.select_output(np.array([0.005]), [])[0] == PredictionSignal.FLAT
    assert dummy.select_output(np.array([0.015]), [])[0] == PredictionSignal.UP
    assert dummy.select_output(np.array([-0.015]), [])[0] == PredictionSignal.DOWN

    dynamic = DynamicThresholdClassifier(k=1.0, period=2)
    bars = [
        BarData(
            timestamp=datetime.now(),
            open=100,
            high=102,
            low=98,
            close=100,
            volume=1,
            bar_type=BarType.TIME,
            ticks_count=1,
            dollar_volume=100,
        ),
        BarData(
            timestamp=datetime.now(),
            open=100,
            high=101,
            low=99,
            close=100,
            volume=1,
            bar_type=BarType.TIME,
            ticks_count=1,
            dollar_volume=100,
        ),
    ]
    # ATR_pct = ((4/100) + (2/100)) / 2 = 0.03
    # Threshold = 1.0 * 0.03 = 0.03
    assert dynamic.select_output(np.array([0.02]), bars)[0] == PredictionSignal.FLAT
    assert dynamic.select_output(np.array([0.04]), bars)[0] == PredictionSignal.UP
    assert dynamic.select_output(np.array([-0.04]), bars)[0] == PredictionSignal.DOWN


def test_pytorch_trainers(tmp_path):
    from nets.models import CNNConfig, LSTMConfig, NNTrainingConfig, RNNConfig
    from nets.training import CNNTrainer, LSTMTrainer, RNNTrainer

    # Create dummy data
    bars = [
        BarData(
            timestamp=datetime.now(),
            open=100 + i * 0.1,
            high=101 + i * 0.1,
            low=99 + i * 0.1,
            close=100 + i * 0.1,
            volume=100,
            bar_type=BarType.TIME,
            ticks_count=1,
            dollar_volume=100,
        )
        for i in range(40)
    ]

    # CNN Trainer
    cnn_trainer = CNNTrainer(
        lookback_period=10,
        model_config=CNNConfig(
            out_channels=[8], kernel_sizes=[3], pool_sizes=[None], dense_units=16
        ),
        training_config=NNTrainingConfig(
            epochs=2, batch_size=4, tensorboard_log_dir=str(tmp_path / "cnn_tb")
        ),
    )
    onnx_bytes_cnn = cnn_trainer.train(bars)
    assert onnx_bytes_cnn is not None
    assert len(onnx_bytes_cnn) > 0

    # RNN Trainer
    rnn_trainer = RNNTrainer(
        lookback_period=10,
        model_config=RNNConfig(hidden_dim=8, num_layers=1),
        training_config=NNTrainingConfig(
            epochs=2, batch_size=4, tensorboard_log_dir=str(tmp_path / "rnn_tb")
        ),
    )
    onnx_bytes_rnn = rnn_trainer.train(bars)
    assert onnx_bytes_rnn is not None
    assert len(onnx_bytes_rnn) > 0

    # LSTM Trainer
    lstm_trainer = LSTMTrainer(
        lookback_period=10,
        model_config=LSTMConfig(hidden_dim=8, num_layers=1, bidirectional=True),
        training_config=NNTrainingConfig(
            epochs=2, batch_size=4, tensorboard_log_dir=str(tmp_path / "lstm_tb")
        ),
    )
    onnx_bytes_lstm = lstm_trainer.train(bars)
    assert onnx_bytes_lstm is not None
    assert len(onnx_bytes_lstm) > 0


def test_pytorch_trainers_validation_and_early_stopping(tmp_path):
    from nets.models import CNNConfig, NNTrainingConfig
    from nets.training import CNNTrainer

    # Create dummy data
    bars = [
        BarData(
            timestamp=datetime.now(),
            open=100 + i * 0.1,
            high=101 + i * 0.1,
            low=99 + i * 0.1,
            close=100 + i * 0.1,
            volume=100,
            bar_type=BarType.TIME,
            ticks_count=1,
            dollar_volume=100,
        )
        for i in range(50)
    ]

    # Set epochs to 50 but early stopping patience to 2 so it exits quickly
    cnn_trainer = CNNTrainer(
        lookback_period=10,
        model_config=CNNConfig(
            out_channels=[8], kernel_sizes=[3], pool_sizes=[None], dense_units=16
        ),
        training_config=NNTrainingConfig(
            epochs=50,
            batch_size=4,
            validation_split=0.2,
            early_stopping_patience=2,
            early_stopping_min_delta=0.0,
            clip_grad_norm=0.5,
            tensorboard_log_dir=str(tmp_path / "cnn_tb_es"),
        ),
    )
    onnx_bytes = cnn_trainer.train(bars)
    assert onnx_bytes is not None
    assert len(onnx_bytes) > 0


def test_sklearn_trainers(tmp_path):
    from nets.models import BaseTrainerConfig
    from nets.training import LinearRegressionTrainer, XGBoostTrainer

    bars = [
        BarData(
            timestamp=datetime.now(),
            open=100 + i * 0.1,
            high=101 + i * 0.1,
            low=99 + i * 0.1,
            close=100 + i * 0.1,
            volume=100,
            bar_type=BarType.TIME,
            ticks_count=1,
            dollar_volume=100,
        )
        for i in range(50)
    ]

    # Linear Regression Trainer
    lr_trainer = LinearRegressionTrainer(
        lookback_period=10,
        config=BaseTrainerConfig(
            validation_split=0.2,
            embargo_pct=0.01,
            horizon=1,
        ),
    )
    onnx_bytes_lr = lr_trainer.train(bars)
    assert onnx_bytes_lr is not None
    assert len(onnx_bytes_lr) > 0

    # XGBoost Trainer
    xgb_trainer = XGBoostTrainer(
        lookback_period=10,
        config=BaseTrainerConfig(
            validation_split=0.2,
            embargo_pct=0.01,
            horizon=1,
        ),
    )
    onnx_bytes_xgb = xgb_trainer.train(bars)
    assert onnx_bytes_xgb is not None
    assert len(onnx_bytes_xgb) > 0


def test_output_selectors():
    from nets.output_selectors import (
        ClassificationOutputSelector,
        QuantileOutputSelector,
    )

    # 1. ClassificationOutputSelector
    # Expected probs: [DOWN, FLAT, UP]
    ufd = ClassificationOutputSelector(
        [PredictionSignal.DOWN, PredictionSignal.FLAT, PredictionSignal.UP]
    )
    sig, conf = ufd.select_output(np.array([0.1, 0.2, 0.7]), [])
    assert sig == PredictionSignal.UP
    assert pytest.approx(conf) == 0.7

    sig, conf = ufd.select_output(np.array([0.6, 0.3, 0.1]), [])
    assert sig == PredictionSignal.DOWN
    assert pytest.approx(conf) == 0.6

    # 2. QuantileOutputSelector
    # Inputs: [q10, q50, q90]
    q_selector = QuantileOutputSelector(threshold=0.01, spread_scale=2.0)
    sig, conf = q_selector.select_output(np.array([-0.05, 0.02, 0.05]), [])
    assert sig == PredictionSignal.UP
    # spread = 0.05 - (-0.05) = 0.10. conf = exp(-0.10 * 2.0) = exp(-0.2) = 0.8187
    assert pytest.approx(conf) == np.exp(-0.2)

    # Flat region
    sig, conf = q_selector.select_output(np.array([-0.05, 0.005, 0.05]), [])
    assert sig == PredictionSignal.FLAT


def test_onnx_predictor_shaping():
    from unittest.mock import MagicMock

    from nets.inference import ONNXPredictor

    # Mock ort.InferenceSession
    mock_session = MagicMock()

    # Mock get_inputs
    mock_input = MagicMock()
    mock_input.name = "input"

    # 1. Test 3D input expected (e.g. PyTorch models: [batch, features, lookback])
    mock_input.shape = ["batch_size", 2, 10]
    mock_session.get_inputs.return_value = [mock_input]

    # Instantiate predictor and override session
    predictor = ONNXPredictor("dummy.onnx")
    predictor.session = mock_session
    predictor.input_metadata = {"input": mock_input}

    # Pass a 2D array of (lookback, features) = (10, 2)
    dummy_input = np.ones((10, 2))
    predictor.predict(dummy_input)

    # Check that session.run was called with shape (1, 2, 10)
    called_inputs = mock_session.run.call_args[0][1]
    assert "input" in called_inputs
    assert called_inputs["input"].shape == (1, 2, 10)

    # 2. Test 2D input expected (e.g. Sklearn/XGBoost models: [batch, lookback * features])
    mock_input.shape = ["batch_size", 20]
    predictor.predict(dummy_input)
    called_inputs = mock_session.run.call_args[0][1]
    assert called_inputs["input"].shape == (1, 20)


def test_multidimensional_training(tmp_path):
    from nets.models import BaseTrainerConfig, CNNConfig, NNTrainingConfig
    from nets.training import CNNTrainer, LinearRegressionTrainer

    # Create multi-feature data
    bars = [
        BarData(
            timestamp=datetime.now(),
            open=100 + i * 0.1,
            high=101 + i * 0.1,
            low=99 + i * 0.1,
            close=100 + i * 0.1,
            volume=100 + i,
            bar_type=BarType.TIME,
            ticks_count=1,
            dollar_volume=100,
        )
        for i in range(50)
    ]

    # Use close and volume as features
    config = BaseTrainerConfig(
        lookback_period=10,
        feature_cols=["close", "volume"],
        validation_split=0.2,
        embargo_pct=0.01,
        horizon=1,
    )

    lr_trainer = LinearRegressionTrainer(
        lookback_period=10,
        config=config,
    )
    onnx_bytes_lr = lr_trainer.train(bars)
    assert onnx_bytes_lr is not None
    assert len(onnx_bytes_lr) > 0

    # Test CNN with multiple features and output selector
    from nets.output_selectors import SimpleThresholdClassifier

    selector = SimpleThresholdClassifier(threshold=0.001)

    nn_config = NNTrainingConfig(
        lookback_period=10,
        feature_cols=["close", "volume"],
        validation_split=0.2,
        embargo_pct=0.01,
        horizon=1,
        epochs=2,
        batch_size=4,
        tensorboard_log_dir=str(tmp_path / "multi_tb"),
    )

    cnn_trainer = CNNTrainer(
        lookback_period=10,
        model_config=CNNConfig(
            out_channels=[8], kernel_sizes=[3], pool_sizes=[None], dense_units=16
        ),
        training_config=nn_config,
        output_selector=selector,
    )
    onnx_bytes_cnn = cnn_trainer.train(bars)
    assert onnx_bytes_cnn is not None
    assert len(onnx_bytes_cnn) > 0
