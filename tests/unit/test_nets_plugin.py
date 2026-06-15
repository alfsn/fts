# tests/unit/test_nets_plugin.py

from datetime import datetime

import numpy as np
import pytest
from nets.classifiers import DynamicThresholdClassifier, SimpleThresholdClassifier
from nets.enums import PredictionSignal

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


def test_classifiers():
    dummy = SimpleThresholdClassifier(threshold=0.01)
    assert dummy.classify(0.005, []) == PredictionSignal.FLAT
    assert dummy.classify(0.015, []) == PredictionSignal.UP
    assert dummy.classify(-0.015, []) == PredictionSignal.DOWN

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
    assert dynamic.classify(0.02, bars) == PredictionSignal.FLAT
    assert dynamic.classify(0.04, bars) == PredictionSignal.UP
    assert dynamic.classify(-0.04, bars) == PredictionSignal.DOWN


def test_pytorch_trainers(tmp_path):
    from nets.schemas import CNNConfig, LSTMConfig, NNTrainingConfig, RNNConfig
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
    from nets.schemas import CNNConfig, NNTrainingConfig
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
    from nets.schemas import BaseTrainerConfig
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
