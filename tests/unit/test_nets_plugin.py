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
    from unittest.mock import MagicMock, patch

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
    with patch("onnxruntime.InferenceSession", return_value=mock_session):
        predictor = ONNXPredictor("dummy.onnx")
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


def test_confidence_sizer_zero_size():
    from nets.sizing.confidence_sizer import ConfidenceSizer

    from trading_bot.core.enums import BarType, SignalType
    from trading_bot.core.schemas import (
        MarketData,
        MarketDetails,
        PortfolioState,
        SizingInput,
        TradeSignal,
    )

    sizer = ConfidenceSizer(base_amount_quote=1000.0)

    # Mock input data
    signal_flat = TradeSignal(
        market_id="BTC_USD",
        strategy_name="test_strat",
        signal_type=SignalType.FLAT,
        confidence=0.5,
    )
    signal_hold = TradeSignal(
        market_id="BTC_USD",
        strategy_name="test_strat",
        signal_type=SignalType.HOLD,
        confidence=0.5,
    )
    signal_buy = TradeSignal(
        market_id="BTC_USD",
        strategy_name="test_strat",
        signal_type=SignalType.BUY,
        confidence=0.5,
    )

    bars = [
        BarData(
            timestamp=datetime.now(),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1.0,
            bar_type=BarType.TIME,
            ticks_count=1,
            dollar_volume=100.0,
        )
    ]

    details = MarketDetails(
        market_id="BTC_USD",
        name="BTC_USD",
        end_date=datetime.now(),
        resolution_source="test",
    )
    market_data = MarketData(market_id="BTC_USD", recent_bars=bars, details=details)

    portfolio = PortfolioState(
        total_balance_quote=10000.0,
        available_balance_quote=10000.0,
        positions=[],
        open_orders=[],
    )

    input_flat = SizingInput(
        signal=signal_flat, market_data=market_data, portfolio_state=portfolio
    )
    input_hold = SizingInput(
        signal=signal_hold, market_data=market_data, portfolio_state=portfolio
    )
    input_buy = SizingInput(
        signal=signal_buy, market_data=market_data, portfolio_state=portfolio
    )

    output_flat = sizer.calculate_size(input_flat)
    output_hold = sizer.calculate_size(input_hold)
    output_buy = sizer.calculate_size(input_buy)

    assert output_flat.amount_quote == 0.0
    assert output_flat.size_shares == 0.0
    assert output_hold.amount_quote == 0.0
    assert output_hold.size_shares == 0.0

    assert output_buy.amount_quote == 500.0
    assert output_buy.size_shares == 5.0


def test_calculate_atr_pct_with_gaps():
    from nets.output_selectors.output_selectors import calculate_atr_pct

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
        # Gap up: prev close was 100, now low is 105, high is 110, close is 108
        # High-low range = 5.
        # Gapped true range = max(5, 110-100, |105-100|) = 10.
        # Prev close denom = 100. So TR_pct = 10 / 100 = 0.10.
        BarData(
            timestamp=datetime.now(),
            open=106,
            high=110,
            low=105,
            close=108,
            volume=1,
            bar_type=BarType.TIME,
            ticks_count=1,
            dollar_volume=108,
        ),
    ]

    # calculate_atr_pct over 1 period
    atr_pct = calculate_atr_pct(bars, period=1)
    # Expected TR_pct = 0.10
    assert pytest.approx(atr_pct) == 0.10


def test_bidirectional_lstm_slicing():
    import torch
    from nets.models import LSTMConfig, SimpleLSTM

    config = LSTMConfig(hidden_dim=4, num_layers=1, bidirectional=True)
    # 2 features, mean/std zero/one
    model = SimpleLSTM(
        input_dim=5, n_features=2, config=config, mean=[0.0, 0.0], std=[1.0, 1.0]
    )

    # Input shape: [batch, features, seq_len] = [2, 2, 5]
    x = torch.randn(2, 2, 5)
    out = model(x)
    # Since bidirectional=True and hidden_dim=4, the dense layer gets features of dim 8
    # and outputs shape [batch, 1] = [2, 1]
    assert out.shape == (2, 1)


def test_trainers_with_multi_step_horizon(tmp_path):
    from nets.models import BaseTrainerConfig, CNNConfig, NNTrainingConfig
    from nets.training import CNNTrainer, LinearRegressionTrainer, XGBoostTrainer

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

    # 1. LinearRegressionTrainer with horizon = 3
    config = BaseTrainerConfig(lookback_period=10, validation_split=0.2, horizon=3)
    lr_trainer = LinearRegressionTrainer(lookback_period=10, config=config)
    onnx_bytes_lr = lr_trainer.train(bars)
    assert onnx_bytes_lr is not None

    # 2. XGBoostTrainer with horizon = 3
    xgb_trainer = XGBoostTrainer(lookback_period=10, config=config)
    onnx_bytes_xgb = xgb_trainer.train(bars)
    assert onnx_bytes_xgb is not None

    # 3. CNNTrainer with horizon = 3
    nn_config = NNTrainingConfig(
        lookback_period=10,
        validation_split=0.2,
        horizon=3,
        epochs=2,
        batch_size=4,
        tensorboard_log_dir=str(tmp_path / "cnn_tb_h3"),
    )
    cnn_trainer = CNNTrainer(
        lookback_period=10,
        model_config=CNNConfig(
            out_channels=[4], kernel_sizes=[3], pool_sizes=[None], dense_units=8
        ),
        training_config=nn_config,
    )
    onnx_bytes_cnn = cnn_trainer.train(bars)
    assert onnx_bytes_cnn is not None
