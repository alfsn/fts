# src/plugins/nets/training.py

import io
import logging
from typing import Any, Dict, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skl2onnx import to_onnx
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trading_bot.core.dataset import DatasetBuilder
from trading_bot.core.schemas import BarData
from trading_bot.core.training import BaseModelTrainer
from trading_bot.core.transforms import LogReturnTransform

from .models import SimpleCNN, SimpleLSTM, SimpleRNN
from .schemas import CNNConfig, LSTMConfig, NNTrainingConfig, RNNConfig

logger = logging.getLogger(__name__)


class LinearRegressionTrainer(BaseModelTrainer):
    """
    Trains a Linear Regression model with a StandardScaler pipeline and exports to ONNX.
    """

    def __init__(self, lookback_period: int = 20) -> None:
        self.lookback_period = lookback_period
        self.transform = LogReturnTransform()

    def train(self, historical_bars: Sequence[BarData]) -> Any:
        logger.info(f"Training LinearRegression on {len(historical_bars)} bars.")

        # 1. Structural Conversion (SOLID: Extract window logic to Core)
        matrix = DatasetBuilder.to_matrix(historical_bars, feature_cols=["close"])

        # Apply LogReturn (Python-side preprocessing for now)
        returns = self.transform.transform(matrix)

        if len(returns) < self.lookback_period + 1:
            logger.error("Insufficient data for training.")
            return None

        # Create sliding windows
        X_raw, y_raw = DatasetBuilder.create_sliding_windows(
            returns, lookback=self.lookback_period, horizon=1
        )

        # Reshape X for sklearn (samples, lookback * features)
        X = X_raw.reshape(X_raw.shape[0], -1)
        y = y_raw.flatten()

        # 2. Pipeline Definition (SOLID: Preprocessing in Pipeline)
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        )

        # 3. Fit
        pipeline.fit(X, y)

        # 4. Export to ONNX
        onx = to_onnx(pipeline, X[:1])
        return onx.SerializeToString()


class XGBoostTrainer(BaseModelTrainer):
    """
    Trains an XGBoost model with a StandardScaler pipeline and exports to ONNX.
    """

    def __init__(self, lookback_period: int = 20) -> None:
        self.lookback_period = lookback_period
        self.transform = LogReturnTransform()

    def train(self, historical_bars: Sequence[BarData]) -> Any:
        import xgboost as xgb

        logger.info(f"Training XGBoost on {len(historical_bars)} bars.")

        matrix = DatasetBuilder.to_matrix(historical_bars, feature_cols=["close"])
        returns = self.transform.transform(matrix)

        if len(returns) < self.lookback_period + 1:
            return None

        X_raw, y_raw = DatasetBuilder.create_sliding_windows(
            returns, lookback=self.lookback_period, horizon=1
        )
        X = X_raw.reshape(X_raw.shape[0], -1)
        y = y_raw.flatten()

        # XGBoost Pipeline
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", xgb.XGBRegressor(n_estimators=100, max_depth=3)),
            ]
        )

        pipeline.fit(X, y)

        from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
            convert_xgboost as convert_xgboost_op,
        )
        from skl2onnx import update_registered_converter
        from skl2onnx.common.shape_calculator import (
            calculate_linear_regressor_output_shapes,
        )

        update_registered_converter(
            xgb.XGBRegressor,
            "XGBRegressor",
            calculate_linear_regressor_output_shapes,
            convert_xgboost_op,
        )

        onx = to_onnx(pipeline, X[:1])
        return onx.SerializeToString()


class BasePyTorchTrainer(BaseModelTrainer):
    """
    Abstract trainer base for PyTorch models that handles dataset creation,
    optional TensorBoard logging, training loops, and exporting to ONNX.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        training_config: Union[NNTrainingConfig, Dict[str, Any], None] = None,
    ) -> None:
        self.lookback_period = lookback_period
        self.transform = LogReturnTransform()
        self.training_config = (
            training_config
            if isinstance(training_config, NNTrainingConfig)
            else NNTrainingConfig(**(training_config or {}))
        )

    def _prepare_data(
        self, historical_bars: Sequence[BarData]
    ) -> Union[None, tuple[np.ndarray, np.ndarray, float, float]]:
        matrix = DatasetBuilder.to_matrix(historical_bars, feature_cols=["close"])
        returns = self.transform.transform(matrix)

        if len(returns) < self.lookback_period + 1:
            logger.error("Insufficient data for PyTorch neural network training.")
            return None

        X_raw, y_raw = DatasetBuilder.create_sliding_windows(
            returns, lookback=self.lookback_period, horizon=1
        )

        # Calculate mean & std from raw features for internal model scaling
        X_mean = float(X_raw.mean())
        X_std = float(X_raw.std() + 1e-8)

        return X_raw, y_raw, X_mean, X_std

    def _train_model(
        self,
        model: nn.Module,
        X_raw: np.ndarray,
        y_raw: np.ndarray,
        dummy_input: torch.Tensor,
    ) -> None:
        # X_pt shape: [batch_size, 1, seq_len]
        X_pt = torch.tensor(X_raw, dtype=torch.float32).transpose(1, 2)
        y_pt = torch.tensor(y_raw.flatten(), dtype=torch.float32).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X_pt, y_pt)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.training_config.batch_size, shuffle=True
        )

        # Loss function
        if self.training_config.loss_fn == "mse":
            criterion = nn.MSELoss()
        elif self.training_config.loss_fn == "mae":
            criterion = nn.L1Loss()
        elif self.training_config.loss_fn == "huber":
            criterion = nn.HuberLoss()
        else:
            criterion = nn.MSELoss()

        # Optimizer
        if self.training_config.optimizer == "adam":
            optimizer = optim.Adam(
                model.parameters(), lr=self.training_config.learning_rate
            )
        elif self.training_config.optimizer == "sgd":
            optimizer = optim.SGD(
                model.parameters(), lr=self.training_config.learning_rate
            )
        elif self.training_config.optimizer == "rmsprop":
            optimizer = optim.RMSprop(
                model.parameters(), lr=self.training_config.learning_rate
            )
        else:
            optimizer = optim.Adam(
                model.parameters(), lr=self.training_config.learning_rate
            )

        # Optional TensorBoard logging
        writer = None
        if self.training_config.tensorboard_log_dir:
            import os

            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.training_config.tensorboard_log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=self.training_config.tensorboard_log_dir)
            try:
                writer.add_graph(model, dummy_input)
            except Exception as e:
                logger.warning(f"Could not log model graph to TensorBoard: {e}")

        global_step = 0
        for epoch in range(self.training_config.epochs):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                if writer:
                    writer.add_scalar("Loss/train_step", loss_val, global_step)
                global_step += 1

            epoch_loss /= len(dataloader)
            if writer:
                writer.add_scalar("Loss/train_epoch", epoch_loss, epoch)

        if writer:
            writer.close()

    def _export_to_onnx(self, model: nn.Module) -> bytes:
        model.eval()
        dummy_input = torch.randn(1, 1, self.lookback_period)
        f = io.BytesIO()
        torch.onnx.export(
            model,
            dummy_input,
            f,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        return f.getvalue()


class CNNTrainer(BasePyTorchTrainer):
    """
    Trains a CNN model and exports to ONNX.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        model_config: Union[CNNConfig, Dict[str, Any], None] = None,
        training_config: Union[NNTrainingConfig, Dict[str, Any], None] = None,
    ) -> None:
        super().__init__(
            lookback_period=lookback_period, training_config=training_config
        )
        self.model_config = (
            model_config
            if isinstance(model_config, CNNConfig)
            else CNNConfig(**(model_config or {}))
        )

    def train(self, historical_bars: Sequence[BarData]) -> Any:
        logger.info(f"Training CNN on {len(historical_bars)} bars.")
        data = self._prepare_data(historical_bars)
        if data is None:
            return None

        X_raw, y_raw, X_mean, X_std = data

        # Instantiate SimpleCNN
        model = SimpleCNN(
            input_dim=self.lookback_period,
            config=self.model_config,
            mean=X_mean,
            std=X_std,
        )

        dummy_input = torch.randn(1, 1, self.lookback_period)
        self._train_model(model, X_raw, y_raw, dummy_input)
        return self._export_to_onnx(model)


class RNNTrainer(BasePyTorchTrainer):
    """
    Trains an Elman RNN model and exports to ONNX.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        model_config: Union[RNNConfig, Dict[str, Any], None] = None,
        training_config: Union[NNTrainingConfig, Dict[str, Any], None] = None,
    ) -> None:
        super().__init__(
            lookback_period=lookback_period, training_config=training_config
        )
        self.model_config = (
            model_config
            if isinstance(model_config, RNNConfig)
            else RNNConfig(**(model_config or {}))
        )

    def train(self, historical_bars: Sequence[BarData]) -> Any:
        logger.info(f"Training RNN on {len(historical_bars)} bars.")
        data = self._prepare_data(historical_bars)
        if data is None:
            return None

        X_raw, y_raw, X_mean, X_std = data

        # Instantiate SimpleRNN
        model = SimpleRNN(
            input_dim=self.lookback_period,
            config=self.model_config,
            mean=X_mean,
            std=X_std,
        )

        dummy_input = torch.randn(1, 1, self.lookback_period)
        self._train_model(model, X_raw, y_raw, dummy_input)
        return self._export_to_onnx(model)


class LSTMTrainer(BasePyTorchTrainer):
    """
    Trains a Long Short-Term Memory (LSTM) network and exports to ONNX.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        model_config: Union[LSTMConfig, Dict[str, Any], None] = None,
        training_config: Union[NNTrainingConfig, Dict[str, Any], None] = None,
    ) -> None:
        super().__init__(
            lookback_period=lookback_period, training_config=training_config
        )
        self.model_config = (
            model_config
            if isinstance(model_config, LSTMConfig)
            else LSTMConfig(**(model_config or {}))
        )

    def train(self, historical_bars: Sequence[BarData]) -> Any:
        logger.info(f"Training LSTM on {len(historical_bars)} bars.")
        data = self._prepare_data(historical_bars)
        if data is None:
            return None

        X_raw, y_raw, X_mean, X_std = data

        # Instantiate SimpleLSTM
        model = SimpleLSTM(
            input_dim=self.lookback_period,
            config=self.model_config,
            mean=X_mean,
            std=X_std,
        )

        dummy_input = torch.randn(1, 1, self.lookback_period)
        self._train_model(model, X_raw, y_raw, dummy_input)
        return self._export_to_onnx(model)
