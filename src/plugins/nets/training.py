# src/plugins/nets/training.py

import logging
from typing import Any, Sequence

from skl2onnx import to_onnx
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trading_bot.core.dataset import DatasetBuilder
from trading_bot.core.schemas import BarData
from trading_bot.core.training import BaseModelTrainer
from trading_bot.core.transforms import LogReturnTransform

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
        # We first get raw close prices
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

        # Export (Requires custom handling for XGBoost within Pipeline)
        # For simplicity, we convert the pipeline
        # by treating it as a standard sklearn object
        # skl2onnx handles Pipeline, but it needs to know how to
        # convert the XGBoost part.
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
            convert_xgboost as convert_xgboost_op,
        )
        from skl2onnx import update_registered_converter
        from skl2onnx.common.shape_calculator import (
            calculate_linear_regressor_output_shapes,
        )

        # This registration is sometimes necessary depending on versions
        update_registered_converter(
            xgb.XGBRegressor,
            "XGBRegressor",
            calculate_linear_regressor_output_shapes,
            convert_xgboost_op,
        )

        onx = to_onnx(pipeline, X[:1])
        return onx.SerializeToString()


class CNNTrainer(BaseModelTrainer):
    """
    Trains a CNN model and exports to ONNX.
    Note: PyTorch models usually don't use sklearn Pipelines directly for ONNX export.
    """

    def __init__(self, lookback_period: int = 20) -> None:
        self.lookback_period = lookback_period
        self.transform = LogReturnTransform()

    def train(self, historical_bars: Sequence[BarData]) -> Any:
        import io

        import torch
        import torch.nn as nn
        import torch.optim as optim

        logger.info(f"Training CNN on {len(historical_bars)} bars.")

        matrix = DatasetBuilder.to_matrix(historical_bars, feature_cols=["close"])
        returns = self.transform.transform(matrix)

        if len(returns) < self.lookback_period + 1:
            return None

        X_raw, y_raw = DatasetBuilder.create_sliding_windows(
            returns, lookback=self.lookback_period, horizon=1
        )

        # Scaling (In PyTorch we often do this manually or via a layer)
        X_mean = X_raw.mean()
        X_std = X_raw.std() + 1e-8
        X_scaled = (X_raw - X_mean) / X_std

        X_pt = torch.tensor(X_scaled, dtype=torch.float32).transpose(
            1, 2
        )  # [batch, channels, seq]
        y_pt = torch.tensor(y_raw.flatten(), dtype=torch.float32).unsqueeze(1)

        # 2. Define Model (Including Scaling logic as a constant)
        class SimpleCNN(nn.Module):
            def __init__(self, input_dim, mean, std):
                super().__init__()
                self.mean = nn.Parameter(torch.tensor(mean), requires_grad=False)
                self.std = nn.Parameter(torch.tensor(std), requires_grad=False)
                self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
                self.relu = nn.ReLU()
                self.fc = nn.Linear(16 * input_dim, 1)

            def forward(self, x):
                # Apply scaling inside the model for ONNX portability
                x = (x - self.mean) / self.std
                x = self.relu(self.conv1(x))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        model = SimpleCNN(self.lookback_period, X_mean, X_std)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(X_pt)
            loss = criterion(outputs, y_pt)
            loss.backward()
            optimizer.step()

        # 4. Export to ONNX
        dummy_input = torch.randn(1, 1, self.lookback_period)
        f = io.BytesIO()
        torch.onnx.export(
            model,
            dummy_input,
            f,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        return f.getvalue()
