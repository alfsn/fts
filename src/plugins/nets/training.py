# src/plugins/nets/training.py

import io
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skl2onnx import to_onnx
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from trading_bot.core.dataset import DatasetBuilder
from trading_bot.core.schemas import BarData
from trading_bot.core.training import BaseModelTrainer
from trading_bot.core.transforms import LogReturnTransform

from .classifiers import BaseOutputSelector
from .evaluator import MetricsCalculator, ValidationEvaluator
from .models import SimpleCNN, SimpleLSTM, SimpleRNN
from .schemas import (
    BaseTrainerConfig,
    CNNConfig,
    LSTMConfig,
    NNTrainingConfig,
    RNNConfig,
)

logger = logging.getLogger(__name__)


def extract_validation_bars(
    historical_bars: Sequence[BarData],
    val_size: int,
    lookback_period: int,
    horizon: int,
) -> List[Sequence[BarData]]:
    """Helper to extract validation bars chronologically corresponding to the validation split."""
    n_returns = len(historical_bars) - 1
    n_raw_samples = n_returns - lookback_period - horizon + 1
    val_start_idx = n_raw_samples - val_size
    val_bars = []
    for j in range(val_size):
        idx = val_start_idx + j
        window = historical_bars[idx : idx + lookback_period + 1]
        val_bars.append(window)
    return val_bars


class LinearRegressionTrainer(BaseModelTrainer):
    """
    Trains a Linear Regression model with a StandardScaler pipeline and exports to ONNX.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        config: Optional[BaseTrainerConfig] = None,
        output_selector: Optional[BaseOutputSelector] = None,
    ) -> None:
        self.config = config or BaseTrainerConfig(lookback_period=lookback_period)
        self.lookback_period = self.config.lookback_period
        self.transform = LogReturnTransform()
        self.output_selector = output_selector

    def train(self, historical_bars: Sequence[BarData]) -> Any:
        logger.info(f"Training LinearRegression on {len(historical_bars)} bars.")

        # 1. Structural Conversion (SOLID: Extract window logic to Core)
        matrix = DatasetBuilder.to_matrix(
            historical_bars, feature_cols=self.config.feature_cols
        )

        # Apply LogReturn (Python-side preprocessing for now)
        returns = self.transform.transform(matrix)

        if len(returns) < self.lookback_period + 1:
            logger.error("Insufficient data for training.")
            return None

        # Create sliding windows
        X_raw, y_raw = DatasetBuilder.create_sliding_windows(
            returns, lookback=self.lookback_period, horizon=self.config.horizon
        )

        # Split into training and validation sets
        X_train, y_train, X_val, y_val = (
            DatasetBuilder.split_train_val_purged_embargoed(
                X_raw,
                y_raw,
                val_ratio=self.config.validation_split,
                horizon=self.config.horizon,
                embargo_pct=self.config.embargo_pct,
            )
        )

        # Reshape X for sklearn (samples, lookback * features)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        y_train_flat = y_train[:, :, 0].flatten()

        # 2. Pipeline Definition (SOLID: Preprocessing in Pipeline)
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        )

        # 3. Fit
        pipeline.fit(X_train_flat, y_train_flat)

        # Evaluate validation metrics
        if len(X_val) > 0:
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            y_val_flat = y_val[:, :, 0].flatten()
            val_preds = pipeline.predict(X_val_flat)

            val_bars = None
            if self.output_selector is not None:
                val_bars = extract_validation_bars(
                    historical_bars=historical_bars,
                    val_size=len(X_val),
                    lookback_period=self.lookback_period,
                    horizon=self.config.horizon,
                )

            metrics = MetricsCalculator.calculate_metrics(
                preds=val_preds,
                targets=y_val_flat,
                output_selector=self.output_selector,
                val_bars=val_bars,
            )
            logger.info(
                f"LinearRegression Validation: Loss = {metrics['loss']:.6f}, "
                f"IC = {metrics['ic']:.4f}, Dir Acc = {metrics['directional_accuracy']:.4f}"
            )
            if "selector_accuracy" in metrics:
                logger.info(
                    f"LinearRegression Validation Selector Accuracy = {metrics['selector_accuracy']:.4f}"
                )

        # 4. Export to ONNX
        onx = to_onnx(pipeline, X_train_flat[:1])
        return onx.SerializeToString()


class XGBoostTrainer(BaseModelTrainer):
    """
    Trains an XGBoost model with a StandardScaler pipeline and exports to ONNX.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        config: Optional[BaseTrainerConfig] = None,
        output_selector: Optional[BaseOutputSelector] = None,
    ) -> None:
        self.config = config or BaseTrainerConfig(lookback_period=lookback_period)
        self.lookback_period = self.config.lookback_period
        self.transform = LogReturnTransform()
        self.output_selector = output_selector

    def train(self, historical_bars: Sequence[BarData]) -> Any:
        import xgboost as xgb

        logger.info(f"Training XGBoost on {len(historical_bars)} bars.")

        matrix = DatasetBuilder.to_matrix(
            historical_bars, feature_cols=self.config.feature_cols
        )
        returns = self.transform.transform(matrix)

        if len(returns) < self.lookback_period + 1:
            return None

        X_raw, y_raw = DatasetBuilder.create_sliding_windows(
            returns, lookback=self.lookback_period, horizon=self.config.horizon
        )

        # Split into training and validation sets
        X_train, y_train, X_val, y_val = (
            DatasetBuilder.split_train_val_purged_embargoed(
                X_raw,
                y_raw,
                val_ratio=self.config.validation_split,
                horizon=self.config.horizon,
                embargo_pct=self.config.embargo_pct,
            )
        )

        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        y_train_flat = y_train[:, :, 0].flatten()

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)

        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=3, early_stopping_rounds=10
        )

        if len(X_val) > 0:
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            y_val_flat = y_val[:, :, 0].flatten()
            X_val_scaled = scaler.transform(X_val_flat)

            model.fit(
                X_train_scaled,
                y_train_flat,
                eval_set=[(X_val_scaled, y_val_flat)],
                verbose=False,
            )

            # Evaluate metrics on validation set
            val_preds = model.predict(X_val_scaled)

            val_bars = None
            if self.output_selector is not None:
                val_bars = extract_validation_bars(
                    historical_bars=historical_bars,
                    val_size=len(X_val),
                    lookback_period=self.lookback_period,
                    horizon=self.config.horizon,
                )

            metrics = MetricsCalculator.calculate_metrics(
                preds=val_preds,
                targets=y_val_flat,
                output_selector=self.output_selector,
                val_bars=val_bars,
            )
            logger.info(
                f"XGBoost Validation: Best Iteration = {model.best_iteration}, "
                f"Loss = {metrics['loss']:.6f}, IC = {metrics['ic']:.4f}, Dir Acc = {metrics['directional_accuracy']:.4f}"
            )
            if "selector_accuracy" in metrics:
                logger.info(
                    f"XGBoost Validation Selector Accuracy = {metrics['selector_accuracy']:.4f}"
                )
        else:
            model.fit(X_train_scaled, y_train_flat, verbose=False)

        # Re-create pipeline with fitted components for ONNX export
        pipeline = Pipeline([("scaler", scaler), ("model", model)])

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

        onx = to_onnx(pipeline, X_train_flat[:1], target_opset={"ai.onnx.ml": 3})
        return onx.SerializeToString()


class TimeSeriesDataset(Dataset):
    """
    Idiomatic PyTorch Dataset for time-series forecasting.
    Converts and transposes features/targets into tensors during initialization.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32).transpose(1, 2)
        # Select target feature (index 0) and flatten if 3D
        if len(y.shape) == 3:
            y_target = y[:, :, 0]
        else:
            y_target = y
        self.y = torch.tensor(y_target.flatten(), dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class BasePyTorchTrainer(BaseModelTrainer):
    """
    Abstract trainer base for PyTorch models that handles dataset creation,
    optional TensorBoard logging, training loops, and exporting to ONNX.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        training_config: Union[NNTrainingConfig, Dict[str, Any], None] = None,
        output_selector: Optional[BaseOutputSelector] = None,
    ) -> None:
        self.lookback_period = lookback_period
        self.transform = LogReturnTransform()
        self.training_config = (
            training_config
            if isinstance(training_config, NNTrainingConfig)
            else NNTrainingConfig(**(training_config or {}))
        )
        self.output_selector = output_selector

    def _prepare_data(self, historical_bars: Sequence[BarData]) -> Union[
        None,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        feature_cols = self.training_config.feature_cols
        matrix = DatasetBuilder.to_matrix(historical_bars, feature_cols=feature_cols)
        returns = self.transform.transform(matrix)

        if len(returns) < self.lookback_period + 1:
            logger.error("Insufficient data for PyTorch neural network training.")
            return None

        X_raw, y_raw = DatasetBuilder.create_sliding_windows(
            returns, lookback=self.lookback_period, horizon=self.training_config.horizon
        )

        # Split into training and validation sets
        X_train, y_train, X_val, y_val = (
            DatasetBuilder.split_train_val_purged_embargoed(
                X_raw,
                y_raw,
                val_ratio=self.training_config.validation_split,
                horizon=self.training_config.horizon,
                embargo_pct=self.training_config.embargo_pct,
            )
        )

        # Calculate mean & std from training features only to prevent data leakage
        n_features = len(feature_cols)
        if len(X_train) > 0:
            X_mean = X_train.mean(axis=(0, 1))
            X_std = X_train.std(axis=(0, 1)) + 1e-8
        else:
            X_mean = np.zeros(n_features)
            X_std = np.ones(n_features)

        return X_train, y_train, X_val, y_val, X_mean, X_std

    def _train_model(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        dummy_input: torch.Tensor,
        val_bars: Optional[Sequence[Sequence[BarData]]] = None,
    ) -> None:
        if len(X_train) == 0:
            logger.error("No training data available after split.")
            return

        train_dataset = TimeSeriesDataset(X_train, y_train)
        dataloader = DataLoader(
            train_dataset, batch_size=self.training_config.batch_size, shuffle=False
        )

        val_loader = None
        if len(X_val) > 0:
            val_dataset = TimeSeriesDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset, batch_size=self.training_config.batch_size, shuffle=False
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

        import copy

        best_val_loss = float("inf")
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_since_improvement = 0

        global_step = 0
        for epoch in range(self.training_config.epochs):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # B) Gradient clipping
                if self.training_config.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.training_config.clip_grad_norm
                    )

                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                if writer:
                    writer.add_scalar("Loss/train_step", loss_val, global_step)
                global_step += 1

            epoch_loss /= len(dataloader)
            if writer:
                writer.add_scalar("Loss/train_epoch", epoch_loss, epoch)

            # Validation step (SRP: delegated to ValidationEvaluator)
            if val_loader is not None:
                val_metrics = ValidationEvaluator.evaluate(
                    model=model,
                    dataloader=val_loader,
                    criterion=criterion,
                    output_selector=self.output_selector,
                    val_bars=val_bars,
                )

                # Log to TensorBoard
                if writer:
                    ValidationEvaluator.log_to_tensorboard(writer, val_metrics, epoch)

                val_loss = val_metrics["loss"]
                logger.info(
                    f"Epoch {epoch}: Train Loss = {epoch_loss:.6f}, Val Loss = {val_loss:.6f}, "
                    f"Val IC = {val_metrics['ic']:.4f}, Val Dir Acc = {val_metrics['directional_accuracy']:.4f}"
                )
                if "selector_accuracy" in val_metrics:
                    logger.info(
                        f"Epoch {epoch}: Val Selector Accuracy = {val_metrics['selector_accuracy']:.4f}"
                    )

                # Early stopping & model checkpointing
                if (
                    val_loss
                    < best_val_loss - self.training_config.early_stopping_min_delta
                ):
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

                if (
                    epochs_since_improvement
                    >= self.training_config.early_stopping_patience
                ):
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break
            else:
                logger.info(
                    f"Epoch {epoch}: Train Loss = {epoch_loss:.6f} (No validation data)"
                )

        # Restore best model state (checkpointing)
        if val_loader is not None and best_model_state is not None:
            logger.info(
                f"Restoring best model state with validation loss: {best_val_loss:.6f}"
            )
            model.load_state_dict(best_model_state)

        if writer:
            writer.close()

    def _export_to_onnx(self, model: nn.Module) -> bytes:
        model.eval()
        n_features = len(self.training_config.feature_cols)
        dummy_input = torch.randn(1, n_features, self.lookback_period)
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
        output_selector: Optional[BaseOutputSelector] = None,
    ) -> None:
        super().__init__(
            lookback_period=lookback_period,
            training_config=training_config,
            output_selector=output_selector,
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

        X_train, y_train, X_val, y_val, X_mean, X_std = data
        n_features = len(self.training_config.feature_cols)

        # Instantiate SimpleCNN
        model = SimpleCNN(
            input_dim=self.lookback_period,
            n_features=n_features,
            config=self.model_config,
            mean=X_mean,
            std=X_std,
        )

        dummy_input = torch.randn(1, n_features, self.lookback_period)

        val_bars = None
        if self.output_selector is not None and len(X_val) > 0:
            val_bars = extract_validation_bars(
                historical_bars=historical_bars,
                val_size=len(X_val),
                lookback_period=self.lookback_period,
                horizon=self.training_config.horizon,
            )

        self._train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            dummy_input=dummy_input,
            val_bars=val_bars,
        )
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
        output_selector: Optional[BaseOutputSelector] = None,
    ) -> None:
        super().__init__(
            lookback_period=lookback_period,
            training_config=training_config,
            output_selector=output_selector,
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

        X_train, y_train, X_val, y_val, X_mean, X_std = data
        n_features = len(self.training_config.feature_cols)

        # Instantiate SimpleRNN
        model = SimpleRNN(
            input_dim=self.lookback_period,
            n_features=n_features,
            config=self.model_config,
            mean=X_mean,
            std=X_std,
        )

        dummy_input = torch.randn(1, n_features, self.lookback_period)

        val_bars = None
        if self.output_selector is not None and len(X_val) > 0:
            val_bars = extract_validation_bars(
                historical_bars=historical_bars,
                val_size=len(X_val),
                lookback_period=self.lookback_period,
                horizon=self.training_config.horizon,
            )

        self._train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            dummy_input=dummy_input,
            val_bars=val_bars,
        )
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
        output_selector: Optional[BaseOutputSelector] = None,
    ) -> None:
        super().__init__(
            lookback_period=lookback_period,
            training_config=training_config,
            output_selector=output_selector,
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

        X_train, y_train, X_val, y_val, X_mean, X_std = data
        n_features = len(self.training_config.feature_cols)

        # Instantiate SimpleLSTM
        model = SimpleLSTM(
            input_dim=self.lookback_period,
            n_features=n_features,
            config=self.model_config,
            mean=X_mean,
            std=X_std,
        )

        dummy_input = torch.randn(1, n_features, self.lookback_period)

        val_bars = None
        if self.output_selector is not None and len(X_val) > 0:
            val_bars = extract_validation_bars(
                historical_bars=historical_bars,
                val_size=len(X_val),
                lookback_period=self.lookback_period,
                horizon=self.training_config.horizon,
            )

        self._train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            dummy_input=dummy_input,
            val_bars=val_bars,
        )
        return self._export_to_onnx(model)
