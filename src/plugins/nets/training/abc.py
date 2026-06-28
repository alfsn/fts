import io
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from trading_bot.core.dataset import DatasetBuilder
from trading_bot.core.schemas import BarData
from trading_bot.core.training import BaseModelTrainer
from trading_bot.core.transforms import LogReturnTransform

from ..models import (
    NNTrainingConfig,
    ONNXModelMetadata,
)
from ..output_selectors import BaseOutputSelector
from .dataset import TimeSeriesDataset
from .evaluator import ValidationEvaluator

logger = logging.getLogger(__name__)


def _get_training_metadata(
    historical_bars: Sequence[BarData],
    lookback_period: int,
    horizon: int,
    val_ratio: float,
) -> ONNXModelMetadata:
    """
    Computes metadata about the training period, ensuring we exactly identify
    the training range boundary of the split.
    """
    if not historical_bars:
        raise ValueError("Cannot compute metadata for empty historical bars.")

    n_returns = len(historical_bars) - 1
    n_samples = n_returns - lookback_period - horizon + 1
    if n_samples <= 2:
        train_end_idx = len(historical_bars) - 1
    else:
        val_size = int(n_samples * val_ratio)
        if val_size == 0:
            val_size = min(1, n_samples // 5)
            if val_size == 0:
                val_size = 1
        train_size = n_samples - val_size
        last_train_sample_idx = train_size - horizon - 1
        train_end_idx = max(
            0,
            min(
                len(historical_bars) - 1,
                last_train_sample_idx + lookback_period + horizon,
            ),
        )

    return ONNXModelMetadata(
        train_start_date=historical_bars[0].timestamp,
        train_end_date=historical_bars[train_end_idx].timestamp,
        lookback_period=lookback_period,
        horizon=horizon,
        val_ratio=val_ratio,
    )


def _add_onnx_metadata(onnx_bytes: bytes, metadata: ONNXModelMetadata) -> bytes:
    """Loads ONNX bytes, adds metadata key-value properties, and returns serialized bytes."""
    model = onnx.load_model_from_string(onnx_bytes)
    custom_map = metadata.to_custom_metadata()
    for key, val in custom_map.items():
        prop = None
        for p in model.metadata_props:
            if p.key == key:
                prop = p
                break
        if prop is None:
            prop = model.metadata_props.add()
            prop.key = key
        prop.value = str(val)
    return model.SerializeToString()


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


class BaseONNXModelTrainer(BaseModelTrainer, ABC):
    """
    Base trainer that implements the Template Method Pattern to automatically
    inject training metadata into exported ONNX models.
    """

    @abstractmethod
    def _train_to_onnx(self, historical_bars: Sequence[BarData]) -> bytes:
        """Trains the model and returns the raw ONNX bytes representation."""
        pass

    def log_to_tensorboard(self) -> None:
        """Logs hyperparameters and final validation metrics to TensorBoard HParams."""
        config = getattr(self, "config", getattr(self, "training_config", None))
        if config is None or not getattr(config, "tensorboard_log_dir", None):
            return

        metrics = getattr(self, "best_val_metrics", None)
        if not metrics:
            return

        import os

        from torch.utils.tensorboard import SummaryWriter

        log_dir = config.tensorboard_log_dir
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        # Build hparam_dict
        hparam_dict = {
            "lr": getattr(config, "learning_rate", 0.0),
            "batch_size": getattr(config, "batch_size", 0),
            "epochs": getattr(config, "epochs", 0),
            "optimizer": getattr(config, "optimizer", "none"),
            "loss_fn": getattr(config, "loss_fn", "none"),
            "lookback": self.lookback_period,
            "horizon": config.horizon,
        }

        # Add model config details if available (PyTorch models)
        model_config = getattr(self, "model_config", None)
        if model_config:
            from pydantic import BaseModel

            cfg_dict = (
                model_config.model_dump()
                if isinstance(model_config, BaseModel)
                else dict(model_config)
            )
            for k, v in cfg_dict.items():
                if isinstance(v, (int, float, str, bool)):
                    hparam_dict[f"model/{k}"] = v
                elif isinstance(v, list):
                    hparam_dict[f"model/{k}"] = str(v)

        # Build metric_dict
        metric_dict = {
            "hparam/val_loss": metrics.get("loss", 999.0),
            "hparam/val_ic": metrics.get("ic", 0.0),
            "hparam/val_directional_accuracy": metrics.get("directional_accuracy", 0.5),
        }
        if "selector_accuracy" in metrics:
            metric_dict["hparam/val_selector_accuracy"] = metrics["selector_accuracy"]

        writer.add_hparams(hparam_dict, metric_dict)
        writer.close()

    def train(self, historical_bars: Sequence[BarData]) -> bytes:
        onnx_bytes = self._train_to_onnx(historical_bars)
        if not onnx_bytes:
            return onnx_bytes

        config = getattr(self, "config", getattr(self, "training_config", None))
        if config is not None:
            self.log_to_tensorboard()

        metadata = _get_training_metadata(
            historical_bars=historical_bars,
            lookback_period=self.lookback_period,
            horizon=config.horizon if config else 1,
            val_ratio=config.validation_split if config else 0.2,
        )
        return _add_onnx_metadata(onnx_bytes, metadata)


class BasePyTorchTrainer(BaseONNXModelTrainer):
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
        self.model_class = None
        self.model_config = None

    def _train_to_onnx(self, historical_bars: Sequence[BarData]) -> Any:
        if self.model_class is None:
            raise NotImplementedError("Subclass must set model_class")
        if self.model_config is None:
            raise NotImplementedError("Subclass must set model_config")

        model_name = self.model_class.__name__.replace("Simple", "")
        logger.info(f"Training {model_name} on {len(historical_bars)} bars.")
        data = self._prepare_data(historical_bars)
        if data is None:
            return None

        X_train, y_train, X_val, y_val, X_mean, X_std = data
        n_features = len(self.training_config.feature_cols)

        model = self.model_class(
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

        self._train_model_loop(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            dummy_input=dummy_input,
            val_bars=val_bars,
        )
        return self._export_to_onnx(model)

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

    def _train_model_loop(
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
        best_val_metrics_dct = None
        epochs_since_improvement = 0

        global_step = 0
        epoch_loss = 0.0
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
                    best_val_metrics_dct = copy.deepcopy(val_metrics)
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
            self.best_val_loss = best_val_loss
            self.best_val_metrics = (
                best_val_metrics_dct if best_val_metrics_dct is not None else {}
            )
        else:
            self.best_val_loss = epoch_loss
            self.best_val_metrics = {"loss": epoch_loss}

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
