import logging
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from skl2onnx import to_onnx
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trading_bot.core.dataset import DatasetBuilder
from trading_bot.core.schemas import BarData
from trading_bot.core.transforms import BaseTransform

from ..models import (
    BaseTrainerConfig,
    CNNConfig,
    LSTMConfig,
    NNTrainingConfig,
    RNNConfig,
    SimpleCNN,
    SimpleLSTM,
    SimpleRNN,
)
from ..output_selectors import BaseOutputSelector
from .abc import BaseONNXModelTrainer, BasePyTorchTrainer, extract_validation_bars
from .evaluator import MetricsCalculator

logger = logging.getLogger(__name__)


class LinearRegressionTrainer(BaseONNXModelTrainer):
    """
    Trains a Linear Regression model with a StandardScaler pipeline and exports to ONNX.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        config: Optional[BaseTrainerConfig] = None,
        output_selector: Optional[BaseOutputSelector] = None,
        transform: Optional[BaseTransform] = None,
    ) -> None:
        self.config = config or BaseTrainerConfig(lookback_period=lookback_period)
        self.lookback_period = self.config.lookback_period
        from trading_bot.core.transforms import LogReturnTransform

        self.transform = transform or LogReturnTransform()
        self.output_selector = output_selector

    def _train_to_onnx(self, historical_bars: Sequence[BarData]) -> Any:
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
        y_train_flat = y_train[:, -1, 0]

        # 2. Pipeline Definition (SOLID: Preprocessing in Pipeline)
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        )

        # 3. Fit
        pipeline.fit(X_train_flat, y_train_flat)

        # Evaluate validation metrics
        if len(X_val) > 0:
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            y_val_flat = y_val[:, -1, 0]
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
            self.best_val_loss = metrics["loss"]
            self.best_val_metrics = metrics
        else:
            self.best_val_loss = 999.0
            self.best_val_metrics = {"loss": 999.0}

        # 4. Export to ONNX
        onx = to_onnx(pipeline, X_train_flat[:1])
        return onx.SerializeToString()


class XGBoostTrainer(BaseONNXModelTrainer):
    """
    Trains an XGBoost model with a StandardScaler pipeline and exports to ONNX.
    """

    def __init__(
        self,
        lookback_period: int = 20,
        config: Optional[BaseTrainerConfig] = None,
        output_selector: Optional[BaseOutputSelector] = None,
        transform: Optional[BaseTransform] = None,
    ) -> None:
        self.config = config or BaseTrainerConfig(lookback_period=lookback_period)
        self.lookback_period = self.config.lookback_period
        from trading_bot.core.transforms import LogReturnTransform

        self.transform = transform or LogReturnTransform()
        self.output_selector = output_selector

    def _train_to_onnx(self, historical_bars: Sequence[BarData]) -> Any:
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
        y_train_flat = y_train[:, -1, 0]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)

        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=3, early_stopping_rounds=10
        )

        if len(X_val) > 0:
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            y_val_flat = y_val[:, -1, 0]
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
            self.best_val_loss = metrics["loss"]
            self.best_val_metrics = metrics
        else:
            model.fit(X_train_scaled, y_train_flat, verbose=False)
            self.best_val_loss = 999.0
            self.best_val_metrics = {"loss": 999.0}

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
        self.model_class = SimpleCNN
        self.model_config = (
            model_config
            if isinstance(model_config, CNNConfig)
            else CNNConfig(**(model_config or {}))
        )


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
        self.model_class = SimpleRNN
        self.model_config = (
            model_config
            if isinstance(model_config, RNNConfig)
            else RNNConfig(**(model_config or {}))
        )


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
        self.model_class = SimpleLSTM
        self.model_config = (
            model_config
            if isinstance(model_config, LSTMConfig)
            else LSTMConfig(**(model_config or {}))
        )
