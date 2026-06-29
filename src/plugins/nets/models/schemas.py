from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class BaseTrainerConfig(BaseModel):
    """Base configuration model that applies to all model trainers."""

    lookback_period: int = Field(
        default=20, ge=1, description="Lookback window size in bars"
    )
    feature_cols: List[str] = Field(
        default_factory=lambda: ["close"],
        description="Bar attributes to use as features",
    )
    validation_split: float = Field(
        default=0.2,
        ge=0.0,
        le=0.9,
        description="Fraction of data to use for validation",
    )
    embargo_pct: float = Field(
        default=0.01,
        ge=0.0,
        le=0.1,
        description="Percentage of data to use as embargo buffer",
    )
    horizon: int = Field(default=1, ge=1, description="Forecasting horizon")


class NNTrainingConfig(BaseTrainerConfig):
    """Configuration for neural network training loops and TensorBoard monitoring."""

    epochs: int = Field(default=10, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=1, description="Training batch size")
    learning_rate: float = Field(
        default=0.001, gt=0.0, description="Optimizer learning rate"
    )
    optimizer: Literal["adam", "sgd", "rmsprop"] = Field(
        default="adam", description="Optimizer type"
    )
    loss_fn: Literal["mse", "mae", "huber"] = Field(
        default="mse", description="Loss function type"
    )
    tensorboard_log_dir: Optional[str] = Field(
        default=None,
        description="Directory to save TensorBoard logs (relative to project root/artifacts)",
    )
    early_stopping_patience: int = Field(
        default=5,
        ge=1,
        description="Epochs to wait for validation loss improvement before stopping",
    )
    early_stopping_min_delta: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum change in validation loss to qualify as an improvement",
    )
    clip_grad_norm: Optional[float] = Field(
        default=1.0, gt=0.0, description="Gradient clipping max norm value"
    )


class CNNConfig(NNTrainingConfig):
    """Configuration for Convolutional Neural Network architectures."""

    out_channels: List[int] = Field(
        default_factory=lambda: [16, 32],
        description="Number of output channels for each conv layer",
    )
    kernel_sizes: List[int] = Field(
        default_factory=lambda: [3, 3], description="Kernel size for each conv layer"
    )
    pool_sizes: List[Optional[int]] = Field(
        default_factory=lambda: [None, None],
        description="Max pooling size for each conv layer, or None to skip",
    )
    dense_units: int = Field(
        default=64, ge=1, description="Number of units in the final dense layer"
    )
    dropout: float = Field(
        default=0.0, ge=0.0, le=0.9, description="Dropout rate after pooling/dense"
    )


class RNNConfig(NNTrainingConfig):
    """Configuration for Recurrent Neural Network (Elman RNN) architectures."""

    hidden_dim: int = Field(
        default=32, ge=1, description="Hidden dimension size for the recurrent layer"
    )
    num_layers: int = Field(default=1, ge=1, description="Number of recurrent layers")
    dropout: float = Field(
        default=0.0, ge=0.0, le=0.9, description="Dropout rate between layers"
    )
    nonlinearity: Literal["tanh", "relu"] = Field(
        default="tanh", description="The non-linearity function to use in RNN"
    )


class LSTMConfig(NNTrainingConfig):
    """Configuration for Long Short-Term Memory Network architectures."""

    hidden_dim: int = Field(
        default=32, ge=1, description="Hidden dimension size for the LSTM layer"
    )
    num_layers: int = Field(default=1, ge=1, description="Number of recurrent layers")
    dropout: float = Field(
        default=0.0, ge=0.0, le=0.9, description="Dropout rate between layers"
    )
    bidirectional: bool = Field(
        default=False, description="Whether the LSTM is bidirectional"
    )


class ONNXModelMetadata(BaseModel):
    """Schema representing the metadata serialized into exported ONNX models."""

    train_start_date: datetime = Field(
        ..., description="Start timestamp of model training data"
    )
    train_end_date: datetime = Field(
        ..., description="End timestamp of model training data"
    )
    lookback_period: int = Field(
        ..., description="The lookback period used for features"
    )
    horizon: int = Field(..., description="Forecasting target horizon")
    val_ratio: float = Field(..., description="Validation split ratio")
    feature_pipeline: Optional[str] = Field(
        None, description="Serialized FeaturePipeline configuration JSON"
    )

    def to_custom_metadata(self) -> dict[str, str]:
        """Converts class attributes to string dictionary for ONNX serialization."""
        meta = {
            "train_start_date": self.train_start_date.isoformat(),
            "train_end_date": self.train_end_date.isoformat(),
            "lookback_period": str(self.lookback_period),
            "horizon": str(self.horizon),
            "val_ratio": str(self.val_ratio),
        }
        if self.feature_pipeline is not None:
            meta["feature_pipeline"] = self.feature_pipeline
        return meta

    @classmethod
    def from_custom_metadata(
        cls, custom_metadata: dict[str, str]
    ) -> "ONNXModelMetadata":
        """Parses ONNX custom metadata properties into a validated Pydantic model."""
        required_keys = [
            "train_start_date",
            "train_end_date",
            "lookback_period",
            "horizon",
            "val_ratio",
        ]
        for key in required_keys:
            if key not in custom_metadata:
                raise ValueError(f"Missing required ONNX model metadata key: '{key}'")
        return cls(
            train_start_date=datetime.fromisoformat(
                custom_metadata["train_start_date"]
            ),
            train_end_date=datetime.fromisoformat(custom_metadata["train_end_date"]),
            lookback_period=int(custom_metadata["lookback_period"]),
            horizon=int(custom_metadata["horizon"]),
            val_ratio=float(custom_metadata["val_ratio"]),
            feature_pipeline=custom_metadata.get("feature_pipeline"),
        )

    def validate_timestamp(
        self,
        timestamp: datetime,
        allow_in_sample: bool = False,
        strategy_name: str = "strategy",
    ) -> None:
        """
        Validates that the given timestamp is chronologically after the model's training range.
        Raises a ValueError if lookahead leakage is detected and allow_in_sample is False.
        """
        if allow_in_sample:
            return

        from datetime import timezone

        # Align timezones
        tick_time = timestamp
        train_end_date = self.train_end_date

        if tick_time.tzinfo is not None and train_end_date.tzinfo is None:
            train_end_date = train_end_date.replace(tzinfo=timezone.utc)
        elif tick_time.tzinfo is None and train_end_date.tzinfo is not None:
            train_end_date = train_end_date.replace(tzinfo=None)

        if tick_time <= train_end_date:
            raise ValueError(
                f"Lookahead Guardrail Violation: Tick timestamp {tick_time} "
                f"is within the training period (ended {train_end_date.isoformat()}) "
                f"for strategy {strategy_name}. To bypass this check for diagnostic runs, "
                f"set allow_in_sample=True in the strategy configuration."
            )
