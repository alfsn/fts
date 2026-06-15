# src/plugins/nets/schemas.py

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


class NNTrainingConfig(BaseModel):
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


class CNNConfig(BaseModel):
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


class RNNConfig(BaseModel):
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


class LSTMConfig(BaseModel):
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
