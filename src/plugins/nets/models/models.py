# src/plugins/nets/models.py

from typing import Any

import torch
import torch.nn as nn

from .schemas import CNNConfig, LSTMConfig, RNNConfig


def prepare_scaling_parameters(
    n_features: int, mean: Any, std: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Helper to convert mean and std inputs into PyTorch parameters of shape (n_features, 1)
    to facilitate correct broadcasting over input tensors of shape (batch, n_features, seq_len).
    """
    mean_tensor = torch.tensor(mean, dtype=torch.float32)
    std_tensor = torch.tensor(std, dtype=torch.float32)
    if mean_tensor.dim() == 0 or mean_tensor.numel() == 1:
        mean_tensor = mean_tensor.expand(n_features)
    if std_tensor.dim() == 0 or std_tensor.numel() == 1:
        std_tensor = std_tensor.expand(n_features)
    return mean_tensor.view(-1, 1), std_tensor.view(-1, 1)


class SimpleCNN(nn.Module):
    """
    A configurable Convolutional Neural Network (CNN) for time-series forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int = 1,
        config: CNNConfig = None,
        mean: Any = 0.0,
        std: Any = 1.0,
    ) -> None:
        super().__init__()
        mean_param, std_param = prepare_scaling_parameters(n_features, mean, std)
        self.mean = nn.Parameter(mean_param, requires_grad=False)
        self.std = nn.Parameter(std_param, requires_grad=False)

        # Build Conv1D layers
        layers = []
        in_channels = n_features
        for i, (out_chan, kernel_sz, pool_sz) in enumerate(
            zip(config.out_channels, config.kernel_sizes, config.pool_sizes)
        ):
            # Use 'same' padding for conv1d
            padding = kernel_sz // 2
            layers.append(
                nn.Conv1d(in_channels, out_chan, kernel_size=kernel_sz, padding=padding)
            )
            layers.append(nn.ReLU())

            if pool_sz is not None and pool_sz > 1:
                layers.append(nn.MaxPool1d(kernel_size=pool_sz))

            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))

            in_channels = out_chan

        self.conv_net = nn.Sequential(*layers)

        # Compute flattened feature size dynamically using a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_features, input_dim)
            dummy_output = self.conv_net(dummy_input)
            flat_size = dummy_output.numel()

        self.dense = nn.Sequential(
            nn.Linear(flat_size, config.dense_units),
            nn.ReLU(),
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
            nn.Linear(config.dense_units, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale inside model for ONNX portability
        x = (x - self.mean) / self.std
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x


class SimpleRNN(nn.Module):
    """
    A configurable Elman Recurrent Neural Network (RNN) for time-series forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int = 1,
        config: RNNConfig = None,
        mean: Any = 0.0,
        std: Any = 1.0,
    ) -> None:
        super().__init__()
        mean_param, std_param = prepare_scaling_parameters(n_features, mean, std)
        self.mean = nn.Parameter(mean_param, requires_grad=False)
        self.std = nn.Parameter(std_param, requires_grad=False)

        self.rnn = nn.RNN(
            input_size=n_features,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            nonlinearity=config.nonlinearity,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        self.fc = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch, n_features, seq_len] -> Scale & transpose to [batch, seq_len, n_features]
        x = (x - self.mean) / self.std
        x = x.transpose(1, 2)
        out, _ = self.rnn(x)
        # Take output of the last time step
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


class SimpleLSTM(nn.Module):
    """
    A configurable Long Short-Term Memory (LSTM) network for time-series forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int = 1,
        config: LSTMConfig = None,
        mean: Any = 0.0,
        std: Any = 1.0,
    ) -> None:
        super().__init__()
        mean_param, std_param = prepare_scaling_parameters(n_features, mean, std)
        self.mean = nn.Parameter(mean_param, requires_grad=False)
        self.std = nn.Parameter(std_param, requires_grad=False)

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        num_directions = 2 if config.bidirectional else 1
        self.fc = nn.Linear(config.hidden_dim * num_directions, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch, n_features, seq_len] -> Scale & transpose to [batch, seq_len, n_features]
        x = (x - self.mean) / self.std
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        # Take output of the last time step
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out
