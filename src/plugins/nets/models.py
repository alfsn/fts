# src/plugins/nets/models.py

import torch
import torch.nn as nn

from .schemas import CNNConfig, LSTMConfig, RNNConfig


class SimpleCNN(nn.Module):
    """
    A configurable Convolutional Neural Network (CNN) for time-series forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        config: CNNConfig,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std), requires_grad=False)

        # Build Conv1D layers
        layers = []
        in_channels = 1
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
            dummy_input = torch.zeros(1, 1, input_dim)
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
        config: RNNConfig,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std), requires_grad=False)

        self.rnn = nn.RNN(
            input_size=1,
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
        # Input shape: [batch, 1, seq_len] -> Scale & transpose to [batch, seq_len, 1]
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
        config: LSTMConfig,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std), requires_grad=False)

        self.lstm = nn.LSTM(
            input_size=1,
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
        # Input shape: [batch, 1, seq_len] -> Scale & transpose to [batch, seq_len, 1]
        x = (x - self.mean) / self.std
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        # Take output of the last time step
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out
