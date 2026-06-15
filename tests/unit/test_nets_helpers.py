# tests/unit/test_nets_helpers.py

import numpy as np
import torch
import torch.nn as nn
from nets.evaluator import ValidationEvaluator
from nets.training import TimeSeriesDataset
from torch.utils.data import DataLoader


def test_time_series_dataset():
    X = np.random.rand(10, 20, 1)  # 10 samples, 20 sequence length, 1 feature
    y = np.random.rand(10, 1)

    dataset = TimeSeriesDataset(X, y)
    assert len(dataset) == 10

    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Check shape: should be [batch, 1, 20]
    for batch_X, batch_y in loader:
        assert batch_X.shape == (4, 1, 20) or batch_X.shape == (2, 1, 20)
        assert batch_y.shape == (4, 1) or batch_y.shape == (2, 1)
        break


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(20, 1)

    def forward(self, x):
        # x is [batch, 1, 20]
        # squeeze to [batch, 20]
        return self.linear(x.squeeze(1))


def test_validation_evaluator():
    X = np.random.rand(10, 20, 1)
    y = np.random.rand(10, 1)

    dataset = TimeSeriesDataset(X, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)
    model = SimpleModel()
    criterion = nn.MSELoss()

    metrics = ValidationEvaluator.evaluate(model, loader, criterion)
    assert "loss" in metrics
    assert "ic" in metrics
    assert "directional_accuracy" in metrics
    assert "ic_decay" in metrics
    assert len(metrics["ic_decay"]) == 6
