import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Idiomatic PyTorch Dataset for time-series forecasting.
    Converts and transposes features/targets into tensors during initialization.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32).transpose(1, 2)
        # Select target feature (index 0) and final step of the horizon if 3D
        if len(y.shape) == 3:
            y_target = y[:, -1, 0]
        else:
            y_target = y
        self.y = torch.tensor(y_target, dtype=torch.float32).flatten().unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
