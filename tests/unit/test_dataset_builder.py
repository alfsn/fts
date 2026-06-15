# tests/unit/test_dataset_builder.py

import numpy as np
import pytest

from trading_bot.core.dataset import DatasetBuilder


def test_split_train_val_purged_embargoed():
    # Setup sequential data
    X = np.arange(100).reshape(100, 1)
    y = np.arange(100).reshape(100, 1)

    # Split with 20% validation, horizon=5, embargo_pct=0.02 (which rounds to 2 bars)
    X_train, y_train, X_val, y_val = DatasetBuilder.split_train_val_purged_embargoed(
        X, y, val_ratio=0.2, horizon=5, embargo_pct=0.02
    )

    # 100 samples -> val_size is 20, train_size is 80.
    # Validation indices are 80 to 99.
    # Initial train indices are 0 to 79.
    # Purging: remove training indices where i >= 80 - 5 (i.e. i >= 75).
    # So purged train indices should be 0 to 74.
    # Embargoing: since all training is before validation here, embargoing after validation does not affect training.
    # So final train indices should be 0 to 74.
    assert len(X_val) == 20
    assert len(X_train) == 75
    assert X_val[0, 0] == 80
    assert X_train[-1, 0] == 74


def test_split_train_val_purged_embargoed_general():
    # Test fallback behavior when data is too small
    X = np.arange(5).reshape(5, 1)
    y = np.arange(5).reshape(5, 1)

    X_train, y_train, X_val, y_val = DatasetBuilder.split_train_val_purged_embargoed(
        X, y, val_ratio=0.2, horizon=1, embargo_pct=0.0
    )
    # n_samples = 5. val_ratio=0.2 -> val_size=1. train_size=4.
    # val_indices = [4]
    # initial train = [0, 1, 2, 3]
    # purging: horizon=1 -> index < 4 - 1 = 3. So purged train = [0, 1, 2].
    assert len(X_val) == 1
    assert len(X_train) == 3
    assert X_val[0, 0] == 4
    assert X_train[-1, 0] == 2
