# tests/unit/test_transforms.py

import numpy as np
import pytest

from trading_bot.core.transforms import (
    BaseTransform,
    FeaturePipeline,
    LogReturnTransform,
    RatioTransform,
)


def test_log_return_transform_1d():
    transform = LogReturnTransform(col_idx=0)
    data = [10.0, 11.0, 12.1]
    expected = np.diff(np.log(data))

    res = transform.transform(data)
    assert np.allclose(res, expected)


def test_log_return_transform_2d():
    transform = LogReturnTransform(col_idx=1)
    data = np.array([[1.0, 10.0], [2.0, 11.0], [3.0, 12.1]], dtype=np.float32)
    # Column 1 returns: log(11/10) and log(12.1/11)
    expected = np.diff(np.log(data[:, 1])).reshape(-1, 1)

    res = transform.transform(data)
    assert res.shape == (2, 1)
    assert np.allclose(res, expected)


def test_log_return_transform_error():
    transform = LogReturnTransform(col_idx=2)
    data = np.array([[1.0, 10.0], [2.0, 11.0]])
    with pytest.raises(ValueError, match="Column index 2 out of bounds"):
        transform.transform(data)


def test_ratio_transform():
    transform = RatioTransform(num_idx=1, den_idx=0)
    data = np.array([[2.0, 10.0], [4.0, 12.0]], dtype=np.float32)
    # Expected: ln(10/2) and ln(12/4)
    expected = np.array([[np.log(10.0 / 2.0)], [np.log(12.0 / 4.0)]], dtype=np.float32)

    res = transform.transform(data)
    assert res.shape == (2, 1)
    assert np.allclose(res, expected)


def test_ratio_transform_error():
    transform = RatioTransform(num_idx=0, den_idx=1)
    with pytest.raises(ValueError, match="RatioTransform requires a 2D input matrix"):
        transform.transform([1.0, 2.0])


def test_feature_pipeline_alignment():
    # LogReturn reduces length by 1. Ratio does not.
    # Input has length 3. LogReturn output will have length 2. Ratio output will have length 3.
    # FeaturePipeline should slice Ratio output to length 2 to align.
    t_log = LogReturnTransform(col_idx=0)
    t_ratio = RatioTransform(num_idx=1, den_idx=0)
    pipeline = FeaturePipeline(transforms=[t_log, t_ratio])

    data = np.array([[10.0, 20.0], [11.0, 23.0], [12.1, 26.62]], dtype=np.float32)

    # Manual calculation
    # Log return of col 0: [ln(11/10), ln(12.1/11)] = [0.09531018, 0.09531018]
    # Ratio of col 1 / col 0: [ln(2), ln(23/11), ln(26.62/12.1)] = [0.69314718, 0.73759894, 0.78845736]
    # Aligned ratio (last 2 elements): [0.73759894, 0.78845736]
    # Horizontally stacked:
    # [[0.09531018, 0.73759894],
    #  [0.09531018, 0.78845736]]
    expected = np.array(
        [
            [np.log(11.0 / 10.0), np.log(23.0 / 11.0)],
            [np.log(12.1 / 11.0), np.log(26.62 / 12.1)],
        ],
        dtype=np.float32,
    )

    res = pipeline.transform(data)
    assert res.shape == (2, 2)
    assert np.allclose(res, expected)


def test_serialization_roundtrip():
    t_log = LogReturnTransform(col_idx=3)
    t_ratio = RatioTransform(num_idx=1, den_idx=3)
    pipeline = FeaturePipeline(transforms=[t_log, t_ratio])

    serialized = pipeline.to_dict()

    # Reconstruct from dict
    pipeline_reconstructed = BaseTransform.from_dict(serialized)

    assert isinstance(pipeline_reconstructed, FeaturePipeline)
    assert len(pipeline_reconstructed.transforms) == 2
    assert isinstance(pipeline_reconstructed.transforms[0], LogReturnTransform)
    assert pipeline_reconstructed.transforms[0].col_idx == 3
    assert isinstance(pipeline_reconstructed.transforms[1], RatioTransform)
    assert pipeline_reconstructed.transforms[1].num_idx == 1
    assert pipeline_reconstructed.transforms[1].den_idx == 3
