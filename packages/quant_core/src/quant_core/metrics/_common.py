"""Helpers and defaults shared by the metrics pipelines."""

import pandas as pd

# 252 *observations*: a year of stock trading days, about 8 months of crypto
# (365 a year). Below it a beta or a forecast score is mostly noise (PR 7 D5,
# PR 10 D6). Kept as an observation count on purpose, not one year of the data.
MIN_SAMPLE_OBS = 252


def _empty_frame(columns) -> pd.DataFrame:
    return pd.DataFrame(columns=columns, dtype=float)
