import math

import numpy as np
import pandas as pd
from quant_core.analytics import ccl_from_pairs, implied_ccl


def test_implied_ccl_scalar():
    # GGAL.BA 5420 ARS, GGAL ADR 32.075 USD, 10 shares per ADR
    assert math.isclose(implied_ccl(5420.0, 32.075, 10.0), 1689.79, rel_tol=1e-4)


def test_implied_ccl_invalid_prices_are_nan():
    assert math.isnan(implied_ccl(0.0, 32.0, 10.0))
    assert math.isnan(implied_ccl(5420.0, -1.0, 10.0))


def test_implied_ccl_series_aligns_on_dates():
    local = pd.Series(
        [1000.0, 0.0, 1200.0],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    adr = pd.Series(
        [10.0, 10.0, 12.0],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    ccl = implied_ccl(local, adr, 1.0)

    assert list(ccl.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert np.isnan(ccl.iloc[0])  # local price 0 is not a price
    assert ccl.iloc[1] == 120.0


PAIRS = (("GGAL.BA", "GGAL", 10.0), ("YPFD.BA", "YPF", 10.0))


def _closes(ggal_local, ypf_local):
    idx = pd.to_datetime(["2024-01-01"])
    return pd.DataFrame(
        {
            "GGAL.BA": [ggal_local],
            "GGAL": [10.0],
            "YPFD.BA": [ypf_local],
            "YPF": [10.0],
        },
        index=idx,
    )


def test_ccl_from_pairs_averages_agreeing_pairs():
    ccl = ccl_from_pairs(_closes(1000.0, 1020.0), PAIRS)  # 1000 and 1020 ARS/USD
    assert ccl.iloc[0] == 1010.0


def test_ccl_from_pairs_rejects_diverging_pairs():
    # A 10x ratio error must surface as NaN, never be averaged in.
    ccl = ccl_from_pairs(_closes(1000.0, 100.0), PAIRS)
    assert np.isnan(ccl.iloc[0])


def test_ccl_from_pairs_uses_single_available_pair():
    closes = _closes(1000.0, 1020.0).drop(columns=["YPF"])
    assert ccl_from_pairs(closes, PAIRS).iloc[0] == 1000.0


def test_ccl_from_pairs_no_pairs_is_empty():
    assert ccl_from_pairs(pd.DataFrame({"AAPL": [1.0]}), PAIRS).empty
