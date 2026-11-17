"""PR 2a-1: split factors, un-adjusting vendor data, and read-time adjustment.

Oracle: a recorded Yahoo frame for NVDA around the 10:1 split (ex 2024-06-10)
and the first post-split dividend (0.01, ex 2024-06-11). Yahoo's ``Close`` and
``Volume`` are split-adjusted; ``Adj Close`` is split- and dividend-adjusted.
"""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from quant_core.analytics import (
    adjust,
    split_factor,
    unadjust_dividends,
    unadjust_splits,
)
from quant_core.enums import CorporateActionType, PriceAdjustment
from quant_core.models import CorporateAction

FIXTURE = Path(__file__).parent / "fixtures" / "nvda_2024_yahoo.csv"
NY = ZoneInfo("America/New_York")
BA = ZoneInfo("America/Argentina/Buenos_Aires")


def _split(ex: datetime, ratio: float, symbol: str = "NVDA") -> CorporateAction:
    return CorporateAction(
        instrument_id=symbol,
        action_type=CorporateActionType.SPLIT,
        ex_date=ex,
        ratio=ratio,
    )


def _dividend(ex: datetime, amount: float, symbol: str = "NVDA") -> CorporateAction:
    return CorporateAction(
        instrument_id=symbol,
        action_type=CorporateActionType.DIVIDEND,
        ex_date=ex,
        amount=amount,
    )


NVDA_SPLIT = _split(datetime(2024, 6, 10, 9, 30, tzinfo=NY), 10.0)
NVDA_DIVIDEND = _dividend(datetime(2024, 6, 11, 9, 30, tzinfo=NY), 0.01)


@pytest.fixture(scope="module")
def yahoo() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, index_col="Date", parse_dates=True)


# --- split_factor -----------------------------------------------------------


def test_split_factor_counts_splits_strictly_after_each_date():
    index = pd.to_datetime(["2024-06-07", "2024-06-10", "2024-06-11"])
    factor = split_factor(index, [NVDA_SPLIT])
    assert factor.tolist() == [10.0, 1.0, 1.0]


def test_split_factor_compounds_and_handles_reverse_splits():
    index = pd.to_datetime(["2020-01-01", "2022-01-01", "2025-01-01"])
    splits = [
        _split(datetime(2021, 7, 20, tzinfo=NY), 4.0),
        _split(datetime(2024, 6, 10, tzinfo=NY), 0.1),  # 1:10 reverse
    ]
    assert split_factor(index, splits).tolist() == [0.4, 0.1, 1.0]


def test_split_factor_compares_local_dates_not_instants():
    """A tz-aware intraday bar at the BYMA open on the ex-date is post-split;
    a tz-naive daily bar dated the session before is pre-split."""
    ypf_split = _split(datetime(2026, 8, 3, 11, 0, tzinfo=BA), 10.0, "YPFD.BA")
    intraday = pd.DatetimeIndex(
        [
            datetime(2026, 7, 31, 16, 0, tzinfo=BA),
            datetime(2026, 8, 3, 11, 0, tzinfo=BA),
        ]
    )
    assert split_factor(intraday, [ypf_split]).tolist() == [10.0, 1.0]
    daily = pd.to_datetime(["2026-07-31", "2026-08-03"])
    assert split_factor(daily, [ypf_split]).tolist() == [10.0, 1.0]
    # UTC-keyed dates (how the cache stores them) also work.
    utc = pd.DatetimeIndex(["2026-07-31", "2026-08-03"], tz="UTC")
    assert split_factor(utc, [ypf_split]).tolist() == [10.0, 1.0]


def test_split_factor_ignores_dividends():
    index = pd.to_datetime(["2024-06-07"])
    assert split_factor(index, [NVDA_DIVIDEND]).tolist() == [1.0]


def test_actions_of_several_instruments_are_rejected():
    """Callers pass one instrument's actions; mixing them would be a silent bug."""
    other = _split(datetime(2024, 6, 10, tzinfo=NY), 10.0, "AAPL")
    with pytest.raises(ValueError):
        split_factor(pd.to_datetime(["2024-06-07"]), [NVDA_SPLIT, other])


def test_split_without_ratio_is_rejected():
    with pytest.raises(ValidationError):
        CorporateAction(
            instrument_id="X",
            action_type=CorporateActionType.SPLIT,
            ex_date=datetime(2024, 1, 1),
        )


def test_corporate_action_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        CorporateAction(
            instrument_id="X",
            action_type=CorporateActionType.SPLIT,
            ex_date=datetime(2024, 1, 1),
            ratio=2.0,
            ratios=2.0,
        )


# --- unadjust_splits (adapter side) -------------------------------------------


def test_unadjust_splits_restores_traded_prices_and_volumes(yahoo):
    raw = unadjust_splits(yahoo, [NVDA_SPLIT])

    assert raw.loc["2024-06-06", "Close"] == pytest.approx(1209.98, abs=0.01)
    assert raw.loc["2024-06-06", "Volume"] == pytest.approx(66_469_600)
    # The ex-date bar itself is already in new shares.
    assert raw.loc["2024-06-10", "Close"] == pytest.approx(121.79, abs=0.01)
    assert raw.loc["2024-06-10", "Volume"] == pytest.approx(313_434_100)
    # Non-price columns are left alone.
    assert raw["Dividends"].equals(yahoo["Dividends"])


def test_unadjust_splits_is_a_noop_without_splits(yahoo):
    pd.testing.assert_frame_equal(unadjust_splits(yahoo, []), yahoo, check_dtype=False)


def test_unadjust_dividends_scales_by_splits_after_the_ex_date():
    dividends = pd.Series(
        [0.004, 0.01],
        index=pd.DatetimeIndex(
            [
                datetime(2024, 3, 5, 9, 30, tzinfo=NY),
                datetime(2024, 6, 11, 9, 30, tzinfo=NY),
            ]
        ),
    )
    raw = unadjust_dividends(dividends, [NVDA_SPLIT])
    assert raw.tolist() == pytest.approx([0.04, 0.01])


# --- adjust (read side) -------------------------------------------------------


def test_adjust_raw_returns_the_input(yahoo):
    raw = unadjust_splits(yahoo, [NVDA_SPLIT])["Close"]
    assert adjust(raw, [NVDA_SPLIT, NVDA_DIVIDEND], PriceAdjustment.RAW).equals(raw)


def test_adjust_split_reproduces_yahoo_close(yahoo):
    raw = unadjust_splits(yahoo, [NVDA_SPLIT])["Close"]
    adjusted = adjust(raw, [NVDA_SPLIT, NVDA_DIVIDEND], PriceAdjustment.SPLIT)
    pd.testing.assert_series_equal(
        adjusted, yahoo["Close"], rtol=1e-9, check_names=False
    )


def test_adjust_total_return_reproduces_yahoo_adj_close(yahoo):
    """Adj Close also carries dividends after the window, which scale every
    price equally, so compare the series normalised to their last value."""
    raw = unadjust_splits(yahoo, [NVDA_SPLIT])["Close"]
    actions = [NVDA_SPLIT, NVDA_DIVIDEND]
    total = adjust(raw, actions, PriceAdjustment.TOTAL_RETURN)
    expected = yahoo["Adj Close"]
    np.testing.assert_allclose(
        (total / total.iloc[-1]).to_numpy(),
        (expected / expected.iloc[-1]).to_numpy(),
        rtol=1e-5,
    )
    # And the dividend really did something: the day before the ex-date is scaled.
    assert (
        total.loc["2024-06-10"]
        < adjust(raw, [NVDA_SPLIT], PriceAdjustment.SPLIT).loc["2024-06-10"]
    )


def test_adjust_total_return_uses_split_adjusted_dividend_and_previous_close():
    """A dividend paid in old shares, before a split: factor 1 - D/P on a
    consistent basis, applied to every price before the dividend's ex-date."""
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    raw = pd.Series([1000.0, 1000.0, 100.0, 100.0], index=index)  # 10:1 split ex 01-03
    split = _split(datetime(2024, 1, 3, tzinfo=NY), 10.0)
    dividend = _dividend(datetime(2024, 1, 2, tzinfo=NY), 10.0)  # raw: 10 per old share
    total = adjust(raw, [split, dividend], PriceAdjustment.TOTAL_RETURN)
    # Split-adjusted: [100, 100, 100, 100]; dividend 1.0 per new share, P_prev 100.
    assert total.tolist() == pytest.approx([99.0, 100.0, 100.0, 100.0])


def test_adjust_total_return_rejects_dividend_larger_than_price():
    index = pd.to_datetime(["2024-01-01", "2024-01-02"])
    raw = pd.Series([10.0, 10.0], index=index)
    with pytest.raises(ValueError):
        adjust(
            raw,
            [_dividend(datetime(2024, 1, 2, tzinfo=NY), 10.0)],
            PriceAdjustment.TOTAL_RETURN,
        )


def test_adjust_ignores_dividends_outside_the_series():
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    raw = pd.Series([10.0, 11.0], index=index)
    actions = [
        _dividend(
            datetime(2024, 1, 2, tzinfo=NY), 1.0
        ),  # ex on the first date: nothing before it
        _dividend(datetime(2024, 1, 9, tzinfo=NY), 1.0),  # after the series
    ]
    assert adjust(raw, actions, PriceAdjustment.TOTAL_RETURN).tolist() == [10.0, 11.0]


def test_adjust_keeps_nan_and_index(yahoo):
    raw = unadjust_splits(yahoo, [NVDA_SPLIT])["Close"].copy()
    raw.iloc[3] = np.nan
    adjusted = adjust(raw, [NVDA_SPLIT, NVDA_DIVIDEND], PriceAdjustment.TOTAL_RETURN)
    assert adjusted.index.equals(raw.index)
    assert np.isnan(adjusted.iloc[3]) and adjusted.notna().sum() == len(raw) - 1


def test_adjust_frame_applies_each_symbols_actions():
    from quant_core.analytics import adjust_frame

    index = pd.DatetimeIndex(["2026-07-31", "2026-08-03"], tz="UTC")
    closes = pd.DataFrame(
        {"YPFD.BA": [82900.0, 8105.0], "YPF": [52.54, 51.06]}, index=index
    )
    actions = {
        "YPFD.BA": [_split(datetime(2026, 8, 3, 11, 0, tzinfo=BA), 10.0, "YPFD.BA")]
    }
    out = adjust_frame(closes, actions, PriceAdjustment.SPLIT)
    assert out["YPFD.BA"].tolist() == pytest.approx([8290.0, 8105.0])
    assert out["YPF"].tolist() == pytest.approx([52.54, 51.06])
