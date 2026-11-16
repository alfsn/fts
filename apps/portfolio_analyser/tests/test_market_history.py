"""PR 2a-3 §6: CCL is implied from split-adjusted closes (D13).

Raw YPFD.BA closes jump ÷10 on 2026-08-03 (10:1 local split) while the ADR
does not, so today's depositary ratio (10) is only right on split-adjusted data.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from data_layer.market_history import (
    ccl_series,
    fx_context_at,
    history_symbols,
    usd_per_unit,
    valuation_date,
)
from quant_core.enums import CorporateActionType, Currency
from quant_core.models import CashBalance, CorporateAction, Portfolio

INDEX = pd.DatetimeIndex(
    ["2026-07-30", "2026-07-31", "2026-08-03", "2026-08-04"], tz="UTC"
)
RAW_CLOSES = pd.DataFrame(
    {
        "YPFD.BA": [81325.0, 82900.0, 8105.0, 7840.0],  # as traded
        "YPF": [52.15, 52.54, 51.06, 49.64],
        "GGAL.BA": [15600.0, 15780.0, 15870.0, 15800.0],  # agrees with the YPF pair
        "GGAL": [100.0, 100.0, 100.0, 100.0],
    },
    index=INDEX,
)
ACTIONS = {
    "YPFD.BA": [
        CorporateAction(
            instrument_id="YPFD.BA",
            action_type=CorporateActionType.SPLIT,
            ex_date=datetime(2026, 8, 3, tzinfo=timezone.utc),
            ratio=10.0,
        )
    ]
}


def test_ccl_is_continuous_across_the_split_on_split_adjusted_closes():
    ccl = ccl_series(RAW_CLOSES, ACTIONS)

    assert ccl.notna().all()
    # YPF pair on 2026-07-31: (82,900 / 10) x 10 / 52.54 = 1,577.8; GGAL pair 1,578.0.
    np.testing.assert_allclose(ccl.loc["2026-07-31"], (1577.8 + 1578.0) / 2, rtol=1e-3)
    assert abs(ccl.loc["2026-08-03"] / ccl.loc["2026-07-31"] - 1) < 0.02


def test_ccl_from_raw_closes_is_nan_before_the_split():
    """Guard: without the adjustment the YPF pair is 10x off, the pairs diverge
    and the divergence check correctly refuses every pre-split day."""
    ccl = ccl_series(RAW_CLOSES, {})

    assert ccl.loc[:"2026-07-31"].isna().all()
    assert ccl.loc["2026-08-03":].notna().all()


def test_valuation_date_and_fx_share_one_as_of_date():
    portfolio = Portfolio(
        positions=[],
        cash_balances=[CashBalance(currency=Currency.ARS, total=1, available=1)],
    )
    fx = usd_per_unit(ccl_series(RAW_CLOSES, ACTIONS), INDEX)
    as_of = valuation_date(portfolio, RAW_CLOSES, fx)
    assert as_of == INDEX[-1]
    context = fx_context_at(portfolio, fx, as_of)
    assert (
        1 / context.get_rate(Currency.ARS) == ccl_series(RAW_CLOSES, ACTIONS).iloc[-1]
    )


def test_valuation_date_is_the_last_day_every_held_listing_has_a_price():
    from quant_core.models import Position

    portfolio = Portfolio(
        positions=[
            Position(
                instrument_id="YPF",
                provider_symbol="YPFD.BA",
                currency=Currency.ARS,
                quantity=1,
                cost_basis=1,
            )
        ],
        cash_balances=[],
    )
    closes = RAW_CLOSES.copy()
    closes.loc["2026-08-04", "YPFD.BA"] = np.nan  # not traded yet on the last day
    fx = usd_per_unit(ccl_series(closes, ACTIONS), INDEX)
    # Forward-filled within 5 sessions, so the last day still counts.
    assert valuation_date(portfolio, closes, fx) == INDEX[-1]
    closes["YPFD.BA"] = np.nan
    with pytest.raises(ValueError, match="YPFD.BA"):
        valuation_date(portfolio, closes, fx)


def test_history_symbols_include_the_ccl_pairs_for_ars():
    portfolio = Portfolio(
        positions=[],
        cash_balances=[CashBalance(currency=Currency.ARS, total=1, available=1)],
    )
    assert {"GGAL.BA", "GGAL", "YPFD.BA", "YPF"} <= history_symbols(portfolio)
