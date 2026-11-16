"""PR 2a-3 §6: CCL is implied from split-adjusted closes (D13).

Raw YPFD.BA closes jump ÷10 on 2026-08-03 (10:1 local split) while the ADR
does not, so today's depositary ratio (10) is only right on split-adjusted data.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from data_layer.market_history import ccl_series, history_symbols, latest_usd_per_unit
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


def test_latest_usd_per_unit_uses_the_adjusted_series():
    rate, as_of = latest_usd_per_unit(RAW_CLOSES, ACTIONS)[Currency.ARS]
    assert as_of == INDEX[-1]
    assert 1 / rate == ccl_series(RAW_CLOSES, ACTIONS).iloc[-1]


def test_history_symbols_include_the_ccl_pairs_for_ars():
    portfolio = Portfolio(
        positions=[],
        cash_balances=[CashBalance(currency=Currency.ARS, total=1, available=1)],
    )
    assert {"GGAL.BA", "GGAL", "YPFD.BA", "YPF"} <= history_symbols(portfolio)
