"""PR 2b-1: holdings, cash and values replayed from a hand-written ledger.

Conventions under test (plans/PR2b.md): every event on date d is in the close
of d; only deposits, withdrawals and in-kind transfers are external flows;
dividends and fees are internal; a held listing without a price is NaN, never
0; splits come from corporate actions and multiply the quantity held at the
previous close.
"""

import logging
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest
from quant_core.analytics import compute_twr
from quant_core.enums import CorporateActionType, Currency, TransactionType
from quant_core.ledger import (
    MAX_STALE_SESSIONS,
    cash,
    drop_unvalued_days,
    external_flows,
    forward_fill,
    holdings,
    portfolio_values,
    reconcile,
    replay,
    symbols_by_listing,
)
from quant_core.models import (
    CashBalance,
    CorporateAction,
    Portfolio,
    Position,
    Transaction,
)

USD, ARS = Currency.USD, Currency.ARS
DATES = pd.DatetimeIndex(pd.bdate_range("2024-01-02", "2024-01-12"), tz="UTC")


def tx(day, type_, currency=USD, **kw):
    return Transaction(
        date=date(2024, 1, day), type=TransactionType(type_), currency=currency, **kw
    )


def _closes(**series):
    return pd.DataFrame({k: v for k, v in series.items()}, index=DATES)


def _fx(ars_per_usd=1000.0):
    return pd.DataFrame({USD: 1.0, ARS: 1.0 / ars_per_usd}, index=DATES)


FX = _fx()
AAPL = _closes(AAPL=[100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0])


# --- holdings and cash ----------------------------------------------------------


def test_buy_sell_dividend_fee_arithmetic():
    ledger = [
        tx(2, "deposit", amount=10_000),
        tx(3, "buy", instrument_id="AAPL", quantity=10, price=101.0, fees=1.0),
        tx(5, "dividend", instrument_id="AAPL", amount=2.40),
        tx(8, "sell", instrument_id="AAPL", quantity=4, price=105.0, fees=1.0),
        tx(9, "fee", amount=3.0),
    ]
    h = holdings(ledger, {}, DATES)
    c = cash(ledger, DATES)

    assert list(h.columns) == [("AAPL", USD)]
    assert h[("AAPL", USD)].loc["2024-01-02"] == 0
    assert h[("AAPL", USD)].loc["2024-01-03":"2024-01-05"].tolist() == [10, 10, 10]
    assert h[("AAPL", USD)].loc["2024-01-08":].tolist() == [6] * 5
    assert c[USD].loc["2024-01-02"] == 10_000
    assert c[USD].loc["2024-01-03"] == pytest.approx(10_000 - 1011.0)
    assert c[USD].loc["2024-01-05"] == pytest.approx(10_000 - 1011.0 + 2.40)
    assert c[USD].loc["2024-01-08"] == pytest.approx(10_000 - 1011.0 + 2.40 + 419.0)
    assert c[USD].loc["2024-01-09"] == pytest.approx(
        10_000 - 1011.0 + 2.40 + 419.0 - 3.0
    )


def test_transfer_in_adds_shares_without_cash_and_is_a_flow_at_market_value():
    ledger = [tx(2, "transfer_in", instrument_id="AAPL", quantity=5)]
    h, c = holdings(ledger, {}, DATES), cash(ledger, DATES)
    assert h[("AAPL", USD)].iloc[0] == 5
    assert c.empty or (c == 0).all().all()

    flows = external_flows(ledger, AAPL, FX, DATES)
    assert flows.loc["2024-01-02"] == pytest.approx(5 * 100.0)

    values = portfolio_values(h, c, AAPL, FX, symbols_by_listing(ledger))
    # The transfer explains the whole first value; TWR is only the price path.
    assert values.iloc[0] == pytest.approx(500.0)
    assert compute_twr(values, flows) == pytest.approx(108.0 / 100.0 - 1)


def test_transfer_on_a_day_without_a_price_raises():
    closes = AAPL.copy()
    closes.iloc[0] = np.nan
    with pytest.raises(ValueError, match="AAPL"):
        external_flows(
            [tx(2, "transfer_in", instrument_id="AAPL", quantity=5)], closes, FX, DATES
        )


def test_split_multiplies_the_quantity_held_at_the_previous_close():
    ex = datetime(2024, 1, 8, tzinfo=timezone.utc)
    split = CorporateAction(
        instrument_id="NVDA",
        action_type=CorporateActionType.SPLIT,
        ex_date=ex,
        ratio=10.0,
    )
    ledger = [
        tx(2, "deposit", amount=100_000),
        tx(2, "buy", instrument_id="NVDA", quantity=10, price=1200.0),
        tx(
            8, "buy", instrument_id="NVDA", quantity=5, price=121.0
        ),  # ex-date: new shares
    ]
    h = holdings(ledger, {"NVDA": [split]}, DATES)
    q = h[("NVDA", USD)]
    assert q.loc["2024-01-05"] == 10
    assert q.loc["2024-01-08"] == 10 * 10 + 5
    assert q.loc["2024-01-12"] == 105


def test_value_is_continuous_across_a_split_on_raw_closes():
    ex = datetime(2024, 1, 8, tzinfo=timezone.utc)
    split = CorporateAction(
        instrument_id="NVDA",
        action_type=CorporateActionType.SPLIT,
        ex_date=ex,
        ratio=10.0,
    )
    raw = _closes(
        NVDA=[1200.0, 1210.0, 1220.0, 1230.0, 124.0, 125.0, 126.0, 127.0, 128.0]
    )
    ledger = [
        tx(2, "deposit", amount=12_000),
        tx(2, "buy", instrument_id="NVDA", quantity=10, price=1200.0),
    ]
    h, c = holdings(ledger, {"NVDA": [split]}, DATES), cash(ledger, DATES)

    values = portfolio_values(h, c, raw, FX, symbols_by_listing(ledger))

    assert values.loc["2024-01-05"] == pytest.approx(12_300)
    assert values.loc["2024-01-08"] == pytest.approx(12_400)  # 100 x 124, no fake -90%
    # Split-adjusted closes with split-applied quantities would be 10x wrong:
    # that is the "never mix bases" rule, so the raw path is the only valid one.


# --- flows, returns and currencies ---------------------------------------------


def _twr(ledger, closes=AAPL, fx=FX, actions=None):
    result = replay(ledger, closes, fx, actions or {}, DATES)
    return compute_twr(result.values, result.flows)


def test_dividend_raises_twr_and_is_not_a_flow():
    base = [
        tx(2, "deposit", amount=1_000),
        tx(2, "buy", instrument_id="AAPL", quantity=10, price=100.0),
    ]
    with_dividend = base + [tx(5, "dividend", instrument_id="AAPL", amount=10.0)]

    assert _twr(base) == pytest.approx(0.08)
    assert _twr(with_dividend) == pytest.approx(0.09)
    flows = external_flows(with_dividend, AAPL, FX, DATES)
    assert flows.loc["2024-01-05"] == 0.0


def test_deposit_leaves_twr_unchanged_and_fee_lowers_it():
    base = [
        tx(2, "deposit", amount=1_000),
        tx(2, "buy", instrument_id="AAPL", quantity=10, price=100.0),
    ]
    # Deposit and invest it at that day's close: the weights don't change, so
    # TWR doesn't either (a deposit left as idle cash would dilute later returns).
    topped_up = base + [
        tx(5, "deposit", amount=1_030),
        tx(5, "buy", instrument_id="AAPL", quantity=10, price=103.0),
    ]
    assert _twr(topped_up) == pytest.approx(_twr(base))
    assert _twr(base + [tx(5, "fee", amount=10.0)]) == pytest.approx(0.07)


def test_two_currencies_at_a_ccl_that_jumps():
    fx = _fx(1000.0)
    fx.loc["2024-01-09":, ARS] = 1.0 / 1250.0  # ARS loses 20% against USD
    ledger = [
        tx(2, "deposit", amount=1_000),
        tx(2, "deposit", currency=ARS, amount=1_000_000),
        tx(
            2,
            "buy",
            currency=ARS,
            instrument_id="YPF",
            provider_symbol="YPFD.BA",
            quantity=100,
            price=10_000,
        ),
    ]
    closes = _closes(**{"YPFD.BA": [10_000.0] * 9})
    result = replay(ledger, closes, fx, {}, DATES)

    assert result.values.loc["2024-01-08"] == pytest.approx(1_000 + 1_000_000 / 1_000)
    assert result.values.loc["2024-01-09"] == pytest.approx(1_000 + 1_000_000 / 1_250)
    assert result.flows.loc["2024-01-02"] == pytest.approx(2_000)
    assert compute_twr(result.values, result.flows) == pytest.approx(1_800 / 2_000 - 1)


# --- prices: forward fill, NaN days, calendar --------------------------------------


def test_forward_fill_is_limited_to_max_stale_sessions():
    closes = AAPL.copy()
    closes.iloc[1:8] = np.nan  # 7 sessions without a print
    filled = forward_fill(closes, DATES)
    assert (
        filled["AAPL"].iloc[1 : 1 + MAX_STALE_SESSIONS].tolist()
        == [100.0] * MAX_STALE_SESSIONS
    )
    assert filled["AAPL"].iloc[1 + MAX_STALE_SESSIONS : 8].isna().all()
    assert filled["AAPL"].iloc[8] == 108.0


def test_forward_fill_never_looks_ahead():
    closes = AAPL.copy()
    closes.iloc[0] = np.nan
    assert np.isnan(forward_fill(closes, DATES)["AAPL"].iloc[0])


def test_unpriced_held_listing_makes_the_day_nan_and_it_is_dropped(caplog):
    closes = AAPL.copy()
    closes.iloc[4] = np.nan  # 2024-01-08
    ledger = [
        tx(2, "deposit", amount=1_000),
        tx(2, "buy", instrument_id="AAPL", quantity=10, price=100.0),
    ]
    h, c = holdings(ledger, {}, DATES), cash(ledger, DATES)
    values = portfolio_values(h, c, closes, FX, symbols_by_listing(ledger))
    assert np.isnan(values.loc["2024-01-08"])

    with caplog.at_level(logging.WARNING):
        kept = drop_unvalued_days(values)
    assert "2024-01-08" in caplog.text
    assert len(kept) == len(DATES) - 1 and kept.notna().all()


def test_unpriced_listing_with_zero_quantity_does_not_matter():
    closes = AAPL.copy()
    closes.iloc[4] = np.nan
    ledger = [tx(2, "deposit", amount=1_000)]
    h, c = holdings(ledger, {}, DATES), cash(ledger, DATES)
    values = portfolio_values(h, c, closes, FX, symbols_by_listing(ledger))
    assert values.notna().all()


def test_flow_on_a_dropped_day_raises():
    dates = pd.DatetimeIndex(pd.bdate_range("2024-01-02", "2024-01-31"), tz="UTC")
    closes = pd.DataFrame({"AAPL": 100.0}, index=dates)
    closes.loc["2024-01-08":, "AAPL"] = (
        np.nan
    )  # a dead feed: NaN after 5 stale sessions
    fx = pd.DataFrame({USD: 1.0}, index=dates)
    ledger = [
        tx(2, "deposit", amount=1_000),
        tx(2, "buy", instrument_id="AAPL", quantity=10, price=100.0),
        tx(22, "deposit", amount=500),
    ]
    result = replay(ledger, closes, fx, {}, dates)

    assert result.values.index[-1] == pd.Timestamp(
        "2024-01-12", tz="UTC"
    )  # 5 filled sessions
    assert len(result.dropped) == 13  # Jan 15 to Jan 31
    assert result.flows.loc["2024-01-22"] == 500  # kept, so that it raises below
    with pytest.raises(ValueError, match="without a valuation"):
        compute_twr(result.values, result.flows)


def test_transaction_after_the_calendar_is_rejected():
    with pytest.raises(ValueError, match="after"):
        holdings([tx(31, "deposit", amount=1.0)], {}, DATES)


def test_weekend_dated_transaction_takes_effect_on_the_next_session():
    ledger = [tx(6, "deposit", amount=1_000)]  # a Saturday
    c = cash(ledger, DATES)
    assert c[USD].loc["2024-01-05"] == 0 and c[USD].loc["2024-01-08"] == 1_000
    assert external_flows(ledger, AAPL, FX, DATES).loc["2024-01-08"] == 1_000


# --- reconciliation ---------------------------------------------------------------


def _portfolio(positions, cash_balances, transactions):
    return Portfolio(
        positions=positions, cash_balances=cash_balances, transactions=transactions
    )


def test_missing_trade_is_a_quantity_break_naming_the_listing():
    ledger = [
        tx(2, "deposit", amount=10_000),
        tx(3, "buy", instrument_id="AAPL", quantity=10, price=100.0),
    ]
    snapshot = _portfolio(
        [Position(instrument_id="AAPL", currency=USD, quantity=12, cost_basis=100.0)],
        [CashBalance(currency=USD, total=9_000, available=9_000)],
        ledger,
    )
    breaks = reconcile(
        holdings(ledger, {}, DATES).iloc[-1], cash(ledger, DATES).iloc[-1], snapshot
    )
    assert breaks.quantity == {("AAPL", USD): (10.0, 12.0)}
    assert breaks.cash == {}
    assert not breaks.ok
    assert "AAPL" in breaks.describe()


def test_missing_fee_is_a_cash_break_only():
    ledger = [
        tx(2, "deposit", amount=10_000),
        tx(3, "buy", instrument_id="AAPL", quantity=10, price=100.0),
    ]
    snapshot = _portfolio(
        [Position(instrument_id="AAPL", currency=USD, quantity=10, cost_basis=100.0)],
        [
            CashBalance(currency=USD, total=8_990, available=8_990)
        ],  # a 10 USD custody fee
        ledger,
    )
    breaks = reconcile(
        holdings(ledger, {}, DATES).iloc[-1], cash(ledger, DATES).iloc[-1], snapshot
    )
    assert breaks.quantity == {}
    assert breaks.cash == {USD: (9_000.0, 8_990.0)}
    assert breaks.ok  # cash breaks are warnings (D7)


def test_listing_in_snapshot_but_not_in_ledger_is_a_break():
    ledger = [tx(2, "deposit", amount=10_000)]
    snapshot = _portfolio(
        [Position(instrument_id="YPF", currency=ARS, quantity=100, cost_basis=1.0)],
        [CashBalance(currency=USD, total=10_000, available=10_000)],
        ledger,
    )
    breaks = reconcile(
        holdings(ledger, {}, DATES).iloc[-1], cash(ledger, DATES).iloc[-1], snapshot
    )
    assert breaks.quantity == {("YPF", ARS): (0.0, 100.0)}


# --- model -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            dict(type="buy", instrument_id="AAPL", quantity=1, price=1.0, quantitiy=2),
            "quantitiy",
        ),
        (dict(type="split", instrument_id="AAPL", quantity=1), "corporate"),
        (dict(type="coupon", instrument_id="AL30", amount=1.0), "PR 2c"),
        (dict(type="sell", instrument_id="AAPL", quantity=1), "price"),
        (
            dict(type="transfer_in", instrument_id="AAPL", quantity=1, price=1.0),
            "price",
        ),
        (dict(type="deposit", amount=1.0, instrument_id="AAPL"), "instrument_id"),
        (dict(type="dividend", amount=1.0), "instrument_id"),
        (
            dict(type="buy", instrument_id="AAPL", quantity=-1, price=1.0),
            "greater than 0",
        ),
        (dict(type="fee", amount=1.0, fees=1.0), "fees"),
    ],
)
def test_transaction_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Transaction(date=date(2024, 1, 2), currency=USD, **kwargs)


def test_ledger_must_start_with_a_deposit_or_transfer():
    with pytest.raises(ValueError, match="deposit or transfer_in"):
        _portfolio([], [], [tx(2, "buy", instrument_id="AAPL", quantity=1, price=1.0)])
    _portfolio([], [], [tx(2, "transfer_in", instrument_id="AAPL", quantity=1)])
    _portfolio(
        [],
        [],
        [
            tx(3, "buy", instrument_id="AAPL", quantity=1, price=1.0),
            tx(2, "deposit", amount=5.0),
        ],
    )


def test_listing_maps_to_one_market_symbol_across_positions_and_ledger():
    with pytest.raises(ValueError, match="YPF"):
        _portfolio(
            [
                Position(
                    instrument_id="YPF",
                    currency=ARS,
                    quantity=1,
                    cost_basis=1.0,
                    provider_symbol="YPFD.BA",
                )
            ],
            [],
            [
                tx(
                    2,
                    "transfer_in",
                    currency=ARS,
                    instrument_id="YPF",
                    quantity=1,
                    provider_symbol="YPF.BA",
                )
            ],
        )
    # The same instrument in two currencies is two listings with two symbols.
    _portfolio(
        [
            Position(
                instrument_id="YPF",
                currency=ARS,
                quantity=1,
                cost_basis=1.0,
                provider_symbol="YPFD.BA",
            )
        ],
        [],
        [tx(2, "transfer_in", currency=USD, instrument_id="YPF", quantity=1)],
    )


def test_portfolio_without_transactions_still_validates():
    assert Portfolio(positions=[], cash_balances=[]).transactions == []
