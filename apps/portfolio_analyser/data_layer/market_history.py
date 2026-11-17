"""Binds a broker snapshot to cached market history (prices, corporate actions, FX)."""

from datetime import datetime, timedelta, timezone
from typing import Mapping, Sequence

import pandas as pd
from quant_core.analytics import adjust_frame, ccl_from_pairs
from quant_core.enums import Currency, PriceAdjustment
from quant_core.ledger import forward_fill
from quant_core.models import CorporateAction, FXContext, MissingFXRateError, Portfolio
from quant_data.reference import ARS_CCL_PAIRS

HISTORY_YEARS = 5
LEDGER_LEAD_DAYS = 7  # history starts a week before the first transaction

Actions = Mapping[str, Sequence[CorporateAction]]


def history_window(
    portfolio: Portfolio | None = None, now: datetime | None = None
) -> tuple[datetime, datetime]:
    """Five years back, or from just before the ledger's first line if earlier."""
    end = now or datetime.now(timezone.utc)
    start = end - timedelta(days=round(365.25 * HISTORY_YEARS))
    if portfolio is not None and portfolio.transactions:
        first = min(t.date for t in portfolio.transactions)
        ledger_start = datetime.combine(
            first - timedelta(days=LEDGER_LEAD_DAYS), datetime.min.time(), timezone.utc
        )
        start = min(start, ledger_start)
    return start, end


def held_currencies(portfolio: Portfolio) -> set[Currency]:
    return {p.currency for p in portfolio.positions} | {
        c.currency for c in portfolio.cash_balances
    }


def history_symbols(portfolio: Portfolio) -> set[str]:
    """Position and ledger symbols plus whatever FX needs for the currencies held
    (and for the benchmark's currency; the benchmark symbol itself is fetched
    only by `client-review`)."""
    symbols = {p.market_symbol for p in portfolio.positions}
    symbols |= {t.market_symbol for t in portfolio.transactions if t.instrument_id}
    currencies = (
        held_currencies(portfolio)
        | {t.currency for t in portfolio.transactions}
        | {portfolio.benchmark.currency}
    )
    if Currency.ARS in currencies | {portfolio.base_currency}:
        symbols |= {s for local, adr, _ in ARS_CCL_PAIRS for s in (local, adr)}
    return symbols


def ccl_series(closes: pd.DataFrame, actions: Actions) -> pd.Series:
    """
    Daily CCL in ARS per USD. The pairs' depositary ratios are today's, so the
    raw closes are split-adjusted first (PR 2a §6): on raw closes the YPF pair
    is 10x off before 2026-08-03 and the divergence check NaNs every day.
    """
    return ccl_from_pairs(
        adjust_frame(closes, actions, PriceAdjustment.SPLIT), ARS_CCL_PAIRS
    )


def usd_per_unit(ccl: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """USD value of one unit of each supported currency per session, forward-
    filled over local holidays (PR 2b D9). Columns are Currency."""
    # USD is 1 on every session: built on the CCL's index it would be missing
    # whenever no ARS pair was fetched (a USD-only portfolio).
    frame = forward_fill(pd.DataFrame({Currency.ARS: 1.0 / ccl}), dates)
    frame.insert(0, Currency.USD, 1.0)
    return frame


def fx_in_base(fx: pd.DataFrame, base: Currency) -> pd.DataFrame:
    """`usd_per_unit` rebased: units of `base` per unit of each currency."""
    return fx.div(fx[base], axis=0)


def valuation_date(
    portfolio: Portfolio, closes: pd.DataFrame, fx: pd.DataFrame
) -> pd.Timestamp:
    """
    The latest session on which every held listing has a (forward-filled)
    close and every held currency an FX rate: price and FX share one as-of date.
    """
    needed = sorted({p.market_symbol for p in portfolio.positions})
    missing = [
        s for s in needed if s not in closes.columns or closes[s].notna().sum() == 0
    ]
    if missing:
        raise ValueError(f"No market price for {', '.join(missing)}")
    priced = forward_fill(closes, closes.index)[needed].notna().all(axis=1)
    currencies = sorted(held_currencies(portfolio) | {portfolio.base_currency})
    rates = fx.reindex(closes.index)[currencies].notna().all(axis=1)
    valued = priced & rates
    if not valued.any():
        raise MissingFXRateError(
            "No market FX rate for "
            + ", ".join(
                c.value for c in currencies if c not in fx or fx[c].notna().sum() == 0
            )
        )
    return valued[valued].index[-1]


def fx_context_at(
    portfolio: Portfolio, fx: pd.DataFrame, as_of: pd.Timestamp
) -> FXContext:
    """FXContext rates = units of base currency per unit of each held currency.

    Raises MissingFXRateError instead of falling back to a guessed rate.
    """
    base = portfolio.base_currency
    needed = held_currencies(portfolio) | {base}
    rates = {}
    for currency in needed:
        rate = fx.at[as_of, currency] if currency in fx.columns else float("nan")
        if not rate > 0:
            raise MissingFXRateError(f"No market FX rate for {currency.value}")
        rates[currency] = rate
    return FXContext(
        base_currency=base, rates={c: rates[c] / rates[base] for c in needed}
    )


def price_positions(
    portfolio: Portfolio, closes: pd.DataFrame, as_of: pd.Timestamp
) -> None:
    """Set every position's `current_price` to its raw close as of `as_of`;
    the YAML price, if any, is ignored for valuation (PR 2b §4)."""
    filled = forward_fill(closes, closes.index)
    for position in portfolio.positions:
        position.current_price = float(filled.at[as_of, position.market_symbol])
