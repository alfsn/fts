"""Binds a broker snapshot to cached market history (prices + FX)."""

from datetime import datetime, timedelta, timezone

import pandas as pd
from quant_core.analytics import ARS_CCL_PAIRS, ccl_from_pairs
from quant_core.enums import Currency
from quant_core.models import FXContext, MissingFXRateError, Portfolio

HISTORY_YEARS = 5


def history_window(now: datetime | None = None) -> tuple[datetime, datetime]:
    end = now or datetime.now(timezone.utc)
    return end - timedelta(days=round(365.25 * HISTORY_YEARS)), end


def held_currencies(portfolio: Portfolio) -> set[Currency]:
    return {p.currency for p in portfolio.positions} | {
        c.currency for c in portfolio.cash_balances
    }


def history_symbols(portfolio: Portfolio) -> set[str]:
    """Position symbols plus whatever FX needs for the currencies held."""
    symbols = {p.market_symbol for p in portfolio.positions}
    if Currency.ARS in held_currencies(portfolio) | {portfolio.base_currency}:
        symbols |= {s for local, adr, _ in ARS_CCL_PAIRS for s in (local, adr)}
    return symbols


def latest_usd_per_unit(
    closes: pd.DataFrame,
) -> dict[Currency, tuple[float, pd.Timestamp]]:
    """USD value of one unit of each supported currency, with the date it is from."""
    rates = {}
    ccl = ccl_from_pairs(closes).dropna()  # ARS per USD
    if not ccl.empty:
        rates[Currency.ARS] = (1.0 / ccl.iloc[-1], ccl.index[-1])
    return rates


def build_fx_context(portfolio: Portfolio, closes: pd.DataFrame) -> FXContext:
    """FXContext rates = units of base currency per unit of each held currency.

    Raises MissingFXRateError instead of falling back to a guessed rate.
    """
    usd_per = {c: r for c, (r, _) in latest_usd_per_unit(closes).items()}
    usd_per[Currency.USD] = 1.0
    base = portfolio.base_currency
    needed = held_currencies(portfolio) | {base}
    missing = needed - set(usd_per)
    if missing:
        names = ", ".join(sorted(c.value for c in missing))
        raise MissingFXRateError(f"No market FX rate for {names}")
    return FXContext(
        base_currency=base,
        rates={c: usd_per[c] / usd_per[base] for c in needed},
    )
