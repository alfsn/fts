"""Binds a broker snapshot to cached market history (prices, corporate actions, FX)."""

from datetime import datetime, timedelta, timezone
from typing import Mapping, Sequence

import pandas as pd
from quant_core.analytics import adjust_frame, ccl_from_pairs
from quant_core.enums import Currency, PriceAdjustment
from quant_core.models import CorporateAction, FXContext, MissingFXRateError, Portfolio
from quant_data.reference import ARS_CCL_PAIRS

HISTORY_YEARS = 5

Actions = Mapping[str, Sequence[CorporateAction]]


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


def ccl_series(closes: pd.DataFrame, actions: Actions) -> pd.Series:
    """
    Daily CCL in ARS per USD. The pairs' depositary ratios are today's, so the
    raw closes are split-adjusted first (PR 2a §6): on raw closes the YPF pair
    is 10x off before 2026-08-03 and the divergence check NaNs every day.
    """
    return ccl_from_pairs(
        adjust_frame(closes, actions, PriceAdjustment.SPLIT), ARS_CCL_PAIRS
    )


def latest_usd_per_unit(
    closes: pd.DataFrame, actions: Actions
) -> dict[Currency, tuple[float, pd.Timestamp]]:
    """USD value of one unit of each supported currency, with the date it is from."""
    rates = {}
    ccl = ccl_series(closes, actions).dropna()
    if not ccl.empty:
        rates[Currency.ARS] = (1.0 / ccl.iloc[-1], ccl.index[-1])
    return rates


def build_fx_context(
    portfolio: Portfolio, closes: pd.DataFrame, actions: Actions
) -> FXContext:
    """FXContext rates = units of base currency per unit of each held currency.

    Raises MissingFXRateError instead of falling back to a guessed rate.
    """
    usd_per = {c: r for c, (r, _) in latest_usd_per_unit(closes, actions).items()}
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
