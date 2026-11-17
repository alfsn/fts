"""Binds the hand-written ledger to market history (PR 2b)."""

import pandas as pd
from quant_core.ledger import Breaks, ReplayResult, cash, holdings, reconcile, replay
from quant_core.models import Portfolio

from .market_history import Actions


def ledger_calendar(portfolio: Portfolio, closes: pd.DataFrame, as_of: pd.Timestamp):
    """Sessions from the first ledger line up to the as-of date."""
    first = pd.Timestamp(
        min(t.date for t in portfolio.transactions), tz=closes.index.tz
    )
    return closes.index[(closes.index >= first) & (closes.index <= as_of)]


def reconcile_snapshot(
    portfolio: Portfolio, closes: pd.DataFrame, actions: Actions, as_of: pd.Timestamp
) -> Breaks:
    """Ledger holdings and cash replayed up to `as_of`, against the snapshot (D4, D7)."""
    dates = ledger_calendar(portfolio, closes, as_of)
    held = holdings(portfolio.transactions, actions, dates)
    balances = cash(portfolio.transactions, dates)
    return reconcile(held.loc[as_of], balances.loc[as_of], portfolio)


def replay_ledger(
    portfolio: Portfolio,
    closes: pd.DataFrame,
    actions: Actions,
    fx: pd.DataFrame,
    as_of: pd.Timestamp,
) -> ReplayResult:
    """
    Daily value series and external flows in the base currency, from the first
    ledger line to `as_of`. `fx` is USD per unit of each currency; it is
    rebased to the portfolio's base currency here.
    """
    base = portfolio.base_currency
    fx_base = fx.div(fx[base], axis=0)
    return replay(
        portfolio.transactions,
        closes,
        fx_base,
        actions,
        ledger_calendar(portfolio, closes, as_of),
    )
