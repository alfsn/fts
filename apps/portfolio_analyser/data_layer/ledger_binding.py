"""Binds the hand-written ledger to market history (PR 2b)."""

import pandas as pd
from quant_core.ledger import Breaks, cash, holdings, reconcile
from quant_core.models import Portfolio

from .market_history import Actions


def reconcile_snapshot(
    portfolio: Portfolio, closes: pd.DataFrame, actions: Actions, as_of: pd.Timestamp
) -> Breaks:
    """Ledger holdings and cash replayed up to `as_of`, against the snapshot (D4, D7)."""
    dates = closes.index[closes.index <= as_of]
    held = holdings(portfolio.transactions, actions, dates)
    balances = cash(portfolio.transactions, dates)
    return reconcile(held.loc[as_of], balances.loc[as_of], portfolio)
