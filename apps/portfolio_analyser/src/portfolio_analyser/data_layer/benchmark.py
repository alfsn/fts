"""Binds the benchmark and the risk-free rate to cached market history (PR 7).
Mapping only: the maths is `quant_core.analytics` (`adjust`, `net_of_withholding`,
`tbill_bond_equivalent_yield`)."""

from typing import Optional

import pandas as pd
from quant_core.analytics import adjust, net_of_withholding, tbill_bond_equivalent_yield
from quant_core.enums import Currency, PriceAdjustment
from quant_core.models import Benchmark
from quant_data.reference import RISK_FREE_SYMBOL

from .market_history import Actions, fx_in_base


def _closes(closes: pd.DataFrame, symbol: str) -> Optional[pd.Series]:
    series = closes[symbol].dropna() if symbol in closes else None
    return None if series is None or series.empty else series


def benchmark_level(
    closes: pd.DataFrame,
    actions: Actions,
    benchmark: Benchmark,
    fx: pd.DataFrame,
    base: Currency,
    withholding: float = 0.0,
) -> Optional[pd.Series]:
    """
    Total-return level of the benchmark (dividends reinvested, net of
    `withholding`) in the portfolio's base currency, on the benchmark's own
    sessions. `fx` is `usd_per_unit` on the closes' calendar. None if the
    benchmark has no prices; NaN on days without an FX rate.
    """
    raw = _closes(closes, benchmark.symbol)
    if raw is None:
        return None
    net = net_of_withholding(actions.get(benchmark.symbol, []), withholding)
    total_return = adjust(raw, net, PriceAdjustment.TOTAL_RETURN)
    rate = fx_in_base(fx, base)[benchmark.currency].reindex(total_return.index)
    return total_return * rate


def treasury_bill_rate(closes: pd.DataFrame) -> Optional[pd.Series]:
    """Annual bond-equivalent yield of the 13-week T-bill (`^IRX`), or None."""
    quotes = _closes(closes, RISK_FREE_SYMBOL)
    return None if quotes is None else tbill_bond_equivalent_yield(quotes / 100)


def benchmark_label(benchmark: Benchmark, withholding: float) -> str:
    dividends = (
        "gross dividends"
        if withholding == 0
        else f"net of {withholding:.0%} withholding"
    )
    return f"{benchmark.symbol} (total return, {dividends})"
