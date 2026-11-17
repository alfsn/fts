# src/trading_bot/core/performance.py

"""
Adapter from persisted backtest logs to quant_core performance analytics.
All KPI math lives in quant_core; this module only maps ORM rows to its inputs.
"""

from collections import defaultdict
from typing import Iterable, List

import pandas as pd
from quant_core.analytics import compute_win_rate, round_trip_pnls
from quant_core.enums import OrderSide
from quant_core.metrics import compute_performance_metrics
from quant_core.models import PerformanceMetrics

from .models import BacktestEquityLog, TradeLog


def equity_series(equity_records: Iterable[BacktestEquityLog]) -> pd.Series:
    records = [e for e in equity_records if e.timestamp is not None]
    return pd.Series(
        [e.equity for e in records],
        index=pd.DatetimeIndex([e.timestamp for e in records]),
        dtype=float,
    )


def trade_win_rate(trade_records: Iterable[TradeLog]) -> float:
    fills_by_book = defaultdict(list)
    for t in sorted(trade_records, key=lambda t: t.fill_timestamp):
        sign = 1.0 if t.side == OrderSide.BUY else -1.0
        fills_by_book[(t.instrument_id, t.outcome)].append(
            (sign * t.quantity, t.fill_price)
        )
    pnls = [p for fills in fills_by_book.values() for p in round_trip_pnls(fills)]
    return compute_win_rate(pnls)


def build_performance_metrics(
    equity_records: List[BacktestEquityLog], trade_records: List[TradeLog]
) -> PerformanceMetrics:
    metrics = compute_performance_metrics(equity_series(equity_records))
    return metrics.model_copy(
        update={
            "win_rate": trade_win_rate(trade_records),
            "total_trades": len(trade_records),
        }
    )
