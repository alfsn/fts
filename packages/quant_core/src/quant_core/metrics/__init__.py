"""
Single entry point composing the analytics primitives into ``PerformanceMetrics``,
``BenchmarkMetrics``, ``VolForecastMetrics`` and ``ForecastMetrics``. Both fts
backtests and the portfolio analyser go through here so a KPI means the same
thing everywhere.
One module per pipeline; the maths itself lives in ``quant_core.analytics``.
"""

from ._common import MIN_SAMPLE_OBS
from .benchmark import (
    CONFIDENCE,
    MIN_BENCHMARK_OBS,
    BenchmarkAnalysis,
    compute_benchmark_metrics,
)
from .forecast import (
    MIN_FORECAST_OBS,
    compute_fold_ic_summary,
    compute_forecast_metrics,
)
from .performance import compute_performance_metrics
from .volatility import (
    MIN_VOL_EVAL_OBS,
    VolForecastAnalysis,
    compute_rolling_vols,
    compute_vol_forecast,
)
