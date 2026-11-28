"""Reference data: tickers, ratios and other facts about markets. The maths
is quant_core's; ``ars_ccl`` only applies it to the reference pairs."""

from .ars import ARS_CCL_PAIRS, ars_ccl, ars_implied
from .benchmarks import RISK_FREE_SYMBOL
from .gaps import gap_rule_for

__all__ = [
    "ARS_CCL_PAIRS",
    "RISK_FREE_SYMBOL",
    "ars_ccl",
    "ars_implied",
    "gap_rule_for",
]
