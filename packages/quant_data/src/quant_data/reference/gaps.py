"""Which gap rule a cached series follows, by the vendor that filled it (M2-D5).

The rule belongs to the vendor class (``BaseMarketDataProvider.gap_rule``);
this table repeats it by source name so the read-only cache inspector can use
it without importing the providers (a test keeps the two in step).
"""

from quant_core.intervals import BarGap, GapRule, SessionGap

DEFAULT_GAP_RULE: GapRule = SessionGap()  # BaseMarketDataProvider's default

GAP_RULES: dict[str, GapRule] = {
    "YFinanceMarketDataProvider": SessionGap(),
    "CCXTMarketDataProvider": BarGap(),
    "LocalCSVMarketDataProvider": SessionGap(),
}


def gap_rule_for(source: str) -> GapRule:
    """The rule of the vendor named ``source``; exchange hours if unknown."""
    return GAP_RULES.get(source, DEFAULT_GAP_RULE)
