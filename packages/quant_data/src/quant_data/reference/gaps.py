"""Which gap rule a cached series follows, by the vendor that filled it (M2-D5).

This table is the one home of each vendor's rule, keyed by source name (the
vendor class's name, as stored in cached bars). Each provider class sets
``gap_rule = GAP_RULES["<its name>"]``, so the read-only cache inspector reads
the same object without importing the providers. A test checks that every
vendor in ``quant_data.providers`` takes its rule from here.
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
