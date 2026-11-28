"""Argentine market reference data, and the one CCL recipe built on it."""

from typing import Mapping, Sequence

import pandas as pd
from quant_core.analytics import adjust_frame, ccl_from_pairs, implied_ccl_frame
from quant_core.enums import PriceAdjustment
from quant_core.models import CorporateAction

Actions = Mapping[str, Sequence[CorporateAction]]

# (local listing, USD ADR, local shares per ADR) used to imply the ARS CCL rate.
#
# Ratios are *today's* depositary ratios, so they are only right across all
# history on SPLIT-adjusted closes (`quant_core.analytics.adjust`): YPF's ADR
# was 1 local share until the 10:1 local split on 2026-08-03 and is 10 since.
# Checked against each other and the official rate over 2021-2026.
ARS_CCL_PAIRS: tuple[tuple[str, str, float], ...] = (
    ("GGAL.BA", "GGAL", 10.0),
    ("YPFD.BA", "YPF", 10.0),
)


def _split_adjusted(closes: pd.DataFrame, actions: Actions) -> pd.DataFrame:
    """The pairs' ratios are today's, so the raw closes are split-adjusted
    first (PR 2a §6): on raw closes the YPF pair is 10x off before 2026-08-03
    and the divergence check NaNs every day."""
    return adjust_frame(closes, actions, PriceAdjustment.SPLIT)


def ars_ccl(closes: pd.DataFrame, actions: Actions) -> pd.Series:
    """Daily CCL in ARS per USD from raw daily closes (columns = symbols)."""
    return ccl_from_pairs(_split_adjusted(closes, actions), ARS_CCL_PAIRS)


def ars_implied(closes: pd.DataFrame, actions: Actions) -> pd.DataFrame:
    """The CCL each pair implies on its own (ARS per USD), one column per pair."""
    return implied_ccl_frame(_split_adjusted(closes, actions), ARS_CCL_PAIRS)
