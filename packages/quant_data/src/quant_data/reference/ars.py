"""Argentine market reference data."""

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
