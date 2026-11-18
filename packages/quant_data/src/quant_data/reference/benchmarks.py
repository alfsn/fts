"""Risk-free reference symbol (PR 7). The default benchmark lives on
`quant_core.models.Benchmark`."""

# Yahoo's 13-week US T-bill: the close is a bank *discount* yield in *percent*
# (4.12 == 4.12%), so divide by 100 and convert with
# `quant_core.analytics.tbill_bond_equivalent_yield` before use. A USD rate:
# wrong as the risk-free rate of an ARS-based portfolio.
RISK_FREE_SYMBOL = "^IRX"
