# AGENTS.md

Workspace-wide rules for agents working in this repo (fts, portfolio_analyser, and `packages/`).
For the fts agent constitution (SOLID, Pydantic, `uv`), see `docs/architecture/fts/AGENTS.md`.
Roadmap status lives in `apps/portfolio_analyser/MASTER_PR_BACKLOG.md`; keep it updated there.

## Performance & risk analytics

- **One home for the maths.** All KPI maths lives in `quant_core.analytics` (returns / drawdown / ratios / trades). Both apps get KPIs through `quant_core.metrics.compute_performance_metrics`. Apps only map their data to its inputs (fts: `trading_bot/core/performance.py`; analyser: `visualization/plot_factory.py`). Never reimplement a metric in an app.
- **Fractions, not percents.** Store `0.05`, not `5.0`. Multiplying by 100 and `%` formatting happen only in the UI.
- **Undefined is NaN, never 0.0.** 0.0 is a legitimate result. Write threshold filters so NaN fails them: `not x >= threshold`.
- **Sample statistics use `ddof=1`.**
- **No default `periods_per_year`.** Pass it, or infer it with `analytics.periods_per_year(index)`. 252 is wrong for crypto (365) and for non-daily data.
- **Ratios use daily closes; path metrics use full resolution.** Sharpe, Sortino and volatility come from `to_daily(wealth)`, because annualising tick returns inflates them. Total return and drawdown use the full-resolution, flow-adjusted wealth index.
- **Always handle cash flows.** Return functions take `cash_flows`: deposits are positive, and each flow is already included in the value at its timestamp. Never compute performance as last value / first value.
- **Win rate counts closed round trips only** (FIFO, per instrument and outcome). A single fill is never a win. `TradeLog.outcome` is the prediction-market side (YES/NO), not win/loss.
- **UI code never calculates.** `quant_ui` only draws what `quant_core` gives it.

## Environment

- Sync with `uv sync --all-extras --dev --all-packages`. A plain `uv sync` removes pytest and the workspace packages' dependencies.
- Test `packages/` as well as the apps: `uv run pytest packages apps/portfolio_analyser/tests apps/fts/tests`.
- Pydantic ignores unknown fields by default, so a test can pass while setting fields that don't exist. Use `model_config = ConfigDict(extra="forbid")` on domain models whose field names matter (e.g. `PerformanceMetrics`).
- In sandboxed runs, some fts SQLite tests fail with "unable to open database file". That's the environment, not a regression.
