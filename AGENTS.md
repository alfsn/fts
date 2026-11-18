# AGENTS.md

Workspace-wide rules for agents working in this repo (fts, portfolio_analyser, and `packages/`).
For the fts agent constitution (SOLID, Pydantic, `uv`), see `docs/architecture/fts/AGENTS.md`.
Roadmap status lives in `apps/portfolio_analyser/MASTER_PR_BACKLOG.md`; keep it updated there.

## Documentation

- **Docs follow the code, in the same PR.** When a change touches a command, flag, config/YAML field, output or workflow, update the docs that describe it (`HOW_TO_USE.md`, READMEs, `docs/`, this file). Delete sections that describe code which no longer exists rather than leaving them.
- **Documented commands must run.** Before committing, run each command or example you changed in the docs, or cover it with a test.

## Performance & risk analytics

- **One home for the maths.** All KPI maths lives in `quant_core.analytics` (returns / drawdown / ratios / trades / benchmark / regression / volatility). Both apps get KPIs through `quant_core.metrics` (`compute_performance_metrics`, `compute_benchmark_metrics`, `compute_rolling_vols`, `compute_vol_forecast`). Daily returns for any dispersion statistic come from `analytics.daily_returns(wealth)`. Apps only map their data to its inputs (fts: `trading_bot/core/performance.py`; analyser: `visualization/plot_factory.py`). Never reimplement a metric in an app.
- **Fractions, not percents.** Store `0.05`, not `5.0`. Multiplying by 100 and `%` formatting happen only in the UI.
- **Undefined is NaN, never 0.0.** 0.0 is a legitimate result. Write threshold filters so NaN fails them: `not x >= threshold`.
- **Sample statistics use `ddof=1`.**
- **No default `periods_per_year`.** Pass it, or infer it with `analytics.periods_per_year(index)`. 252 is wrong for crypto (365) and for non-daily data.
- **Ratios use daily closes; path metrics use full resolution.** Sharpe, Sortino and volatility come from `to_daily(wealth)`, because annualising tick returns inflates them. Total return and drawdown use the full-resolution, flow-adjusted wealth index.
- **Always handle cash flows.** Return functions take `cash_flows`: deposits are positive, and each flow is already included in the value at its timestamp. Never compute performance as last value / first value.
- **Win rate counts closed round trips only** (FIFO, per instrument and outcome). A single fill is never a win. `TradeLog.outcome` is the prediction-market side (YES/NO), not win/loss.
- **UI code never calculates.** `quant_ui` only draws what `quant_core` gives it.
- **Prices are cached raw (as traded), with corporate actions beside them.** Valuation uses raw prices × share counts; returns, charts, features and CCL use `quant_core.analytics.adjust(closes, actions, mode)` (`SPLIT` or `TOTAL_RETURN`). A model trained on raw equity bars learns fake −90% split days. Every cached bar records its `source`; one vendor per symbol.

## Environment

- Sync with `uv sync --all-extras --dev --all-packages`. A plain `uv sync` removes pytest and the workspace packages' dependencies.
- Test `packages/` as well as the apps, as three runs (a single run fails at collection: `apps/fts/tests` and `packages/quant_core/tests` are both a package named `tests`, and `--import-mode=importlib` breaks the analyser's imports): `uv run pytest packages` · `uv run pytest apps/portfolio_analyser/tests` · `cd apps/fts && uv run pytest tests`.
- The market-data cache tables (`bar_data_logs`, `corporate_actions`, `fx_rates`) are disposable; there are no migrations. After a schema change run `uv run python -m quant_data.db.reset --db-url <url>`, which drops only those three tables (fts `dev.db` also holds the model registry and Optuna studies).
- Pydantic ignores unknown fields by default, so a test can pass while setting fields that don't exist. Use `model_config = ConfigDict(extra="forbid")` on domain models whose field names matter (e.g. `PerformanceMetrics`).
- In sandboxed runs, some fts SQLite tests fail with "unable to open database file". That's the environment, not a regression.
