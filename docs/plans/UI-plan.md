# UI plan: Workbench, one click-around UI for fts + portfolio_analyser (pulls PR 26 forward)

> Planned 2026-11-18. Written against `5fa7eeb`; re-checked against `beac7fb`
> (PR 10b): `quant_core.metrics` is now a package (`performance`, `benchmark`, `volatility`), and
> minimum samples are stored on the metrics as `min_obs`. The UI prints that field, never its own constant.

## Context

Before resuming backlog PRs 8–21 we want to *see* what the system already does. Today:

- fts has a Streamlit dashboard (`trading_bot/dashboard/app.py`), but its backtest charts are empty
  (`backtest_equity_logs` = 0 rows in `dev.db` and `backtests.db`). `backtests.db` still has the old
  `bar_data_logs` schema (no `source` column). Its renderers do their own drawdown/return maths.
- The analyser is CLI-only (Rich tables + a static HTML file).
- Nobody can browse the market-data cache. `dev.db` holds 500 AAPL daily bars and 0 corporate actions;
  `fx_rates` is never written (the CCL is recomputed from the GGAL/YPF pairs every run).
- The analyser pipeline (prefetch → FX → as-of → reconcile → replay → benchmark → rf) lives inside the
  click commands in `main.py`, mixed with `console.print`, so a UI today would duplicate it.
- The repo has no ledger: `portfolio.yaml` is a snapshot, so `client-review` refuses to run on it.

**In plain words:** we add one Streamlit app, `apps/workbench`, with five tabs (Portfolio, Cache,
FX, Models, Backtests). It contains no maths: it calls one function per screen that each app exposes,
and draws with `quant_ui` figures. Along the way the analyser becomes a real package, fts's own
dashboard folder goes away (its charts move to `quant_ui` and their maths to `quant_core`), and four
small rule violations get fixed.

## Decisions

| # | Decision |
|---|---|
| D1 | Unified workbench, shipped as **stacked PRs** W1–W4 |
| D2 | Shell = new `apps/workbench`, the **only** code importing `streamlit`. It imports only each app's use-case functions, never their internals. Remove `streamlit` from the fts and root `pyproject.toml` |
| D3 | Layering is forced by import direction (below): `quant_ui` = Plotly figures only (no `streamlit`, no `quant_data`); use cases live in each app; the shell lives in the workbench. Generic widgets stay in the workbench until a second UI exists |
| D4 | Delete `trading_bot/dashboard/` (`app.py`, `cli.py`, `renderers.py`); fts has no standalone UI |
| D5 | Commit a demo ledger: USD + ARS, real tickers, ~3 years (2023-11 → today) |
| D6 | Date range included; it **rebases** (KPIs computed on the window only) |
| D7 | FX tab shows the computed `ccl_series`; delete the dead `fx_rates` table |
| D8 | Gap = two consecutive bars more than N calendar days apart (default 5) |
| D9 | DB URLs come from the existing env vars (`DATABASE_URL`, `PA_DATABASE_URL`) and can be edited in the sidebar. `BACKTEST_DATABASE_URL` is dropped by M1 (`modularization.md` D16); until then the Backtests tab reads `DATABASE_URL`, where the existing runs are |
| D10 | Keep fts's Promote/Archive write action |
| D11 | Cleanups in the stack: fts renderers → `quant_ui`, delete `fx_rates`, hide mock brokers, fix `visualizer.py` `fillna(0.0)` |

## Why charts and shell are separate (both are "UI")

Imports flow one way: packages ← apps.

```
quant_core ← quant_data ← quant_ui(figures)          packages: never import apps
      ↑            ↑           ↑
   fts (engine, CLI, CatalogQueryService)   portfolio_analyser (CLI, review use cases)
      ↑                                        ↑
      └──────────── apps/workbench (Streamlit shell) ┘
```

- The **shell** must import both apps (it calls `prepare_review` and `CatalogQueryService`), so it can't
  be in a package: `quant_ui` importing fts would make a loop.
- The **charts** are imported by the apps (the HTML report, fts's visualizer, notebooks), so they can't
  live in the workbench.
- Rule you can grep: code that uses `st.` goes in the workbench; code that turns data into a `go.Figure`
  goes in `quant_ui`; code that calculates goes in `quant_core`.

## W1: package the analyser (no behaviour change)

- Move `main.py`, `data_layer/`, `visualization/` → `apps/portfolio_analyser/src/portfolio_analyser/`
  with a build backend (setuptools, mirroring `apps/fts/src/trading_bot/pyproject.toml`). Imports become
  `portfolio_analyser.data_layer...`. Add `[project.scripts] portfolio-analyser = "portfolio_analyser.main:cli"`
  (closes PR 7 D7).
- Tests: change the imports only (`from main import cli` → `from portfolio_analyser.main import cli`).
  Delete `conftest.py` if it only does sys.path setup. Update the AGENTS.md test-run note if the
  `--import-mode` caveat changes.
- Guard: add `portfolio_analyser`, `trading_bot` and `nets` to import-linter's `root_packages` and a contract
  that the two apps never import each other; drop the old top-level names (`data_layer`, `visualization`)
  from the "shared packages never import an app" list (`modularization.md` "Guards per PR").
- Docs: `HOW_TO_USE.md` and the README commands become `uv run portfolio-analyser ...`; fix the stale PR 26
  text in `docs/BACKLOG.md` (it lists `domain/` and `config/`, which no longer exist).
- **Hide mock brokers:** `IBKRConnector`/`InviuConnector`/`IOLConnector` return hard-coded fakes
  (e.g. IOL → 200 YPFD). The CLI help says "(mock)"; the workbench offers YAML only.

## W2: use cases + date window (the shared pipeline)

- New `portfolio_analyser/review.py`, the only analyser API the workbench calls:
  - `prepare_history(portfolio, db_url, extra_symbols) -> (closes, actions)`: today's `prefetch_history`, moved.
  - `prepare_snapshot(portfolio, db_url) -> Snapshot` (frozen dataclass: `as_of`, `fx_context`, priced
    positions, cash rows, history rows, `breaks`, `total_value`).
  - `prepare_review(portfolio, db_url, benchmark_symbol, risk_free_rate, withholding, start, end) -> Review`
    (`snapshot`, `ReplayResult`, benchmark level, rf, label, `warnings: list[str]` replacing the yellow
    `console.print`s, and `figures` from `backward_analytics_figures`).
  - Errors stay exceptions (`MissingFXRateError`, `ValueError`, quantity `Breaks`); the CLI maps them to
    `ClickException`, the workbench to `st.error`.
- `main.py` `analyze` / `client-review` become: build the args → call the use case → print. Add
  `--start/--end` to `client-review` (docs follow the code).
- **Window maths in `quant_core`:** `quant_core.analytics.returns.window(values, cash_flows, start, end)
  -> (values, flows)`: an inclusive slice of both on the values' index, raising if fewer than 2 values remain.
  `period_returns` already drops the first return, so a flow *on* the start date becomes part of the
  starting capital with no special case, and `compute_mwr` already treats the first value as the initial
  contribution. `backward_analytics_figures` receives the sliced series; the benchmark and rf are aligned
  to the portfolio dates inside the benchmark pipeline (`align_daily`).
- **Demo ledger** `apps/portfolio_analyser/examples/demo_ledger.yaml`:
  - USD deposit → AAPL buys.
  - ARS deposit → GGAL.BA and YPFD.BA buys, held across YPFD's 10:1 local split on 2026-08-03, so the
    split path is exercised.
  - AAPL dividend lines on pay dates, one partial sell, one withdrawal.
  - A snapshot that reconciles.
  - Trade prices = real cached closes on those dates, filled in during implementation from
    `prepare_history`. A header comment says the trades are synthetic.
- **fts use cases:** `CatalogQueryService` stays the facade. Add the backtest curves (`wealth_index(equity)`,
  `drawdown_series(wealth)` from `quant_core`) and the T=0 index alignment for comparisons to the DTO or
  a `backtest_curves(run_id)` method, so no page or figure computes.
- **Cache (`quant_data`):** `MarketDataRepository.cached_series() -> list[CachedSeries]` (symbol, interval,
  source, bar count, first/last). Analyser `review.inspect_symbol(db_url, symbol) -> SymbolView`: raw closes,
  `adjust(..., SPLIT)`, `adjust(..., TOTAL_RETURN)`, the actions list, gaps. It is read-only: it uses
  `get_bars`/`load_corporate_actions`, never a provider.
- **Gaps maths:** `quant_core.analytics.calendar_gaps(index, max_days=5) -> DataFrame(start, end, days)`.

## W3: the workbench shell

```
apps/workbench/
  pyproject.toml          # deps: streamlit, fts, portfolio-analyser, quant-ui
  src/workbench/
    app.py                # st.navigation over the pages; sidebar: DB URLs (D9)
    settings.py           # reads the env vars, defaults
    pages/portfolio.py    # YAML path (default: demo), benchmark, rf (blank = ^IRX), withholding, date range
                          #   → prepare_snapshot / prepare_review → tables + st.plotly_chart(fig)
    pages/cache.py        # pick DB + cached symbol → raw/SPLIT/TOTAL_RETURN chart, actions, gaps table
    pages/fx.py           # ccl_series from cached GGAL/YPF pairs + per-pair implied CCL
    pages/models.py       # ported from dashboard/app.py tab 2 (filters + Promote/Archive)
    pages/backtests.py    # ported from tab 3
    widgets.py            # the hyperparameter range filter, written once (today it's copy-pasted twice)
  tests/                  # streamlit.testing.v1.AppTest smoke tests
```

- Launch: `uv run streamlit run apps/workbench/src/workbench/app.py --server.address 127.0.0.1`
  (localhost only: reports contain client PII, Epic 8), plus a `workbench` script entry.
- Page rule: `st.*` plus calls to use cases and `quant_ui` only. `% ×100` formatting happens only here or
  in `quant_ui` (fractions everywhere else). Minimum-sample captions print `metrics.min_obs`.
- New `quant_ui` figures (draw-only): `plot_price_adjustments(raw, split, total_return)`, `plot_ccl(series, pairs)`,
  the fts trade overlay, the prediction scatter, and a multi-series `plot_cumulative_returns` for the run comparison.
- Add `apps/workbench` to the root workspace members; delete `trading_bot/dashboard/` and
  `apps/fts/tests/test_dashboard_renderers.py` (its cases move to `quant_ui` tests).
- Docs: README, `DUMB_HOW_TO_USE.md`, `HOW_TO_USE.md` (replace the `trading_bot.dashboard.cli` commands); a
  short `apps/workbench/README.md`. Update the AGENTS.md test-run list with the workbench run.

## W4: cleanups

- Check that grepping `cummax|pct_change|initial_eq` in `quant_ui` and `workbench` finds nothing (the
  renderer maths is gone after W2/W3).
- `backtesting/visualizer.py` `calculate_pnl`: `pct_change().fillna(0.0)` → leave NaN; the cumulative uses
  `wealth_index` (a missing close is not a 0% day).
- Delete `FXRateRecord`, `get_fx_rate` and their test; after merging, run `quant_data.db.reset`
  (**the user confirms the `dev.db` reset first**).
- `docs/BACKLOG.md`: mark PR 26 as partly done by the workbench (Docker still open).

## Does each call do what we intend?

| Call | Intended | Verified |
|---|---|---|
| `wealth_index(values, flows)` on a window slice | TWR from the window's first day | `period_returns` = `(v − f)/v.shift(1) − 1`, then `.iloc[1:]`, so a first-day flow drops out ✅ |
| `_aligned_flows` | flows must sit on valuation dates | raises on unmatched dates, so `window` must slice the flows too ✅ |
| Benchmark pipeline (`align_daily`) | aligns the benchmark to the portfolio's dates | inner-join on real prices (PR 7 notes) ✅ |
| `adjust(closes, actions, SPLIT/TOTAL_RETURN)` | adjusted views for the cache tab | `analytics/corporate_actions.py` ✅ |
| `ccl_series(closes, actions)` | SPLIT-adjusts first, because of the YPF ratio change | `data_layer/market_history.py` ✅ |
| `drawdown_series(wealth)` vs fts `renderers.py` | replaces UI maths | same peak-to-trough as a fraction ✅ |
| `TieredDataProvider` in `prepare_history` | fills the cache on first fetch | writes to the DB (expected); the cache tab never calls a provider |

## Quant pitfalls

**Handled**
1. A window rebases the starting capital, and a flow on the start date is not counted twice.
2. Short windows: alpha/beta and the forecast scores show NaN below `min_obs`, with n always displayed. The page explains the n/a using `metrics.min_obs`.
3. The EWMA re-seeds at the window start (the seed returns are NaN). Expected under rebase, and labelled.
4. The cache tab shows **raw** prices beside the adjusted ones, so a fake −90% split day is visible, not hidden.
5. The rf series is aligned to the window, not forward-filled across it (7-day staleness rule kept).
6. The demo holds YPFD.BA across its 10:1 split, so reconciliation must apply the split from corporate actions (a quantity break stops the run, which makes it a good test).
7. fts normalised return: the `initial_eq = 1.0` fallback when equity ≤ 0 (`renderers.py`) is replaced by a NaN-aware `wealth_index`.
8. `fillna(0.0)` on returns removed: a missing close is not a 0% day.

**Known limits (documented, not fixed)**
1. Gap > 5 calendar days misses single missing sessions; there's no exchange calendar.
2. The demo ledger's first run needs Yahoo (network); tests use the existing `_stub_get_bars` monkeypatch.
3. Pay-date vs ex-date dividends: the demo's cash timing follows the ledger (PR 7 D10).
4. The demo uses real tickers but synthetic trades; its KPIs are illustrative only. Client-facing content rules (no capital/fees/returns in client PDFs) don't apply to this internal explorer.
5. Streamlit reruns: use-case results are cached with `st.cache_data` keyed on the inputs.
6. Old fts runs with `close = 0.0` equity rows still render (treated as missing); the backtest tab stays empty until runs are re-done (`backtest_equity_logs` = 0).

## Verification

- `uv sync --all-extras --dev --all-packages`, then: `uv run pytest packages` ·
  `uv run pytest apps/portfolio_analyser/tests` · `cd apps/fts && uv run pytest tests` · `uv run pytest apps/workbench/tests`.
- New tests: `window` (flow on the start date, a 1-value window raises), `calendar_gaps`, `cached_series`,
  `prepare_review` with a stubbed provider (same figures as the CLI), the demo YAML reconciling offline,
  and the `quant_ui` figure tests moved from fts. Workbench smoke tests: each page renders on the stubbed demo
  without an exception (`AppTest`).
- CLI unchanged: `uv run portfolio-analyser client-review --broker yaml --file examples/demo_ledger.yaml`
  gives the same KPI values as before W1.
- Manual: launch the workbench and click every tab; check the YPFD raw/SPLIT chart around 2026-08-03;
  check that a 1-year window changes the KPIs and a 3-month window shows n/a for alpha.
- Guards (import-linter contracts, see `modularization.md` "Guards per PR"): W1 adds "the apps never import
  each other"; W3 adds "`streamlit` only in the workbench" and "the workbench never imports `quant_data` or
  `quant_core.analytics`, and reaches the apps only through their use-case modules". `quant_ui` ↛ `quant_data`
  is already a contract.
