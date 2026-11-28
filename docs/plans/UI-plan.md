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
| D9 | DB URLs come from the env vars (`DATABASE_URL` for fts research, `MARKET_CACHE_URL` for the shared price cache, `modularization.md` D18; `PA_DATABASE_URL` is gone, M2b) and can be edited in the sidebar. Runs live in the one research database (`DATABASE_URL`, `modularization.md` D16); `BACKTEST_DATABASE_URL` is gone (M1) |
| D10 | Keep fts's Promote/Archive write action |
| D11 | Cleanups in the stack: fts renderers → `quant_ui`, delete `fx_rates`, hide mock brokers, fix `visualizer.py` `fillna(0.0)` |
| D12 | (2026-11-28) W3 ships as two PRs: **W3a** shell + Portfolio, Models, Backtests; **W3b** Cache + FX. The cache inspector lives in `quant_data.inspect`, not the analyser: both apps share the cache, and fts's intraday series are in it. It is the one `quant_data` module the workbench may import |
| D13 | (2026-11-28) The run comparison lines runs up by **days since each run's first bar**, never by bar number (a 1h run would look 24× longer than a 1d run) |

## Why charts and shell are separate (both are "UI")

Imports flow one way: packages ← apps.

```
               quant_core                    packages: never import apps
               ↑        ↑
      quant_data        quant_ui(figures)    quant_data and quant_ui never import each other
          ↑                  ↑
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
  with a build backend (hatchling with a `src/` layout, like `packages/quant_*`; no extra config). Imports become
  `portfolio_analyser.data_layer...`. Add `[project.scripts] portfolio-analyser = "portfolio_analyser.main:cli"`
  (closes PR 7 D7).
- Tests: change the imports only (`from main import cli` → `from portfolio_analyser.main import cli`).
  Delete `conftest.py` if it only does sys.path setup. Update the AGENTS.md test-run note if the
  `--import-mode` caveat changes.
- Guard: add `portfolio_analyser`, `trading_bot`, `nets` and `argentina` to import-linter's `root_packages` and two
  contracts (one per direction) that the two apps never import each other; drop the old top-level names (`main`, `data_layer`, `visualization`)
  from the "shared packages never import an app" list (`modularization.md` "Guards per PR").
- Docs: `HOW_TO_USE.md` and the README commands become `uv run portfolio-analyser ...`.
- **Hide mock brokers:** `IBKRConnector`/`InviuConnector`/`IOLConnector` return hard-coded fakes
  (e.g. IOL → 200 YPFD). The CLI help says "(mock)"; the workbench offers YAML only.

## W2: use cases + date window (the shared pipeline)

> Re-planned 2026-11-27 against `dfbbf1c`. W2 covers the analyser only. The cache helpers and the fts
> backtest curves moved to W3 (the Backtests page sits on M1's `RunStore`).

- New `portfolio_analyser/review.py`, the only analyser API the CLI and the workbench call. It returns
  data. It never prints, never draws (no plotly or `quant_ui` import) and never calculates:
  - `prepare_history(portfolio, db_url, extra_symbols, sources=None) -> (closes, actions)`: the old
    `prefetch_history`. `sources` are the providers behind the cache (default Yahoo).
  - `history_rows(portfolio, closes) -> list[HistoryRow]`: cache coverage per symbol.
  - `build_snapshot(portfolio, closes, actions) -> Snapshot` (frozen: `portfolio` priced at `as_of`, `fx`,
    `fx_context`, `total_value`, `breaks`), and `prepare_snapshot(portfolio, db_url)`, which runs both steps.
    `analyze` uses the two steps, so the history table still prints when valuation fails.
  - `prepare_review(portfolio, db_url, *, benchmark_symbol, risk_free_rate, withholding, start, end) -> Review`
    (`snapshot`; the windowed `values`, `flows` and `dropped`; the `benchmark` level and its name; `risk_free`;
    `period`; and `warnings`, which replace the yellow prints).
  - Errors stay exceptions: `ReviewError(ValueError)` (no ledger, no rf for a non-USD base) and
    `QuantityBreakError(ReviewError)`, which carries `breaks`. `MissingFXRateError` is unchanged. The CLI
    maps them to `ClickException`; the workbench will use `st.error`.
- `main.py` builds the arguments, calls the use case and prints. `client-review` gets `--start/--end`, and it
  calls `backward_analytics_figures(review.values, ...)` itself.
- **Window maths in `quant_core`:** `quant_core.analytics.window(values, cash_flows, start, end)`. The
  base is the **last priced row before `start`** (the prior close, as in a calendar-year return), so
  adjacent windows compound to the whole period. Without an earlier row, the base is the first row. Both
  series are cut by the same dates, and flows are not reindexed, so a flow on an unpriced day still
  raises. Fewer than 2 valued rows raises.
  - A flow on the base row is starting capital: `period_returns` drops the first return, and `compute_mwr`
    overwrites the first flow with the first value.
  - The benchmark and the rf are passed in full and aligned by the benchmark pipeline (`align_daily`) and by
    `excess_returns`.
- A window outside the ledger is clipped, with a warning. Reconciliation always runs at the as-of date.
- **Demo ledger** `apps/portfolio_analyser/examples/demo_ledger.yaml`: a USD sleeve (AAPL, dividends on
  approximate pay dates, a partial sell, a withdrawal) and an ARS sleeve (GGAL.BA, YPFD.BA held across the
  10:1 split on 2026-08-03). It has real raw closes and synthetic trades, and a snapshot that reconciles
  exactly.
- Guard: `portfolio_analyser.main` never imports `portfolio_analyser.data_layer` (import-linter).

## W3: the workbench shell

> Re-planned 2026-11-28 against `2ab0656` and split (D12). **W3a** (shipped first): the shell, `settings`,
> `state`, `filters` (written once, NaN-safe), `views/portfolio.py`, `views/models.py`, `views/backtests.py`;
> `CatalogQueryService.open(research_url, market_cache_url)` builds the sessions and defaults, so the workbench
> holds neither; `run_curves` (cumulative, drawdown, top episodes, `close_series`, fills, predictions, from the
> same `equity_series` as the KPIs) and `compare_curves(align="calendar"|"elapsed")`; `review.load_portfolio`,
> `cash_rows`, `position_rows`, `Snapshot.ars_per_usd` (the `analyze` CLI prints the same rows); `quant_ui.plots.backtest`
> (trade overlay, prediction scatter that never puts a missing return at 0, comparison). The Backtests page's
> minimum Sharpe defaults to blank: the old −10 hid every run too short for a Sharpe. Pages live in `views/`
> (not `pages/`, which Streamlit would also auto-load). **W3b** (2026-11-28, stacked on W3a): `views/cache.py`,
> `views/fx.py` on the read-only `quant_data.inspect` (`cached_series`, `inspect_series`, `ccl_view`). Three
> things got one home each: the gap comparison (`quant_core.intervals.find_gaps`, now also used by
> `TieredDataProvider` and `ingest_historical`), the CCL recipe (`quant_data.reference.ars_ccl` / `ars_implied`;
> the analyser's `ccl_series` calls it) and bars → daily closes (`quant_data.history.daily_closes`). The vendor's
> gap rule is looked up by source name in `quant_data.reference.gaps.GAP_RULES`, the one home of the rules:
> each provider class takes its `gap_rule` from that table (a test checks every vendor in
> `quant_data.providers`), so the inspector never imports the providers. The pages never fetch: fill the cache with
> `ingest_historical` or a Portfolio review.

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
- **Cache use cases (W3b, D12):** `MarketDataRepository.series_summary() -> list[CachedSeries]` (symbol,
  interval, source, bar count, first/last) and `quant_data.inspect.inspect_series(symbol, interval) -> SeriesView`
  (raw closes, `adjust(..., SPLIT)`, `adjust(..., TOTAL_RETURN)`, the actions, gaps). It is read-only:
  `load_bars`/`load_corporate_actions`, never a provider. Gaps maths:
  `quant_core.intervals.find_gaps(timestamps, max_gap) -> DataFrame(start, end, length)`, with `max_gap` from the
  vendor's `GapRule` (M2-D5). `ccl_view()` gives the CCL, each pair's rate, and the rejected and single-pair days.
- **fts backtest curves (moved from W2, on M1's `RunStore`):** `CatalogQueryService` stays the facade. It adds
  `wealth_index(equity)`, `drawdown_series(wealth)` and the T=0 alignment for comparisons, so no page or
  figure computes.
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
| `wealth_index(values, flows)` on a `window` slice | TWR from the last close before the start | `period_returns` = `(v − f)/v.shift(1) − 1`, then `.iloc[1:]`, so a base-day flow drops out ✅ |
| `_aligned_flows` | flows must sit on valuation dates | raises on unmatched dates, so `window` cuts the flows by the same dates, without reindexing ✅ |
| Benchmark pipeline (`align_daily`) | aligns the benchmark to the portfolio's dates | inner-join on real prices (PR 7 notes) ✅ |
| `adjust(closes, actions, SPLIT/TOTAL_RETURN)` | adjusted views for the cache tab | `analytics/corporate_actions.py` ✅ |
| `ccl_series(closes, actions)` | SPLIT-adjusts first, because of the YPF ratio change | `data_layer/market_history.py` ✅ |
| `drawdown_series(wealth)` vs fts `renderers.py` | replaces UI maths | same peak-to-trough as a fraction ✅ |
| `TieredDataProvider` in `prepare_history` | fills the cache on first fetch | writes to the DB (expected); the cache tab never calls a provider |

## Quant pitfalls

**Handled**
1. A window rebases on the last close before its start (adjacent windows compound to the whole period), and a flow on that base day is starting capital, not counted twice.
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
- New tests: `window` (prior-close base, adjacent windows compound, base-day flow, a 1-value window raises), the
  demo YAML reconciling offline (W2); `calendar_gaps`, `cached_series`,
  and the `quant_ui` figure tests moved from fts. Workbench smoke tests: each page renders on the stubbed demo
  without an exception (`AppTest`).
- CLI unchanged: `uv run portfolio-analyser client-review --broker yaml --file examples/demo_ledger.yaml`
  gives the same KPI values as before W1.
- Manual: launch the workbench and click every tab; check the YPFD raw/SPLIT chart around 2026-08-03;
  check that a 1-year window changes the KPIs and a 3-month window shows n/a for alpha.
- Guards (import-linter contracts, see `modularization.md` "Guards per PR"): W1 adds "the apps never import
  each other"; W3a adds "`streamlit` only in the workbench" and "the workbench never imports `quant_core.analytics`
  / `metrics`, `quant_data.db` / `providers` / `history`, `sqlalchemy`, or an app's internals" (`quant_data.inspect`
  stays open for W3b), plus a text test: no `cummax`, `pct_change`, `cumprod` or `* 100` in the workbench. W3b adds "`quant_data.inspect`
  never reaches `quant_data.providers` / `yfinance` / `ccxt`" (indirect imports included) and a text test: only
  `find_gaps` compares a hole with `max_gap`. `quant_ui` ↛ `quant_data` is already a contract.
