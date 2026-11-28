# Plan: modularize by hidden decisions (Parnas review)

> Agreed in discussion 2026-11-26. Written against `beac7fb` (PR 10b).
> It interleaves with the workbench plan (`UI-plan.md`, steps W1–W4) and doesn't replace it.

## In plain words

No big rewrite. Following Parnas ("On the Criteria To Be Used in Decomposing Systems into
Modules", 1972), we list the decisions likely to change and give each to **one** module that
hides it. Today several of them are known by many modules:

- How a backtest run is stored is known by about six modules, and the code that deletes a run is written twice.
- fts reads the price-cache tables directly instead of going through `quant_data`.
- The sweep builds its own trading pipeline by hand. It ignores the execution settings in its YAML,
  calculates win rate itself and scores every trial on the same test period.
- The IBKR API is split between a read side (`quant_data`) and an order side (fts).

The fix is a series of small stacked PRs. Each one hides one decision behind a small interface,
and every rule that might change (fill price, commissions, data split, search sampler, scoring,
acceptance) is a pluggable class chosen in YAML, never an if/else or a hardcoded default.

## The test we apply

For each decision: who knows it today, and who should?

| Decision likely to change | Known today by | Should be known by |
|---|---|---|
| Which tables make up a run | backtest engine, sweep runner, results, visualizer, sweep visualizer, catalog | `RunStore` (M1) |
| The price-cache schema | `quant_data`, fts `readers.py`, fts `core/models.py` | `quant_data` only (M2) |
| What counts as a strategy version | nobody (a new ID per process start) | Deployment ID (M3) |
| How fills and costs are simulated | simulator if/else, sweep runner `.get` defaults | `ExecutionSpec` + pluggable classes + one builder (M4) |
| How IC and its uncertainty are computed | `plugins/nets/training/evaluator.py` | `quant_core.analytics.forecast` (R1) |
| How experiments are split, searched, scored, accepted | sweep runner, orchestrator, hparam search (overlapping) | one experiment loop (R2) |
| How models are trained (framework-free parts) | `plugins/nets` | `trading_bot.modeling` (M8) |
| The IBKR API | `quant_data` brokers (fake) + fts execution handlers | `quant_broker` (M5) |
| Where live state lives | fts `Portfolio` + DB, no reconcile | live book + `ledger.reconcile` (M6) |

## Decisions

| # | Decision |
|---|---|
| D1 | No big rewrite and no NautilusTrader. Incremental stacked PRs. |
| D2 | Order: workbench first (W1, W2, M1, M2, W3, W4), then M3, M4, R1, R2, M8, then the IBKR path (M5, M6, M7, M9). A1 any time after W2. No live trading is planned soon. |
| D3 | **A live session is not a run.** Live trading is an append-only book of the fills IBKR reports. The broker is the source of truth. Live is compared with backtests through an on-demand shadow backtest, joined on the deployment ID. Backtests and live save the same **result shape** (transactions + equity + cash flows) but not the same storage or lifecycle. |
| D4 | **One ID per deployment** = hash(code commit, model file, config). A retrained model is a new deployment. The UI groups deployments into a **strategy line** (strategy + config without the model). |
| D5 | Default simulated fill = **next bar's open**. The fill price, slippage and commission are pluggable classes chosen in YAML. The commission model starts with the simplest (per share, IBKR fixed pricing: check the numbers on IBKR's site first). |
| D6 | New package `quant_broker`: `AccountReader` (read-only, OK for client accounts) and `Venue` (orders and fills, only the user's own account at first). IBKR via `ib_async`. Paper account first. One deployment per account at first. |
| D7 | The `Venue` interface lives in `quant_broker`, `SimVenue` lives in fts, and the cost models (commission, slippage) live in `quant_core` so the analyser's what-if can reuse them. |
| D8 | When live positions don't match IBKR (a quantity break), the bot **stops trading**. A cash break is a warning (as in `ledger.reconcile`). |
| D9 | Searches tune **model hyperparameters** only (not strategy settings). **Optuna is the only search tool.** A grid sweep is `sampler: grid` (`GridSampler`). |
| D10 | Every trial is scored with **walk-forward** splits. `Splitter` is pluggable (`ExpandingWindow`, `RollingWindow`). Window, step (default `1M`), gap (in bars, default = horizon, rejected if smaller), `min_train` (bars), holdout (**bars**) and `max_folds` are config. |
| D11 | Objective = **rank IC** (Spearman), aggregated over folds by a pluggable `FoldAggregate` (default `mean`). Std, n, IR and the share of positive folds are always shown beside it. |
| D12 | **Every displayed IC carries SE bars.** Per-fold and holdout IC: Newey-West SE (lags ≥ horizon − 1, because targets overlap). Mean across folds: std(fold ICs, ddof=1)/√folds. Each result stores which SE it used. |
| D13 | After a search: retrain the best settings on all data before the holdout, score the holdout **once**, and save the model as `candidate` if a pluggable `AcceptanceRule` passes (default `IC > 0`), else as `rejected` with the reason. |
| D14 | Search budget: 10 minutes on a laptop by default (Optuna `timeout`, median pruner, `max_folds`). |
| D15 | Existing backtest runs and trial models are **disposable**: new schema, reset, no migration. The reset of `dev.db` still needs the user's go-ahead at that step. |
| D16 | **One fts research database today; the live book gets its own later.** Drop `BACKTEST_DATABASE_URL` and the `"test"`-in-URL switch in `BacktestEngine.run`. Every write of a run goes to the one session passed in. Checked 2026-11-26: `backtests.db` is empty, and the existing backtest rows (980 predictions, 280 orders) are already in `dev.db`. Each store (`RunStore`, `MarketDataRepository`, the model registry, later the live book) gets its session from one place in config, so moving a store to its own file later is a config change. The live book (M6) gets its own database: it's never reset and must be backed up. R2 saves full runs only for the final candidate and holdout; other trials keep their scores in Optuna. (UI-plan D9 updated to match.) Positions are not stored: they are derived from fills (D19). |
| D17 | One instrument per model and per search for now. Multi-instrument models are in the backlog under "Future stuff". |
| D18 | **One price cache shared by both apps**, as the last step of M2. Bars, corporate actions and the cache's market rows move from `dev.db` and `portfolio_analyser.db` into one file (e.g. `market_cache.db`), set by one env var used by both apps and the workbench. fts then has two files (research + cache) because they have different lifecycles: the cache is shared and disposable. The shared cache starts empty and is refilled from the providers, since cache tables are disposable (`AGENTS.md`). That also avoids merging two files whose symbols may come from different vendors. |
| D19 | **Fills are the truth, positions are derived** (agreed 2026-11-27). No positions table: a backtest checks at the end that replaying its saved fills (`ledger.holdings`, `ledger.cash`) gives the Portfolio's final positions and cash; live checks its fill book against IBKR (M6, D8). A snapshot table is only worth it when replay is slow, which it isn't at our size. Until M6 a restarted live bot starts flat. |

## PR sequence

| # | PR | In brief | Hides |
|---|---|---|---|
| W1 | Package the analyser | As in the UI plan. | — |
| W2 | Use cases + date window | As in the UI plan. | The analyser pipeline |
| **M1** | `RunStore` + one database | One fts module saves, loads and deletes a run (transactions, equity, flows, predictions; which model traded each segment). The visualizer, results, sweep visualizer and `CatalogQueryService` go through it. Removes the duplicated delete code, the raw SQL in `visualizer.py`, `BACKTEST_DATABASE_URL` and the `"test"` switch (D16). Disposable runs are reset (D15). | Which tables make up a run |
| **M2a** | Bars via `quant_data` | `SQLBacktestDataReader`, the nets loaders, the visualizer and ingest read through `MarketDataRepository`, which returns `BarData`. The interval is required, a split in the window raises, and the gap rule comes from the vendor. fts stops importing `BarDataLog` / `Market`. | The price-cache schema |
| **M2b** | One cache file | Both apps point at one shared cache file through `MARKET_CACHE_URL` (D18). fts gets its own `Base`, and the FKs to `markets` go. | Where the cache lives |
| W3 | Workbench shell | As in the UI plan; the Backtests page is built on M1. | — |
| W4 | Cleanups | As in the UI plan. | — |
| **M3** | Deployment ID | `deployment_id` = hash(commit, model file, config) and `strategy_line_id` = hash(strategy + config without the model), saved on runs, predictions and orders. The live runner stops making a new ID per process start. | What a version is |
| **M4** | Configurable fills | One `ExecutionSpec` (fractions, no `.get` defaults). `FillPrice` (`NextBarOpen` default, `NextBarClose`), `PriceSlip` (exists), `CommissionModel` (`PerShareCommission`) in `quant_core`. One builder turns the spec into `SimVenue`; the live runner and every backtest use it. | How fills and costs are simulated |
| **R1** | Forecast maths in `quant_core` | `quant_core.analytics.forecast`: `rank_ic(pred, target)` (NaN when undefined), `ic_se_newey_west(pred, target, horizon)` built on the existing `hac_ols`, fold aggregates and their SE. A `ForecastMetrics` model (`extra="forbid"`, `min_obs`). `evaluator.py` calls these; its Pearson IC and `0.0` fallbacks go. | How IC and its SE are computed |
| **R2** | One experiment loop | `ExperimentSpec` → builder → `run_experiment(spec)`. `search(base_spec, space, sampler, splitter, objective, aggregate, acceptance)` on Optuna, with the study name including a spec hash. Walk-forward folds, holdout scored once, candidate/rejected. Deletes `sweep_runner.py`, the duplicated parts of `orchestrator.py` and `hparam_search.py`, and `SweepSpec` (a sweep becomes a YAML with `sampler: grid`). Win rate comes from `quant_core` (closed round trips, NaN when none). | How experiments are split, searched, scored and accepted |
| **M8** | Generic training out of `nets` | Trainer registry, `BaseONNXModelTrainer` and the framework-free training base move to `trading_bot.modeling`. `nets` keeps the torch trainers and architectures. A second ONNX family then plugs in a trainer only. | How models are trained |
| **M5** | `quant_broker` package | `AccountReader` + `Venue` interfaces. The IBKR implementation via `ib_async`, plus the fake IBKR/IOL/Inviu readers and `LocalYamlConnector` moved from `quant_data`. The analyser uses `AccountReader` only. | The IBKR API |
| **M6** | Live book | IBKR fills saved as `Transaction`s, append-only, tagged with `deployment_id` (IBKR `orderRef`). Reconcile on startup and every bar with `ledger.reconcile`. A quantity break stops trading (D8). Instruments with open orders are skipped. | Where live state lives |
| **M7** | IBKR paper | One deployment on a paper account, from YAML. | — |
| **M9** | Shadow backtest + Live page | A workbench button re-runs a deployment over its live period, and the Live page overlays live vs shadow, grouped by strategy line. | — |
| A1 | Analyser what-if | Weights + `quant_core.rebalance` → `Transaction`s → `ledger.replay` → existing metrics and figures. Costs from the M4 cost models once they exist. | How what-ifs are simulated |

## M1 design: `RunStore` + one database

> Written 2026-11-27 against `b749df3` (W2); the live DBs were read read-only.

**In plain words.** One package, `trading_bot.runs`, knows which tables make up a run. Everyone else asks it
to "write this fill", "load run X" or "delete run X". There's one database, and fills are the truth (D19).

**What we found**
- `trade_logs` is never written, so the catalog win rate is always NaN.
- The 280 orders in `dev.db` have no `run_id`, so deleting a run leaves them behind.
- `PositionRepository.save_position` ignores `run_id`, and the live bot reloads every open position.
- `order_logs` has two writers (`ExecutionEngine` and `Portfolio` via `OrderRepository`).
- The visualizer uses the wall-clock `updated_at` as the trade time.
- `BacktestEngine.run` swaps sessions, but the Portfolio's repos keep the old one, so a run can end up split across two files.
- The catalog guesses the model from the run name.
- The sweep runner reads `TradeLog.realized_pnl`, which doesn't exist.

**Modules** (`apps/fts/src/trading_bot/runs/`)

| Module | Job |
|---|---|
| `tables.py` | The run schema, hidden: `RunLog` (run card: kind, strategy, instrument, start, end), `RunSegmentLog` (which model traded from when to when; one row per backtest today, one per fold in R2), `OrderLog`, `TradeLog` (one row per fill, stamped with the bar time), `PredictionLog` and its subclasses, `BacktestEquityLog`. `RUN_TABLES` is the only list of them, parents before children, so deletes run in reverse and never trip a foreign key (`trade_logs` → `order_logs`). |
| `records.py` | Pydantic records callers see (`extra="forbid"`): `RunHeader` (the card, with `initial_cash`), `Segment`, `EquityPoint`, `Fill`, `Prediction`, `RunRecord`. No ORM objects leave the package. |
| `recorder.py` | Write side during a run: the `RunRecorder` Protocol and `SqlRunRecorder(run_id, kind)`. Every method takes the session to write into, so one run can't be split across two sessions; it never commits (the loop or the execution engine owns the transaction). A `run_id` is started once: `start` raises `RunExistsError` for a saved run, so a rerun clears it first (`clear_previous_run=True`). |
| `store.py` | Read/delete side: `RunStore(session_factory)` with `list`, `markets`, `load`, `delete` and `counts`. `markets` reads the predictions, not the card, so the run picker lists a run under every instrument and strategy it predicted. |
| `integrity.py` | Fills → `quant_core` `Transaction`s (the run card's `initial_cash` is one `DEPOSIT`, the run's cash flow; the instrument key includes the outcome, so YES and NO never merge). The end-of-run check against the Portfolio raises `RunIntegrityError`, and `BacktestEngine` then deletes the run (orders and fills are committed as they happen, so a rollback can't). A run without fills must end flat with its initial cash. |
| `reset.py` | `python -m trading_bot.runs.reset --db-url <url>` drops the run tables (and the legacy `positions` table) only. |

`ExecutionEngine`, the prediction logger and `BacktestEngine` receive a `RunRecorder`. The readers (`BacktestResult`,
the catalog, the visualizers, the sweep runner, the dashboard) receive a `RunStore`. `Portfolio` is in-memory only.
`core/performance.py` takes records, not ORM rows. The bar SQL in the visualizer stays until M2.

**Quant pitfalls.**
Handled:
1. Fill time is the bar time.
2. Win rate counts round trips over fills that now exist.
3. The outcome is part of the replay key.
4. The initial balance is a cash flow.
5. One session per run.
6. No backtest state can leak into live.
7. A test checks that `RUN_TABLES` covers every table with a `run_id` column.
8. A fill is committed before the Portfolio applies it. If that commit fails, the Portfolio is untouched and the order stays open, and the next status check saves and applies it, once (`record_fill` is idempotent per order).

Known limits:
1. A restarted live bot starts flat until M6.
2. The replay check compares the final state only.
3. `Transaction` is dated, so only end-of-run totals are compared, and those don't depend on the order within a day.
4. The quote currency is one of `quant_core.enums.Currency` (USD, ARS, EUR). `ExecutionRiskSpec` and `Portfolio` refuse anything else up front; adding one (e.g. USDT) means adding it to that enum.

**Tests.**
1. A fixture backtest lands in one DB, and `delete` empties every run table.
2. Fills carry the bar time, and the win rate is a number on a closed round trip.
3. Removing a fill row makes the integrity check raise.

## M2 design: bars via `quant_data` (M2a + M2b)

> Written 2026-11-28 against `e738aca` (M1). Two stacked PRs: M2a hides the cache schema and changes no data or files; M2b gives both apps one cache file.

**In plain words.** Only `quant_data` knows the cache tables. fts asks it for "the 1h bars of BTC/USDT from A to B" and gets `BarData` back. The interval is always required, so one symbol's 1h and 1d bars never mix. fts uses raw prices, so a split inside the window stops the load with a clear error. Afterwards both apps share one `market_cache.db`, and `dev.db` keeps only research data (models, runs, Optuna).

**What we found**
- Five hand-written row → `BarData` mappings: `readers.py` (×2), `orchestrator.py`, `hparam_search.py` and `TieredDataProvider._cached_bars`.
- `SQLBacktestDataReader` and `visualizer._bars` don't filter on interval.
- `MarketDataRepository.get_bars` returns ORM rows, treats `interval=None` as "all intervals", and compares aware datetimes with SQLite's naive UTC as given.
- fts never adjusts for splits or dividends. `dev.db` holds raw AAPL 1d bars and no corporate actions, because ingest never fetches them.
- `calculate_dataset_hash` hashes `timestamp.isoformat()`: naive and aware give different hashes.
- `order_logs`, `trade_logs` and `prediction_logs` have `ForeignKey("markets.instrument_id")`. The relationships are unused, and SQLite never enforced the FKs (`foreign_keys` is never set).
- fts and the cache share one `Base`, so `init_db` creates the cache tables in `dev.db`.
- `TieredDataProvider` has one gap rule for every vendor and interval (`MAX_CALENDAR_GAP = 5 days`), so it misses multi-day holes in 1h crypto data.

**Decisions**

| # | Decision |
|---|---|
| M2-D1 | Two stacked PRs: M2a (one door to the bars) and M2b (one cache file, D18). |
| M2-D2 | Bars stay raw, and a split inside the loaded window (warm-up included) raises `UnadjustedSplitError`. Adjusted research data is a backlog item. |
| M2-D3 | `MARKET_CACHE_URL`. The one default lives in `quant_data.db.cache.cache_url`: `market_cache.db` in the workspace root (the folder holding `uv.lock`). `PA_DATABASE_URL` goes. |
| M2-D4 | The shared cache starts empty. The old cache tables and `markets` are dropped from `dev.db` (after the user's go-ahead), and `portfolio_analyser.db` is deleted. |
| M2-D5 | The gap rule comes from the data vendor: `GapRule.max_gap(interval)`. `BarGap(5)` (5 × interval) for 24/7 venues (ccxt); `SessionGap(5, 5 days)` (max of 5 × interval and 5 days) for exchange-hours vendors (yfinance, the default). An exchange calendar can replace it later. |
| M2-D6 | Dividends: no guard and no warning. It's a known limit, fixed by the backlog item. |

**M2a modules**

| Where | Job |
|---|---|
| `quant_data.db.repositories.MarketDataRepository` | `load_bars(instrument_id, interval, start, end, bar_type)` and `load_bars_before(…, before, count)` return `BarData`, which is the only row mapping. Bounds go through `_naive_utc`, timestamps come back UTC-aware, and both bounds are inclusive. Also `intervals`, `instruments`, `market_name` and `counts`. `get_bars` becomes private. |
| `quant_core.intervals` | `interval_to_timedelta` (fts's `timeframe_to_seconds` now delegates to it), the `GapRule` ABC, `BarGap` and `SessionGap`. `BaseMarketDataProvider.gap_rule` defaults to `SessionGap()`, and `CCXTMarketDataProvider` sets `BarGap()`. `TieredDataProvider` and ingest's gap warning ask the rule of the series' vendor. |
| `trading_bot/core/bars.py` | `load_research_bars(repo, instrument_id, interval, start, end, warmup)` is fts's raw-bar policy: it loads the bars, then runs `quant_core.analytics.split_factor` over them, and a factor that isn't constant raises. |
| Callers | The reader (`interval` required, YAML included), the sweep runner (`model_entry.interval`), the nets orchestrator and hparam search, the visualizer (interval picker), the catalog counts and ingest (which also caches corporate actions when the vendor has them). `calculate_dataset_hash` hashes naive UTC, so existing `dataset_id`s don't change. |

**M2b modules**

| Where | Job |
|---|---|
| `quant_data.db.cache` | `cache_url(override=None)` returns the override, else `$MARKET_CACHE_URL`, else the one default, `sqlite:///<workspace root>/market_cache.db` (the workspace root is the first parent folder holding `uv.lock`). `init_cache(engine)` creates the cache tables only (`markets`, `bar_data_logs`, `corporate_actions`, `fx_rates`). Both are exported from `quant_data.db`. |
| `trading_bot.config.Settings` | `MARKET_CACHE_URL: str \| None = None`, with no default path in fts. pydantic reads `.env.{APP_ENV}` into `settings`, not into `os.environ`, so fts always passes `settings.MARKET_CACHE_URL` as the override. |
| `trading_bot.core.database` | fts gets its own registry, `class ResearchBase(DeclarativeBase)`; `runs/tables.py` and `core/models.py` inherit from it, so `init_db` no longer creates cache tables in `dev.db`. `cache_engine` / `CacheSessionLocal` are built from `cache_url(settings.MARKET_CACHE_URL)`, and `init_market_cache()` runs `init_cache` on it wherever bars are read or written. |
| `runs/tables.py` | The three `ForeignKey("markets.instrument_id")` on `order_logs`, `trade_logs` and `prediction_logs` go: a FK can't point into another file, nothing uses them (the relationships went in M2a), and SQLite never enforced them (`PRAGMA foreign_keys` is never set). `instrument_id` stays an indexed string. `model_registry → time_series_datasets` stays: both are research tables. |
| fts callers | Every bar or market read and write uses the cache session; the registry, datasets and runs use the research session (`SessionLocal`). The reader's `session=None` fallback is `CacheSessionLocal()`, and the sweep runner and notebook 03 pass a cache session. Ingest writes bars and actions to the cache and the dataset to research. The nets orchestrator and hparam search load bars in a cache-session block and register the dataset in a research-session block; Optuna stays on `DATABASE_URL`. The visualizer gets a second sessionmaker for the cache (bars, intervals, instruments); `RunStore` stays on research. `CatalogQueryService` gets a cache session factory injected for `counts()` instead of opening URLs itself. The dashboard wires the cache URL the same way as `DATABASE_URL`. |
| Analyser | `APP_DIR`'s `DEFAULT_DB_URL` and `PA_DATABASE_URL` go. `--db-url` takes `envvar="MARKET_CACHE_URL", default=None`; `review.prepare_history` resolves it with `cache_url` and calls `init_cache` instead of `Base.metadata.create_all`. |
| `quant_data.db.reset` | `markets` joins `CACHE_TABLES` (safe now that nothing outside the cache points at it), and `--db-url` defaults to `cache_url()`. `trading_bot.runs.reset` is unchanged. |
| Guards | A text test scanning `trading_bot`, `plugins` and the analyser's `src`: no `market_cache.db`, `portfolio_analyser.db` or `PA_DATABASE_URL` in app sources, and no `MARKET_CACHE_URL` default outside `quant_data/db/cache.py`. A `quant_data` test that `cache_url` honours the override, then the env var, then the workspace-root default. |
| Data (M2-D4) | The shared cache starts empty and refills from the vendors. After the user's go-ahead, a one-off step drops the cache tables and `markets` from `dev.db`, and `apps/portfolio_analyser/portfolio_analyser.db` (cache only) is deleted. The two old files are never merged. |

**Quant pitfalls.**
Handled:
1. Mixed intervals (interval required).
2. Unadjusted splits (raise).
3. Naive/aware bound bugs.
4. The dataset hash stays stable.
5. Warm-up is strictly before the start.
6. One vendor per series (writes still go through `save_bars`/`replace_bars`).
7. No fill or interpolation in the door.
8. Intraday holes are caught (gap rule per vendor).
9. Two apps' vendors are never merged (the cache starts empty).

Known limits:
1. "No actions cached" isn't "no splits" (bars cached before M2a, or a failed vendor call).
2. Dividends: raw prices drop on the ex-date and the backtest never credits the cash, so P&L is low by about the yield.
3. Ingest may cache a still-forming last bar.
4. Nothing checks that a YAML backtest's interval matches the model's.
5. The reader hardcodes `TradFiDetails(STOCK, US)`.

## Search config (R2, sketch)

```yaml
split:
  window: expanding          # expanding | rolling
  train_length: null         # rolling only, in bars
  step: "1M"                 # walk-forward step = live retrain cadence
  gap_bars: null             # null → horizon; rejected if < horizon
  min_train: null            # bars; folds below it are skipped and reported
  holdout_bars: 250          # locked final period, scored once
  max_folds: null
search:
  sampler: tpe               # tpe | random | grid
  space: {...}
  objective: rank_ic         # rank_ic | val_loss
  aggregate: mean            # mean | median | ir
  timeout: "10m"
  pruner: median
acceptance:
  rule: ic_above
  threshold: 0.0
```

## Does each call do what we intend?

| Call | Intended | Verified |
|---|---|---|
| `ledger.reconcile(holdings_at, cash_at, portfolio)` | live vs IBKR check, quantity break = stop | `Breaks.ok` is false only on a quantity break; a cash break is a warning ✅ |
| `hac_ols(y, X, lags)` | Newey-West SE for the IC | takes explicit `lags`; Bartlett, n/(n−k), matches statsmodels ✅. The rank IC is the slope of standardized target ranks on standardized prediction ranks, so its HAC SE is that slope's SE |
| `newey_west_lags(n)` | lag count | gives `floor(4(n/100)^(2/9))`, i.e. 4 lags at n = 250: **too few when horizon > 5**, so use `max(newey_west_lags(n), horizon − 1)` |
| `SimulatedExecutionHandler(execution_price_source=...)` | next-bar open | today a string flag, default `"close"`, never set by any spec or YAML: replaced by `FillPrice` (M4) |
| `SweepSpec.execution` → `run_model_backtest` | apply the YAML delay and slippage | keys `execution_delay_k`/`slippage_pct` vs read keys `execution_delay_bars`/`slippage_bps`: **ignored today**, defaults k=1, 5 bp used (removed in R2) |
| `optuna.samplers.GridSampler` | a sweep as a search | tries every combination of the given lists ✅ |
| `optuna.create_study(load_if_exists=True)` | resume a study | joins old trials of a changed spec under the same name: study name must include the spec hash |
| `quant_core.rebalance` | A1 what-if | exists, has no users yet ✅ |

## Quant pitfalls

**Handled**
1. Selection bias: trials are chosen on walk-forward folds; the holdout is never seen by the search and is scored once.
2. Leakage across the train/test boundary: `gap_bars ≥ horizon` is enforced.
3. Overlapping targets make IC look more significant than it is: Newey-West SE with lags ≥ horizon − 1.
4. Pruned trials have fewer folds: the default aggregate (mean) doesn't depend on n; t-stat and lower-bound scores aren't offered as objectives.
5. IC IR blows up when std is tiny: it's shown, and only used as the objective if chosen explicitly.
6. Undefined IC (constant predictions) is NaN, not 0.0.
7. Win rate: closed round trips only, from `quant_core`, NaN with no trades.
8. Commissions are costs, not cash flows (otherwise returns would look better).
9. Fill rule changes (close → open) make old runs incomparable: the config hash in the deployment ID records the rule.
10. Short holdouts: `holdout_bars` is in bars, and the holdout IC is always shown with its SE and n.
11. Retrains split the live record: deployments are grouped into strategy lines.
12. Live reconcile false alarms: instruments with open orders are skipped. Splits overnight still stop the bot (safe), and corporate actions are fetched before reconciling.

**Known limits (documented, not fixed)**
1. IBKR paper fills are optimistic (no real liquidity).
2. One deployment per account, one instrument per model.
3. `PerShareCommission` is IBKR fixed pricing only; tiered pricing, exchange fees and FX costs are left out.
4. 500 daily bars (`dev.db`, per the UI plan) give few folds after a 250-bar holdout; early folds may be skipped by `min_train`.

## Must verify before R2

1. **`FeaturePipeline` is fitted on the training fold only** (`trading_bot/core/transforms.py`, `fit` at line 145). If a scaler sees test data, every walk-forward score leaks.
2. The ONNX export carries the fitted pipeline (`predictor.pipeline` in `sweep_runner.py`), so live inference uses training-fold statistics.
3. IBKR fixed per-share pricing numbers, from IBKR's site.
4. **The target matches the fill rule.** With `NextBarOpen` and delay k, a signal at bar t's close fills at bar t+k's open, so the target the model learns (and the IC that scores it) should start there, not at close t. Otherwise the search rewards a return the strategy can't capture. Raised before in `docs/fts/presentation.md` §17 ("Target Realism").

## Verification

- The three test runs from `AGENTS.md` after each PR.
- M1: a backtest's equity, trades and predictions land in the one database; deleting a run removes every row of it (one test that lists all run tables).
- M4: a YAML with `NextBarClose` and one with `NextBarOpen` give different fill prices on a fixture; changing `slippage_pct` in a sweep YAML changes the result (regression test for the ignored keys).
- R1: `rank_ic` vs `scipy.stats.spearmanr`; constant predictions → NaN; HAC SE vs statsmodels on a fixture.
- R2: a grid search on a fixture reproduces the old sweep's trial list; the holdout is never passed to the objective (assert on fold indices); gap < horizon raises.
- Each PR adds the guards listed for it in "Guards per PR" below.

## Guards per PR

Every PR in this plan ends by turning its boundary into a check, so later changes (by people or agents)
can't quietly undo it. A guard arrives **in the PR that makes it true**; adding it earlier fails CI.

- **Import rules** → an import-linter contract in the root `pyproject.toml` (`[[tool.importlinter.contracts]]`),
  checked by `uv run lint-imports` in CI.
- **Rules that aren't imports** (a setting name, a function call) → a small pytest test that searches the
  source files, with a narrow pattern so comments and docs don't trip it.
- **Prove each new guard bites:** plant the forbidden import or text in a temporary file, run the check, see
  it fail, delete the file.
- **Never weaken a contract to make a change pass** (`AGENTS.md`). If a boundary is wrong, change the plan first.

**How to write them**
- `root_packages` lists only `quant_core`, `quant_data` and `quant_ui` today. A contract about app code needs
  the app added there first (`trading_bot`, `nets`; `portfolio_analyser` after W1; `quant_broker` after M5;
  `workbench` after W3). Libraries such as `streamlit`, `optuna` or `ib_async` can be named as forbidden
  modules because `include_external_packages = true`.
- `forbidden` contracts also follow **indirect** imports: importing `quant_data.db` reaches `quant_data.db.models`
  through its `__init__`. Set `allow_indirect_imports = true` when the rule is about direct imports only, or
  use `ignore_imports` for one named, explained exception.
- import-linter sees only `import` statements. Components loaded from YAML by `PluginLoader`
  (`importlib` + `class_path`) are invisible to it, so a boundary crossed through config needs a pytest test
  that loads the specs and checks their `class_path`s.

| PR | Guard | Kind |
|---|---|---|
| Existing (in `pyproject.toml`) | Shared packages never import an app; `quant_core` imports no workspace package; `quant_data` and `quant_ui` never import each other; `quant_ui` never imports a data library (`sqlalchemy`, `yfinance`, `ccxt`) | contracts |
| W1 | The apps never import each other: `portfolio_analyser` ↛ `trading_bot`/`nets`/`argentina`, and the reverse. The analyser's old top-level names (`main`, `data_layer`, `visualization`) leave the first contract's list | contract |
| W2 | The analyser CLI reaches the pipeline only through `review`: `portfolio_analyser.main` never imports `portfolio_analyser.data_layer` (the workbench side is checked in W3) | contract |
| M1 | Only `trading_bot.runs` imports `trading_bot.runs.tables` (`RunLog`, `RunSegmentLog`, `OrderLog`, `TradeLog`, `PredictionLog`, `BacktestEquityLog`). Text test: no `BACKTEST_DATABASE_URL`, no `"test" in` URL checks, no `PositionRepository` in `trading_bot/` | contract + test |
| M2a | fts (`trading_bot`, `nets`) never imports `quant_data.db.models` (bars and markets go through the repository); `Base` is re-exported from `quant_data.db` for `trading_bot.core.database`, with `allow_indirect_imports = true` | contract |
| M2b | Text test: no hardcoded cache file path; both apps read the one cache env var (`quant_data.db.cache.cache_url`) | test |
| W3 | `streamlit` is imported only by `workbench` (forbidden for every package and app, `quant_ui` included); `workbench` never imports `quant_data` or `quant_core.analytics` (pages draw, never calculate), and imports the apps only through their use-case modules (`portfolio_analyser.review`, the fts catalog service) | contracts |
| M3 | Text test: only the deployment-ID module computes the ID hash (no other `hashlib` use on commit/model/config) | test |
| M4 | Text test: no `execution_delay_bars`, `slippage_bps` or `execution_price_source` in `trading_bot/` or `plugins/`; a spec test that every execution YAML loads with `extra="forbid"` | test |
| R1 | Text test: no `np.corrcoef`, `spearmanr` or `pearsonr` in `trading_bot/` or `plugins/` (IC maths only in `quant_core`) | test |
| R2 | `optuna` is imported only by the search module (`ignore_imports` lists nothing else); `sweep_runner`, `SweepSpec` and `run_model_backtest` don't exist (text test) | contract + test |
| M8 | `trading_bot` never imports `nets` (core doesn't know the plugin); `trading_bot.modeling` never imports `torch` | contracts |
| M5 | `quant_broker` may import `quant_core` only; `quant_core`/`quant_data`/`quant_ui` never import `quant_broker`; `ib_async` only in `quant_broker`'s IBKR module; `portfolio_analyser` imports the reader side only, never the `Venue` (order) module | contracts |
| M6 | Only the live-book module imports the live-book tables; text test: the live book has no delete or update of fill rows (append-only) | contract + test |
| A1 | `portfolio_analyser` still ↛ `trading_bot` (the W1 contract covers it; the what-if uses `quant_core` only) | — |
