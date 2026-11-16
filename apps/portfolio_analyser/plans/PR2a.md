# PR 2a: Market-Data Integrity (raw prices, corporate actions, provenance)

**Status:** planned · **Blocks:** PR 2b (ledger), PR 7 (total-return benchmark)
**Packages:** `quant_core` (interfaces + pure maths), `quant_data` (adapters + cache), analyser (wiring)

## Why

PR 2 caches Yahoo daily closes called "raw" (`auto_adjust=False`). They aren't raw.
Yahoo's `Close` is **split-adjusted retroactively**, and so are its volumes and dividend amounts.
Checked live on 2026-11-15:

| Symbol | Date | Yahoo `Close` | Actually traded | Event |
|---|---|---|---|---|
| NVDA | 2024-06-06 | 121.00 | ~1,210 | 10:1 split, ex 2024-06-10 |
| NVDA | 2023 dividends | 0.004 | 0.04 | same split |
| YPFD.BA | all history before 2026-08-03 | ÷10 | ×10 | 10:1 split, ex 2026-08-03 |

With today's insert-only cache, the next split leaves the old (pre-split basis) rows in place and appends post-split rows.
That shows up as a fake −90% day. There are three further problems:

- Cached bars don't record which provider they came from, so a series can silently mix vendors' conventions.
- Holes in the middle of a series are never refilled.
- `TieredDataProvider.create_default()` puts mock tiers first. `fts ingest --provider tiered` therefore writes a dummy bar (close 102, timestamp "now") into the real DB for every ticker except TSLA (`apps/fts/src/trading_bot/utils/ingest_historical.py:132`).

**The YPF "10:1" mystery is explained.** `ARS_CCL_PAIRS` uses 10 for YPF, while the depositary ratio was 1. Yahoo's history is ÷10 because of the 2026-08-03 local split, so ratio 10 is right *on adjusted data*. Once we store true raw prices, a constant ratio of 10 is 10× wrong before 2026-08-03. CCL must therefore be computed from **split-adjusted** closes (see §6).

## Decisions

All made with the user on 2026-11-15/16.

| # | Topic | Decision | Rejected alternative (why) |
|---|---|---|---|
| D1 | Price basis | **Store true raw prices + a corporate-actions table, and adjust when reading.** This is the CRSP / zipline / QuantConnect pattern. | Store adjusted and rebuild on every event: history changes under you and doesn't match broker statements. |
| D2 | Where un-adjusting happens | **In each provider adapter** (anti-corruption layer). `get_bars()` always returns raw. | The provider labels its basis and the cache converts: the cache layer would have to know about corporate actions. |
| D3 | Corporate-action interface | **A separate `CorporateActionProvider`** (interface segregation). CCXT and the mocks don't implement it. | A method on `BaseMarketDataProvider`: it means nothing for crypto. |
| D4 | Provenance | **One source per symbol.** Every bar stores `source`. A provider switch refetches the full range and never stitches. | Record the source for auditing only: a series could still silently mix vendors. |
| D5 | Dividends | **Cache them now** (PR 7 needs a total-return benchmark). | — |
| D6 | Delivery | **Three stacked PRs** (see *Delivery*). | One PR: too big to review, and the maths would wait on the plumbing. |
| D7 | History API | **`fetch_daily_closes` stays raw-only.** A new `fetch_corporate_actions(provider, symbols)` sits beside it, and callers combine the two with `adjust(...)`. One job per function. | An `adjustment=` parameter: it hides a corporate-action lookup inside a price fetch. |
| D8 | Corporate-action refresh | **Every run**, one call per symbol. If the vendor fails, serve the cache. | Once a day: a split announced yesterday could be missed, which gives a wrong value for that day. |
| D9 | Gap rule | **A gap of more than 5 calendar days between consecutive cached bars triggers one full refetch.** If the gap persists, log a warning and serve what we have. A real gap (Argentine bridge holidays, illiquid local names) costs one extra call per run, which is harmless. | Exchange calendars (`exchange_calendars`): an extra dependency we don't need yet. |
| D10 | Existing cached data | **Wipe the market-data cache tables and repopulate.** There's no `ALTER TABLE` migration (see *Reset & repopulate*). | Keep and migrate: there's nothing worth keeping (1,507 bars in fts `dev.db`, no analyser DB yet). |
| D11 | API limits | **Each provider declares its rate limit. A shared throttle and backoff enforce it** (see *Rate limits*). | Ad-hoc `sleep()` calls: invisible, and not testable. |
| D12 | fts `--provider tiered` | **Remove the option**, and move `MockIBKRProvider` / `MockTiingoProvider` into `packages/quant_data/tests/`. Delete `TieredDataProvider.create_default()`. | Wire it to a real chain: nothing real uses it; it only demonstrated the fallback. |
| D13 | CCL reference data | **`quant_data/reference/ars.py`** holds `ARS_CCL_PAIRS` (Yahoo symbols + today's depositary ratios). `ccl_from_pairs(closes, pairs)` takes `pairs` as a required argument. | Keep it in `quant_core`: tickers and ratios are reference data, not maths. |
| D14 | Intraday bars | **Un-adjust every interval**, not only `1d`. The factor depends only on the date, so "get_bars returns raw" holds everywhere. | Daily only: it would break the contract for fts's intraday Yahoo requests. |

Sources: [CRSP factor adjustment (UMich)](https://leiq.bus.umich.edu/docs/crsp_factor_adjustment.pdf), [zipline data bundles](https://zipline-trader.readthedocs.io/en/stable/bundles.html), [QuantConnect normalization modes](https://www.quantconnect.com/forum/discussion/508/update-dividends-splits-and-custom-price-normalization/p1), [Yahoo: what is adjusted close](https://help.yahoo.com/kb/SLN28256.html).

## Delivery: three stacked PRs

Stacked on `feature/PR-epic-3`. Each PR targets the branch below it.

| PR | Branch | Scope | Base |
|---|---|---|---|
| 2a-1 | `feature/PR-2a-1-adjust-maths` | §1 contract docs, §2 pure maths, `PriceAdjustment` enum, NVDA fixture tests. No DB, no network. | `feature/PR-epic-3` |
| 2a-2 | `feature/PR-2a-2-corporate-actions` | §3 Yahoo adapter (raw + `CorporateActionProvider`), §7 rate limits, `fetch_corporate_actions`, mocks to tests (D12). | 2a-1 |
| 2a-3 | `feature/PR-2a-3-cache-integrity` | §4 cache rules (`source`, single-source, gaps, upserts), corporate-action cache, §6 CCL on split-adjusted closes (D13), reset script, chores. | 2a-2 |

## Design

### 1. Contract (`quant_core.interfaces`)

- `BaseMarketDataProvider.get_bars` docstring states the rule: **prices and volumes are as traded (raw), for every interval**.
- Add a `source_name` property, defaulting to the class name. This reuses fts's existing convention (`trading_bot/data_ingestion/abc.py:31`, `plugins/argentina/data_providers.py:31`).
- Add a `rate_limit: Optional[RateLimit]` class attribute (see §7).
- New ABC:

```python
class CorporateActionProvider(ABC):
    @abstractmethod
    def get_corporate_actions(self, instrument_id: str) -> Sequence[CorporateAction]:
        """Full history, raw (as-paid) amounts. No date range on purpose: un-adjusting
        a bar needs every split *after* it, however far outside the requested range."""
```

- `CorporateAction` (already in `quant_core.models`) is reused. Document `ratio` as **new shares per old share** (10:1 → 10.0; 1:10 reverse → 0.1). This matches what `yf.Ticker.splits` returns. Set `extra="forbid"`.

### 2. Pure maths (`quant_core.analytics.corporate_actions`, new)

| Function | What it does |
|---|---|
| `split_factor(index, splits) -> Series` | For each date t, the product of `ratio` over splits with `ex_date > t`. Dates are compared in the **exchange's local date** (see §3). That many of today's shares equal one share held at t. |
| `unadjust_splits(df, splits) -> df` | Undoes the vendor's adjustment. Raw OHLC = adjusted × factor; raw volume = adjusted ÷ factor. Used only inside adapters. |
| `adjust(closes, actions, mode)` | `mode` ∈ {`RAW`, `SPLIT`, `TOTAL_RETURN`}. `SPLIT`: raw ÷ factor. `TOTAL_RETURN`: also multiplies every price before an ex-date by `1 − D / P_prev`, CRSP-style, where `D` and `P_prev` are both raw. Charts, CCL, benchmarks and model features use this. |

Add an enum `PriceAdjustment` in `quant_core.enums`.

### 3. Yahoo adapter (`quant_data.providers.yfinance_provider`)

- Implements `BaseMarketDataProvider` and `CorporateActionProvider`.
- In `get_bars`, fetch the split history once per symbol per adapter instance, then run `unadjust_splits` on the downloaded frame. This applies to every interval (D14). `dollar_volume = raw close × raw volume` (a split doesn't change it).
- In `get_corporate_actions`, splits come from `Ticker.splits`. Dividends come from `Ticker.dividends`, **un-adjusted** (amount × `split_factor` at the ex-date), because Yahoo split-adjusts dividends too.
- Fix the misleading comment at `yfinance_provider.py:105`.

**`yf.Ticker(s).splits` verified on yfinance 1.7.0 (2026-11-16):**

| Symbol | Splits | Last ex-date (as returned) | Ratio |
|---|---|---|---|
| NVDA | 6 (full history since 2000) | `2024-06-10 09:30-04:00` | 10.0 |
| AAPL | 5 | `2020-08-31 09:30-04:00` | 4.0 |
| YPFD.BA | 1 | `2026-08-03 11:00-03:00` | 10.0 |
| GGAL.BA, BTC-USD | 0 | — | — |

- The result is a full history with no range, as needed.
- The ratio is a float giving new shares per old share.
- The index is **tz-aware in the exchange's local zone, stamped at the open**. So `ex_date.date()` in that zone is the first post-split session. Compare bars on that local date. (BYMA bars at 14:00 UTC = 11:00 local land on the same date.)
- `Ticker.dividends` also returns the full history (NVDA 56, AAPL 93, GGAL.BA 36, YPFD.BA 16).

### 4. Cache (`quant_data.db`, `TieredDataProvider`)

- `BarDataLog.source` and `CorporateActionRecord.source` are non-null `String` columns, which the reset makes possible (D10).
- **Single-source rule (D4).** If the cached series' `source` differs from the provider that answers a top-up, refetch the full range from that provider and replace the symbol's rows for that interval (`repo.replace_bars`).
- **Gaps (D9).** A gap of more than `MAX_CALENDAR_GAP` (5 days; it replaces `START_COVERAGE_TOLERANCE`) between consecutive cached bars inside the requested range triggers one full-range refetch. If it persists, log a warning and serve what we have.
- **Revisions.** `save_bars` updates existing rows (matched on the unique key) instead of skipping them. The top-up already re-requests the last cached bar, so a late correction to it gets in.
- **Corporate actions (D8).** `TieredDataProvider` also implements `CorporateActionProvider`:
  - It asks the provider named in the cached bars' `source`, so the actions come from the same vendor as the bars.
  - It upserts on `uq_corporate_action`, refreshes on every run, and falls back to the cache when the vendor fails.
- **History API (D7).** `quant_data.history.fetch_corporate_actions(provider, symbols) -> dict[str, list[CorporateAction]]` sits next to `fetch_daily_closes`, which stays raw.

### 5. Remove mocks from production (D12)

- Delete `TieredDataProvider.create_default()`.
- Move `MockIBKRProvider` and `MockTiingoProvider` to `packages/quant_data/tests/`.
- Remove `tiered` from the fts `ingest --provider` choices.

### 6. CCL on split-adjusted closes (D13)

- `latest_usd_per_unit` / `build_fx_context` call `ccl_from_pairs(adjust(closes, actions, SPLIT), ARS_CCL_PAIRS)`. Each pair's ratio is *today's* depositary ratio, which is correct across all history only on split-adjusted data.
- Move `ARS_CCL_PAIRS` to `quant_data/reference/ars.py`. Make `pairs` a required argument. Fix the stale "YPF 1" in the `implied_ccl` docstring.
- fts `CCLProvider` works on live order books, so today's ratio is right there. No change.

### 7. Rate limits (D11)

Yahoo publishes **no** limit. yfinance scrapes unofficial endpoints, and Yahoo throttles with HTTP 429, which yfinance raises as `YFRateLimitError`. Users report blocks lasting more than 24 hours after bursts ([yfinance #2422](https://github.com/ranaroussi/yfinance/issues/2422), [#2431](https://github.com/ranaroussi/yfinance/discussions/2431)). We therefore pick conservative limits ourselves and keep them in one place.

- `quant_core.models.RateLimit(min_interval_s: float, max_retries: int, backoff_s: float)`, a frozen Pydantic model with `extra="forbid"`.
- Each provider declares its limit as a class attribute:

  | Provider | Limit |
  |---|---|
  | `YFinanceMarketDataProvider` | `RateLimit(min_interval_s=1.0, max_retries=3, backoff_s=30)` |
  | CCXT | the exchange's own `rateLimit`, which ccxt already exposes |
  | Mocks | `None` |

- **Enforcement lives in one place:** `quant_data.providers.throttle.Throttle`, keyed by `source_name`, which `TieredDataProvider` uses around each provider call.
  - It waits until `min_interval_s` has passed since that provider's previous call.
  - On a rate-limit error it retries with exponential backoff (`backoff_s · 2^k`) up to `max_retries`. After that it raises, and the waterfall moves to the next tier.
  - Adapters stay free of sleeps.
- Every rule that can cause a refetch (D4 source switch, D9 gap) goes through the throttle, so a mass refetch after the reset can't burst.
- Tests use a fake clock (no real sleeping): calls are spaced, a 429 then success retries, and three 429s fall through to the next tier.

### 8. Chores

- **AGENTS.md test command.** A single pytest run fails at collection: `apps/fts/tests` and `packages/quant_core/tests` are both a package named `tests`, and `--import-mode=importlib` breaks the analyser's imports. Document the three commands CI runs instead:
  `uv run pytest packages` · `uv run pytest apps/portfolio_analyser/tests` · `uv run pytest apps/fts/tests`.
- **New AGENTS.md rule:** *"Valuation uses raw prices × share counts; returns, charts, features and CCL use `adjust(...)`."*

## Does each call do what we intend?

| Call | Intended | Verified / risk |
|---|---|---|
| `yf.download(auto_adjust=False)` `Close` | raw close | **No.** It's split-adjusted (NVDA 2024-06-06 = 121.00). Hence `unadjust_splits`. |
| `yf.download` `Adj Close` | split + dividend adjusted | Yes (NVDA 2024-06-06 = 120.65 vs 121.00). Used as the **test oracle** for `adjust(..., TOTAL_RETURN)`. |
| `yf.Ticker(s).splits` | full split history, new/old ratio | **Yes**, verified 2026-11-16 (table in §3). The index is local-tz at the open, so compare on local dates. |
| `yf.Ticker(s).dividends` | as paid | Full history, yes. Amounts are **split-adjusted** (NVDA 2023: 0.004 vs 0.04 paid), so un-adjust them. |
| `ccl_from_pairs` on raw closes | ARS per USD | **Wrong before 2026-08-03** (YPF pair 10× off, so the divergence check NaNs every day). Feed it split-adjusted closes. |
| `Base.metadata.create_all` | adds the new column | No, it never alters existing tables. Hence the reset (D10). |
| Yahoo rate limit | known quota | **None published.** Hence our own conservative `RateLimit` plus backoff (§7). |

## Reset & repopulate (D10)

Only the **market-data cache** is wiped. fts `dev.db` also holds things that are not a cache: the model registry (10 rows), Optuna studies, prediction, order and trade logs, and backtest equity. Deleting the whole file would lose them.

| DB | Contents today (2026-11-16) | Action |
|---|---|---|
| fts `apps/fts/dev.db` | `bar_data_logs` 1,507 rows (1d: 507 / 3 symbols; 1m: 500; 30m: 500), `corporate_actions` 0, `fx_rates` 0 | Drop and recreate `bar_data_logs`, `corporate_actions`, `fx_rates` |
| fts `apps/fts/backtests.db` | all empty | Same (no-op in practice) |
| analyser `portfolio_analyser.db` | doesn't exist yet | Created fresh by `create_all` |

- **Script:** `quant_data.db.reset_market_cache(engine)` drops exactly these three tables and recreates them through `Base.metadata.create_all(tables=[...])`. A CLI wrapper is `uv run python -m quant_data.db.reset --db-url <url>`. It prints row counts before it drops anything and asks for confirmation (`--yes` skips the prompt).
- **Repopulate.** No new backfill command; the existing entry points fill the cache on demand through the throttle:
  - `uv run python apps/portfolio_analyser/main.py analyze --broker yaml` (the portfolio's symbols, the CCL pairs, and corporate actions);
  - `fts ingest` per ticker and interval for fts data (BTC/USDT 30m via CCXT, any Yahoo tickers).
- This supersedes PR 2's manual `DELETE ... WHERE interval = '1d'` note.

## Tests

- **2a-1:**
  - `split_factor` / `unadjust_splits` reproduce NVDA 2024-06-06: 121.00 → 1,210.0, with volume ÷10.
  - A reverse split (ratio 0.1).
  - A split exactly on a bar's date leaves that bar unadjusted.
  - Comparison uses the local date (a YPFD.BA-style 11:00-03:00 ex-date vs a 14:00 UTC bar).
  - `adjust(raw, actions, TOTAL_RETURN)` ≈ Yahoo `Adj Close` within 1e-3 relative, using a recorded NVDA fixture with no network.
- **2a-2:**
  - The adapter returns raw for `1d` and an intraday interval (mocked `yf`).
  - Dividends are un-adjusted.
  - Throttle: spacing, retry after a 429, and fall-through after `max_retries` (fake clock).
  - fts `ingest` no longer offers `tiered`.
- **2a-3:**
  - A provider switch triggers a full refetch, not a stitch.
  - An interior gap triggers exactly one refetch.
  - A revised last bar is updated.
  - Corporate actions are served from the cache when the vendor fails.
  - The reset drops only the three cache tables, so the model registry survives.
  - Analyser: CCL from `adjust(..., SPLIT)` is continuous across 2026-08-03. From raw closes it is NaN before the split, which is a guard test.

## fts (backlog questions 1 and 2)

1. **Prior art:** `source_name` (reused). fts `CCLProvider` (unchanged, live). No split handling, gap detection or rate limiting exists in fts.
2. **Applies to fts:**
   - fts's Yahoo plugin goes through the same adapter, so its equity bars become raw at every interval. Any fts feature built from equity bars must call `adjust(..., SPLIT or TOTAL_RETURN)`, or a model will learn fake −90% returns (new AGENTS.md rule).
   - BTC via CCXT has no splits, so nothing changes there.
   - The throttle also protects `fts ingest`.

## Out of scope / known gaps

- Spin-offs, mergers, rights. Yahoo doesn't provide them; the `CorporateActionType` values are left unused.
- Real IBKR and Tiingo adapters. When they arrive, they only have to obey the raw contract and declare a `RateLimit`.
- Alembic. Adopt it if the schema starts changing more often; for now the cache is disposable.
