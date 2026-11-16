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

## Decisions (made with the user, 2026-11-15)

1. **Store true raw prices + a corporate-actions table, and compute adjusted series when reading.** This is the industry pattern. CRSP ships raw prices plus cumulative adjustment factors (`cfacpr`/`cfacshr`). zipline keeps bars and a separate splits/dividends store. QuantConnect derives Raw / SplitAdjusted / TotalReturn / Adjusted modes from one stored basis.
2. **Each provider adapter returns raw.** This is an anti-corruption layer: a vendor that only sends adjusted data (Yahoo) un-adjusts it inside its own adapter. The cache and `TieredDataProvider` stay unaware of it.
3. **Corporate actions get their own small interface** (`CorporateActionProvider`), following interface segregation. CCXT and the mocks don't implement it.
4. **One source per symbol.** Every bar records its provider. A series is never stitched from two providers.
5. **Cache dividends now** (PR 7 needs them for a total-return benchmark).

Sources: [CRSP factor adjustment (UMich)](https://leiq.bus.umich.edu/docs/crsp_factor_adjustment.pdf), [zipline data bundles](https://zipline-trader.readthedocs.io/en/stable/bundles.html), [QuantConnect normalization modes](https://www.quantconnect.com/forum/discussion/508/update-dividends-splits-and-custom-price-normalization/p1), [Yahoo: what is adjusted close](https://help.yahoo.com/kb/SLN28256.html).

## Design

### 1. Contract (`quant_core.interfaces`)

- `BaseMarketDataProvider.get_bars` docstring states the rule: **prices and volumes are as traded (raw)**.
- Add a `source_name` property, defaulting to the class name. This reuses fts's existing convention (`trading_bot/data_ingestion/abc.py:31`, `plugins/argentina/data_providers.py:31`).
- New ABC:

```python
class CorporateActionProvider(ABC):
    @abstractmethod
    def get_corporate_actions(self, instrument_id: str) -> Sequence[CorporateAction]:
        """Full history, raw (as-paid) amounts. No date range on purpose: un-adjusting
        a bar needs every split *after* it, however far outside the requested range."""
```

- `CorporateAction` (already in `quant_core.models`) is reused. Document `ratio` as **new shares per old share** (10:1 → 10.0; 1:10 reverse → 0.1). Set `extra="forbid"`.

### 2. Pure maths (`quant_core.analytics.corporate_actions`, new)

| Function | What it does |
|---|---|
| `split_factor(index, splits) -> Series` | For each date t, the product of `ratio` over splits with `ex_date > t` (compared on **dates**, not timestamps). That many of today's shares equal one share held at t. |
| `unadjust_splits(df, splits) -> df` | Undoes the vendor's adjustment. Raw OHLC = adjusted × factor; raw volume = adjusted ÷ factor. Used only inside adapters. |
| `adjust(closes, actions, mode)` | `mode` ∈ {`RAW`, `SPLIT`, `TOTAL_RETURN`}. `SPLIT`: raw ÷ factor. `TOTAL_RETURN`: also multiplies every price before an ex-date by `1 − D / P_prev`, CRSP-style, where `D` and `P_prev` are both raw. Charts, CCL, benchmarks and model features use this. |

Add an enum `PriceAdjustment` in `quant_core.enums`.

### 3. Yahoo adapter (`quant_data.providers.yfinance_provider`)

- Implements `BaseMarketDataProvider` and `CorporateActionProvider`.
- In `get_bars`, fetch the full split history once per symbol per instance (`yf.Ticker(s).splits`, which has no range), then `unadjust_splits` on the downloaded frame. `dollar_volume = raw close × raw volume` (a split doesn't change it).
- In `get_corporate_actions`, splits come from `Ticker.splits`. Dividends come from `Ticker.dividends`, **un-adjusted** (amount × `split_factor` at the ex-date), because Yahoo split-adjusts dividends too.
- Fix the misleading comment at `yfinance_provider.py:105`.

### 4. Cache (`quant_data.db`, `TieredDataProvider`)

- `BarDataLog.source` and `CorporateActionRecord.source` are nullable `String` columns.
- **Single-source rule.** If the cached series' `source` differs from the provider that answers a top-up, refetch the full range from that provider and replace the symbol's rows for that interval (`repo.replace_bars`). Legacy rows have `source = NULL`, which counts as a mismatch, so old Yahoo-adjusted rows heal themselves on the next run.
- **Gaps.** If two consecutive cached bars inside the requested range are more than `MAX_CALENDAR_GAP` apart (today's `START_COVERAGE_TOLERANCE`, 5 days, renamed), refetch the full range once. If the gap persists, log a warning and serve what we have.
- **Revisions.** `save_bars` updates bars that already exist (matched on the unique key) instead of skipping them. The top-up already re-requests the last cached bar, so a late correction to it gets in.
- **Corporate actions.** `TieredDataProvider` also implements `CorporateActionProvider`:
  - It asks the provider named in the cached bars' `source`, so the actions come from the same vendor as the bars.
  - It upserts on `uq_corporate_action`.
  - It refreshes on every run (one call per symbol) and falls back to the cache when the vendor fails.
- `quant_data.history.fetch_corporate_actions(provider, symbols) -> dict[str, list[CorporateAction]]` sits next to `fetch_daily_closes`.

### 5. Remove mocks from production

- Delete `TieredDataProvider.create_default()`.
- Move `MockIBKRProvider` and `MockTiingoProvider` to `packages/quant_data/tests/`.
- Remove fts `--provider tiered`, or wire it to a real chain: `[YFinanceMarketDataProvider()]`.

### 6. CCL on split-adjusted closes

- `latest_usd_per_unit` / `build_fx_context` call `ccl_from_pairs(adjust(closes, actions, SPLIT), pairs)`. Each pair's ratio is *today's* depositary ratio, which is correct across all history only on split-adjusted data.
- Move `ARS_CCL_PAIRS` out of `quant_core` into `quant_data` (e.g. `quant_data/reference/ars.py`). Ticker symbols and depositary ratios are reference data, not maths.
- Make `pairs` a required argument of `ccl_from_pairs`.
- Fix the stale "YPF 1" in the `implied_ccl` docstring.
- fts `CCLProvider` works on live order books, so today's ratio is right there. No change.

### 7. Chores

- **AGENTS.md test command.** A single pytest run fails at collection: `apps/fts/tests` and `packages/quant_core/tests` are both a package named `tests`, and `--import-mode=importlib` breaks the analyser's imports. Document the three commands CI runs instead:
  `uv run pytest packages` · `uv run pytest apps/portfolio_analyser/tests` · `uv run pytest apps/fts/tests`.
- **New AGENTS.md rule:** *"Valuation uses raw prices × share counts; returns, charts, features and CCL use `adjust(...)`."*

## Does each call do what we intend?

| Call | Intended | Verified / risk |
|---|---|---|
| `yf.download(auto_adjust=False)` `Close` | raw close | **No.** It's split-adjusted (NVDA 2024-06-06 = 121.00). Hence `unadjust_splits`. |
| `yf.download` `Adj Close` | split + dividend adjusted | Yes (NVDA 2024-06-06 = 120.65 vs 121.00). Used as the **test oracle** for `adjust(..., TOTAL_RETURN)`. |
| `yf.Ticker(s).splits` | full split history | Chart API `events=split` with no range returns all (NVDA: 6 splits since 2000). Recheck `Ticker.splits` on yfinance 1.7.0 during the PR. |
| Yahoo dividend amounts | as paid | **No.** Split-adjusted (NVDA 2023: 0.004 vs 0.04 paid). Un-adjust. |
| Yahoo split timestamp | ex-date | First post-split session (NVDA 2024-06-10 13:30 UTC; YPFD.BA 2026-08-03 14:00 UTC). Compare on dates. |
| `ccl_from_pairs` on raw closes | ARS per USD | **Wrong before 2026-08-03** (YPF pair 10× off, so the divergence check NaNs every day). Feed it split-adjusted closes. |
| `Base.metadata.create_all` | adds the new column | **No.** It never alters existing tables. Use the one-off migration below. |

## Migration (one-off, per DB: `portfolio_analyser.db`, fts `dev.db`)

```sql
ALTER TABLE bar_data_logs ADD COLUMN source VARCHAR;
ALTER TABLE corporate_actions ADD COLUMN source VARCHAR;
```

Legacy rows have `source = NULL` and are replaced automatically on the next fetch. This supersedes PR 2's manual `DELETE ... WHERE interval = '1d'` note.

## Tests

- **`quant_core`:**
  - `split_factor` / `unadjust_splits` reproduce NVDA 2024-06-06: 121.00 → 1,210.0, with volume ÷10.
  - A reverse split (ratio 0.1).
  - A split exactly on a bar's date leaves that bar unadjusted.
  - `adjust(raw, actions, TOTAL_RETURN)` ≈ Yahoo `Adj Close` within 1e-3 relative, using a recorded NVDA fixture with no network in CI.
- **`quant_data`:**
  - A NULL-source cache is replaced.
  - A provider switch triggers a full refetch, not a stitch.
  - An interior gap triggers one refetch.
  - A revised last bar is updated.
  - Corporate actions are served from the cache when the vendor fails.
- **Analyser:** with raw YPFD.BA spanning 2026-08-03, CCL from `adjust(..., SPLIT)` is continuous, and from raw closes it is NaN before the split. That second check is a guard test.

## fts (backlog questions 1 and 2)

1. **Prior art:** `source_name` (reused). fts `CCLProvider` (unchanged, live). No split handling or gap detection exists in fts.
2. **Applies to fts:** fts's Yahoo plugin goes through the same adapter, so its daily equity bars become raw. Any fts feature built from equity bars must call `adjust(..., SPLIT or TOTAL_RETURN)`, or a model will learn fake −90% returns. This is covered by the new AGENTS.md rule. BTC via CCXT has no splits, so nothing changes there.

## Out of scope / known gaps

- Spin-offs, mergers, rights. Yahoo doesn't provide them; the `CorporateActionType` values are left unused.
- Real IBKR and Tiingo adapters. When they arrive, they only have to obey the raw contract.
