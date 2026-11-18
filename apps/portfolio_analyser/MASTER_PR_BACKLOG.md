# Portfolio Analyser: Master PR Backlog (The "JIRA")

*This is the complete, unfiltered backlog translating every single feature from our original brainstorming session into vertical-slice Pull Requests. It adheres to Clean Architecture (`quant_core` for math, `quant_data` for DB/API, `portfolio_analyser` for orchestration).*

For each PR: search:
1. is this (loosely) implemented in fts already? Or something similar? If so, we must extract and generalize usage for both apps.
2. Is this something we could apply in fts? If so, how?

---

## Epic 1: Core Architecture & Tiered Data Engine
**PR 1: Provider Evaluation & Tiered Requesting Engine (`quant_data`)** :done
*   **Goal:** Build a resilient, multi-provider waterfall (shipped as `quant_data.providers.TieredDataProvider`; the planned `SmartDataFetcher` name was never used).
*   **Action:** Move away from relying solely on `yfinance`. Evaluate institutional-grade SSOTs (e.g., Tiingo, EODHD, or the IBKR Advisor Historical API). Implement a Tiered Fallback pattern: check DB -> check Tier 1 (e.g., IBKR API) -> fallback to Tier 2 (e.g., Tiingo) -> fallback to `yfinance`.

**PR 2: Broker Snapshot Parsing & Data Binding (`portfolio_analyser`)** :done
*   **Goal:** Solidify the ingestion of broker data (IBKR, Inviu, IOL, YAML).
*   **Action:** Extract all tickers from the broker snapshot and pre-fetch 5 years of daily history using `TieredDataProvider` during the CLI `analyze` initialization.

> **PR 2 notes.** `Position.provider_symbol` maps broker ids to market symbols (e.g. YPF in ARS -> `YPFD.BA`). `quant_data.history.fetch_daily_closes` returns cached raw (unadjusted) daily closes per symbol, indexed by UTC date and capped at today 00:00 UTC, so partial bars are never cached. `TieredDataProvider` tops up range requests incrementally, and an explicit range now wins over `count`. ARS converts at CCL (`quant_core.analytics.ccl_from_pairs`): GGAL and YPF ADR pairs, both 10:1, with days where the pairs diverge by more than 5% rejected. Yahoo has no AL30/AL30D data, so MEP isn't available. The DB engine factory now lives in `quant_data.db.session`. Broker connectors other than YAML are still mocks. **fts migration:** `dev.db` daily bars cached before this PR may be dividend-adjusted, so delete them once: `DELETE FROM bar_data_logs WHERE interval = '1d';`.

**PR 2a: Market-Data Integrity (`quant_core`, `quant_data`)** :done — plan: [plans/PR2a.md](plans/PR2a.md)
*   **Goal:** Cached prices that never silently change. Yahoo's `auto_adjust=False` `Close` is still split-adjusted retroactively, so the insert-only cache breaks at the next split.
*   **Action:** Providers return true raw prices (each adapter un-adjusts its own vendor data). Corporate actions (splits + dividends) go behind a separate `CorporateActionProvider` and are cached. `adjust(closes, actions, mode)` gives split/total-return series at read time. Each bar records its `source` (one source per symbol). Interior gaps are refetched, last-bar revisions are upserted, and mock tiers are removed from production. CCL is computed on split-adjusted closes (YPFD.BA split 10:1 on 2026-08-03).

**PR 2b: Transaction Ledger & Real Holdings History (`portfolio_analyser`)** :done — plan: [plans/PR2b.md](plans/PR2b.md)
*   **Goal:** Real client performance. A snapshot shows only today's holdings, and backcasting them as performance has survivorship and look-ahead bias.
*   **Data:** Add a hand-written transactions list (trades, dividends, fees, deposits, withdrawals) to `Portfolio` and the YAML format. Splits come from PR 2a's corporate actions, never from the YAML.
*   **Math (`quant_core`):** Daily quantities from the ledger x raw closes x daily FX (CCL from split-adjusted closes, forward-filled across local holidays with a limit) gives the value series in USD. **Only deposits and withdrawals become `cash_flows`.** Dividends, coupons and fees are internal and part of the return. A missing price is NaN, never 0.
*   **App Integration:** `analyze` values positions at the latest market close (not the YAML price) and reconciles ledger quantities against the snapshot. `client-review` feeds the real values and cash flows into `backward_analytics_figures`, replacing the mock random walk and the mock baseline.

> **PR 2a notes (2026-11-16).** Shipped as three stacked PRs (#42 maths, #43 adapter + throttle, #44 cache rules). Yahoo's `auto_adjust=False` frame is still split-adjusted (closes, volumes *and* dividend amounts); the adapter undoes it with `Ticker.splits` on every interval, so `get_bars` is raw everywhere, including fts. Daily bars are keyed by the local trading date at 00:00 UTC (yfinance 1.7.0 returns a tz-naive date index), and so are `ex_date`s. `adjust(..., TOTAL_RETURN)` reproduces Yahoo `Adj Close` to 1e-6. Each bar records its `source`; one vendor per symbol; gaps > 5 days refetch once (BYMA's 2024 Easter + Malvinas bridge is a real 7-day gap, so GGAL.BA/YPFD.BA refetch every run). `RateLimit` per provider, enforced by `Throttle`. **Operational:** fts `dev.db` needs `uv run python -m quant_data.db.reset --db-url sqlite:///apps/fts/dev.db` once (drops only the three cache tables), then `fts ingest` to repopulate; `--provider tiered` is gone.

> **PR 2b notes (2026-11-16).** Shipped as #45 (model + `quant_core.ledger`), #46 (`analyze` at market close + reconciliation, NaN for unpriced) and #47 (`client-review` on the ledger). Decisions D1–D14 in the plan: `transfer_in`/`transfer_out` are external flows at market value; `fee` lines are internal; a quantity break is an error and a cash break a warning; bonds/coupons are rejected until PR 2c; prices and CCL forward-fill 5 sessions; NaN days are dropped with a warning (dropping first keeps the return across the hole); dividends are dated on the pay date. The mock SPY baseline is gone; PR 7 brings the honest one. **fts follow-up:** `holdings` + `cash` + `portfolio_values` can rebuild a backtest's equity from `TradeLog` fills as a consistency test against `BacktestEquityLog`. **Follow-up (PR 7 D10):** accrue a dividend receivable from the ex-date to the pay date, so the ledger and a total-return benchmark move on the same day.

**PR 2c: Bond Prices & Coupons (`quant_data`, `quant_core`)** — TODO
*   **Goal:** Value Argentine sovereign bonds (AL30, GD30, AL30D) and record coupons. PR 2b rejects `coupon` lines and bond listings because no price source exists (Yahoo has none; PR 2 checked).
*   **Research:** a daily-close source for BYMA bonds (candidates: BYMA data, Rava, IOL/PPI APIs, data912). It must obey PR 2a's raw contract and declare a `RateLimit`. Decide how to key the listing (AL30 in ARS vs AL30D in USD).
*   **Action:** adapter + cache; lift the PR 2b validator; `coupon` lines are internal income (cash in, never an external flow); amortising bonds change the quantity on each payment, which is a corporate action, not a trade. A bond pair (AL30 / AL30D) also gives the MEP rate.

---

## Epic 2: Backward Analytics - Return & Risk Basics
**PR 3: Core Returns (TWR & MWR)** :done
*   **Math (`quant_core`):** Implement `compute_twr(df)` and `compute_mwr(cashflows, values)`.
*   **UI (`quant_ui`):** Add Cumulative Return vs Baseline line chart.

**PR 4: Rolling Volatility** :done
*   **Math (`quant_core`):** `compute_rolling_vol(df, window)` using `pandas.DataFrame.rolling`.
*   **UI (`quant_ui`):** Add 30d, 90d, and 252d rolling volatility sub-charts.

**PR 5: Drawdown Analytics** :done
*   **Math (`quant_core`):** Calculate High-Water Marks, Maximum Drawdown (MDD), and "Time Underwater".
*   **UI (`quant_ui`):** Create the red-shaded underwater plot highlighting the top 5 historic drawdowns.

**PR 6: Risk-Adjusted Ratios** :done
*   **Math (`quant_core`):** `compute_sharpe`, `compute_sortino`, `compute_calmar`, and `compute_information_ratio`.
*   **App Integration:** Surface these metrics in a "Risk KPIs" summary table.

> **Epic 2 notes.** Math lives in `quant_core.analytics` (returns / drawdown / ratios / trades), composed by `quant_core.metrics.compute_performance_metrics`, which both fts (`trading_bot.core.performance`) and the analyser (`visualization.plot_factory`) use. Conventions: fractions not percents, NaN for undefined stats, ddof=1, periods-per-year inferred from the data, ratios on daily closes. `client-review` runs on the ledger's real values and flows since PR 2b. `compute_information_ratio` is surfaced by PR 7's benchmark KPIs.

---

## Epic 3: Backward Analytics - Benchmarking & Attribution
**PR 7: Alpha, Beta & Market Capture** :done
*   **Math (`quant_core`):** Regress portfolio returns against a benchmark (e.g., SPY) to extract Alpha and Beta. Calculate Up-Market and Down-Market Capture Ratios.
*   **UI (`quant_ui`):** Scatter plot of Portfolio Returns vs Benchmark Returns with the regression line.
*   **Review notes (2026-11-15):**
    *   **The benchmark must be a fair comparison.** It's the total return (`adjust(closes, actions, TOTAL_RETURN)` from PR 2a, since raw SPY closes miss about 1.3%/yr of dividends), in the portfolio's base currency (USD at CCL), and aligned to the portfolio's dates before any return is taken. Today `plot_factory.backward_analytics_figures` builds the baseline with `wealth_index(raw closes)` on its own index; fix that here.
    *   **Regress daily excess returns** (minus the risk-free rate). Report alpha and beta with **HAC (Newey-West) standard errors / confidence intervals**, because daily returns are autocorrelated and heteroskedastic, and a point estimate alone overstates certainty. Show the number of observations, and show NaN below a minimum sample size.
    *   **Capture ratios** use the same aligned series. Wire `compute_information_ratio` (already in `quant_core`) to this benchmark.

> **PR 7 notes (2026-11-17).** Maths in `quant_core.analytics.benchmark` (`align_daily`, `hac_ols`, `compute_capture_ratios`) and `.rates` (`excess_returns`, `tbill_bond_equivalent_yield`), composed by `quant_core.metrics.compute_benchmark_metrics` into `BenchmarkMetrics` (`extra="forbid"`). Decisions: **D1** rf = daily 13-week T-bill `^IRX` (quoted as a % discount yield → /100 → bond-equivalent yield) unless `--risk-free-rate`; **D2** a non-USD base currency must pass the flag; **D3** the same rf feeds Sharpe/Sortino/alpha (so client-review's Sharpe fell, e.g. 0.93 → 0.55 on a live AAPL+cash ledger), fts keeps 0; **D4** `--dividend-withholding` (default gross, labelled); **D5** n ≥ 252 aligned daily returns, else NaN (n always shown); **D6** missing SPY/`^IRX` → warning + n/a; **D7** docs say `uv run python main.py` (no `[project.scripts]` until PR 26 packages the app); **D9** Dimson β (lag 1) beside β; **D10** ex-date vs pay-date dividends accepted. Method: the flow-adjusted wealth index is built *first*, then levels are inner-joined on dates where both have a real price (no fake 0% holiday returns), then returns are taken. rf accrues ACT/365 over each interval's actual length, at the rate known at its start (7-day staleness → NaN). OLS + Newey-West (Bartlett, `floor(4(n/100)^(2/9))` lags, n/(n−k)), matching statsmodels `use_correction=True` to 1e-10; t(n−k) 95% CIs; alpha = daily intercept × inferred periods/year (arithmetic). Captures: Morningstar geometric form on total returns. R² = squared correlation of daily returns. `compute_sharpe`/`compute_sortino` now take excess returns. Also fixed: `usd_per_unit` lost USD for a USD-only portfolio (no CCL pairs fetched); `analyze` reuses `_print_reconciliation`. **fts:** `build_benchmark_metrics` (buy-and-hold of the logged `close`; `close = 0.0` from the engine is treated as missing) on `BacktestRunCatalogItem` / `BacktestDetailDTO.benchmark_metrics`; `visualizer.calculate_pnl` uses `wealth_index` (NaN, not 0.0, when undefined).
> **Known limits.** BYMA vs NYSE closes, CCL from NYSE ADRs and forward-filled prices bias daily β down (compare Dimson β); flow-day timing and ARS devaluation days are regression outliers (not winsorised); holding GGAL/YPF makes their USD returns partly mechanical (they define the CCL); daily capture ratios are noisier than monthly; cash dilutes β (correct); SPY may be the wrong comparator for an ARS book (R² shows it). fts: raw `close` is wrong across equity splits; α of in-sample backtests (`allow_in_sample: true`) is fitted noise; the best of many Optuna trials has an inflated α t-stat (multiple testing; don't filter the catalog on α significance without a deflation correction); a long/flat/short strategy's β is roughly its average exposure. The fts engine's `close = 0.0` equity bug is fixed in PR 7b.

**PR 7b: One equity number in fts** :done
*   **Problem:** fts valued the portfolio twice, with different prices. The backtest engine logged `cash + qty × last bar close`, with the close set to **0.0 on a tick without a bar** (a fake crash in the equity curve, and `final_equity` = cash if the last tick had no bar, which feeds the sweep's `total_pnl`). `Portfolio.get_state` used bid/ask, falling back to **cost basis** without a book; backtest readers never build a book. Both sizers also read `portfolio_state.total_balance_usd`, which `Portfolio` doesn't have: the first sized signal raised AttributeError.
*   **Fix (`trading_bot/risk_management/portfolio.py`):** `Portfolio.equity()` is the single number: cash plus every position at its **last bar close**, carried over ticks without a bar (cost basis before any close is seen). The engine log, `get_state` (so the RiskManager's limits and both sizers) all use it. **Bid/ask only for the allocation-limit check** (`calculate_unrealized_pnl`), falling back to the last close. Ticks before the instrument's first bar are not logged. No schema change. The signals-only fallback curve in `quant_ui.plots.performance` is labelled "simulated from signals".
*   **Behaviour change:** backtest sizes and limit checks now use the last close instead of cost basis, so old runs aren't comparable: re-run them. Old stored equity logs keep their 0.0 rows; `build_benchmark_metrics` still treats `close ≤ 0` as missing for them.
*   **Open bug: prediction-market (outcome) positions.** Positions with an outcome are keyed `"{instrument_id}:{outcome}"` (`portfolio.py`), but the engine reads `positions.get(instrument_id)`, so the logged `position` column is 0 for them. `equity()` now includes them, but at **cost basis**: the bars are the instrument's, and the YES/NO side's own price isn't in `MarketData`. A related lookup in `RiskManager` (`pnl_map.get(pos.instrument_id)`, max-allocation check) also misses outcome keys, so their unrealized P&L counts as 0 there. Fix when a prediction-market strategy is backtested: carry a per-outcome price (e.g. YES = last trade / mid, NO = 1 − YES) and key every lookup by `(instrument_id, outcome)`. Test: `test_risk_management.py::TestSingleValuation::test_outcome_positions_are_at_cost_basis` documents today's behaviour.
*   **Also open:** multi-instrument engine runs (`instrument_id=None`) log `close = 0.0`, because there is no single traded instrument; their equity is right.

**PR 8: Brinson-Fachler Attribution**
*   **Math (`quant_core`):** Matrix math to separate excess returns into *Asset Allocation* effects vs. *Security Selection* effects.
*   **UI (`quant_ui`):** Waterfall chart showing the attribution breakdown.

**PR 9: Cost & Friction Drag Engine**
*   **Math (`quant_core`):** Compute Portfolio Turnover Ratio and estimate tax/commission drag based on historical trade frequency.

---

## Epic 4: Forward Projections & Tail Risk

> **Review notes (2026-11-15). Suggested order after PR 7:** PR 10 → PR 11 → PR 19 → PR 18. Each is cheap (about a day) and stays valid on short histories. Do them before the model-heavy PRs (8, 12, 16, 17). Inputs for all four are the ledger's daily wealth-index returns (PR 2b), never back-cast snapshot weights.
> *   **PR 10:** EWMA (RiskMetrics λ=0.94 for daily data, stated as a parameter, not a magic number) on daily returns, annualised with the observed `periods_per_year`. Backtest the forecast (realised vs forecast vol) so it's checked against reality.
> *   **PR 11:** Lead with **historical** VaR/CVaR. Parametric Gaussian VaR understates ARS tail risk, so show it only next to the historical figure, labelled as such. State the horizon (1-day) and the sign convention (loss as a positive fraction). With fewer than about 250 observations, 99% figures rest on 2–3 points: show the observation count, and NaN below a minimum.
> *   **PR 19:** Correlations on aligned daily returns in base currency (pairwise-complete obs, minimum overlap, otherwise NaN). Expect ARS listings to look highly correlated, because they share the CCL factor. Optionally show local-currency correlations next to them.
> *   **PR 18:** Drift = actual − target weights at market value (PR 2b prices). An unpriced position is a violation, not weight 0.
**PR 10: Expected Volatility Forecast** :done
*   **Math (`quant_core`):** Implement an Exponentially Weighted Moving Average (EWMA) volatility forecaster.

> **PR 10 notes (2026-11-18).** Maths in `quant_core.analytics.volatility` (`ewma_variance`, `qlike`, `evaluate_variance_forecast`, `annualise_variance`; `RISKMETRICS_DAILY_LAMBDA = 0.94` as the default of a `lam` parameter, `EWMA_SEED_OBS = 30`), composed by `quant_core.metrics.compute_vol_forecast` into `VolForecastMetrics` holding two `ForecastEvaluation`s (both `extra="forbid"`). Decisions: **D1** baseline = rolling one-year sample variance (`compute_rolling_vol`, reused); **D2** fts exposes it on `BacktestDetailDTO.vol_forecast` only (not computed per catalog row); **D3** the chart's realised line = vol of the next month; **D4** seed = mean r² of the first 30 returns, NaN before; **D5** zero-mean EWMA (RiskMetrics); **D6** EWMA and baseline are scored on the *same* out-of-sample days, NaN below 252 (`MIN_VOL_EVAL_OBS`), counts always shown; **D8** baseline window = `round(ppy)` and horizon = `round(ppy/12)` obs (252/21 equities, 365/30 crypto). Method: daily returns of the flow-adjusted wealth index → pandas `ewm(alpha=1−λ, adjust=False)` from the seed; the value at t forecasts t+1, shifted by one before scoring (no look-ahead). Scores: QLIKE `r²/h + ln h` (Patton 2011, finite at r = 0), Mincer-Zarnowitz r² on h with Newey-West SEs, std of r/√h. On a simulated GARCH(1,1) the EWMA beats the baseline (QLIKE −8.44 vs −8.34) with slope 0.71 ± 0.10 (λ over-reacts vs that process); on a live AAPL ledger (972 days) −8.19 vs −8.10, slope 0.64 ± 0.34 vs 0.14. **Refactors (one home for the maths):** `hac_ols` / `HacFit` / `newey_west_lags` moved to `analytics.regression`; `analytics.daily_returns(wealth)` replaces the copies in `compute_performance_metrics` and `plot_factory`; `metrics.compute_rolling_vols` replaces the app's own rolling-vol loop; `quant_ui` KPI tables share `_kpi_table`. **Known limits:** λ not re-estimated for ARS assets (fitting it by min QLIKE is a follow-up, watch in-sample bias); no mean reversion (IGARCH), so the annualised figure overstates a 1-year vol after shocks; history as held, not current weights (PR 12/19); stale BYMA prices understate vol; fat tails dominate QLIKE; pay-date dividends add small jumps; scores need ~2 years of history; the (30, 90, 252) rolling-vol chart windows are still fixed obs counts.

**PR 11: Tail Risk Modeling (VaR & CVaR)**
*   **Math (`quant_core`):** Parametric Value at Risk (VaR) and Historical Expected Shortfall (CVaR) at 95% and 99%.
*   **UI (`quant_ui`):** Daily returns distribution histogram emphasizing the left-tail risk zone.

**PR 12: Monte Carlo Wealth Simulations**
*   **Math (`quant_core`):** Generate 10,000 random walk paths using historical covariance matrices via `numpy`.
*   **UI (`quant_ui`):** Plotly probability cone chart for 1, 3, and 5-year terminal wealth projections (p5, p50, p95).

**PR 13: Income & Yield Projections**
*   **Data (`quant_data`):** Fetch dividend yield and coupon schedules.
*   **App Integration:** Generate a 12-month forward-looking calendar of expected cash flows.

---

## Epic 5: Stress Testing & Advanced Models
**PR 14: Historical Scenario Replays**
*   **Math (`quant_core`):** Project current portfolio weights backward onto the 2008 Financial Crisis and the 2020 COVID-19 shock timelines.
*   **UI (`quant_ui`):** Overlay charts comparing the portfolio's hypothetical performance against the S&P 500 during those specific periods.

**PR 15: Hypothetical Macro Shocks**
*   **Math (`quant_core`):** Sensitivity testing models (e.g., simulating a +100bps interest rate hike or a 10% equity shock based on historical beta).

**PR 16: Factor Exposures (Smart Beta)**
*   **Math (`quant_core`):** Multi-factor regression against Fama-French factors (Size, Value, Momentum, Quality).
*   **UI (`quant_ui`):** Radar/Spider chart showing the portfolio's factor tilt.

**PR 17: Portfolio Optimization**
*   **Math (`quant_core`):** Markowitz Efficient Frontier calculator and Black-Litterman model implementation using `scipy.optimize`.
*   **UI (`quant_ui`):** Efficient frontier scatter plot highlighting the current portfolio vs. the optimal portfolio.

---

## Epic 6: Operational Governance & Smart Alerts
**PR 18: Policy Corridor & Asset Drift Alerts**
*   **Core (`quant_core`):** Implement Pydantic `TargetPolicy` validation.
*   **App Integration:** CLI console warnings when `actual_weights - target_weights` breaches user-defined thresholds (e.g., > 5%).

**PR 19: Correlation Matrix Clustering**
*   **Math (`quant_core`):** Compute Pearson correlation matrices and apply hierarchical clustering.
*   **UI (`quant_ui`):** Clustered heatmap visualizing hidden asset overlaps.

**PR 20: Tax-Loss Harvesting Radar**
*   **Math (`quant_core`):** Algorithm to flag tax lots with significant unrealized losses.
*   **Data (`quant_data`):** Map primary tickers to highly correlated "Tax Proxy" ETFs.

**PR 21: Liquidity Profiling**
*   **Data (`quant_data`):** Fetch Average Daily Volume (ADV).
*   **Math (`quant_core`):** Calculate "Days to Liquidate" given current position sizes, assuming max 10% participation in daily volume.

---

## Epic 7: Delivery
**PR 22: The Master Report Engine**
*   **UI (`quant_ui`):** Create a robust Jinja2 templating system (`report_template.html`) that binds all 21 preceding components into a beautiful, multi-tab local web dashboard.
*   **App Integration:** The final `main.py analyze --full-report` execution command.

**PR 26: Local Dashboard & Deployment (`quant_ui`, fts dashboard, `portfolio_analyser`)**
*   **Goal:** One locally deployed browser UX for both apps: the fts backtest catalog plus the analyser's `analyze` / `client-review`, without the CLI.
*   **Prerequisite:** make the analyser importable. Move `main.py`, `data_layer/`, `domain/`, `visualization/` and `config/` under a `portfolio_analyser/` package with a build backend, which also unblocks `[project.scripts] portfolio-analyser` (PR 7 D7). The Streamlit app can't cleanly import generic top-level `config` / `domain` modules.
*   **Action:** Add Streamlit pages to `trading_bot/dashboard/app.py`, or lift the dashboard shell into a shared package if both apps' pages grow. Pages: upload/select a portfolio YAML, pick the benchmark / risk-free rate / withholding, and render the same `backward_analytics_figures`. No new maths: the pages call the same functions as the CLI. fts pages show PR 7's `benchmark_metrics`.
*   **Deploy:** `uv run streamlit run …` for development; a `Dockerfile` + `docker-compose.yml` with the market-data cache DB on a mounted volume. Bind to localhost only (`--server.address 127.0.0.1`, `127.0.0.1:` port mapping) because reports contain client PII (Epic 8). No credentials baked into the image (PR 23).
*   **fts:** the same container serves the existing backtest explorer.

---

## Epic 8: INFOSEC & Advisor-Grade Security
*Objective: Harden the application for Advisor API usage. Since this connects to institutional master accounts and generates reports for end-clients, preventing PII leaks and securing credentials is a top priority.*

**PR 23: Secrets Management & API Hardening**
*   **Action:** Remove `.env` or hardcoded dependencies for broker APIs. Integrate a secure secrets management workflow (e.g., HashiCorp Vault, AWS Secrets Manager, or heavily restricted environment injection) for handling high-value IBKR/Inviu/IOL Advisor credentials. Ensure all API connections enforce strict TLS.

**PR 24: Client Data Isolation & PII Redaction**
*   **Action:** Implement sanitization middleware. Scrub client names, account numbers, and PII from all application logs, error traces, and database caches. The system should internally reference accounts via anonymized UUIDs, injecting client names only at the final PDF/HTML rendering stage.

**PR 25: Supply Chain Security & Dependency Auditing**
*   **Action:** Integrate `pip-audit` or `safety` into the CI/CD pipeline to continuously scan `quant_core`, `quant_data`, and other packages for CVEs. Lock down and verify `uv.lock` hashes to prevent malicious dependency injections.
