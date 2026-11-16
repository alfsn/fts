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

> **PR 2b notes (2026-11-16).** Shipped as #45 (model + `quant_core.ledger`), #46 (`analyze` at market close + reconciliation, NaN for unpriced) and #47 (`client-review` on the ledger). Decisions D1–D14 in the plan: `transfer_in`/`transfer_out` are external flows at market value; `fee` lines are internal; a quantity break is an error and a cash break a warning; bonds/coupons are rejected until PR 2c; prices and CCL forward-fill 5 sessions; NaN days are dropped with a warning (dropping first keeps the return across the hole); dividends are dated on the pay date. The mock SPY baseline is gone; PR 7 brings the honest one. **fts follow-up:** `holdings` + `cash` + `portfolio_values` can rebuild a backtest's equity from `TradeLog` fills as a consistency test against `BacktestEquityLog`.

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

> **Epic 2 notes.** Math lives in `quant_core.analytics` (returns / drawdown / ratios / trades), composed by `quant_core.metrics.compute_performance_metrics`, which both fts (`trading_bot.core.performance`) and the analyser (`visualization.plot_factory`) use. Conventions: fractions not percents, NaN for undefined stats, ddof=1, periods-per-year inferred from the data, ratios on daily closes. `client-review` runs on the ledger's real values and flows since PR 2b. **Open:** `compute_information_ratio` has no UI consumer until PR 7 brings in a benchmark.

---

## Epic 3: Backward Analytics - Benchmarking & Attribution
**PR 7: Alpha, Beta & Market Capture**
*   **Math (`quant_core`):** Regress portfolio returns against a benchmark (e.g., SPY) to extract Alpha and Beta. Calculate Up-Market and Down-Market Capture Ratios.
*   **UI (`quant_ui`):** Scatter plot of Portfolio Returns vs Benchmark Returns with the regression line.
*   **Review notes (2026-11-15):**
    *   **The benchmark must be a fair comparison.** It's the total return (`adjust(closes, actions, TOTAL_RETURN)` from PR 2a, since raw SPY closes miss about 1.3%/yr of dividends), in the portfolio's base currency (USD at CCL), and aligned to the portfolio's dates before any return is taken. Today `plot_factory.backward_analytics_figures` builds the baseline with `wealth_index(raw closes)` on its own index; fix that here.
    *   **Regress daily excess returns** (minus the risk-free rate). Report alpha and beta with **HAC (Newey-West) standard errors / confidence intervals**, because daily returns are autocorrelated and heteroskedastic, and a point estimate alone overstates certainty. Show the number of observations, and show NaN below a minimum sample size.
    *   **Capture ratios** use the same aligned series. Wire `compute_information_ratio` (already in `quant_core`) to this benchmark.

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
**PR 10: Expected Volatility Forecast**
*   **Math (`quant_core`):** Implement an Exponentially Weighted Moving Average (EWMA) volatility forecaster.

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

---

## Epic 8: INFOSEC & Advisor-Grade Security
*Objective: Harden the application for Advisor API usage. Since this connects to institutional master accounts and generates reports for end-clients, preventing PII leaks and securing credentials is a top priority.*

**PR 23: Secrets Management & API Hardening**
*   **Action:** Remove `.env` or hardcoded dependencies for broker APIs. Integrate a secure secrets management workflow (e.g., HashiCorp Vault, AWS Secrets Manager, or heavily restricted environment injection) for handling high-value IBKR/Inviu/IOL Advisor credentials. Ensure all API connections enforce strict TLS.

**PR 24: Client Data Isolation & PII Redaction**
*   **Action:** Implement sanitization middleware. Scrub client names, account numbers, and PII from all application logs, error traces, and database caches. The system should internally reference accounts via anonymized UUIDs, injecting client names only at the final PDF/HTML rendering stage.

**PR 25: Supply Chain Security & Dependency Auditing**
*   **Action:** Integrate `pip-audit` or `safety` into the CI/CD pipeline to continuously scan `quant_core`, `quant_data`, and other packages for CVEs. Lock down and verify `uv.lock` hashes to prevent malicious dependency injections.
