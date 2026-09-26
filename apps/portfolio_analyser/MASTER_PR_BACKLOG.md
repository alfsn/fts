# Portfolio Analyser: Master PR Backlog (The "JIRA")

*This is the complete, unfiltered backlog translating every single feature from our original brainstorming session into vertical-slice Pull Requests. It adheres to Clean Architecture (`quant_core` for math, `quant_data` for DB/API, `portfolio_analyser` for orchestration).*

---

## Epic 1: Core Architecture & Tiered Data Engine
**PR 1: Provider Evaluation & Tiered Requesting Engine (`quant_data`)**
*   **Goal:** Build `quant_data.engine.SmartDataFetcher` with a resilient, multi-provider waterfall.
*   **Action:** Move away from relying solely on `yfinance`. Evaluate institutional-grade SSOTs (e.g., Tiingo, EODHD, or the IBKR Advisor Historical API). Implement a Tiered Fallback pattern: check DB -> check Tier 1 (e.g., IBKR API) -> fallback to Tier 2 (e.g., Tiingo) -> fallback to `yfinance`.

**PR 2: Broker Snapshot Parsing & Data Binding (`portfolio_analyser`)**
*   **Goal:** Solidify the ingestion of broker data (IBKR, Inviu, IOL, YAML).
*   **Action:** Extract all tickers from the broker snapshot and pre-fetch 5 years of daily history using the new `SmartDataFetcher` during the CLI `analyze` initialization.

---

## Epic 2: Backward Analytics - Return & Risk Basics
**PR 3: Core Returns (TWR & MWR)**
*   **Math (`quant_core`):** Implement `compute_twr(df)` and `compute_mwr(cashflows, values)`.
*   **UI (`quant_ui`):** Add Cumulative Return vs Baseline line chart.

**PR 4: Rolling Volatility**
*   **Math (`quant_core`):** `compute_rolling_vol(df, window)` using `pandas.DataFrame.rolling`.
*   **UI (`quant_ui`):** Add 30d, 90d, and 252d rolling volatility sub-charts.

**PR 5: Drawdown Analytics**
*   **Math (`quant_core`):** Calculate High-Water Marks, Maximum Drawdown (MDD), and "Time Underwater".
*   **UI (`quant_ui`):** Create the red-shaded underwater plot highlighting the top 5 historic drawdowns.

**PR 6: Risk-Adjusted Ratios**
*   **Math (`quant_core`):** `compute_sharpe`, `compute_sortino`, `compute_calmar`, and `compute_information_ratio`.
*   **App Integration:** Surface these metrics in a "Risk KPIs" summary table.

---

## Epic 3: Backward Analytics - Benchmarking & Attribution
**PR 7: Alpha, Beta & Market Capture**
*   **Math (`quant_core`):** Regress portfolio returns against a benchmark (e.g., SPY) to extract Alpha and Beta. Calculate Up-Market and Down-Market Capture Ratios.
*   **UI (`quant_ui`):** Scatter plot of Portfolio Returns vs Benchmark Returns with the regression line.

**PR 8: Brinson-Fachler Attribution**
*   **Math (`quant_core`):** Matrix math to separate excess returns into *Asset Allocation* effects vs. *Security Selection* effects.
*   **UI (`quant_ui`):** Waterfall chart showing the attribution breakdown.

**PR 9: Cost & Friction Drag Engine**
*   **Math (`quant_core`):** Compute Portfolio Turnover Ratio and estimate tax/commission drag based on historical trade frequency.

---

## Epic 4: Forward Projections & Tail Risk
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
