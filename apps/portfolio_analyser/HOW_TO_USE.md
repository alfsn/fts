# 💼 Portfolio Analyser: How To Use

The **Portfolio Analyser** is a standalone CLI tool for extracting your broker data, calculating your overall portfolio value (in a standardized base currency), and generating beautiful client-facing review dashboards.

> [!IMPORTANT]
> Always run these commands from the `portfolio_analyser` app folder!
> ```bash
> cd /home/alfred/github/fts/apps/portfolio_analyser
> ```

---

## 🚀 Running the Commands

### 1. Simple Value Analysis
If you just want to ping your broker and calculate your total portfolio value:
```bash
uv run python main.py analyze --broker yaml --file portfolio.yaml
```
*(You can pass `--broker ibkr`, `--broker inviu`, `--broker iol` or `--broker yaml`; the first three are still mocks.)*

Positions are valued at the **latest market close** on which every holding and the CCL rate have data (the date is printed). A `current_price` in the YAML is ignored. Market data is cached in `portfolio_analyser.db` (`--db-url` to change it).

### 2. Generate a Client Dashboard
The review is built from a **transactions ledger**, replayed day by day and valued at market, so it needs a `transactions:` list in the YAML (see below):
```bash
uv run python main.py client-review --broker yaml --file portfolio.yaml
```
This will generate a file named `client_review_yaml.html` in your current folder. Just double-click the file to open it in your browser!

Before anything is drawn the ledger is **reconciled** with the snapshot on the valuation date: a share-count difference stops the run (a trade or a split is missing), a cash difference is printed as a warning.

### 3. The transactions ledger
```yaml
transactions:
  - {date: 2024-01-02, type: transfer_in, currency: USD, instrument_id: AAPL, quantity: 5}   # held before the ledger starts
  - {date: 2024-01-02, type: deposit,     currency: USD, amount: 10000}
  - {date: 2024-01-03, type: buy,         currency: USD, instrument_id: AAPL, quantity: 10, price: 185.6, fees: 1.0}
  - {date: 2024-02-16, type: dividend,    currency: USD, instrument_id: AAPL, amount: 3.60}  # pay date
  - {date: 2024-03-01, type: deposit,     currency: ARS, amount: 1500000}
  - {date: 2024-03-04, type: buy,         currency: ARS, instrument_id: YPF, provider_symbol: YPFD.BA, quantity: 100, price: 23500}
  - {date: 2024-06-28, type: fee,         currency: ARS, amount: 12000, note: custody Q2}
```
- Every number is positive; `type` gives the direction (`buy`, `sell`, `transfer_in`, `transfer_out`, `deposit`, `withdrawal`, `dividend`, `fee`).
- `date` is the local trade date (the pay date for a dividend). Prices are as traded, per share.
- Don't type splits: they come from the cached corporate actions. Bonds and coupons are not supported yet.
- Only deposits, withdrawals and transfers are external flows; dividends and fees are part of the return.

### 4. Benchmark, alpha and beta
`client-review` compares the portfolio with a **total-return** benchmark (dividends reinvested) in the portfolio's base currency, on the days both have a price:
```bash
uv run python main.py client-review --broker yaml --file portfolio.yaml \
    --benchmark SPY --risk-free-rate 0.04 --dividend-withholding 0.30
```
- `--benchmark` (USD listing; default: the YAML's `benchmark:` block, else `SPY`). A non-USD benchmark goes in the YAML: `benchmark: {symbol: "^MERV", currency: ARS}`.
- `--risk-free-rate`: annual fraction. Without it, the daily 13-week US T-bill (`^IRX`) is used, for Sharpe and Sortino too. A portfolio whose base currency isn't USD must pass it.
- `--dividend-withholding`: tax withheld on the benchmark's dividends (0.30 for a non-US holder of SPY). Default 0, labelled "gross dividends".

How to read it:
- **Alpha** is arithmetic and annualised (daily intercept x periods per year), with a Newey-West (HAC) standard error and 95% interval. **R²** says how much the benchmark explains: with a low R², alpha is mostly just the portfolio's own excess return.
- **Dimson beta** adds the response to the previous day's benchmark move. When it's far above beta, prices aren't synchronous (BYMA vs NYSE closes, holidays) and the daily beta understates exposure.
- Below **252 daily observations** every benchmark statistic shows n/a; the observation count is always shown. That's a year of stock trading days, but only about 8 months for crypto.
- Cash in the portfolio lowers beta: that's real exposure, not an error.
- The benchmark reinvests dividends on the ex-date, while the ledger books them on the pay date, so tracking error is slightly noisier around dividends.

### 5. Expected volatility (EWMA forecast)
`client-review` also forecasts how much the portfolio will move tomorrow and checks that forecast against what really happened. There are no new flags.
- **The forecast** is RiskMetrics EWMA on the daily returns of the flow-adjusted wealth index (deposits and withdrawals aren't volatility). Yesterday's move counts the most; each older day counts 6% less (decay λ = 0.94). It is shown annualised ("next-day volatility, ann."), using the portfolio's own trading calendar (about 252 days a year for stocks, 365 for crypto). It starts after 30 daily returns.
- **The chart** draws each day's forecast next to the volatility actually realised over the **following month**, and a lazy baseline: the average squared daily return of the **last year**. All three lines assume an average daily return of 0, like EWMA, so they measure the same thing.
- **The table** scores both forecasts on the same out-of-sample days (each forecast only uses data up to the day before):
  - **QLIKE**: lower is better. Only the comparison between the two columns means something, not the level.
  - **Mincer-Zarnowitz slope** (regression of the realised squared return on the forecast, Newey-West SE): 1 is ideal. Below 1 the forecast over-reacts, and near 0 it carries no information.
  - **R²** is low by construction: one day's squared return is a very noisy measure of that day's variance. Don't judge the forecast by it.
  - **RMS of standardized returns** (root mean square of return / forecast vol, taken about zero like the forecast): about 1 if the forecast is calibrated. Above 1, it understates risk (or the returns have fat tails).
- Below **252 common out-of-sample days** (about 8 months of crypto) the scores show n/a; the day count is always shown. Days with a zero forecast (a flat stretch, e.g. all cash) are left out of the count. The baseline needs a year to start, so the scores need about two years of history.
- Limits: λ = 0.94 is tuned on developed markets, not re-estimated for ARS assets. EWMA has no mean reversion, so after a shock the annualised number overstates a true 1-year volatility. It describes the portfolio as it was held, not today's weights. Stale BYMA prices make it look calmer.

---

## 🛠️ How to Add a Portfolio (Without Real Broker Data)

The `ibkr`, `inviu` and `iol` connectors are still mocks. To analyse a real portfolio, write it as a YAML file and pass `--broker yaml --file <path>`; no code changes are needed. `portfolio.yaml` in this folder is a minimal example:

```yaml
base_currency: USD
positions:
  - {instrument_id: AAPL, currency: USD, quantity: 10, cost_basis: 150.0}
  - {instrument_id: YPF, provider_symbol: YPFD.BA, currency: ARS, quantity: 100, cost_basis: 12000.0}
cash_balances:
  - {currency: USD, total: 5000.0, available: 5000.0}
  - {currency: ARS, total: 250000.0, available: 250000.0}
# optional
transactions: []                      # the ledger (section 3); client-review needs it
benchmark: {symbol: SPY, currency: USD}
```
- `instrument_id` is the broker's name for the holding; `provider_symbol` is the market-data symbol when it differs (YPF in ARS trades as `YPFD.BA`).
- `current_price` may be present but is ignored: positions are valued at the cached market close (section 1). FX comes from the market too (the CCL rate for ARS), never typed by hand.
- `base_currency` is the reporting currency (`USD` or `ARS`).
