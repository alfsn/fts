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
- Below **252 daily observations** every benchmark statistic shows n/a; the observation count is always shown.
- Cash in the portfolio lowers beta: that's real exposure, not an error.
- The benchmark reinvests dividends on the ex-date, while the ledger books them on the pay date, so tracking error is slightly noisier around dividends.

---

## 🛠️ How to Add a Portfolio (Without Real Broker Data)

If you haven't actually implemented the real API connections inside the `quant_data` broker connectors yet, the CLI will probably crash or return empty data when trying to connect.

To test the tool and bypass the real broker connections, you can easily hardcode a **Mock Portfolio**. Here is how you manually construct a valid `Portfolio` object using the `quant_core` domain models:

```python
from datetime import datetime, timezone
from quant_core.enums import Currency
from quant_core.models import Portfolio, Position, CashBalance, FXContext

# 1. Create Mock Positions
my_positions = [
    Position(
        instrument_id="AAPL",
        currency=Currency.USD,
        quantity=10.0,
        cost_basis=150.0,
        current_price=175.50
    ),
    Position(
        instrument_id="YPF",
        currency=Currency.ARS,
        quantity=100.0,
        cost_basis=12000.0,
        current_price=15500.0
    )
]

# 2. Create Mock Cash Balances
my_cash = [
    CashBalance(currency=Currency.USD, total=5000.0, available=5000.0),
    CashBalance(currency=Currency.ARS, total=250000.0, available=250000.0)
]

# 3. Build the Portfolio
mock_portfolio = Portfolio(
    timestamp=datetime.now(timezone.utc),
    base_currency=Currency.USD,
    positions=my_positions,
    cash_balances=my_cash
)
```

### Injecting it into `main.py`
To use this mock portfolio to test the `analyze` command without touching the `quant_data` package, simply modify `apps/portfolio_analyser/main.py`:

```python
@cli.command()
@click.option("--broker", type=click.Choice(["ibkr", "inviu", "iol"]), required=True)
def analyze(broker):
    click.echo(f"Analyzing portfolio from {broker}...")

    # Override the real connector with the mock portfolio you built above!
    portfolio = mock_portfolio

    # Setup your FX rates to convert ARS and EUR to your base currency (USD)
    fx_context = FXContext(
        base_currency=portfolio.base_currency,
        rates={
            Currency.ARS: 0.001,  # Example: 1 ARS = 0.001 USD
            Currency.EUR: 1.1,
            Currency.USD: 1.0,
        },
    )

    total_value = portfolio.total_value(fx_context)
    click.echo(f"Total Portfolio Value ({portfolio.base_currency.value}): {total_value:.2f}")
```
