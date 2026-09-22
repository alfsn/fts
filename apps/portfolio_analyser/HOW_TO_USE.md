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
uv run portfolio-analyser analyze --broker ibkr
```
*(You can pass `--broker ibkr`, `--broker inviu`, or `--broker iol`)*

### 2. Generate a Client Dashboard
To generate the "Bloomberg-style" HTML review dashboard:
```bash
uv run portfolio-analyser client-review --broker ibkr
```
This will generate a file named `client_review_ibkr.html` in your current folder. Just double-click the file to open it in your browser!

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
