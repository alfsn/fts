# Portfolio Analyser

`portfolio_analyser` is an institutional-grade quantitative portfolio analysis, backtesting, and client performance review engine.

It provides:
- **Exploratory Portfolio Backtesting**: Compound CAGR, periodic & deviation corridor rebalancing with turnover friction, high-water mark drawdowns, and rolling correlation analytics.
- **Client Performance Accounting**: Ingestion from Interactive Brokers (IBKR), Inviu, and InvertirOnline (IOL), computing both forward-looking **Snapshots** (allocation, Pydantic policy compliance audits) and backward-looking **Movies** (Time-Weighted Return vs Money-Weighted Return / XIRR).
- **Multi-Currency Normalization**: Support for USD, ARS, and financial exchange rates (MEP and CCL).
- **Persistent Data Layer**: PostgreSQL caching (inspectable in pgAdmin) with tiered market data providers (Yahoo Finance + local Argentine price feeds) and in-memory SQLite support for automated unit testing.

Run it with `uv run portfolio-analyser --help` (usage: [HOW_TO_USE.md](HOW_TO_USE.md)).

## Layout

```
apps/portfolio_analyser/
├── src/portfolio_analyser/
│   ├── main.py          # Click CLI (`portfolio-analyser` script): analyze, client-review; prints only
│   ├── review.py        # Use cases (history, snapshot, review): the one pipeline the CLI and the workbench call
│   ├── data_layer/      # Market history and FX (CCL), benchmark, ledger replay and reconciliation
│   └── visualization/   # Maps the data to quant_core metrics and quant_ui figures; writes the HTML report
├── tests/
├── examples/demo_ledger.yaml  # Demo ledger: real tickers and prices, synthetic trades
└── portfolio.yaml       # Sample snapshot
```

KPI maths lives in `quant_core`, data access in `quant_data`, figures in `quant_ui`; this app only wires them.
