# Portfolio Analyser

`portfolio_analyser` is an institutional-grade quantitative portfolio analysis, backtesting, and client performance review engine.

It provides:
- **Exploratory Portfolio Backtesting**: Compound CAGR, periodic & deviation corridor rebalancing with turnover friction, high-water mark drawdowns, and rolling correlation analytics.
- **Client Performance Accounting**: Ingestion from Interactive Brokers (IBKR), Inviu, and InvertirOnline (IOL), computing both forward-looking **Snapshots** (allocation, Pydantic policy compliance audits) and backward-looking **Movies** (Time-Weighted Return vs Money-Weighted Return / XIRR).
- **Multi-Currency Normalization**: Support for USD, ARS, and financial exchange rates (MEP and CCL).
- **Persistent Data Layer**: PostgreSQL caching (inspectable in pgAdmin) with tiered market data providers (Yahoo Finance + local Argentine price feeds) and in-memory SQLite support for automated unit testing.

## Master Architecture & Staged Implementation Plan

For the full system design, decoupled seams, rubberducked edge case resolutions, and prompt-engineered execution stages for Gemini LLMs, see:

👉 **[ARCHITECTURE_AND_PLAN.md](file:///home/alfred/github/portfolio_analyser/ARCHITECTURE_AND_PLAN.md)**

## Architectural Seams

```
portfolio_analyser/
├── domain/            # Pure mathematical core (Models, Rebalancing ABC, Calculators, Rule Engine)
├── data_layer/        # Persistence (Postgres/SQLite), Tiered Market Providers, Financial FX
├── brokers/           # Broker Ingestion (IBKR, Inviu, IOL) & Client Accounting (TWR/MWR)
├── visualization/     # PlotFactory (Matplotlib/Seaborn) & ReportExporter (HTML/PDF)
├── config/            # Pydantic Settings & Themes
└── main.py            # Click CLI entry point
```
