# fts: Automated Trading System

An automated, event-driven trading system.

This repository contains the complete architectural skeleton for a modular trading bot, built with a focus on type-safety, extensibility, and clear data contracts.

---

## Core Features

* **Data Contracts**: All data flowing through the system is validated by Pydantic schemas (`schemas.py`), ensuring type safety and explicit "contracts" between modules.
* **Extensible by Design**: Abstract Base Classes (ABCs) are used for all core components (e.g., `BaseMarketDataProvider`, `BaseStrategy`, `BaseExecutionHandler`), allowing different services or strategies to be plugged in without changing the core logic.
* **Persistent & Observable**: The system is built to log all actions—from signals to orders and fills—into a SQL database using SQLAlchemy ORM models (`models.py`).
* **Modern Tooling**: The project is configured with `pre-commit`, `black`, `isort`, and `flake8` for high code quality and consistency.

---

## Modular Architecture

The system is broken down into six core modules:

1.  **Data Ingestion**: Connects to market APIs (e.g., order books, trades) and external data sources (e.g., news, sentiment).
2.  **Strategy Engine**: Receives data and generates trading recommendations (`TradeSignal`).
3.  **Risk & Position Management**: Manages the portfolio, enforces risk limits, and calculates order size using strategies like the Kelly Criterion.
4.  **Execution Engine**: Takes a sized, risk-checked order and handles the technical execution (e.g., signing transactions, placing orders via API).
5.  **Monitoring & Alerting**: Provides system-wide logging and sends critical alerts (`Alert`) to external services.
6.  **Backtesting Engine**: Simulates strategy performance against historical data.

---

## Core Data Flow

The Pydantic schemas create a logical data pipeline:

`IngestionEngineOutput` → **(Strategy)** → `TradeSignal` → **(Risk)** → `OrderRequest` → **(Execution)** → `ExecutionResult`

This `ExecutionResult` is then used to update the `PortfolioState`, which feeds back into the Risk module for the next decision.

---

## Technology Stack

* **Python 3.12+**
* **Pydantic**: For all data contracts and settings management.
* **SQLAlchemy**: For database session management and ORM models.
* **uv**: For dependency management (see `uv.lock`).