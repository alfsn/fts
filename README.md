# fts: Automated Trading System (Core & Plugins)

An event-driven, modular quantitative trading system built in Python. **This repository serves as the core system (`/fts/`) and plugin ecosystem.**

The system provides an end-to-end pipeline for data ingestion, machine learning / neural network model training, hyperparameter search, candidate model registry, position & risk management, order execution, monitoring, and historical backtesting with visual HTML reporting.

---

## Key Architectural Highlights

* **Data Contracts (Pydantic)**: All data flowing through the pipeline (market bars, signals, orders, fills) is strictly validated via Pydantic schemas, enforcing explicit data contracts between modules.
* **Modular Plugin Architecture (`uv` Workspace)**: Supports decoupled plugins for data providers (`ccxt`, `yfinance`, `argentina`) and ML model strategies (`nets` featuring LSTM, RNN, CNN, Linear Regression, and XGBoost).
* **Model Registry & Promotion Pipeline**: Trains models into ONNX format, logs performance metrics (validation loss, IC, directional accuracy) to a SQLite Model Registry database, and enables one-command production model promotion.
* **Persistent & Observable**: System actions, predictions, portfolio positions, and trade logs are persisted via SQLAlchemy ORM into SQLite.
* **Backtesting & Analytics**: Event-driven historical replay engine with configurable execution delay, slippage models, performance calculation (Sharpe, drawdown, PnL), and HTML report exporter.

---

## Modular System Architecture

The core architecture is organized into six primary engine modules located in `src/trading_bot/`:

1. **Data Ingestion** (`data_ingestion/`): Interface for real-time and historical market data providers, bar resampling, and provider factories.
2. **Strategy Engine** (`strategy/`): Signal generation engine supporting custom strategies and investable universe filtering.
3. **Risk & Position Management** (`risk_management/`): Manages cash balances, positions, portfolio states, and dynamic order sizing (Fixed amount, Fixed %, Kelly Criterion).
4. **Execution Engine** (`execution/`): Handles order routing, latency delay simulation, slippage modeling, and execution handlers (Polymarket CLOB API, Simulated handler).
5. **Monitoring & Alerting** (`monitoring/`): Configures logging, database prediction tracking, alerter hooks (e.g., Telegram), and terminal dashboard displays.
6. **Backtesting & Research Engine** (`backtesting/`): Historical replay loop runner, spec configuration parser, metrics generation, and interactive HTML report exporter.

### Core Data Flow Pipeline

```
Data Ingestion (Bars) → Strategy Engine (TradeSignal) → Risk Manager (OrderRequest) → Execution Engine (ExecutionResult) → Portfolio Update
```

---

## Plugin Ecosystem (`src/plugins/`)

The repository leverages a `uv` workspace structure with plugins defined under `src/plugins/`:

* **`nets`**: Neural networks plugin containing model architectures (LSTM, RNN, CNN, Linear Regression, XGBoost), dataset builders, inference engines, output selectors, confidence sizers, and Optuna hyperparameter search pipelines.
* **`ccxt`**: Data provider integration for CCXT exchange APIs.
* **`yfinance`**: Data provider integration for Yahoo Finance equity and asset history.
* **`argentina`**: Data provider plugin for Argentine financial market instruments.

---

## Quickstart & CLI Usage

### 1. Installation

Install all workspace packages and dependencies using `uv`:

```bash
uv sync --all-packages
```

### 2. Ingest Historical Market Data

Ingest historical bars into the database:

```bash
python -m trading_bot.utils.ingest_historical
```

### 3. Run Hyperparameter Search & Model Training

Execute an Optuna hyperparameter optimization trial based on a YAML spec:

```bash
python -m plugins.nets.training.hparam_search --spec specs/train/BTCUSDT/lstm_hparam_search.yaml
```

Candidate models will be exported as ONNX artifacts into `models/registry/trials/` and registered in the database model registry.

### 4. Promote Candidate Model to Production

Promote the best candidate model ID to production status:

```bash
python -m trading_bot.utils.promote <model_id>
```

### 5. Run a Trading Task or Backtest

Run the main event loop or historical backtest simulation using a task YAML config:

```bash
python -m trading_bot --config nets_task.yaml
```

Backtest summaries and visual HTML reports are exported to `runs/reports/`.

### 6. Run Test Suite

Run unit and integration test suites:

```bash
uv run pytest
```

---

## Project Structure

For a full annotated sitemap of the codebase, see [folder_structure.txt](file:///home/alfred/github/fts/folder_structure.txt).

```
fts/
├── src/
│   ├── plugins/         # Modular plugins (nets, ccxt, yfinance, argentina)
│   └── trading_bot/     # Core trading system engine
├── specs/               # YAML task & backtest specifications
├── models/              # Registered ONNX models and Optuna search trials
├── notebooks/           # Jupyter research notebooks
├── runs/                # TensorBoard logs and backtest HTML reports
└── tests/               # Unit and integration test suite
```