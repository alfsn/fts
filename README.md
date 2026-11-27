# Quant Workspace (`fts` + `portfolio_analyser`)

A `uv` monorepo workspace containing two quantitative finance applications sharing a common core domain, market-data layer, and Plotly visualization library.

---

## Workspace Structure

```text
fts/
├── packages/
│   ├── quant_core/              # Pure domain models, ledger replay, corporate-action & KPI math
│   ├── quant_data/              # SQLAlchemy market-data cache, TieredDataProvider, broker loaders
│   └── quant_ui/                # Shared Plotly chart builders and themes
└── apps/
    ├── fts/                     # Event-driven ML trading bot, ONNX registry, backtester & Streamlit UI
    │   ├── src/trading_bot/     # Core engine (ingestion, strategy, risk, execution, backtesting, dashboard)
    │   ├── src/plugins/         # Plugins (nets, ccxt, yfinance, argentina)
    │   ├── specs/               # YAML training, sweep & backtest specifications
    │   ├── models/              # Registered ONNX models and Optuna trial artifacts
    │   ├── notebooks/           # Jupyter research & backtest notebooks
    │   ├── runs/                # Optuna/TensorBoard logs and HTML backtest/sweep reports
    │   └── tests/               # Unit and integration tests
    └── portfolio_analyser/      # CLI for broker snapshots, ledger replay & HTML client reviews
        ├── src/portfolio_analyser/
        │   ├── main.py          # Click CLI (`portfolio-analyser` script)
        │   ├── data_layer/      # Market history, benchmark & ledger reconciliation binding
        │   └── visualization/   # Static HTML client-review report builder
        ├── tests/               # Analyser test suite
        └── HOW_TO_USE.md        # Portfolio Analyser usage guide
docs/
├── BACKLOG.md                   # Workspace roadmap: PR order and status (the only place for status)
├── plans/                       # One design per PR or epic: decisions, why and how
├── architecture/                # fts architecture notes and agent constitution
└── fts/                         # fts presentation and research notes
```

---

## Applications & Shared Packages

### Shared Packages (`packages/`)

* **`quant_core`**: Pure domain schemas (Pydantic), FIFO trade/ledger replay (`ledger.py`), split & total-return corporate-action adjustments, CCL FX extraction, returns/drawdown/ratio/benchmark math (`analytics/`), and unified KPI composition (`compute_performance_metrics`, `compute_benchmark_metrics`).
* **`quant_data`**: SQLAlchemy cache (`bar_data_logs`, `corporate_actions`, `fx_rates`), `TieredDataProvider` with rate-limiting throttle and cache-integrity checks, Yahoo Finance & CCXT adapters, and broker connectors (`LocalYamlConnector` + mock `IBKR`, `Inviu`, `IOL`).
* **`quant_ui`**: Reusable Plotly figures for performance curves, drawdowns, rolling volatility, benchmark regressions, and hyperparameter sweeps.

### `apps/fts` (Automated Trading & Research System)

An event-driven, modular quantitative trading and ML experimentation system:

```text
Data Ingestion (Bars) → Strategy Engine (TradeSignal) → Risk Manager (OrderRequest) → Execution Engine (ExecutionResult) → Portfolio Update
```

1. **Data Ingestion** (`trading_bot/data_ingestion/`): Market data provider registry and bar resampling.
2. **Strategy Engine** (`trading_bot/strategy/`): Signal generation and investable universe filtering.
3. **Risk & Position Management** (`trading_bot/risk_management/`): Single market-close portfolio equity tracking, allocation limits, and sizing (Fixed amount, Fixed %, Kelly Criterion).
4. **Execution Engine** (`trading_bot/execution/`): Order routing, bar-delay simulation, slippage models, and handlers (Simulated, Polymarket CLOB).
5. **Monitoring** (`trading_bot/monitoring/`): Structured logging and SQLite prediction tracking (`DatabasePredictionLogger`).
6. **Backtesting & Dashboard** (`trading_bot/backtesting/`, `trading_bot/dashboard/`): Historical replay loop, HTML report/sweep exporters, and a Streamlit explorer for the model registry and backtest runs.

Plugins in `apps/fts/src/plugins/`:
* **`nets`**: PyTorch/sklearn/XGBoost training (`LSTM`, `RNN`, `CNN`, `LinearRegression`, `XGBoost`), Optuna hyperparameter search, ONNX export & inference, and confidence sizing.
* **`ccxt`**, **`yfinance`**, **`argentina`**: Market data provider plugins.

### `apps/portfolio_analyser`

Standalone Click CLI (`uv run portfolio-analyser`, package `portfolio_analyser`) for valuing holdings at the latest market close, converting multi-currency books (USD/ARS at CCL), reconciling transaction ledgers against snapshots, and exporting interactive HTML client reviews (`client_review_yaml.html`). See [apps/portfolio_analyser/HOW_TO_USE.md](apps/portfolio_analyser/HOW_TO_USE.md) for full details.

---

## Quickstart & CLI Usage

### 1. Install Workspace Dependencies

Run from the repository root (`/home/alfred/github/fts`):

```bash
uv sync --all-extras --dev --all-packages
```

*(A plain `uv sync` omits dev tools like `pytest` and some workspace package extras.)*

### 2. Portfolio Analyser (`apps/portfolio_analyser`)

```bash
cd apps/portfolio_analyser

# Value a portfolio at the latest market close
uv run portfolio-analyser analyze --broker yaml --file portfolio.yaml

# Generate an interactive HTML client review (requires a transactions ledger in YAML)
uv run portfolio-analyser client-review --broker yaml --file portfolio.yaml
```

### 3. FTS Trading Bot (`apps/fts`)

All `fts` commands assume `apps/fts/` as the working directory:

```bash
cd apps/fts
```

1. **Ingest Historical Market Data** into `dev.db`:

   ```bash
   uv run python -m trading_bot.utils.ingest_historical -p yfinance -t AAPL -f 1d -d 1y
   ```

2. **Run Hyperparameter Search & Train Candidate Models** (exports ONNX files to `apps/fts/models/registry/trials/` and logs to `dev.db` & `apps/fts/runs/optuna/`):

   ```bash
   uv run python -m nets.training.hparam_search --spec specs/train/BTCUSDT/lstm_hparam_close.yaml
   ```

3. **Promote a Candidate Model to Production**:

   ```bash
   uv run python -m trading_bot.utils.promote <model_id>
   ```

4. **Run a Trading Task or Backtest** (exports HTML reports to `apps/fts/runs/reports/`):

   ```bash
   uv run python -m trading_bot --config nets_task.yaml
   ```

5. **Launch the Interactive Streamlit Dashboard** (browse DB counts, promote/archive models, compare backtest equity curves):

   ```bash
   uv run python -m trading_bot.dashboard.cli --db dev.db --port 8501
   ```

---

## Running the Test Suites

Because `apps/fts/tests` and `packages/quant_core/tests` both define a `tests` package, run the three suites separately:

```bash
uv run pytest packages
uv run pytest apps/portfolio_analyser/tests
cd apps/fts && uv run pytest tests
```
