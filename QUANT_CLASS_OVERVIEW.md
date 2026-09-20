# Quantitative Architecture & Business Class Overview
*Mapping and domain logic for [`02_strategy_hparam_training.ipynb`](file:///home/alfred/github/fts/notebooks/02_strategy_hparam_training.ipynb), [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb), and [`04_parameter_sweep_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/04_parameter_sweep_backtest.ipynb)*

---

## 1. System High-Level Workflows

```mermaid
graph TD
    subgraph Workflow 1: Model Optimization & Selection (02.ipynb)
        A[Market Data Storage] -->|MarketDataRepository| B[DatasetBuilder]
        B -->|Feature Matrices| C[Optuna Search Engine]
        C -->|HParamStudySpec| D[NN Training & Evaluation]
        D -->|Best Model Weights & Config| E[ModelRepository Registry]
    end

    subgraph Workflow 2: Event-Driven Single Backtest (03.ipynb)
        F[Historical Bar Replay] -->|HistoricalReplayLoop| G[TradingPipeline / Transforms]
        G -->|Engineered Features| H[ONNXPredictor & Classifier]
        H -->|ML Predictions & Signals| I[NetsStrategy / StrategyEngine]
        I -->|Target Positions| J[Portfolio & RiskManager]
        J -->|Validated Orders| K[ExecutionEngine & Slippage Handler]
        K -->|Fills & Cash Updates| F
        K -->|Logs & Metric Metrics| L[BacktestVisualizer & Exporter]
    end

    subgraph Workflow 3: Multi-Run Parameter Sweep (04.ipynb)
        M[SweepSpec Config Space] -->|Parameter Grid| N[Parallel Backtest Runner]
        N -->|Run Iterations| Workflow2
        N -->|Aggregated SweepResult| O[SweepVisualizer & Heatmaps]
    end
```

---

## 2. Quant Domain 1: Data Management & Feature Engineering

### [MarketDataRepository](file:///home/alfred/github/fts/src/trading_bot/core/repository.py)
* **Used in:** [`02_strategy_hparam_training.ipynb`](file:///home/alfred/github/fts/notebooks/02_strategy_hparam_training.ipynb), [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** Serves as the single source of truth for historical market data retrieval. It abstracts raw database queries to provide clean, windowed OHLCV (Open, High, Low, Close, Volume) time-series data for training and simulation while guaranteeing point-in-time consistency.

### [DatasetBuilder](file:///home/alfred/github/fts/src/trading_bot/core/dataset.py)
* **Used in:** [`02_strategy_hparam_training.ipynb`](file:///home/alfred/github/fts/notebooks/02_strategy_hparam_training.ipynb)
* **Quant Business Logic:** Converts raw historical market bars into tabular or sequential machine learning feature datasets (X) and forward return targets (y). Handles windowing, sequence creation, feature transformation pipelines, and splitting train/validation splits without look-ahead leakage.

### [BaseTransform](file:///home/alfred/github/fts/src/trading_bot/core/transforms.py) & [LogReturnTransform](file:///home/alfred/github/fts/src/trading_bot/core/transforms.py)
* **Used in:** [`02_strategy_hparam_training.ipynb`](file:///home/alfred/github/fts/notebooks/02_strategy_hparam_training.ipynb), [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** Standardizes price time series into stationary financial inputs. By computing log returns and normalized feature representations, it removes non-stationary price drifts so deep learning models can generalize across varying volatility regimes.

### [SQLBacktestDataReader](file:///home/alfred/github/fts/src/trading_bot/backtesting/readers.py)
* **Used in:** [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** A high-throughput data reader designed to stream historical price bars from database storage into the replay engine bar-by-bar, ensuring memory efficiency during large-scale backtesting.

---

## 3. Quant Domain 2: Model Training, Selection & Inference

### [HParamStudySpec](file:///home/alfred/github/fts/src/plugins/nets/spec.py) & [NNTrainingConfig](file:///home/alfred/github/fts/src/plugins/nets/models.py)
* **Used in:** [`02_strategy_hparam_training.ipynb`](file:///home/alfred/github/fts/notebooks/02_strategy_hparam_training.ipynb)
* **Quant Business Logic:** Defines the hyperparameter search space (e.g., neural net depth, dropout rates, learning rates, lookback horizons) and concrete training execution configs for automated Optuna optimization studies.

### [ModelRepository](file:///home/alfred/github/fts/src/trading_bot/core/repository.py)
* **Used in:** [`02_strategy_hparam_training.ipynb`](file:///home/alfred/github/fts/notebooks/02_strategy_hparam_training.ipynb), [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** Model asset registry and catalog. Stores trained model weights (ONNX format), input schema definitions, feature specs, and out-of-sample score performance logs for versioning and seamless backtest loading.

### [ONNXPredictor](file:///home/alfred/github/fts/src/plugins/nets/inference/inference.py)
* **Used in:** [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** High-performance, low-latency ML inference wrapper. Loads compiled ONNX models to produce raw model outputs (e.g., directional return probability distributions) on each new bar during live simulation.

### [DynamicThresholdClassifier](file:///home/alfred/github/fts/src/plugins/nets/output_selectors/output_selectors.py)
* **Used in:** [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** Quant decision rule layer that translates raw model probabilities into discrete directional predictions (Long / Short / Flat). Uses adaptive probability thresholds (e.g., confidence quantiles) to filter out weak noise signals.

---

## 4. Quant Domain 3: Signal Generation & Strategy Engine

### [TradingPipeline](file:///home/alfred/github/fts/src/trading_bot/core/pipeline.py)
* **Used in:** [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** Stateful real-time feature orchestrator. Maintains rolling lookback buffers of market data during bar-by-bar execution and applies feature transformations dynamically prior to inference.

### [NetsStrategy](file:///home/alfred/github/fts/src/plugins/nets/strategies/nets_strategy.py)
* **Used in:** [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** Specialized quantitative trading strategy class. Integrates the feature transformation pipeline, ONNX inference predictor, and dynamic signal selector to evaluate market conditions and output actionable alpha signals.

### [StrategyEngine](file:///home/alfred/github/fts/src/trading_bot/strategy/engine.py)
* **Used in:** [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** Master strategy coordinator. Manages multiple underlying strategy instances, converts raw alpha signals into target portfolio position allocations, and dispatches target requests to risk management.

---

## 5. Quant Domain 4: Portfolio, Sizing & Risk Management

### [Portfolio](file:///home/alfred/github/fts/src/trading_bot/risk_management/portfolio.py)
* **Used in:** [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** Keeps track of real-time account state: available cash balance, gross/net position exposure, unrealized/realized PnL, and current held contract quantities.

### [FixedPercentageSizer](file:///home/alfred/github/fts/src/trading_bot/risk_management/sizing/fixed_percentage.py)
* **Used in:** [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** Position sizing algorithm determining order contract counts based on a fixed percentage of current portfolio equity, enforcing controlled position allocation.

### [RiskManager](file:///home/alfred/github/fts/src/trading_bot/risk_management/manager.py)
* **Used in:** [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** Risk control gateway. Evaluates proposed strategy trades against account constraints (max position size, drawdown limits, leverage limits) and adjusts or blocks orders that violate portfolio risk guidelines.

---

## 6. Quant Domain 5: Execution Simulation & Market Impact

### [KBarExecuteDelay](file:///home/alfred/github/fts/src/trading_bot/execution/delay.py)
* **Used in:** [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** Simulates execution latency. Delays order fill matching by $K$ bars (e.g., signal generated at Bar $N$ close is filled at Bar $N+1$ open), preventing look-ahead bias and modeling realistic execution lag.

### [FlatPriceSlip](file:///home/alfred/github/fts/src/trading_bot/execution/slippage.py)
* **Used in:** [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** Slippage model simulating market impact and bid-ask spread costs by applying fixed or ratio-based price adjustments to order fill prices.

### [SimulatedExecutionHandler](file:///home/alfred/github/fts/src/trading_bot/execution/handlers/simulated_handler.py) & [ExecutionEngine](file:///home/alfred/github/fts/src/trading_bot/execution/engine.py)
* **Used in:** [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** Virtual exchange matching engine. Receives validated orders, applies delay and slippage rules, calculates transaction fees/commissions, updates position records, and returns execution fill notifications.

---

## 7. Quant Domain 6: Backtesting Engine, Audit Logging & Parameter Sweeps

### [HistoricalReplayLoop](file:///home/alfred/github/fts/src/trading_bot/core/loop.py)
* **Used in:** [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** Event loop dispatcher. Drives time forward tick-by-tick or bar-by-bar during backtests, synchronizing data ingestion, feature generation, strategy evaluation, and order matching in strict chronological order.

### [DatabasePredictionLogger](file:///home/alfred/github/fts/src/trading_bot/monitoring/prediction_logger.py)
* **Used in:** [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** Quantitative audit system. Records raw model prediction scores, threshold classifications, generated signals, and resulting trade executions into a persistent database table for diagnostic post-hoc analysis.

### [BacktestSpec](file:///home/alfred/github/fts/src/trading_bot/backtesting/spec.py)
* **Used in:** [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** Complete declarative configuration for a single backtest run (start/end dates, initial account equity, strategy selection, risk parameters, execution cost assumptions).

### [BacktestEngine](file:///home/alfred/github/fts/src/trading_bot/backtesting/engine.py)
* **Used in:** [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** Orchestration facade for single backtest execution. Assembles the replay loop, pipeline, strategy engine, risk manager, and execution handler, and calculates overall strategy statistics (Sharpe Ratio, Max Drawdown, Win Rate, CAGR).

### [BacktestVisualizer](file:///home/alfred/github/fts/src/trading_bot/backtesting/visualizer.py) & [HTMLBacktestExporter](file:///home/alfred/github/fts/src/trading_bot/backtesting/exporter.py)
* **Used in:** [`03_custom_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/03_custom_backtest.ipynb)
* **Quant Business Logic:** Reporting and analytics suite. Renders interactive equity curves, underwater drawdown charts, trade entry/exit markers, and exports complete self-contained HTML performance report dashboards.

### [SweepSpec](file:///home/alfred/github/fts/src/plugins/nets/spec.py) & [SweepResult](file:///home/alfred/github/fts/src/trading_bot/backtesting/sweep_results.py)
* **Used in:** [`04_parameter_sweep_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/04_parameter_sweep_backtest.ipynb)
* **Quant Business Logic:** Defines parameter grid spaces for sensitivity testing across multiple strategy/risk configurations, and captures aggregated performance matrix results across all backtest runs.

### [SweepVisualizer](file:///home/alfred/github/fts/src/trading_bot/backtesting/sweep_visualizer.py) & [HTMLSweepExporter](file:///home/alfred/github/fts/src/trading_bot/backtesting/sweep_exporter.py)
* **Used in:** [`04_parameter_sweep_backtest.ipynb`](file:///home/alfred/github/fts/notebooks/04_parameter_sweep_backtest.ipynb)
* **Quant Business Logic:** Comparative analytics dashboard. Visualizes cross-run parameter sensitivity heatmaps, metric distribution plots (e.g. Sharpe vs. Drawdown tradeoffs), and exports summary reports for multi-parameter robustness evaluation.
