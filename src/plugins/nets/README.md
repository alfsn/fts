# Nets Plugin (Forecasting & ML)

The `nets` plugin provides a modular framework for implementing machine learning based forecasting strategies within the FTS (Financial Trading System) core. It specifically addresses the complexities of the Argentinian market (CCL conversion) and implements S.O.L.I.D. and DRY principles for ML inference and training.

## 1. Pending Implementation (To-Do)

While the architectural foundation is complete, the following data and logic components are pending:

*   **Real Data Ingestion**: The plugin currently relies on placeholders for market data. Integration with a live Argentinian stock provider (e.g., BYMA/Rofex API) via `BaseMarketDataProvider` is required.
*   **Historical Data Population**: Scripts to fetch and store historical ticks/trades into `BarDataLog` (Dollar Bars) to enable model training.
*   **Concrete Model Artifacts**: Transition from "Mock Models" to actual serialized model files (`.json` for XGBoost, `.pt` for PyTorch/CNN).
*   **Offline Training Pipeline**: A dedicated script to process `BarDataLog` data, generate features via `LogReturnTransform`, and save trained models.
*   **Real inference in NetsStrategy**.

## 2. Current v0 Status

The current implementation is a functional "v0" prototype:

*   **NetsStrategy**: Uses a mock inference logic (extrapolating from the last return).
*   **DynamicFlat**: Implements a simplified ATR-based neutral zone calculation. It currently uses a rolling average of (High-Low) as an approximation.
*   **CCLProvider**: Implements a midpoint approach for `GGAL` (Local vs. ADR) with a fixed 10:1 ratio.
*   **ModelTrainer**: Interfaces are defined as placeholders (`XGBoostTrainer`, `CNNTrainer`) but do not yet contain training algorithms.
*   **ConfidenceSizer**: Performs linear scaling of position sizes based on a 0.0-1.0 confidence score.

## 3. Usage & Implementation

To use the `nets` plugin, you must define a **Task** in a YAML configuration file. The `PluginLoader` will handle the instantiation of all components.

### Example Configuration (`nets_task.yaml`)

```yaml
name: "nets_forecasting_ggal"
market_ids: ["GGAL_ARS"]
extra_models: ["plugins.nets.db_models"] # If plugin defines extra tables

market_provider:
  class_path: "trading_bot.data_ingestion.providers.MockProvider" # Replace with real provider
  params: {}

strategies:
  - class_path: "nets.strategies.nets_strategy.NetsStrategy"
    params:
      lookback_period: 20
      name_suffix: "v1"
      transform:
        class_path: "trading_bot.core.transforms.LogReturnTransform"
      flat_bucket:
        class_path: "nets.flat_buckets.DynamicFlat"
        params:
          k: 0.5
          period: 10
      model: "path/to/model.json"

sizing_strategy:
  class_path: "nets.sizing.confidence_sizer.ConfidenceSizer"
  params:
    base_amount_quote: 1000.0
```

### Running a Backtest with the Plugin

```python
from trading_bot.config import TaskConfig, PluginLoader
from trading_bot.backtesting.simulator import BacktestSimulator
import yaml

# 1. Load Config
with open("nets_task.yaml") as f:
    config = TaskConfig(**yaml.safe_load(f))

# 2. Instantiate Components via PluginLoader
market_provider = PluginLoader.instantiate(config.market_provider)
strategies = [PluginLoader.instantiate(s) for s in config.strategies]
# ... (instantiate other components)

# 3. Run Simulator
simulator = BacktestSimulator(db=db, strategy_engine=engine, ...)
simulator.run()
```

## 4. Design Philosophy

*   **Decoupled Transforms**: Pre-processing (Log-returns) lives in the Core to be shared across any ML plugin.
*   **Interface Segregation**: The plugin only knows about `BaseStrategy` and `BaseSizingStrategy` abstractions from the Core.
*   **Market Isolation**: All Argentinian-specific logic (CCL) is encapsulated within the plugin's `CCLProvider`.
