# Master Architecture & Staged Implementation Plan (Revised - Two Stages)

## Goal Description
Following the "Best code is no code" and SOLID/DRY principles, the original plan for `portfolio_analyser` has been structurally transformed into a two-stage process. 

The strategy ensures we do not reinvent the wheel:
1. **Non-UI Additive Work**: We will expand `quant_core` and `quant_data` with domain models, math, risk rules, and broker ingestion without breaking existing code. `portfolio_analyser` becomes a thin orchestration CLI wrapper.
2. **UI Abstraction & Recycling**: We discovered rich interactive Plotly and IPyWidget visualization code inside `apps/fts/src/trading_bot/backtesting/` (`visualizer.py`, `sweep_visualizer.py`). Instead of building a parallel Matplotlib/Seaborn stack, we will extract these Plotly functions into a new shared `quant_ui` package. We will decouple them from database logic so both `fts` and `portfolio_analyser` can share the exact same UI components. 

## User Review Required
> [!IMPORTANT]  
> By recycling `apps/fts` Plotly visualizations into `quant_ui`, we transition `portfolio_analyser` from a static PDF generator (using Matplotlib) to an interactive, HTML-first reporting system (using Plotly).
> Existing `fts` visualizer classes will be modified to import pure UI functions from `quant_ui`, keeping the `fts` tests and notebooks green.

---

## Stage 1: Non-UI Work (Purely Additive & Shared Logic)
Expand the shared libraries with portfolio management and broker ingestion capabilities. This is purely additive and will not break existing `fts` infrastructure.

### Component 1: `packages/quant_core` (Domain Math & Rules)
#### [MODIFY] `packages/quant_core/src/quant_core/enums.py`
Add missing enums `Currency`, `RateType` (MEP/CCL/OFFICIAL), and `CorporateActionType`.

#### [MODIFY] `packages/quant_core/src/quant_core/models.py`
Add `FXRate` and `CorporateAction` domain models.

#### [NEW] `packages/quant_core/src/quant_core/rebalance.py`
Implement `RebalanceStrategy(ABC)`, `PeriodicRebalanceStrategy`, and `DeviationRebalanceStrategy`.

#### [NEW] `packages/quant_core/src/quant_core/calculator.py`
Implement quant math (CAGR, TWR, MWR/XIRR, Sharpe, Drawdowns).

#### [NEW] `packages/quant_core/src/quant_core/risk.py`
Implement the declarative Policy/Rules Engine (`PortfolioRule`, `MaxAssetWeight`, etc.).

---

### Component 2: `packages/quant_data` (Data & Broker Ingestion)
#### [MODIFY] `packages/quant_data/src/quant_data/db/models.py`
Add SQLAlchemy schemas `FXRateRecord` and `CorporateActionRecord`.

#### [MODIFY] `packages/quant_data/src/quant_data/db/repositories.py`
Add methods to query FX rates and Corporate Actions.

#### [NEW] `packages/quant_data/src/quant_data/providers/brokers/`
Create broker integrations (API & CSV parsing) returning standard `quant_core.models.Portfolio`.
- `base.py` (`BrokerConnector(ABC)`)
- `ibkr.py`
- `inviu.py`
- `iol.py`

---

### Component 3: `apps/portfolio_analyser` (The Thin Orchestrator)
This application will house no core logic, only CLI routing.

#### [MODIFY] `apps/portfolio_analyser/pyproject.toml`
Depends on `quant_core`, `quant_data`, `quant_ui` (created in Stage 2), and `click`.

#### [NEW] `apps/portfolio_analyser/main.py`
Implements the CLI commands (`analyze`, `client_review`) wiring `quant_data` brokers to `quant_core` math.

---

## Stage 2: UI Work (Abstraction & Recycling)
Extract existing `fts` Plotly logic into a generic presentation package.

### Component 4: `packages/quant_ui` (New Shared Presentation Layer)
#### [NEW] `packages/quant_ui/pyproject.toml`
Dependencies: `plotly`, `pandas`, `ipywidgets`.

#### [NEW] `packages/quant_ui/src/quant_ui/plots/performance.py`
Abstracted function from `fts/visualizer.py`:
```python
def plot_candlestick_with_trades(df_ohlc: pd.DataFrame, df_trades: pd.DataFrame) -> go.Figure:
    # Pure UI function: takes Pandas dataframes, returns Plotly Figure. No DB queries.
    pass

def plot_cumulative_pnl(df_returns: pd.DataFrame) -> go.Figure:
    pass
```

#### [NEW] `packages/quant_ui/src/quant_ui/plots/sweeps.py`
Abstracted functions from `fts/sweep_visualizer.py`:
```python
def plot_parameter_sensitivity(trials_df: pd.DataFrame, sweep_param: str) -> go.Figure:
    pass
```

---

### Component 5: Refactoring `apps/fts` Visualizers
#### [MODIFY] `apps/fts/src/trading_bot/backtesting/visualizer.py`
Remove the hardcoded Plotly `go.Figure` generation. Modify the `BacktestVisualizer` class to query the DB as it currently does, but delegate charting to `quant_ui`.
```python
from quant_ui.plots.performance import plot_candlestick_with_trades

class BacktestVisualizer:
    def render_charts(self, df: pd.DataFrame, instrument_id: str) -> go.Figure:
        df_pnl = self.calculate_pnl(df)
        return plot_candlestick_with_trades(df_pnl, df_trades=df_pnl.dropna(subset=["side"]))
```

#### [MODIFY] `apps/fts/src/trading_bot/backtesting/sweep_visualizer.py`
Similarly refactor `SweepVisualizer` to delegate to `quant_ui.plots.sweeps`.

---

## Verification Plan

### Automated Tests
1. **Stage 1 (Core)**: Run `pytest packages/quant_core/` and `pytest packages/quant_data/` to verify rebalancing, math, and broker parsing.
2. **Stage 2 (UI Recycling)**: Run `pytest apps/fts/tests/unit/test_backtest_visualizer.py` to guarantee the extraction to `quant_ui` did not break existing `fts` notebook workflows. All `fts` tests must remain green.
3. **Type Safety**: `mypy --strict packages/quant_core packages/quant_ui`.

### Manual Verification
1. Open existing `fts` Jupyter notebooks and trigger `show_dashboard()` to verify the IPyWidgets and Plotly charts still render perfectly.
2. Run `portfolio_analyser client_review` from the CLI, which will generate an HTML dashboard utilizing the newly shared `quant_ui` Plotly components.
