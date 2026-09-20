# Implementation Plan: Unified Quant Workspace Architecture (Phased)

> **Note:** This is the Master Plan for the Quant Workspace redesign. 
> For legacy context and component-specific designs, see:
> - [Portfolio Analyser v2 Architecture](./portfolio_analyser_v2_plan.md)
> - [FTS Agent Overview](./fts/AGENTS.md)
> - [FTS Quant Classes](./fts/QUANT_CLASS_OVERVIEW.md)
> - [FTS Data Flow](./fts/data_flow.md)


## Goal Description
To safely eliminate duplication and establish a Single Source of Truth (SSOT), we will split the architecture redesign into two distinct phases. 

* **Phase 1** focuses strictly on file movement and workspace configuration with **zero code changes**, ensuring Git tracks history properly and nothing breaks.
* **Phase 2** handles the actual extraction of the core logic into decoupled packages (`quant_core` and `quant_data`) and the additive modifications of the existing `fts` schemas.

---

## Phase 1: Pure Structural Migration (Zero Code Changes)
In this phase, we solely reorganize the repositories into a `uv` workspace. The `fts` application will continue to run exactly as it does today, using its internal imports.

### 1. Workspace Initialization
Initialize the root workspace.
#### [NEW] pyproject.toml (Root)
```toml
[project]
name = "quant-workspace"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[tool.uv.workspace]
members = ["apps/*", "packages/*"]
```

### 2. Isolate FTS (The Trading Bot)
Using `git mv`, we will move all contents of the current `fts` repository into an `apps/fts` directory.
#### [NEW] apps/fts/
Moves the following from root to `apps/fts/`:
- `src/`
- `tests/`
- `specs/`
- `pyproject.toml`
- `uv.lock`

*Verification:* Running `cd apps/fts && uv run pytest` should pass 100% since no imports were touched.

### 3. Scaffold Portfolio Analyser
#### [NEW] apps/portfolio_analyser/
Copy the empty skeleton files from `/home/alfred/github/portfolio_analyser/` into this directory. Give it a basic `pyproject.toml` to register it in the workspace.

**At the end of Phase 1, we commit to Git. This guarantees the file renames are cleanly tracked before we start ripping out code.**

---

## Phase 2: Extraction & Refactoring
With the workspace secured, we extract the reusable components out of `apps/fts` and into shared packages, giving strict precedence to the existing `fts` models.

### 1. Create `quant_core` (Layer 1: Pure Domain)
This package has zero dependencies on databases or network APIs.

#### [NEW] packages/quant_core/pyproject.toml
Dependencies: `pydantic`, `pandas`, `numpy`, `scipy`.

#### [NEW] packages/quant_core/src/quant_core/models.py
Move schemas from `apps/fts/src/trading_bot/core/schemas.py`. Apply **Additive Modifications** to support `portfolio_analyser` without breaking `fts`:
```python
class MarketDetails(BaseModel):
    market_id: str
    name: str
    end_date: Optional[datetime] = None  # Relaxed for TradFi
    resolution_source: Optional[str] = None
    # Additive fields
    asset_type: Optional[str] = None
    geography: Optional[str] = None
    currency: Optional[str] = "USD"

class PortfolioState(BaseModel):
    # Existing fts fields
    total_balance_quote: float 
    available_balance_quote: float
    positions: List[Position]
    open_orders: List[OrderRequest]
    # Additive field for analyser
    cash_balances: Dict[str, float] = Field(default_factory=dict)
```

### 2. Create `quant_data` (Layer 2: Infrastructure)
This package handles SQLAlchemy and API fetching, implementing interfaces from `quant_core`.

#### [NEW] packages/quant_data/pyproject.toml
Dependencies: `quant_core`, `sqlalchemy`, `psycopg`, `yfinance`.

#### [NEW] packages/quant_data/src/quant_data/db/
- `database.py`: Moved from `apps/fts/.../database.py`.
- `models_sql.py`: Extract the SQLAlchemy definitions from `apps/fts/.../models.py` (e.g., `MarketSQL`, `BarDataLogSQL`).

#### [NEW] packages/quant_data/src/quant_data/providers/
- `yfinance_provider.py` (Moved from `fts/plugins`).
- `local_provider.py` (Scaffolded for Argentine CSVs).

### 3. Refactor FTS and Analyser to use Packages
Update the applications to consume the newly extracted packages.

#### [MODIFY] apps/fts/pyproject.toml & apps/portfolio_analyser/pyproject.toml
Link the workspace packages:
```toml
[tool.uv.sources]
quant-core = { workspace = true }
quant-data = { workspace = true }
```

#### [MODIFY] apps/fts/src/trading_bot/**/*.py
Update all internal imports. For example, replace:
```python
from trading_bot.core.schemas import MarketDetails
```
with:
```python
from quant_core.models import MarketDetails
```
Refactor logic that previously passed SQLAlchemy objects directly into the strategy engine to strictly cast to the pure `quant_core` Pydantic models first.

## Verification Plan

### Phase 1 Verification
- Run `uv sync` at the workspace root.
- Run `uv run pytest apps/fts/tests/`. Ensure 100% pass rate before starting Phase 2.

### Phase 2 Verification
- Strict type checking: `uv run mypy packages/quant_core packages/quant_data apps/fts --strict` to ensure `quant_core` is entirely decoupled from SQLAlchemy.
- Run an existing `fts` backtest spec to ensure the additive schema modifications did not alter the mathematical outcomes of the bot.
