# Master Architecture & Staged Implementation Plan: `portfolio_analyser` (v2 - Rubberducked)

## Executive Architecture & Natural Seams

To prevent scope bloat while supporting both exploratory quantitative backtesting and real-world multi-broker client performance reviews (Interactive Brokers, Inviu, InvertirOnline), the system is structured into **5 strictly decoupled architectural layers (seams)** adhering to Clean Architecture and SOLID principles.

```mermaid
flowchart TD
    subgraph Presentation ["Layer 4: Presentation & Interface"]
        CLI["Click CLI (main.py)"]
        Plots["PlotFactory (Matplotlib/Seaborn)"]
        Exporter["ReportExporter (HTML / PDF / PNG)"]
    end

    subgraph BrokerIntegration ["Layer 3: Broker Ingestion & Client Accounting"]
        BrokerABC["BrokerConnector (ABC)"]
        IBKR["IBKRAdapter (Flex / REST / CSV)"]
        Inviu["InviuAdapter (REST / CSV)"]
        IOL["IOLAdapter (OAuth2 / CSV)"]
        SnapMovie["ClientPortfolioService (Snapshot vs Movie)"]
        RulesEngine["ClientInformation & Declarative Rules"]
    end

    subgraph DomainCore ["Layer 1: Domain Core (Zero External Deps, Pure Math)"]
        Models["Domain Models (Asset, Position, CashFlow, Transaction, Metrics)"]
        RebalanceABC["RebalanceStrategy (ABC)"]
        TimeRebal["PeriodicRebalanceStrategy (Monthly/Quarterly/Annual)"]
        DevRebal["DeviationRebalanceStrategy (Tolerance Corridor)"]
        Calc["QuantCalculator (Compound CAGR, TWR, MWR/XIRR, Sharpe, Drawdown)"]
        Aligner["CalendarAligner (Union Calendar ffill)"]
    end

    subgraph DataStorage ["Layer 2: Market Data & Persistence Layer"]
        RepoABC["MarketDataRepository (ABC)"]
        SQLRepo["SQLAlchemyRepository (PostgreSQL / SQLite)"]
        Taxonomy["AssetTaxonomy & Metadata Store"]
        ProviderABC["MarketDataProvider (ABC)"]
        TieredData["TieredDataProvider (yfinance + Local Argentine CSV/Feeds)"]
        FXService["FXRateService (Financial MEP / CCL / USD)"]
    end

    CLI --> SnapMovie
    CLI --> DomainCore
    CLI --> Exporter
    SnapMovie --> BrokerABC
    SnapMovie --> RulesEngine
    BrokerABC --> IBKR & Inviu & IOL
    SnapMovie --> RepoABC
    SnapMovie --> DomainCore
    DataStorage --> DomainCore
    Plots --> DomainCore
    Exporter --> Plots
```

---

## Rubberducked Edge Cases & Resolved Hidden Assumptions

During rigorous architectural stress-testing, six critical hidden assumptions were identified and resolved:

| # | Domain Area | Hidden Assumption Trap | Resolved Architectural Decision |
| :--- | :--- | :--- | :--- |
| **1** | **Argentine Market Data** | `yfinance` lacks historical prices for Argentine corporate bonds (ONs), Bopreals, Lecaps, and MEP/CCL rates. Relying solely on Yahoo crashes Inviu/IOL ingestion. | **Tiered Market Data Fallback**: Use `yfinance` for global equities/ETFs, but route Argentine fixed income/bills to a secondary local price/FX file ingestion engine. |
| **2** | **Market Holidays & Calendars** | US (NYSE) and Argentine (BYMA) holiday calendars diverge. A naive `dropna()` discards 15–20% of valid trading days. | **Union Calendar Alignment (`ffill`)**: Align data to the union of all business days, forward-filling last known closing prices when an exchange is closed. |
| **3** | **Rebalancing Lag** | Rebalancing instantaneously at day $T$ close introduces lookahead bias for realistic advisory execution. | **Configurable Execution Timing**: Instantaneous $T$ close by default (for quant research), with an optional `execution_lag=1` parameter to model realistic next-day ($T+1$) execution. |
| **4** | **Cash Drag & Dividends** | Ignoring uninvested cash or assuming immediate synthetic dividend reinvestment distorts client return vs manager skill. | **Full Cash Ledger Accounting**: Track explicit cash balances. Dividends and bond coupons flow into cash as organic return. External capital deposits/withdrawals are isolated to calculate both **TWR** (manager skill) and **MWR/XIRR** (dollar return). |
| **5** | **Argentine FX Valuation** | Official Argentine exchange rate (BCRA A3500) creates a 30–50% artificial distortion in client net worth. | **Financial Dollar Valuation**: ARS cash and local instruments are converted to USD base using market-driven **MEP** (local bonds/cash) and **CCL** (CEDEARs/offshore) rates. |
| **6** | **Corporate Actions & Splits** | Historical transactions over multi-year periods diverge from current share counts due to stock splits (e.g. NVDA 10:1, AAPL 4:1) or CEDEAR ratio adjustments. | **Snapshot Ground-Truth Anchor**: The broker's current snapshot anchors today's position quantities; corporate action events (splits/spinoffs) in the transaction ledger are parsed to reconcile historical share counts. |

---

## Directory & Package Structure

```
portfolio_analyser/
├── pyproject.toml               # Poetry/pip-tools configuration, mypy strict, ruff, pytest
├── config/
│   ├── __init__.py
│   ├── settings.py              # Pydantic Settings (DB URLs, API keys, base currency, risk-free rate)
│   └── theme.py                 # Color schemes, typography, plot formatting configs
├── domain/                      # PURE DOMAIN - Zero 3rd party network/DB dependencies
│   ├── __init__.py
│   ├── models.py                # Asset, Position, Transaction, CashFlow, Currency, Metrics
│   ├── rebalance/
│   │   ├── __init__.py
│   │   ├── base.py              # RebalanceStrategy (ABC) with turnover cost & lag modeling
│   │   ├── periodic.py          # PeriodicRebalanceStrategy (monthly, quarterly, annual)
│   │   └── deviation.py         # DeviationRebalanceStrategy (corridor / tolerance bands)
│   ├── calculator.py            # Geometric CAGR, TWR, MWR/XIRR, Sharpe, Drawdown, Rolling Corr
│   ├── calendar.py              # Union calendar alignment and price forward-filling
│   └── rules.py                 # Declarative Pydantic Rule Engine (Client policy compliance)
├── data_layer/                  # PERSISTENCE & DATA INGESTION
│   ├── __init__.py
│   ├── interfaces.py            # MarketDataRepository(ABC), MarketDataProvider(ABC)
│   ├── database.py              # SQLAlchemy 2.0 Base, engine, session factory (Postgres / SQLite)
│   ├── models_sql.py            # ORM tables (assets, daily_prices, fx_rates, corporate_actions)
│   ├── repository.py            # SQLAlchemyMarketDataRepository (caching, deduplication)
│   ├── taxonomy.py              # Geography, AssetType, Sector/Industry taxonomy service
│   ├── currency.py              # Multi-currency & Financial FX (MEP / CCL) normalizer
│   └── providers/
│       ├── __init__.py
│       ├── yfinance_provider.py # YFinance with Tenacity exponential backoff & rate limiting
│       └── local_provider.py    # Local Argentine CSV/JSON provider for ONs, bonds, and MEP/CCL
├── brokers/                     # BROKER INTEGRATION & CLIENT ACCOUNTING
│   ├── __init__.py
│   ├── base.py                  # BrokerConnector (ABC)
│   ├── ibkr/                    # Interactive Brokers (Flex Web Service, Client Portal API, CSV)
│   │   ├── __init__.py
│   │   ├── connector.py
│   │   └── parser.py
│   ├── inviu/                   # Inviu Argentina (REST API & CSV export parser)
│   │   ├── __init__.py
│   │   ├── connector.py
│   │   └── parser.py
│   ├── iol/                     # InvertirOnline (OAuth2 Bearer token REST & CSV parser)
│   │   ├── __init__.py
│   │   ├── connector.py
│   │   └── parser.py
│   └── client_service.py        # Snapshot (forward-looking) vs Movie (backward-looking ledger)
├── visualization/               # REPORTING & VISUALIZATION
│   ├── __init__.py
│   ├── styles.py                # Seaborn/Matplotlib themes and color palettes
│   ├── plot_factory.py          # Rebased performance, correlation heatmap, underwater drawdown
│   └── exporter.py              # HTML, PDF, PNG exporter
├── tests/                       # UNIT & INTEGRATION TESTS
│   ├── conftest.py              # In-memory SQLite fixtures, synthetic market data, mock brokers
│   ├── test_domain_rebalance.py # Tests for periodic & deviation rebalancing strategies
│   ├── test_calculator.py       # Math validation (CAGR, TWR, MWR, Sharpe, Drawdown)
│   ├── test_calendar.py         # Union calendar alignment and forward-fill tests
│   ├── test_repository.py       # SQL repository caching & deduplication tests
│   ├── test_brokers.py          # Mock broker API and CSV statement parsers
│   ├── test_rules.py            # Client policy engine validation tests
│   └── test_cli.py              # Click CLI command-line argument testing
└── main.py                      # Click CLI application entry point
```

---

## Staged Implementation Plan & LLM Prompts

Each stage below contains an exact, production-grade **LLM Prompt** designed for Gemini (or Antigravity) to execute systematically.

---

### Stage 1: Domain Core, Rebalancing Engine & Quant Math

#### Objectives:
- Implement pure domain entities using Pydantic v2 and Python dataclasses.
- Implement the `RebalanceStrategy(ABC)` supporting turnover transaction costs and configurable execution lag ($T$ vs $T+1$).
- Implement concrete rebalancing subclasses:
  - `PeriodicRebalanceStrategy` (Daily, Monthly, Quarterly, Annual, BuyAndHold).
  - `DeviationRebalanceStrategy` (Tolerance corridors).
- Fix arithmetic return scaling: implement true geometric Compound Annual Growth Rate ($\text{CAGR} = (V_T/V_0)^{252/N} - 1$), High-Water Mark Drawdown, Sharpe Ratio, and Rolling Correlation.
- Implement `CalendarAligner`: forward-fill (`ffill`) on the union of calendar trading days.
- Ensure 100% strict type safety (`mypy --strict`).

#### Gemini Execution Prompt: Stage 1

```markdown
You are a Senior Quantitative Developer and Python Systems Architect.
Execute Stage 1 of the portfolio_analyser project: The Domain Core, Rebalancing Engine, and Quant Math.

### Requirements:
1. Environment & Config:
   - Configure `pyproject.toml` with Python >= 3.10, `pydantic>=2.5.0`, `numpy>=1.26.0`, `pandas>=2.1.0`, `scipy>=1.11.0`, `mypy>=1.8.0`, `pytest>=7.4.0`.
   - Configure `mypy` with `strict = true`.

2. File: `portfolio_analyser/domain/models.py`
   - Define immutable Domain entities:
     - `Currency` (Enum: USD, ARS, EUR, etc.)
     - `AssetType` (Enum: STOCK, ETF, BOND, COMMODITY, CRYPTO, CASH)
     - `Geography` (Enum: US, ARGENTINA, DEVELOPED_EX_US, EMERGING_MARKETS, GLOBAL)
     - `TransactionType` (Enum: BUY, SELL, DEPOSIT, WITHDRAWAL, DIVIDEND, COUPON, SPLIT)
     - `Asset` (ticker: str, name: str, asset_type: AssetType, geography: Geography, industry: str, currency: Currency)
     - `Position` (asset: Asset, quantity: float, cost_basis: float, current_price: float)
     - `CashFlow` (timestamp: datetime, amount: float, currency: Currency, flow_type: TransactionType)
     - `PortfolioSnapshot` (timestamp: datetime, positions: list[Position], cash_balances: dict[Currency, float])
     - `PerformanceMetrics` (cagr: float, annualized_volatility: float, sharpe_ratio: float, max_drawdown: float, calmar_ratio: float, total_return: float)

3. File: `portfolio_analyser/domain/rebalance/base.py`
   - Create abstract base class `RebalanceStrategy(ABC)`:
     ```python
     class RebalanceStrategy(ABC):
         def __init__(self, transaction_cost_bps: float = 0.0, execution_lag: int = 0): ...
         @abstractmethod
         def compute_portfolio_series(
             self, 
             prices: pd.DataFrame, 
             target_weights: dict[str, float], 
             initial_capital: float = 100000.0
         ) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
             """Returns (portfolio_value_series, asset_weights_series, turnover_costs_series)"""
             pass
     ```

4. Files: `portfolio_analyser/domain/rebalance/periodic.py` & `deviation.py`
   - Implement `PeriodicRebalanceStrategy`:
     - Frequencies: `daily`, `monthly`, `quarterly`, `annual`, `buy_and_hold`.
     - Tracks drifted weights daily. Rebalances on schedule. Applies transaction cost basis points on turnover. Supports `execution_lag` (0 = execute on close, 1 = execute next trading day).
   - Implement `DeviationRebalanceStrategy`:
     - Parameters: `drift_threshold: float` (e.g. 0.05), `check_frequency: str = 'daily'`.
     - Rebalances to target weights when $\max_i |w_i(t) - w_i^*| \ge \text{threshold}$.

5. File: `portfolio_analyser/domain/calendar.py`
   - Implement `CalendarAligner`:
     - `align_price_series(prices_dict: dict[str, pd.Series]) -> pd.DataFrame`:
       - Takes price series with divergent calendars (e.g. NYSE vs BYMA), constructs the union of all business dates, and applies forward-filling (`ffill`) so no valid trading days are discarded.

6. File: `portfolio_analyser/domain/calculator.py`
   - `calculate_cagr(value_series: pd.Series, trading_days: int = 252) -> float`:
     - Computes geometric CAGR: $(V_{end} / V_{start})^{(252 / N)} - 1$.
   - `calculate_volatility(returns: pd.Series, trading_days: int = 252) -> float`:
     - $\sigma \times \sqrt{252}$.
   - `calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, trading_days: int = 252) -> float`.
   - `calculate_drawdowns(value_series: pd.Series) -> tuple[pd.Series, float]`:
     - Running high-water mark, drawdown series, and maximum drawdown.
   - `calculate_twr(subperiod_returns: list[float]) -> float`:
     - Time-Weighted Return chain-linking: $\prod (1 + R_t) - 1$.
   - `calculate_mwr(cash_flows: list[tuple[date, float]], final_value: float, end_date: date) -> float`:
     - Money-Weighted Return via exact XIRR (using `scipy.optimize.newton`).
   - `calculate_rolling_correlation(returns: pd.DataFrame, window: int = 30) -> dict[tuple[str, str], pd.Series]`.

7. Tests: `tests/test_domain_rebalance.py`, `tests/test_calculator.py`, `tests/test_calendar.py`
   - Validate CAGR vs arithmetic mean under high volatility.
   - Verify execution lag parameter delays turnover by exact days specified.
   - Verify calendar aligner preserves multi-market union without losing trade days.
   - Ensure all code passes `mypy --strict`.
```

---

### Stage 2: Persistence, Caching, Tiered Data & Financial FX Layer

#### Objectives:
- Implement SQLAlchemy 2.0 repository pattern to persist market prices, FX rates, and asset metadata.
- Support PostgreSQL via connection string (inspectable in pgAdmin v0) and in-memory SQLite for tests.
- Deduplication: Query database first; only download non-existent date intervals.
- Tiered Market Data Providers:
  - `YFinanceProvider` wrapped with exponential backoff (`tenacity`) for global equities/ETFs.
  - `LocalMarketDataProvider` for local Argentine instruments (ONs, Bopreals, Lecaps).
- Financial FX Rate Service: normalize ARS cash and assets to USD base using market-driven **MEP** and **CCL** rates.

#### Gemini Execution Prompt: Stage 2

```markdown
You are a Senior Backend and Data Engineer.
Execute Stage 2 of the portfolio_analyser project: Persistence, Caching, Tiered Data, and Financial FX.

### Requirements:
1. Dependencies:
   - Add `sqlalchemy>=2.0.25`, `psycopg[binary]>=3.1.18`, `alembic>=1.13.0`, `tenacity>=8.2.3`, `yfinance>=0.2.35`.

2. File: `portfolio_analyser/config/settings.py`
   - Implement `Settings(BaseSettings)`:
     - `DATABASE_URL: str = "postgresql+psycopg://postgres:postgres@localhost:5432/portfolio_db"` (fallback: `sqlite:///portfolio.db`)
     - `BASE_CURRENCY: str = "USD"`
     - `DEFAULT_RISK_FREE_RATE: float = 0.02`
     - `CACHE_EXPIRY_HOURS: int = 24`
     - `LOCAL_DATA_DIR: Path = Path("./data")`

3. File: `portfolio_analyser/data_layer/models_sql.py`
   - SQLAlchemy 2.0 Declarative Models:
     - `AssetRecord`: `ticker` (PK), `name`, `asset_type`, `geography`, `industry`, `sector`, `currency`, `created_at`.
     - `PriceRecord`: `id` (PK, autoincrement), `ticker` (FK), `date` (Date, indexed), `adjusted_close` (Float), `volume` (BigInteger). Unique `(ticker, date)`.
     - `FXRateRecord`: `from_currency`, `to_currency`, `rate_type` (Enum: MEP, CCL, OFFICIAL), `date`, `rate`. Unique `(from_currency, to_currency, rate_type, date)`.
     - `CorporateActionRecord`: `ticker`, `date`, `action_type` (SPLIT, DIVIDEND), `ratio`.

4. File: `portfolio_analyser/data_layer/interfaces.py`
   - `MarketDataRepository(ABC)`:
     - `get_prices(ticker: str, start_date: date, end_date: date) -> pd.Series`
     - `save_prices(ticker: str, prices: pd.Series) -> None`
     - `get_asset(ticker: str) -> Optional[Asset]`
     - `save_asset(asset: Asset) -> None`
     - `get_missing_date_ranges(ticker: str, start_date: date, end_date: date) -> list[tuple[date, date]]`
     - `get_fx_rate(from_curr: Currency, to_curr: Currency, rate_type: str, date_val: date) -> float`

5. File: `portfolio_analyser/data_layer/repository.py`
   - Implement `SQLAlchemyMarketDataRepository(MarketDataRepository)`:
     - Implements upserts / deduplication: only queries missing dates.
     - Supports both PostgreSQL and SQLite.

6. Providers: `portfolio_analyser/data_layer/providers/`
   - `YFinanceProvider`:
     - Decorated with `@retry` from `tenacity` (`stop_after_attempt(5)`, `wait_exponential(multiplier=1, min=2, max=30)`).
     - Extracts `.info` to auto-populate metadata.
   - `LocalMarketDataProvider`:
     - Ingests Argentine local prices and MEP/CCL historical rates from local CSV/JSON feeds.
   - `TieredMarketDataProvider`:
     - Routes global tickers to `YFinanceProvider` and Argentine fixed income/bills/FX to `LocalMarketDataProvider`.

7. File: `portfolio_analyser/data_layer/currency.py`
   - `FinancialFXConverter`:
     - Normalizes ARS assets and cash balances to USD base using MEP (local bonds/cash) and CCL (CEDEARs/offshore).

8. Tests: `tests/test_repository.py`
   - SQLite in-memory engine fixture (`sqlite:///:memory:`).
   - Verify cache deduplication (provider called only for missing dates).
   - Test tiered routing and financial MEP/CCL conversion.
```

---

### Stage 3: Broker Ingestion, Client Accounting (Snapshot vs Movie) & Policy Engine

#### Objectives:
- Implement unified `BrokerConnector(ABC)` supporting hybrid mode (API auth + CSV statement parsing).
- Concrete Adapters:
  - `IBKRAdapter`: Interactive Brokers (Flex Query XML/CSV parser & Client Portal REST client).
  - `InviuAdapter`: Inviu Argentina (REST token auth & CSV parser handling CEDEARs, ONs, ARS, MEP).
  - `IOLAdapter`: InvertirOnline (OAuth2 bearer token API & CSV parser).
- Build the **Snapshot** vs **Movie** analytics:
  - **Snapshot (Forward-Looking)**: Current asset allocation, geographic/asset-class/currency exposure, top concentration risk, dividend yield. Anchors net worth as ground truth.
  - **Movie (Backward-Looking)**: Historical NAV series, Time-Weighted Return (TWR) vs Money-Weighted Return (MWR / XIRR) isolating client deposits/withdrawals from performance, drawdown duration, and benchmark comparison.
- Implement `ClientInformation` & Declarative Pydantic Rule Engine (e.g., `MaxAssetWeight`, `MinAssetTypeWeight`, `MaxGeographyWeight`).

#### Gemini Execution Prompt: Stage 3

```markdown
You are a Senior Financial Software Engineer specializing in Portfolio Accounting and Broker Integrations.
Execute Stage 3 of the portfolio_analyser project: Broker Ingestion, Client Accounting, and Policy Engine.

### Requirements:
1. File: `portfolio_analyser/brokers/base.py`
   - Define `BrokerConnector(ABC)`:
     ```python
     class BrokerConnector(ABC):
         @abstractmethod
         def authenticate(self) -> bool: ...
         @abstractmethod
         def get_current_positions(self) -> list[Position]: ...
         @abstractmethod
         def get_transaction_history(self, start_date: date, end_date: date) -> list[Transaction]: ...
         @abstractmethod
         def parse_statement(self, file_path: Path) -> tuple[list[Position], list[Transaction], list[CashFlow]]: ...
     ```

2. Concrete Adapters:
   - `portfolio_analyser/brokers/ibkr/`: Flex Query CSV/XML and Activity Statement parser.
   - `portfolio_analyser/brokers/inviu/`: REST auth & parser for Argentine holdings (CEDEARs, ONs, ARS, MEP).
   - `portfolio_analyser/brokers/iol/`: IOL OAuth2 bearer token API (`/api/token`) with refresh token handling, and CSV export parser.

3. File: `portfolio_analyser/domain/rules.py`
   - Declarative Policy Engine using Pydantic:
     - `PortfolioRule(ABC)`:
       - `validate(snapshot: PortfolioSnapshot) -> RuleEvaluationResult(passed: bool, current_value: float, threshold: float, message: str)`
     - Built-in rules:
       - `MaxAssetWeight(max_weight=0.10)`
       - `MinAssetTypeWeight(asset_type=AssetType.BOND, min_weight=0.20)`
       - `MaxGeographyWeight(geography=Geography.ARGENTINA, max_weight=0.15)`
       - `MaxCashWeight(max_weight=0.05)`
     - `ClientPolicy`: Collection of rules with client metadata (risk profile). Emits compliance audit report.

4. File: `portfolio_analyser/brokers/client_service.py`
   - `ClientPortfolioService`:
     - `generate_snapshot(client_id: str, connector: BrokerConnector) -> ClientSnapshotReport`:
       - Ground-truth snapshot positions, normalized base currency, categorical allocations (% equity, % bond, % geography, % currency), and rule violations.
     - `generate_movie(client_id: str, connector: BrokerConnector, benchmark_ticker: str = "SPY") -> ClientMovieReport`:
       - Reconstructs historical NAV from transactions.
       - Reconciles historical share counts against current snapshot anchor via corporate action records.
       - Calculates **TWR (Time-Weighted Return)** isolating manager skill from cash deposit timing.
       - Calculates **MWR (Money-Weighted Return / XIRR)** measuring client dollar-weighted return.
       - Benchmark comparison curve and drawdown analysis.

5. Tests: `tests/test_brokers.py` & `tests/test_rules.py`
   - Provide synthetic sample statements for IBKR, Inviu, and IOL.
   - Verify TWR calculation separates cash deposit timing from investment return.
   - Test rule engine emits pass/fail alerts accurately when portfolio breaches limits.
```

---

### Stage 4: Visualizations, Click CLI & Reporting Pipeline

#### Objectives:
- Implement institutional-grade visualizations using Matplotlib & Seaborn.
- Implement `ReportExporter` generating HTML and PDF reports.
- Implement Click CLI with rich terminal formatting:
  - Command: `analyze`: Run exploratory analysis with tickers, weights, dates, rebalancing strategy.
  - Command: `client-review`: Generate forward-looking snapshot, backward-looking movie, or both.
- Wire all components into `main.py`.

#### Gemini Execution Prompt: Stage 4

```markdown
You are a Senior Full-Stack Python Engineer and CLI Developer.
Execute Stage 4 of the portfolio_analyser project: Visualizations, Click CLI, and Reporting Pipeline.

### Requirements:
1. Dependencies:
   - Add `click>=8.1.7`, `matplotlib>=3.8.0`, `seaborn>=0.13.0`, `jinja2>=3.1.3`, `weasyprint>=61.0`, `rich>=13.7.0`.

2. File: `portfolio_analyser/visualization/plot_factory.py`
   - Implement `PlotFactory`:
     - `plot_rebased_performance(prices: pd.DataFrame, base: float = 100) -> Figure`
     - `plot_correlation_heatmap(returns: pd.DataFrame) -> Figure`
     - `plot_cumulative_and_drawdown(portfolio_value: pd.Series, drawdown: pd.Series, benchmark_value: Optional[pd.Series] = None) -> Figure`
     - `plot_asset_allocation_pies(snapshot: PortfolioSnapshot) -> Figure` (Asset type, geography, currency breakdown)
     - `plot_rolling_correlation(rolling_corr: pd.Series, ticker1: str, ticker2: str) -> Figure`

3. File: `portfolio_analyser/visualization/exporter.py`
   - `ReportExporter`:
     - Compiles metrics and plots into a clean Jinja2 HTML report template.
     - Exports to `.html` and converts to publication-ready `.pdf`.

4. File: `portfolio_analyser/main.py`
   - Implement Click CLI with subcommands:
     - `python main.py analyze`:
       - `-t`, `--tickers` (multiple): e.g. `-t SPY -t GLD -t TIP -t BRKB`
       - `-w`, `--weights` (multiple): e.g. `-w 0.3 -w 0.3 -w 0.15 -w 0.25`
       - `-s`, `--start-date`: e.g. `2020-01-01`
       - `-e`, `--end-date`: e.g. `2025-01-01`
       - `-r`, `--rebalance`: choices `[daily, monthly, quarterly, annual, buy_and_hold, deviation]` (default: `quarterly`)
       - `--threshold`: deviation corridor if rebalance=deviation (default: 0.05)
       - `--lag`: execution lag days (default: 0)
       - `--export`: choices `[terminal, html, pdf]`
     - `python main.py client-review`:
       - `--broker`: choices `[ibkr, inviu, iol]`
       - `--file`: path to statement export (optional if using live API)
       - `--client-id`: client identifier
       - `--mode`: choices `[snapshot, movie, both]`
       - `--benchmark`: default `SPY`
       - `--export`: choices `[terminal, html, pdf]`

5. Tests: `tests/test_cli.py`
   - Test CLI invocations using Click's `CliRunner`.
   - Verify argument validation (weights sum to 1.0, ticker count matches weight count).
```

---

### Stage 5: Verification, Benchmarking & End-to-End Test Suite

#### Objectives:
- Full test suite execution across unit, integration, and CLI layers.
- Strict type checking with `mypy --strict`.
- End-to-end regression validation against the original [`portfolio_creator/portfolio.ipynb`](file:///home/alfred/github/portfolio_creator/portfolio.ipynb) data.

#### Gemini Execution Prompt: Stage 5

```markdown
You are a Quality Assurance and Performance Engineer.
Execute Stage 5 of the portfolio_analyser project: Verification, Benchmarking, and Mypy Strict Validation.

### Requirements:
1. Run static analysis:
   - `mypy --strict portfolio_analyser`
   - Resolve any typing issues, ensuring zero `Any` leakages in core domain math.

2. Run test suite:
   - `pytest -v --cov=portfolio_analyser tests/`
   - Ensure >= 90% code coverage across domain and data layers.

3. End-to-End Validation:
   - Run CLI command matching the original prototype:
     `python main.py analyze -t GLD -t SPY -t TIP -t BRKB -w 0.3 -w 0.3 -w 0.15 -w 0.25 -s 2019-11-01 -e 2025-11-01 --export html`
   - Verify generated metrics match expectations:
     - Compare arithmetic vs compound CAGR.
     - Confirm database cache stores all downloaded price rows.
     - Confirm subsequent runs execute instantly from cache without hitting network requests.
```

---

## Future Enhancements Roadmap

1. **Macroeconomic Data Module (`data_layer/macro/`)**:
   - Ingest interest rate cycles (Fed Funds, BCRA Leliq/BOPREAL rates), inflation prints (US CPI, Argentine CER/INDEC), and yield curve spreads (10Y - 2Y Treasury).
2. **Elections & Geopolitical Risk Module (`data_layer/events/`)**:
   - Event calendar tracking scheduled election dates, FOMC meetings, and IMF review deadlines.
3. **Advanced Optimization Engine (`domain/optimizer.py`)**:
   - Mean-Variance Efficient Frontier, Black-Litterman model, and Hierarchical Risk Parity (HRP).
