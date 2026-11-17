## Data Flow

The schemas connect the entire system. The data flows logically from one module to the next:

1.  **Ingestion (Module 1)** produces `MarketData` and `ExternalData`.
2.  These are wrapped in `IngestionEngineOutput`, which is the input for...
3.  **Strategy (Module 2)**, which consumes `IngestionEngineOutput` and produces `TradeSignal`.
4.  `TradeSignal` is passed to...
5.  **Risk Management (Module 3)**, which wraps it in a `SizingInput` object, calculates the size, and produces a final `OrderRequest`.
6.  `OrderRequest` is the input for...
7.  **Execution (Module 4)**, which consumes it and produces an `ExecutionResult`.
8.  `ExecutionResult` is then used (along with `Position`) to build the `PortfolioState`, which feeds back into the Risk module for future decisions.