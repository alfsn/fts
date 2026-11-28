# Workbench

One Streamlit app over both apps (W3a, [plan](../../docs/plans/UI-plan.md#w3-the-workbench-shell)):

| Page | What it shows | Calls |
|---|---|---|
| Portfolio | a YAML ledger's review over a date window: holdings, warnings, the `client-review` charts | `portfolio_analyser.review` + `visualization.plot_factory` |
| Models | the fts model registry, parameter filters, Promote / Archive | `CatalogQueryService.model_repo` |
| Backtests | runs with KPIs, one run's curves, fills and predictions, a run comparison | `CatalogQueryService.backtest_repo` (`run_curves`, `compare_curves`) |
| Cache | every cached series; one series raw beside split- and total-return-adjusted, its actions and gaps (by its vendor's rule) | `quant_data.inspect` (`cached_series`, `inspect_series`) |
| FX | the ARS CCL from the cached GGAL/YPF pairs (the analyser's recipe), each pair's own rate, rejected and single-pair days | `quant_data.inspect.ccl_view` |

```bash
uv run workbench            # http://127.0.0.1:8501
uv run workbench --port 8600
```

It listens on 127.0.0.1 only, because reviews hold client data.

**Databases.** The sidebar starts from `DATABASE_URL` (fts research: models, runs) and `MARKET_CACHE_URL`
(the shared price cache). Blank means each app's own default (fts settings; the workspace `market_cache.db`).
The Portfolio page fills the price cache from Yahoo the first time it values a ledger. The Cache and FX pages only
read it; they never call a vendor.

**Rules** (import-linter contracts in the root `pyproject.toml`):
- This is the only code that imports `streamlit`.
- Pages draw: they call one use case per screen and plot with `quant_ui`. They never import `quant_core.analytics`
  / `quant_core.metrics`, `quant_data`'s db/providers/history, `sqlalchemy`, or an app's internals. The one
  `quant_data` module they use is the read-only `quant_data.inspect`.

Tests: `uv run pytest apps/workbench/tests` (Streamlit `AppTest` smoke tests on temp databases).
