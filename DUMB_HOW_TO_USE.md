# 🦍 Dumb HOW TO USE: FTS & Portfolio Analyser

Here is the simplest way to run this repo without blowing up your brain.

> [!NOTE]
> Looking for the **Portfolio Analyser** (`analyze` / `client-review` HTML reports)? Jump to [apps/portfolio_analyser/HOW_TO_USE.md](apps/portfolio_analyser/HOW_TO_USE.md).
> The steps below walk through **FTS** (the trading bot, model training, backtesting, and dashboard).

---

## Step 1: Install the Stuff 📦

From the **repository root** (`/home/alfred/github/fts`), install all workspace apps, packages, and dev tools with `uv`:

```bash
cd /home/alfred/github/fts
uv sync --all-extras --dev --all-packages
```

After syncing, move into the `fts` app folder for all the steps below:

```bash
cd /home/alfred/github/fts/apps/fts
```

---

## Step 2: Get the Market Data 📊

The bot needs to know what happened in the past so it can guess the future. Run this to download historical bars into the shared price cache, `market_cache.db` at the repo root (the analyser uses the same file; `apps/fts/dev.db` only gets the dataset entry):

```bash
uv run python -m trading_bot.utils.ingest_historical -p yfinance -t AAPL -f 1d -d 1y
```

### 🎛️ Extra CLI Parameters you can use here:
- `-p` / `--provider`: Data source provider (`yfinance` or `ccxt`).
- `-t` / `--ticker`: The market symbol (e.g. `AAPL` or `BTC/USDT`).
- `-f` / `--timeframe`: Interval of bars (e.g., `1m`, `5m`, `1d`).
- `-d` / `--period`: Time period to download for `yfinance` (e.g., `5d`, `1mo`, `1y`).
- `-s` / `--since` & `-u` / `--until`: ISO dates to define exact download ranges.

Prices are saved raw (as traded). For `yfinance` the splits and dividends are saved too. A training run or backtest whose window contains a split stops with an error, because a raw 4:1 split would look like a −75% day. Pick a window after the split.

---

## Step 3: Train the Brain 🧠

Now teach the bot how to trade by running an Optuna hyperparameter search (e.g. LSTM, RNN, CNN, Linear Regression, or XGBoost). See `apps/fts/specs/train/BTCUSDT/` for available specs:

```bash
uv run python -m nets.training.hparam_search --spec specs/train/BTCUSDT/lstm_hparam_close.yaml
```
*(This exports ONNX model files to `apps/fts/models/registry/trials/`, TensorBoard logs to `apps/fts/runs/optuna/`, and saves the candidate models in `apps/fts/dev.db`.)*

---

## Step 4: Promote the Smartest Brain 🎓

Once training is done, pick a candidate `model_id` and "promote" it to production (either via CLI or by clicking **Promote** in the Streamlit dashboard below):

```bash
uv run python -m trading_bot.utils.promote <model_id_here>
```

---

## Step 5: Run the Bot! 🚀

Run the bot in backtest (historical replay) or live mode using a YAML task config:

```bash
uv run python -m trading_bot --config nets_task.yaml
```

### 🎛️ Extra CLI Parameters:
- `--config`: Specify a different YAML task configuration file (default is `nets_task.yaml`).

The backtest's `data_reader` needs an `interval` (e.g. `"30m"`), the same one the model was trained on. A symbol cached at two intervals is two separate series.

> [!NOTE]
> Want to see how you did? Check `apps/fts/runs/reports/` — backtests export standalone interactive `.html` reports you can open directly in your browser.

---

## Step 6: Click Around in the Dashboard 🖥️

Want a UI to inspect your SQLite databases (`dev.db` and the price cache), filter models by hyperparameters, promote a model with a button click, or compare backtest equity curves side-by-side?

```bash
uv run python -m trading_bot.dashboard.cli --db dev.db --port 8501
```

Then open `http://localhost:8501` in your browser.

---

## Step 7: Did I Break It? (Testing) 🛠️

Run the three test suites from the workspace root (each runs separately so `tests` package names don't collide):

```bash
cd /home/alfred/github/fts
uv run pytest packages
uv run pytest apps/portfolio_analyser/tests
cd apps/fts && uv run pytest tests
```

---

## 💡 What is the "SSOT" for Configs?

**SSOT** stands for **Single Source of Truth**. In `fts`, the SSOT for configuration is `apps/fts/src/trading_bot/config.py` (powered by `pydantic-settings`).

What does this mean for you?
- The app loads environment variables automatically from `.env` files inside `apps/fts/` (like `.env.dev` and `.env.prod`, selected via `APP_ENV`).
- `config.py` validates those variables and resolves paths (`ROOT_DIR`, `MODELS_DIR`, `RUNS_DIR`, and SQLite URLs) relative to `apps/fts/`.
- Every module asks `settings` for its paths and database URLs instead of hardcoding them.

---

## 📓 Using Notebooks for Strategies (Training & Productizing)

If you prefer experimenting in Jupyter, check `apps/fts/notebooks/`:
- `02_strategy_hparam_training.ipynb`: Run hyperparameter sweeps interactively and register models into `dev.db`.
- `03_custom_backtest.ipynb`: Run backtests programmatically and inspect trade overlays and metrics.
- `04_parameter_sweep_backtest.ipynb`: Compare parameter sweep backtests side-by-side and export sweep HTML reports to `apps/fts/runs/reports/`.
