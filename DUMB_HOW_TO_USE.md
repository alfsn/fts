# 🦍 Dumb HOW TO USE: FTS (Automated Trading System)

So, you want to make money (or at least pretend to) using the FTS Trading Bot. Here is the absolute simplest way to run this thing without blowing up your brain.

> [!IMPORTANT]
> **Always make sure you are in the right folder first!**
> Open your terminal and run:
> ```bash
> cd /home/alfred/github/fts/apps/fts
> ```

---

## Step 1: Install the Stuff 📦

Before anything works, you need to install the dependencies. The system uses a tool called `uv` to make this fast.

```bash
uv sync --all-packages
```

## Step 2: Get the Market Data 📊

The bot needs to know what happened in the past so it can guess the future. Run this to download historical data:

```bash
python -m trading_bot.utils.ingest_historical -p yfinance -t AAPL -f 1d -d 1y
```

### 🎛️ Extra CLI Parameters you can use here:
- `-p` / `--provider`: Data source provider (`yfinance` or `ccxt`).
- `-t` / `--ticker`: The market symbol (e.g. `AAPL` or `BTC/USDT`).
- `-f` / `--timeframe`: Interval of bars (e.g., `1m`, `5m`, `1d`).
- `-d` / `--period`: Time period to download (e.g. `5d`, `1mo`, `1y`).
- `-s` / `--since` & `-u` / `--until`: ISO dates to define exact download ranges.

## Step 3: Train the Brain 🧠

Now we need to teach the bot how to trade by running some machine learning models (like LSTMs). It will test a bunch of settings to see what works best.

```bash
python -m plugins.nets.training.hparam_search --spec specs/train/BTCUSDT/lstm_hparam_search.yaml
```
*(This will spit out some ONNX model files and save them in your database).*

## Step 4: Promote the Smartest Brain 🎓

Once training is done, you pick the best model and "promote" it to be the one you actually use.
Find the `model_id` (a bunch of numbers/letters it gave you in Step 3) and run:

```bash
python -m trading_bot.utils.promote <model_id_here>
```

## Step 5: Run the Bot! 🚀

Now that your bot is smart and has data, let it run. You can run it in backtest (fake historical trading) or live mode using a config file.

```bash
python -m trading_bot --config nets_task.yaml
```

### 🎛️ Extra CLI Parameters you can use here:
- `--config`: Specify a different YAML task configuration file (default is `nets_task.yaml`).

> [!NOTE]
> Want to see how you did? Check the `runs/reports/` folder. It will generate a fancy HTML report with your profits, losses, and charts.

## Step 6: Did I Break It? (Testing) 🛠️

If you changed code and want to know if the bot is broken, run the tests:

```bash
uv run pytest
```

---

## 💡 What is the "SSOT" for Configs?

**SSOT** stands for **Single Source of Truth**. In this app, the SSOT for configuration is the `config.py` file (powered by `pydantic-settings`). 

What does this mean for you?
- The app loads environment variables automatically from `.env` files (like `.env.dev` and `.env.prod`).
- `config.py` validates all those variables and provides them to the entire app.
- Instead of passing random variables all over the code, every part of the application (database, logger, engine) asks the single `config.py` for its configuration settings. This keeps everything consistent and prevents bugs where half the app connects to the wrong database!

---

## 📓 Using Notebooks for Strategies (Training & Productizing)

If you don't like using the command line for everything, there is a `notebooks/` folder made exactly for you! 

Notebooks (like `02_strategy_hparam_training.ipynb`) let you:
1. **Experiment interactively**: See charts and debug your machine learning models step-by-step instead of waiting for a massive script to finish.
2. **Train your strategies**: You can run hyperparameter sweeps directly in your browser.
3. **Productize**: Once you are happy with a model in the notebook, you can register it into the database right there. It saves the ONNX model so the main bot can pick it up and use it live!

## 📈 Generating and Visualizing Backtests

Backtesting is just the bot "pretending" to trade in the past to see if your strategy is actually profitable. 

**Two ways to do this:**
1. **Via the CLI (The Fast Way):** 
   When you run `python -m trading_bot --config nets_task.yaml` with a backtest configuration, it automatically exports a beautiful HTML report containing your equity curve, Sharpe ratio, and drawdowns. You can find these inside the `runs/reports/` folder. Just double-click the `.html` file to view it!

2. **Via Notebooks (The Deep Dive Way):**
   If you want to dig deeper into *why* the bot made a trade, open `03_custom_backtest.ipynb` or `04_parameter_sweep_backtest.ipynb`. These notebooks let you run the backtest engine programmatically. They generate custom charts, show exactly where the bot bought and sold on the price graph, and let you compare different backtests side-by-side!
