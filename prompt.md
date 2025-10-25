You are an expert crypto software engineer.

Help me generate a skeleton of the following repo.
 This should focus on abstract classes, pydantic base models and general integration logic. It will not be used for concrete implementation code but as a backbone for abstractions and logic. All classes and models should be fully documented.


An automated trading system for Polymarket or Kalshi would be comprised of five core

 modules. Such a system, often called a "bot," interacts directly with 

Polymarket's on-chain smart contracts and off-chain order book to 

execute trades based on a pre-defined strategy.


Here is a detailed explanation of the modules you would require.

Core Prerequisites


## Module 1: The Data Ingestion Engine

This module's sole purpose is to gather all the data the bot needs to make a decision. It pulls from two primary sources:


a) Market Data: This is the internal data from Polymarket or Kalshi itself. The bot would use Polymarket's official client libraries (e.g., py-clob-client for Python or @polymarket/sdk for TypeScript) to connect to their APIs and get:

Order Books: The current list of all buy ("bid") and sell ("ask") orders for a given market.

Market Prices: The last traded price, best bid, and best ask.

Trade History: A stream of recently executed trades.

Market Details: Information like the event question, resolution source, and end date.

This would need to have an abstract base class to switch between one and another market.

b) External Data (Event Data):

This is the external, real-world data your bot will use to form a 

prediction. For now this External Data Pipeline should be set as abstract classes for down the line implementation.

This engine constantly feeds all this data into the Strategy Engine.

Think about the classes and pydantic data structures we will need.

This module should be loosely coupled to be able to connect to different systems.


## Module 2: The Strategy Engine 

This is the core logic of the bot. 

It receives data from the Data Ingestion Engine and decides

a) what is the investable universe according to some rules.

b) what to do. 

The output of this module is a simple "trade signal" (e.g., BUY_YES, SELL_NO, HOLD).

Currently, this should run as a placeholder dummy abstract class.


## Module 3: Risk & Position Management

Once the Strategy Engine generates a "buy" signal, this module answers the crucial follow-up question: "How much?" It manages the bot's bankroll to ensure it doesn't go broke on a single bad trade.

Its key functions include:

Position Sizing: Calculating the exact amount of USDC to spend on the trade. This is where you would implement a formula like the Kelly Criterion to determine the optimal bet size based on your perceived "edge." There should be an abstract implementation of a sizing formula and a concrete one, for the Kelly Criterion.

Portfolio Management: Keeping track of all current open positions across all markets.

Risk Limits: Enforcing rules like:

"Never allocate more than 20% of the total bankroll to a single market."

"Never hold more than 10 open positions at once."

Stop-Loss/Take-Profit:

 "If I buy shares at $0.60, automatically place an order to sell them if

 the price drops to $0.40 (stop-loss) or rises to $0.90 (take-profit)."

This module takes the signal from the Strategy Engine (e.g., BUY_YES) and passes a refined order (e.g., BUY 150 'Yes' shares) to the Execution Engine.

The position Management module should have a specific class to act as a Portfolio class which allows for easy tracking and management


## Module 4: The Execution Engine 

This

 module's job is to take the precise order from the Risk module and make

 it happen on Polymarket. This is a highly technical module that 

interacts directly with the blockchain.

Here is the step-by-step flow:


Receive Order: It receives the command (e.g., BUY 150 'Yes' shares of Token ID 0x123... at a limit price of $0.60).

Build Transaction: It uses the Polymarket client library (py-clob-client) to build the technical order data.

Sign Transaction: It fetches the bot's private key from a secure location (like an environment variable or a password manager) and uses it to cryptographically sign the order. This proves to the network that the bot, and only the bot, authorized this trade.

Send Order:

 It sends this signed order to the Polymarket Central Limit Order Book 

(CLOB) API, which then places it in the market. If the order is a 

"market order" (buy at any price), it will be settled on-chain almost 

immediately.

This module also handles the complexities 

of the blockchain, such as estimating gas fees (in MATIC), handling 

failed transactions, and retrying if necessary.


## Module 5: Monitoring, Logging, & Alerting 

A bot cannot run in a black box. This module is your interface to the bot's activities.

Logging: The bot must log everything:

INFO: New data received from Twitter.

INFO: Strategy Engine generated BUY signal for Market X.

INFO: Risk Engine calculated order size: 150 shares.

INFO: Execution Engine submitted order. Transaction hash: 0xabc...

ERROR: Transaction failed. Reason: 'Insufficient gas'.

Monitoring: A simple dashboard (which could be a web page or just a terminal printout) that shows your bot's real-time status:

Current P&L (Profit & Loss)

All open positions

Bot status (e.g., RUNNING, PAUSED, ERROR)

Alerting: An automated notification system (e.g., a simple bot that sends messages to you on Telegram or Discord) for critical events:

"A large trade was just executed."

"Your MATIC balance is running low"

"The bot has crashed and requires a restart."


## Module 6: Research and Backtesting engine

this should be fully integrated to the pydantic objects.


This is the project structure
trading_bot/
├── .env.example               # Example environment variables (API keys, DB URL, private key)
├── .gitignore                 # Standard Python .gitignore
├── README.md                  # Project overview, setup, and run instructions
├── pyproject.toml             # Project dependencies (pydantic, sqlalchemy, py-clob-client)
├── data/                      # For storing historical data for backtesting
│   ├── market_data/           # e.g., historical order books, trades
│   └── external_data/         # e.g., historical event data
├── notebooks/                 # Jupyter notebooks for research and backtest analysis
│   ├── 01_data_exploration.ipynb
│   └── 02_strategy_backtest.ipynb
├── src/
│   └── trading_bot/
│       ├── __init__.py
│       ├── __main__.py            # Main entrypoint to start the bot
│       ├── config.py              # Loads configuration from .env using Pydantic BaseSettings
│       │
│       ├── core/                  # Shared data structures and database logic
│       │   ├── __init__.py
│       │   ├── database.py        # SQLAlchemy engine, session setup, and Base
│       │   ├── enums.py           # Core enumerations (e.g., OrderSide, Market, SignalType)
│       │   ├── models.py          # SQLAlchemy ORM Models (e.g., Trade, Position, EventLog)
│       │   └── schemas.py         # Pydantic data contracts (e.g., MarketData, OrderBook, TradeSignal)
│       │
│       ├── data_ingestion/        # Module 1: Data Ingestion Engine
│       │   ├── __init__.py
│       │   ├── abc.py             # Abstract Base Classes (BaseMarketDataProvider, BaseExternalDataProvider)
│       │   ├── engine.py          # Main engine to coordinate data providers and feed the strategy
│       │   └── clients/           # Concrete implementations of data providers
│       │       ├── __init__.py
│       │       ├── polymarket_client.py # Concrete class for Polymarket's CLOB API
│       │       └── kalshi_client.py     # Concrete class for Kalshi's API
│       │
│       ├── strategy/              # Module 2: Strategy Engine
│       │   ├── __init__.py
│       │   ├── abc.py             # Abstract BaseStrategy class
│       │   ├── engine.py          # The core engine that receives data and produces signals
│       │   ├── universe.py        # Logic for filtering and defining the "investable universe"
│       │   └── strategies/        # Folder for all concrete strategy implementations
│       │       ├── __init__.py
│       │       └── dummy_strategy.py # A simple placeholder strategy
│       │
│       ├── risk_management/       # Module 3: Risk & Position Management
│       │   ├── __init__.py
│       │   ├── abc.py             # Abstract BaseSizingStrategy
│       │   ├── manager.py         # Main RiskManager class (applies limits, sizing)
│       │   ├── portfolio.py       # The specific Portfolio class for tracking positions, P&L
│       │   └── sizing/            # Concrete sizing formula implementations
│       │       ├── __init__.py
│       │       └── kelly_criterion.py # Concrete implementation of Kelly Criterion sizer
│       │
│       ├── execution/             # Module 4: Execution Engine
│       │   ├── __init__.py
│       │   ├── abc.py             # Abstract BaseExecutionHandler
│       │   ├── engine.py          # Main engine to process orders, handle retries
│       │   ├── wallet.py          # Securely manages private keys for signing
│       │   └── handlers/          # Concrete execution implementations
│       │       ├── __init__.py
│       │       └── polymarket_handler.py # Uses py-clob-client to sign & send orders
│       │
│       ├── monitoring/            # Module 5: Monitoring, Logging, & Alerting
│       │   ├── __init__.py
│       │   ├── alerter.py         # Service for sending alerts (e.g., to Telegram)
│       │   ├── dashboard.py       # (Optional) A simple terminal dashboard
│       │   └── logger.py          # Configures system-wide logging
│       │
│       ├── backtesting/           # Module 6: Research and Backtesting Engine
│       │   ├── __init__.py
│       │   ├── engine.py          # Core backtesting event loop
│       │   ├── results.py         # Calculates and presents backtest metrics
│       │   └── simulator.py       # Simulates the execution engine and market fills
│       │
│       └── utils/                 # General helper functions
│           ├── __init__.py
│           └── time_utils.py
│
└── tests/                         # Unit and integration tests
    ├── __init__.py
    ├── test_core_schemas.py
    ├── test_data_ingestion.py
    ├── test_execution.py
    ├── test_risk_management.py
    └── test_strategy.py


ALWAYS KEEP IN MIND SOLID BEST PRINCIPLES

ALWAYS KEEP IN MIND ARCHITECTURAL BEST PRACTICES

ALWAYS KEEP IN MIND PYDANTIC OBJECTS FOR EXPLICIT TYPING AND DATA VALIDATION

WHENEVER USEFUL, REGISTER INFORMATION IN AN SQL TABLE. All tables should have an SQLAlchemy ORM.


START RIGHT NOW ONLY WITH THE BASIC FOLDER AND FILE STRUCTURE
In a later chat I will require from you:
i) Data Contracts.
ii) Abstract Base Classes
iii) ORMs and Concrete Implementation