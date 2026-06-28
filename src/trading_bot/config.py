# src/trading_bot/config.py

"""
Handles all configuration management for the application.

This module uses pydantic-settings to load configuration from
environment variables and .env files, providing a single, validated
source of truth for all other modules.
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ComponentConfig(BaseModel):
    """Configuration for a dynamically loaded component."""

    class_path: str = Field(
        ...,
        description="Full module path to the class "
        "(e.g., 'trading_bot.strategy.strategies.dummy_strategy.DummyStrategy')",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the constructor"
    )


class TaskConfig(BaseModel):
    """Configuration for a specific trading or backtesting task."""

    name: str
    loop_driver: Optional[ComponentConfig] = Field(
        None, description="Dynamic event loop timing/scheduling configuration"
    )
    market_provider: Optional[ComponentConfig] = Field(
        None, description="Dynamic market data provider configuration"
    )
    external_providers: List[ComponentConfig] = Field(default_factory=list)
    strategies: List[ComponentConfig]
    sizing_strategy: ComponentConfig
    execution_handler: Optional[ComponentConfig] = Field(
        None,
        description="Dynamic execution handler configuration (e.g., PolymarketHandler)",
    )
    market_ids: List[str]
    extra_models: List[str] = Field(
        default_factory=list, description="Plugin DB models to register"
    )


from trading_bot.core.loader import PluginLoader

ROOT_DIR = str(Path(__file__).resolve().parents[2])


def get_env_filename() -> str:
    """
    Determines which .env file to load based on the APP_ENV variable.
    """
    # We need to manually check the OS environ first for APP_ENV
    # to decide which .env file to load.
    app_env = os.environ.get("APP_ENV", "dev")
    return f".env.{app_env}"


class Settings(BaseSettings):
    """
    Defines the application's configuration settings.

    This class is environment-aware and will load variables from the
    correct .env file (e.g., .env.dev or .env.prod) based on the
    `APP_ENV` environment variable.
    """

    # --- Environment Configuration ---
    # This setting determines which .env file to load.
    # Set APP_ENV=prod in your production environment.
    APP_ENV: Literal["dev", "prod"] = "dev"

    # --- Core Application Settings ---
    LOG_LEVEL: str = "INFO"

    # --- Database Configuration ---
    # TheDATABASE_URL is used by SQLAlchemy in core/database.py.
    # Example for dev (SQLite): "sqlite+pysqlite:///./dev.db"
    # Example for prod (PostgreSQL): "postgresql+psycopg2://user:pass@db:5432/trading"
    DATABASE_URL: str = "sqlite+pysqlite:///./dev.db"

    # --- Path Configuration ---
    # Central repository paths, resolved relative to project root
    ROOT_DIR: str = ROOT_DIR
    MODELS_DIR: str = os.path.join(ROOT_DIR, "models")
    RUNS_DIR: str = os.path.join(ROOT_DIR, "runs")

    # Configure the settings model to load from the .env file
    # determined by the `get_env_filename` method.
    model_config = SettingsConfigDict(
        env_file=get_env_filename(),
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields not defined in the model
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings object.

    Using lru_cache ensures that the .env file is read and
    settings are validated only once, improving performance.

    :return: A singleton Settings instance.
    """
    return Settings()


# Create a single, globally accessible settings instance.
# Other modules (like database.py or logger.py) can import this
# object directly: `from .config import settings`
settings = get_settings()
