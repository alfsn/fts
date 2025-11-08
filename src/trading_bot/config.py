# src/trading_bot/config.py

"""
Handles all configuration management for the application.

This module uses pydantic-settings to load configuration from
environment variables and .env files, providing a single, validated
source of truth for all other modules.
"""

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    DATABASE_URL: str

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
