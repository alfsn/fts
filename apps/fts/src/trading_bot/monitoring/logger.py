# src/trading_bot/monitoring/logger.py

"""
Configures the system-wide logger for the trading bot.

This module provides a centralized `setup_logging` function
that should be called once on application startup (e.g., in __main__.py).
It reads the log level from the central config.
"""

import logging
import sys

# Import the centralized settings object
from ..config import settings


def setup_logging():
    """
    Configures the root logger for the entire application.

    This function sets the logging level, format, and output
    handler (streaming to stdout) based on the values in the
    `settings` object.

    This adheres to SOLID by:
    - Single Responsibility: Its only job is to configure logging.
    - Open-Closed: It's driven by configuration (settings.LOG_LEVEL)
      without needing modification to change the log level.
    """
    # Get the root logger
    logger = logging.getLogger()

    # Set the logging level from the configuration
    # This allows us to set "DEBUG" in dev and "INFO" in prod
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a stream handler to output to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Create a standard log formatter
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the formatter for the handler
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    logging.info(f"Logging successfully configured with level: {settings.LOG_LEVEL}")
