"""The workbench (W3): a Streamlit window over fts and portfolio_analyser.

It is the only code that imports ``streamlit``. Pages call one use case per
screen (``trading_bot.core.catalog_repository.CatalogQueryService``,
``portfolio_analyser.review``) and draw the answer with ``quant_ui``: no
maths, no database code (import-linter contracts in the root pyproject).
"""
