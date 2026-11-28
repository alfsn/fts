"""What every page shares: the database URLs from the sidebar and the
use-case objects built from them (cached per URL, so a rerun reuses them)."""

import streamlit as st
from trading_bot.core.catalog_repository import CatalogQueryService
from workbench import settings

RESEARCH_KEY = "research_url"
CACHE_KEY = "market_cache_url"


def research_url() -> str:
    return st.session_state.get(RESEARCH_KEY, settings.research_url())


def market_cache_url() -> str:
    return st.session_state.get(CACHE_KEY, settings.market_cache_url())


@st.cache_resource
def _catalog(research: str, cache: str) -> CatalogQueryService:
    return CatalogQueryService.open(research or None, cache or None)


def catalog() -> CatalogQueryService:
    """fts's catalog over the sidebar's research DB and price cache."""
    return _catalog(research_url(), market_cache_url())
