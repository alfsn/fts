import pytest
from quant_data.db import create_db_engine, init_cache
from trading_bot.config import settings
from trading_bot.core import database


@pytest.fixture(scope="module", autouse=True)
def isolated_market_cache(tmp_path_factory):
    """Point the shared price cache at a temp file, so tests never
    write into the real workspace cache (D18)."""
    original_url = settings.MARKET_CACHE_URL
    settings.MARKET_CACHE_URL = (
        f"sqlite+pysqlite:///{tmp_path_factory.mktemp('cache') / 'market_cache.db'}"
    )
    engine = create_db_engine(settings.MARKET_CACHE_URL)
    init_cache(engine)
    database.CacheSessionLocal.configure(bind=engine)
    yield settings.MARKET_CACHE_URL
    settings.MARKET_CACHE_URL = original_url
    database.CacheSessionLocal.configure(bind=database.cache_engine)


@pytest.fixture(scope="module", autouse=True)
def isolated_models_dir(tmp_path_factory):
    """Trained ONNX files go to a temp folder, never the real models/."""
    original = settings.MODELS_DIR
    settings.MODELS_DIR = str(tmp_path_factory.mktemp("models"))
    yield settings.MODELS_DIR
    settings.MODELS_DIR = original
