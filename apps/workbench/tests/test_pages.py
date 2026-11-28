"""Smoke tests: every page renders on the temp databases without an exception."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

SRC = Path(__file__).resolve().parents[1] / "src" / "workbench"


def _run(script: str) -> AppTest:
    app = AppTest.from_file(str(SRC / script), default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    return app


@pytest.mark.usefixtures("databases")
def test_the_shell_renders_the_portfolio_page_first():
    app = _run("app.py")
    assert app.title[0].value == "Portfolio review"


@pytest.mark.usefixtures("databases")
def test_models_page_lists_the_model_and_can_promote_it(databases):
    app = _run("views/models.py")
    assert app.dataframe[0].value["Model ID"].tolist() == ["model_1"]
    app.button[0].click().run()
    assert not app.exception
    assert databases.model_repo.get_model_details("model_1").status == "production"


@pytest.mark.usefixtures("databases")
def test_backtests_page_draws_the_run():
    app = _run("views/backtests.py")
    assert app.dataframe[0].value["Run ID"].tolist() == ["run_1"]


@pytest.mark.usefixtures("databases")
def test_portfolio_page_shows_an_error_for_a_missing_file(tmp_path):
    app = _run("views/portfolio.py")
    app.text_input[0].set_value(str(tmp_path / "missing.yaml"))
    app.button[0].click().run()
    assert not app.exception
    assert "not found" in app.error[0].value


@pytest.mark.usefixtures("databases")
def test_cache_and_fx_pages_explain_an_empty_cache():
    assert "empty" in _run("views/cache.py").info[0].value
    assert "No cached daily bars" in _run("views/fx.py").warning[0].value


def test_cache_and_fx_pages_draw_a_seeded_cache(databases):
    import os
    from datetime import datetime, timedelta, timezone

    from quant_core.enums import BarType
    from quant_core.models import BarData
    from quant_data.db import MarketDataRepository, create_db_session

    day0 = datetime(2026, 7, 1, 20, tzinfo=timezone.utc)
    with create_db_session(os.environ["MARKET_CACHE_URL"]) as session:
        repo = MarketDataRepository(session)
        for symbol, close in (("GGAL.BA", 100.0), ("GGAL", 1.0)):
            repo.save_bars(
                symbol,
                [
                    BarData(
                        timestamp=day0 + timedelta(days=i),
                        open=close,
                        high=close,
                        low=close,
                        close=close,
                        volume=1.0,
                        bar_type=BarType.TIME,
                        interval="1d",
                        ticks_count=1,
                        dollar_volume=close,
                    )
                    for i in range(3)
                ],
                "YFinanceMarketDataProvider",
            )
        session.commit()
    cache = _run("views/cache.py")
    assert cache.dataframe[0].value["instrument_id"].tolist() == ["GGAL", "GGAL.BA"]
    fx = _run("views/fx.py")
    assert fx.metric[0].value == "3"
