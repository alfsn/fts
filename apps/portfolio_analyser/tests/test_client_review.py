"""`client-review` reports on the real ledger (PR 2b-3): values and external
flows replayed from `transactions:`, no random walk, no mock baseline."""

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from main import cli
from quant_data.providers import YFinanceMarketDataProvider
from test_analyze import CLOSES, LEDGER, PORTFOLIO, _stub_get_bars
from visualization.plot_factory import backward_analytics_figures


def test_backward_analytics_figures_without_a_baseline():
    dates = pd.bdate_range("2022-01-03", periods=400)
    values = pd.Series(
        100 * np.cumprod(1 + np.random.default_rng(2).normal(0, 0.01, 400)),
        index=dates,
    )
    figures = backward_analytics_figures(values)
    assert len(figures) == 4


def _review(monkeypatch, tmp_path, yaml, closes=CLOSES):
    monkeypatch.setattr(YFinanceMarketDataProvider, "get_bars", _stub_get_bars(closes))
    monkeypatch.setattr(YFinanceMarketDataProvider, "rate_limit", None)
    monkeypatch.setattr(
        YFinanceMarketDataProvider, "get_corporate_actions", lambda self, s: []
    )
    path = tmp_path / "portfolio.yaml"
    path.write_text(yaml)
    monkeypatch.chdir(tmp_path)
    args = [
        "client-review",
        "--broker",
        "yaml",
        "--file",
        str(path),
        "--db-url",
        "sqlite://",
    ]
    return CliRunner().invoke(cli, args, terminal_width=200)


def test_client_review_reports_on_the_ledger(monkeypatch, tmp_path):
    result = _review(monkeypatch, tmp_path, PORTFOLIO + LEDGER % 100)

    assert result.exit_code == 0, result.output
    assert "Reconciliation: OK" in result.output
    assert "from 2024-01" in result.output and "valued days" in result.output
    html = (tmp_path / "client_review_yaml.html").read_text(encoding="utf-8")
    assert "Risk KPIs" in html
    assert "SPY" not in html  # the mock baseline is gone (PR 7 brings a real one)


def test_client_review_refuses_without_a_ledger(monkeypatch, tmp_path):
    result = _review(monkeypatch, tmp_path, PORTFOLIO)

    assert result.exit_code != 0
    assert "transactions" in result.output
    assert not (tmp_path / "client_review_yaml.html").exists()


def test_client_review_refuses_on_a_quantity_break(monkeypatch, tmp_path):
    result = _review(monkeypatch, tmp_path, PORTFOLIO + LEDGER % 90)

    assert result.exit_code != 0
    assert "quantity break YPF (ARS)" in result.output
    assert not (tmp_path / "client_review_yaml.html").exists()


def test_client_review_values_follow_the_market(monkeypatch, tmp_path):
    """A flat market gives a flat value series and a zero total return."""
    from data_layer.ledger_binding import replay_ledger
    from data_layer.market_history import ccl_series, usd_per_unit, valuation_date
    from quant_core.analytics import compute_twr
    from quant_data.providers.brokers import LocalYamlConnector

    monkeypatch.setattr(YFinanceMarketDataProvider, "get_bars", _stub_get_bars(CLOSES))
    monkeypatch.setattr(YFinanceMarketDataProvider, "rate_limit", None)
    monkeypatch.setattr(
        YFinanceMarketDataProvider, "get_corporate_actions", lambda self, s: []
    )
    # Bought at the (flat) close, so the only thing that can move TWR is the market.
    at_market = LEDGER.replace("amount: 1450000", "amount: 1000000").replace(
        "price: 12000", "price: 7500"
    )
    path = tmp_path / "portfolio.yaml"
    path.write_text(PORTFOLIO + at_market % 100)
    portfolio = LocalYamlConnector(str(path)).get_portfolio()
    from main import prefetch_history

    closes, actions = prefetch_history(portfolio, "sqlite://")
    fx = usd_per_unit(ccl_series(closes, actions), closes.index)
    as_of = valuation_date(portfolio, closes, fx)

    result = replay_ledger(portfolio, closes, actions, fx, as_of)

    assert result.values.index[0] >= pd.Timestamp("2024-01-02", tz="UTC")
    assert result.values.index[-1] == as_of
    # 100 x 7,500 ARS + 250,000 ARS cash, at 1,666.67 ARS per USD = 600 USD
    assert result.values.iloc[-1] == pytest.approx(600.0, rel=1e-4)
    assert result.flows.iloc[0] == pytest.approx(1_000_000 / 1_666.67, rel=1e-4)
    assert compute_twr(result.values, result.flows) == pytest.approx(0.0, abs=1e-9)
