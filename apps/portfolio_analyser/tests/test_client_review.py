"""`client-review` reports on the real ledger (PR 2b-3): values and external
flows replayed from `transactions:`, against a total-return benchmark (PR 7)."""

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
    assert len(figures) == 6
    forecast, table = figures[3], figures[4]  # after rolling vol (PR 10)
    assert forecast.layout.title.text.startswith("Expected Volatility")
    assert table.layout.title.text == "Volatility Forecast vs Realised"


def test_backward_analytics_figures_against_an_unaligned_benchmark():
    """The benchmark has its own calendar (a US holiday the portfolio values
    through); its line starts at 0 on the portfolio's first date."""
    dates = pd.bdate_range("2022-01-03", periods=400, tz="UTC")
    rng = np.random.default_rng(4)
    spy_r = rng.normal(0.0004, 0.01, 400)
    spy = pd.Series(100 * np.cumprod(1 + spy_r), index=dates).drop(dates[10])
    values = pd.Series(
        1000 * np.cumprod(1 + 0.7 * spy_r + rng.normal(0, 0.003, 400)), index=dates
    )

    figures = backward_analytics_figures(values, benchmark=spy, benchmark_name="SPY")

    assert len(figures) == 8
    cumulative = figures[1]
    assert [t.name for t in cumulative.data] == ["Portfolio", "SPY"]
    assert cumulative.data[0].y[0] == cumulative.data[1].y[0] == 0
    assert cumulative.data[0].x[0] == cumulative.data[1].x[0]
    beta_row = figures[2].data[0].cells.values[1][1]
    assert beta_row.startswith("0.7")


MARKET = {**CLOSES, "SPY": 500.0, "^IRX": 4.0}


def _review(monkeypatch, tmp_path, yaml, closes=CLOSES, extra_args=()):
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
        *extra_args,
    ]
    return CliRunner().invoke(cli, args, terminal_width=200)


def test_client_review_reports_on_the_ledger(monkeypatch, tmp_path):
    result = _review(monkeypatch, tmp_path, PORTFOLIO + LEDGER % 100)

    assert result.exit_code == 0, result.output
    assert "Reconciliation: OK" in result.output
    assert "from 2024-01" in result.output and "valued days" in result.output
    html = (tmp_path / "client_review_yaml.html").read_text(encoding="utf-8")
    assert "Risk KPIs" in html
    assert "Benchmark KPIs" not in html  # SPY has no prices in this stub
    assert "No prices for benchmark SPY" in result.output
    assert "No ^IRX quotes" in result.output


def test_client_review_compares_against_the_total_return_benchmark(
    monkeypatch, tmp_path
):
    result = _review(monkeypatch, tmp_path, PORTFOLIO + LEDGER % 100, MARKET)

    assert result.exit_code == 0, result.output
    assert "No prices" not in result.output and "No ^IRX" not in result.output
    html = (tmp_path / "client_review_yaml.html").read_text(encoding="utf-8")
    assert "SPY (total return, gross dividends)" in html
    assert "Benchmark KPIs" in html and "Alpha" in html
    assert "min 252" in html  # weekly stub bars: too few for a beta, shown as n/a


def test_client_review_labels_net_dividends(monkeypatch, tmp_path):
    result = _review(
        monkeypatch,
        tmp_path,
        PORTFOLIO + LEDGER % 100,
        MARKET,
        ["--dividend-withholding", "0.3", "--benchmark", "SPY"],
    )

    assert result.exit_code == 0, result.output
    html = (tmp_path / "client_review_yaml.html").read_text(encoding="utf-8")
    assert "SPY (total return, net of 30% withholding)" in html


def test_client_review_refuses_a_us_tbill_for_an_ars_portfolio(monkeypatch, tmp_path):
    ars = (PORTFOLIO + LEDGER % 100).replace("base_currency: USD", "base_currency: ARS")
    result = _review(monkeypatch, tmp_path, ars, MARKET)

    assert result.exit_code != 0
    assert "--risk-free-rate" in result.output
    assert not (tmp_path / "client_review_yaml.html").exists()

    result = _review(monkeypatch, tmp_path, ars, MARKET, ["--risk-free-rate", "0.3"])
    assert result.exit_code == 0, result.output


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


def test_benchmark_level_is_total_return_in_the_base_currency():
    """A dividend is reinvested (net of withholding) and USD converts at CCL."""
    from datetime import datetime, timezone

    from data_layer.benchmark import benchmark_level
    from quant_core.enums import CorporateActionType, Currency
    from quant_core.models import Benchmark, CorporateAction

    dates = pd.bdate_range("2024-03-11", periods=4, tz="UTC")
    closes = pd.DataFrame({"SPY": [100.0, 100.0, 98.0, 98.0]}, index=dates)
    actions = {
        "SPY": [
            CorporateAction(
                instrument_id="SPY",
                action_type=CorporateActionType.DIVIDEND,
                ex_date=datetime(2024, 3, 13, tzinfo=timezone.utc),
                amount=2.0,
            )
        ]
    }
    fx = pd.DataFrame(
        {Currency.USD: 1.0, Currency.ARS: [1 / 1000, 1 / 1000, 1 / 1100, 1 / 1100]},
        index=dates,
    )

    usd = benchmark_level(closes, actions, Benchmark(), fx, Currency.USD)
    assert usd.pct_change().iloc[2] == pytest.approx(
        0.0
    )  # 2.0 dividend offsets the drop

    net = benchmark_level(closes, actions, Benchmark(), fx, Currency.USD, 0.3)
    # Yahoo's Adj Close convention (PR 2a): ex-date return = P_ex / (P_prev - D) - 1
    assert net.pct_change().iloc[2] == pytest.approx(98 / (100 - 1.4) - 1)

    ars = benchmark_level(closes, actions, Benchmark(), fx, Currency.ARS)
    assert ars.iloc[-1] / ars.iloc[0] == pytest.approx(1100 / 1000)
