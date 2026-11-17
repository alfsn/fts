import pytest
from quant_core.enums import Currency
from quant_data.providers.brokers import (
    IBKRConnector,
    InviuConnector,
    IOLConnector,
    LocalYamlConnector,
)


def test_ibkr_connector_mock():
    connector = IBKRConnector()
    assert connector.name == "IBKR"
    portfolio = connector.get_portfolio()

    assert portfolio is not None
    assert len(portfolio.positions) > 0
    assert len(portfolio.cash_balances) > 0
    assert portfolio.positions[0].instrument_id == "AAPL"
    assert portfolio.cash_balances[0].currency == Currency.USD


def test_inviu_connector_mock():
    connector = InviuConnector()
    assert connector.name == "Inviu"
    portfolio = connector.get_portfolio()

    assert portfolio is not None
    assert len(portfolio.positions) > 0
    assert portfolio.positions[0].instrument_id == "GGAL"
    assert portfolio.positions[0].market_symbol == "GGAL.BA"
    assert portfolio.cash_balances[0].currency == Currency.ARS


def test_iol_connector_mock():
    connector = IOLConnector()
    assert connector.name == "IOL"
    portfolio = connector.get_portfolio()

    assert portfolio is not None
    assert len(portfolio.positions) > 0
    assert portfolio.positions[0].instrument_id == "YPFD"
    assert portfolio.positions[0].market_symbol == "YPFD.BA"
    assert portfolio.cash_balances[0].currency == Currency.ARS


def test_local_yaml_connector(tmp_path):
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        """
base_currency: USD
positions:
  - instrument_id: AAPL
    currency: USD
    quantity: 10.0
    cost_basis: 150.0
  - instrument_id: YPF
    provider_symbol: YPFD.BA
    currency: ARS
    quantity: 100.0
    cost_basis: 12000.0
cash_balances:
  - currency: ARS
    total: 1000.0
    available: 1000.0
"""
    )
    portfolio = LocalYamlConnector(file_path=str(path)).get_portfolio()

    assert [p.market_symbol for p in portfolio.positions] == ["AAPL", "YPFD.BA"]
    assert portfolio.positions[1].currency == Currency.ARS
    assert portfolio.cash_balances[0].total == 1000.0


def test_local_yaml_connector_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalYamlConnector(file_path=str(tmp_path / "nope.yaml")).get_portfolio()
