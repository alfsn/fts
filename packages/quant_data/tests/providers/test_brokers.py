from quant_core.enums import Currency
from quant_data.providers.brokers import IBKRConnector, InviuConnector, IOLConnector


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
    assert portfolio.cash_balances[0].currency == Currency.ARS


def test_iol_connector_mock():
    connector = IOLConnector()
    assert connector.name == "IOL"
    portfolio = connector.get_portfolio()

    assert portfolio is not None
    assert len(portfolio.positions) > 0
    assert portfolio.positions[0].instrument_id == "YPFD"
    assert portfolio.cash_balances[0].currency == Currency.ARS
