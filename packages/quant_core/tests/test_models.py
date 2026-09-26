from datetime import datetime, timezone

import pytest
from quant_core.enums import Currency
from quant_core.models import (
    CashBalance,
    FXContext,
    MissingFXRateError,
    Portfolio,
    Position,
)


def test_fx_context_base_currency_1_to_1():
    context = FXContext(base_currency=Currency.USD, rates={})
    assert context.get_rate(Currency.USD) == 1.0


def test_fx_context_missing_rate_raises():
    context = FXContext(base_currency=Currency.USD, rates={Currency.EUR: 1.1})
    with pytest.raises(MissingFXRateError):
        context.get_rate(Currency.ARS)


def test_portfolio_total_value_mixed_currencies():
    portfolio = Portfolio(
        timestamp=datetime.now(timezone.utc),
        base_currency=Currency.USD,
        positions=[
            Position(
                instrument_id="AAPL",
                currency=Currency.USD,
                quantity=10.0,
                cost_basis=100.0,
                current_price=150.0,
            ),
            Position(
                instrument_id="GGAL",
                currency=Currency.ARS,
                quantity=100.0,
                cost_basis=1000.0,
                current_price=2000.0,
            ),
        ],
        cash_balances=[
            CashBalance(currency=Currency.USD, total=500.0, available=500.0),
            CashBalance(currency=Currency.EUR, total=100.0, available=100.0),
        ],
    )

    fx_context = FXContext(
        base_currency=Currency.USD,
        rates={
            Currency.ARS: 0.001,
            Currency.EUR: 1.1,
        },
    )

    # Math:
    # AAPL (USD): 10 * 150 * 1.0 = 1500
    # GGAL (ARS): 100 * 2000 * 0.001 = 200
    # Cash (USD): 500 * 1.0 = 500
    # Cash (EUR): 100 * 1.1 = 110
    # Total = 1500 + 200 + 500 + 110 = 2310.0

    total_value = portfolio.total_value(fx_context)
    assert total_value == 2310.0


def test_portfolio_total_value_missing_rate():
    portfolio = Portfolio(
        timestamp=datetime.now(timezone.utc),
        base_currency=Currency.USD,
        positions=[],
        cash_balances=[
            CashBalance(currency=Currency.ARS, total=1000.0, available=1000.0),
        ],
    )

    fx_context = FXContext(
        base_currency=Currency.USD,
        rates={},  # Missing ARS
    )

    with pytest.raises(MissingFXRateError):
        portfolio.total_value(fx_context)
