from datetime import datetime, timezone

from quant_core.enums import Currency
from quant_core.models import CashBalance, FXContext, Portfolio, Position
from quant_core.risk import MaxAssetWeight, RiskEngine


def test_max_asset_weight_inclusive_of_cash():
    portfolio = Portfolio(
        timestamp=datetime.now(timezone.utc),
        base_currency=Currency.USD,
        positions=[
            Position(
                instrument_id="AAPL",
                currency=Currency.USD,
                quantity=10.0,
                cost_basis=100.0,
                current_price=100.0,
            ),
        ],
        cash_balances=[
            CashBalance(currency=Currency.USD, total=3000.0, available=3000.0),
        ],
    )
    fx_context = FXContext(base_currency=Currency.USD, rates={})

    # Total value = 1000 + 3000 = 4000. AAPL weight = 1000/4000 = 25%

    # 30% max weight should pass
    rule_pass = MaxAssetWeight("AAPL", 0.30)
    assert rule_pass.evaluate(portfolio, fx_context) is None

    # 20% max weight should fail
    rule_fail = MaxAssetWeight("AAPL", 0.20)
    violation = rule_fail.evaluate(portfolio, fx_context)
    assert violation is not None
    assert "exceeds max" in violation.message


def test_max_asset_weight_with_fx():
    portfolio = Portfolio(
        timestamp=datetime.now(timezone.utc),
        base_currency=Currency.USD,
        positions=[
            Position(
                instrument_id="GGAL",
                currency=Currency.ARS,
                quantity=1000.0,
                cost_basis=1000.0,
                current_price=1000.0,
            ),
        ],
        cash_balances=[
            CashBalance(currency=Currency.USD, total=1000.0, available=1000.0),
        ],
    )

    # 1000 ARS = 1 USD -> 1000000 ARS position = 1000 USD.
    # Total = 1000 USD (GGAL) + 1000 USD (Cash) = 2000 USD.
    # GGAL weight = 50%
    fx_context = FXContext(
        base_currency=Currency.USD,
        rates={Currency.ARS: 0.001},
    )

    rule_pass = MaxAssetWeight("GGAL", 0.60)
    assert rule_pass.evaluate(portfolio, fx_context) is None

    rule_fail = MaxAssetWeight("GGAL", 0.40)
    violation = rule_fail.evaluate(portfolio, fx_context)
    assert violation is not None
