from datetime import datetime, timedelta

from quant_core.enums import Currency
from quant_core.models import FXContext, Portfolio
from quant_core.rebalance import PeriodicRebalanceStrategy


def test_periodic_rebalance_strategy():
    strategy = PeriodicRebalanceStrategy(target_weights={}, interval_days=7)
    portfolio = Portfolio(positions=[], cash_balances=[])
    fx_context = FXContext(base_currency=Currency.USD, rates={})

    t0 = datetime(2023, 1, 1)

    # First run: should rebalance because last_rebalance_time is None
    assert strategy.should_rebalance(portfolio, t0, fx_context) is True

    # But it shouldn't mutate state implicitly
    assert strategy.last_rebalance_time is None
    assert strategy.should_rebalance(portfolio, t0, fx_context) is True

    # Explicitly update state
    strategy.update_state(t0)
    assert strategy.last_rebalance_time == t0

    # Second run: less than interval_days
    t1 = t0 + timedelta(days=6)
    assert strategy.should_rebalance(portfolio, t1, fx_context) is False

    # Third run: >= interval_days
    t2 = t0 + timedelta(days=7)
    assert strategy.should_rebalance(portfolio, t2, fx_context) is True


def test_deviation_strategy_treats_an_unpriced_position_as_a_deviation():
    from quant_core.models import CashBalance, Position
    from quant_core.rebalance import DeviationRebalanceStrategy

    priced = Portfolio(
        positions=[
            Position(
                instrument_id="AAPL",
                currency=Currency.USD,
                quantity=1.0,
                cost_basis=1.0,
                current_price=500.0,
            )
        ],
        cash_balances=[
            CashBalance(currency=Currency.USD, total=500.0, available=500.0)
        ],
    )
    unpriced = priced.model_copy(deep=True)
    unpriced.positions[0].current_price = None
    strategy = DeviationRebalanceStrategy(
        target_weights={"AAPL": 0.5}, max_deviation=0.05
    )
    fx_context = FXContext(base_currency=Currency.USD, rates={})

    assert strategy.should_rebalance(priced, datetime(2024, 1, 1), fx_context) is False
    assert strategy.should_rebalance(unpriced, datetime(2024, 1, 1), fx_context) is True
