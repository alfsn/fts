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
