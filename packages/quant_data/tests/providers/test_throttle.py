"""PR 2a-2: one place enforces each provider's declared rate limit (D11)."""

import pytest
from quant_core.interfaces import RateLimitedError
from quant_core.models import RateLimit
from quant_data.providers.throttle import Throttle


class FakeClock:
    def __init__(self):
        self.now = 1_000.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class _Provider:
    def __init__(self, name, rate_limit, failures=0):
        self._name = name
        self.rate_limit = rate_limit
        self.failures = failures
        self.calls = 0

    @property
    def source_name(self):
        return self._name

    def fetch(self, x):
        self.calls += 1
        if self.calls <= self.failures:
            raise RateLimitedError(self._name)
        return x * 2


LIMIT = RateLimit(min_interval_s=1.0, max_retries=3, backoff_s=30.0)


def test_calls_to_one_source_are_spaced_by_min_interval():
    clock = FakeClock()
    throttle = Throttle(clock=clock, sleep=clock.sleep)
    provider = _Provider("yahoo", LIMIT)

    assert throttle.call(provider, provider.fetch, 1) == 2
    assert throttle.call(provider, provider.fetch, 2) == 4
    assert clock.sleeps == [1.0]  # the first call never waits

    clock.now += 5  # long enough since the last call: no wait
    throttle.call(provider, provider.fetch, 3)
    assert clock.sleeps == [1.0]


def test_sources_do_not_wait_for_each_other():
    clock = FakeClock()
    throttle = Throttle(clock=clock, sleep=clock.sleep)
    a, b = _Provider("a", LIMIT), _Provider("b", LIMIT)

    throttle.call(a, a.fetch, 1)
    throttle.call(b, b.fetch, 1)
    assert clock.sleeps == []


def test_rate_limit_error_retries_with_exponential_backoff():
    clock = FakeClock()
    throttle = Throttle(clock=clock, sleep=clock.sleep)
    provider = _Provider("yahoo", LIMIT, failures=2)

    assert throttle.call(provider, provider.fetch, 1) == 2
    assert provider.calls == 3
    assert clock.sleeps == [30.0, 60.0]


def test_gives_up_after_max_retries():
    clock = FakeClock()
    throttle = Throttle(clock=clock, sleep=clock.sleep)
    provider = _Provider("yahoo", LIMIT, failures=10)

    with pytest.raises(RateLimitedError):
        throttle.call(provider, provider.fetch, 1)
    assert provider.calls == 1 + LIMIT.max_retries
    assert clock.sleeps == [30.0, 60.0, 120.0]


def test_provider_without_a_limit_is_called_directly():
    clock = FakeClock()
    throttle = Throttle(clock=clock, sleep=clock.sleep)
    provider = _Provider("mock", None, failures=1)

    with pytest.raises(RateLimitedError):  # no retries either
        throttle.call(provider, provider.fetch, 1)
    assert clock.sleeps == []


def test_other_errors_are_not_retried():
    clock = FakeClock()
    throttle = Throttle(clock=clock, sleep=clock.sleep)
    provider = _Provider("yahoo", LIMIT)

    def boom():
        raise ValueError("bad symbol")

    with pytest.raises(ValueError):
        throttle.call(provider, boom)
    assert clock.sleeps == []


def test_rate_limit_model_is_frozen_and_strict():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RateLimit(min_interval_s=1.0, max_retries=3, backoff_s=30.0, burst=5)
    with pytest.raises(ValidationError):
        LIMIT.min_interval_s = 2.0
