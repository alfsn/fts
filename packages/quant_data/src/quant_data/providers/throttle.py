"""
One place that enforces every provider's declared ``RateLimit`` (D11).

Adapters stay free of sleeps: ``TieredDataProvider`` routes each vendor call
through ``Throttle.call``, which spaces calls per ``source_name`` and retries
``RateLimitedError`` with exponential backoff. The clock and sleep are
injectable so tests never wait.
"""

import logging
import time
from typing import Callable, Dict, TypeVar

from quant_core.interfaces import RateLimitedError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Throttle:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleep
        self._last_call: Dict[str, float] = {}

    def call(self, provider, fn: Callable[..., T], *args, **kwargs) -> T:
        """Run ``fn(*args, **kwargs)`` within ``provider.rate_limit``."""
        limit = getattr(provider, "rate_limit", None)
        if limit is None:
            return fn(*args, **kwargs)

        source = provider.source_name
        attempt = 0
        while True:
            self._wait_for_slot(source, limit.min_interval_s)
            try:
                return fn(*args, **kwargs)
            except RateLimitedError:
                if attempt >= limit.max_retries:
                    raise
                delay = limit.backoff_s * 2**attempt
                attempt += 1
                logger.warning(
                    f"{source} rate-limited; retry {attempt}/{limit.max_retries} in {delay:.0f}s"
                )
                self._sleep(delay)

    def _wait_for_slot(self, source: str, min_interval_s: float) -> None:
        last = self._last_call.get(source)
        if last is not None:
            remaining = min_interval_s - (self._clock() - last)
            if remaining > 0:
                self._sleep(remaining)
        self._last_call[source] = self._clock()
