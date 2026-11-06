"""Usage counters for RedisVL LLM caches.

Every cache counts its own checks, hits, misses, errors and stores, so a
deployment gets cache health signals without wrapping ``check``/``store``
calls in its own instrumentation. See :attr:`BaseLLMCache.stats
<redisvl.extensions.cache.llm.base.BaseLLMCache.stats>`.

The counters are per cache object, in process, and never leave it: nothing is
written to Redis, so two workers running the same cache name do not aggregate.
No prompt, response or metadata content is ever recorded.
"""

import functools
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class CacheStats:
    """A point-in-time snapshot of one cache's counters.

    Attributes:
        checks (int): ``check``/``acheck`` calls, including those that raised.
        hits (int): Checks that returned at least one cached entry.
        misses (int): Checks that returned no entry.
        errors (int): Checks that raised. ``checks`` always equals
            ``hits + misses + errors``.
        stores (int): ``store``/``astore`` calls that completed.
        lookup_seconds (float): Total wall-clock seconds spent inside checks,
            covering the embedding call and the Redis search but not the LLM
            call the caller makes on a miss, including checks that raised.
    """

    checks: int = 0
    hits: int = 0
    misses: int = 0
    errors: int = 0
    stores: int = 0
    lookup_seconds: float = 0.0

    @property
    def hit_rate(self) -> float | None:
        """Hits over completed checks, or None before the first one.

        Checks that raised are excluded from the denominator, since an error is
        not a cache miss.
        """
        completed = self.hits + self.misses
        return self.hits / completed if completed else None

    @property
    def avg_lookup_seconds(self) -> float | None:
        """Mean seconds per check, or None if the cache has not been checked."""
        return self.lookup_seconds / self.checks if self.checks else None


@dataclass
class _CacheCounters:
    """The mutable counters a cache increments; :meth:`snapshot` freezes them."""

    checks: int = 0
    hits: int = 0
    misses: int = 0
    errors: int = 0
    stores: int = 0
    lookup_seconds: float = 0.0

    def snapshot(self) -> CacheStats:
        return CacheStats(
            checks=self.checks,
            hits=self.hits,
            misses=self.misses,
            errors=self.errors,
            stores=self.stores,
            lookup_seconds=self.lookup_seconds,
        )


def observe_check(func: F) -> F:
    """Count and time one synchronous check. See :func:`observe_acheck`."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            results = func(self, *args, **kwargs)
        except Exception:
            self._record_error(start)
            raise
        self._record_check(start, results)
        return results

    return cast(F, wrapper)


def observe_acheck(func: F) -> F:
    """Count and time one asynchronous check.

    Errors are counted before they propagate, so a backend outage shows up in
    ``CacheStats.errors`` rather than as a silently flat check count.
    """

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            results = await func(self, *args, **kwargs)
        except Exception:
            self._record_error(start)
            raise
        self._record_check(start, results)
        return results

    return cast(F, wrapper)


def count_store(func: F) -> F:
    """Count one synchronous store, only if it completed."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self._stats.stores += 1
        return result

    return cast(F, wrapper)


def count_astore(func: F) -> F:
    """Count one asynchronous store, only if it completed."""

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        result = await func(self, *args, **kwargs)
        self._stats.stores += 1
        return result

    return cast(F, wrapper)
