"""SemanticCache telemetry must never change what the caller sees."""

import pytest

from redisvl.extensions.cache.llm import _telemetry
from redisvl.utils import otel


class _Cache:
    name = "cache"

    @_telemetry.timed("check", lookup=True)
    def check(self, error=None):
        if error:
            raise error
        return ["hit"]

    @_telemetry.timed("check", lookup=True)
    async def acheck(self, error=None):
        return self.check(error)


async def test_a_failing_meter_never_reaches_the_caller(monkeypatch):
    # e.g. an opentelemetry-api older than 1.30 rejecting the bucket advice
    def broken_histogram(name):
        raise TypeError("unexpected keyword argument")

    monkeypatch.setattr(otel, "histogram", broken_histogram)
    cache, error = _Cache(), KeyError("missing")

    assert cache.check() == ["hit"]
    assert await cache.acheck() == ["hit"]
    with pytest.raises(KeyError) as raised:
        cache.check(error)
    assert raised.value is error
    _telemetry.record_nearest_distance("cache", [{"vector_distance": "0.1"}], [])
