"""SemanticCache telemetry must never change what the caller sees."""

import pytest

from redisvl.extensions.cache.llm import _telemetry
from redisvl.utils import otel
from redisvl.utils.vectorize import CustomVectorizer


class _Cache:
    name = "cache"

    @_telemetry.operation("check", lookup=True)
    def check(self, error=None):
        if error:
            raise error
        return ["hit"]

    @_telemetry.operation("check", lookup=True)
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
    assert CustomVectorizer(embed=lambda text: [1.0]).embed("x") == [1.0]
    with _telemetry.retrieval("cache", 1, 0.1) as span:
        _telemetry.record_lookup(span, [{"vector_distance": "0.1"}], None)
    _telemetry.record_nearest_distance("cache", [{"vector_distance": "0.1"}], [])
