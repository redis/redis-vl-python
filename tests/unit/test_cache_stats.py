"""``stats`` counters for the LLM caches.

``LangCacheSemanticCache``'s client is autospecced and mocked, so these run
without Redis or an API call. The ``SemanticCache`` counter paths, which need a
real index, are covered in tests/integration/test_llmcache.py.
"""

import importlib.util
from unittest.mock import MagicMock, create_autospec, patch

import pytest

from redisvl.extensions.cache.llm import CacheStats, LangCacheSemanticCache

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("langcache") is None,
    reason="langcache package not installed",
)


@pytest.fixture
def client():
    # Local import: at module scope this would break collection in an
    # environment without the optional langcache dependency.
    from langcache import LangCache

    return create_autospec(LangCache, instance=True)


def _cache(client) -> LangCacheSemanticCache:
    with patch.object(LangCacheSemanticCache, "_create_client", return_value=client):
        return LangCacheSemanticCache(
            name="stats_test", cache_id="cache-id", api_key="api-key"
        )


def _hit() -> MagicMock:
    entry = MagicMock()
    entry.model_dump.return_value = {
        "id": "entry-1",
        "prompt": "prompt",
        "response": "response",
        "similarity": 0.95,
        "created_at": 1.0,
        "updated_at": 1.0,
    }
    return entry


def _search(*entries) -> MagicMock:
    response = MagicMock()
    response.data = list(entries)
    return response


def test_stats_start_at_zero(client):
    stats = _cache(client).stats

    assert stats == CacheStats()
    # No completed check yet, so there is no rate or mean to report.
    assert stats.hit_rate is None
    assert stats.avg_lookup_seconds is None


def test_check_counts_a_hit_then_a_miss(client):
    client.search.side_effect = [_search(_hit()), _search()]
    cache = _cache(client)

    assert cache.check(prompt="prompt")
    assert cache.check(prompt="other") == []

    stats = cache.stats
    assert (stats.checks, stats.hits, stats.misses, stats.errors) == (2, 1, 1, 0)
    assert stats.hit_rate == 0.5
    assert stats.lookup_seconds > 0
    assert stats.avg_lookup_seconds is not None
    assert stats.avg_lookup_seconds > 0


def test_check_counts_an_argument_error(client):
    cache = _cache(client)

    with pytest.raises(ValueError):
        cache.check()

    stats = cache.stats
    # An error is counted as a check, but not as a miss.
    assert (stats.checks, stats.errors, stats.hits, stats.misses) == (1, 1, 0, 0)
    assert stats.hit_rate is None


def test_check_counts_a_backend_failure(client):
    client.search.side_effect = RuntimeError("backend down")
    cache = _cache(client)

    with pytest.raises(RuntimeError):
        cache.check(prompt="prompt")

    assert (cache.stats.checks, cache.stats.errors) == (1, 1)


def test_store_counted_only_when_it_completes(client):
    client.set.return_value = MagicMock(entry_id="entry-1")
    cache = _cache(client)

    cache.store(prompt="prompt", response="response")
    assert cache.stats.stores == 1

    with pytest.raises(ValueError):
        cache.store(prompt="", response="response")
    assert cache.stats.stores == 1


@pytest.mark.asyncio
async def test_async_check_and_store_counted(client):
    client.search_async.side_effect = [_search(_hit()), _search()]
    client.set_async.return_value = MagicMock(entry_id="entry-1")
    cache = _cache(client)

    await cache.astore(prompt="prompt", response="response")
    assert await cache.acheck(prompt="prompt")
    assert await cache.acheck(prompt="other") == []

    stats = cache.stats
    assert (stats.stores, stats.checks, stats.hits, stats.misses) == (1, 2, 1, 1)


def test_stats_are_a_snapshot(client):
    client.search.side_effect = [_search(_hit())]
    cache = _cache(client)
    before = cache.stats

    cache.check(prompt="prompt")

    assert before.checks == 0
    assert cache.stats.checks == 1


def test_counters_are_per_cache(client):
    busy = _cache(client)
    client.search.return_value = _search(_hit())
    busy.check(prompt="prompt")

    assert busy.stats.checks == 1
    assert _cache(client).stats.checks == 0


@pytest.mark.asyncio
async def test_async_check_error_counted_as_error_only(client):
    client.search_async.side_effect = RuntimeError("backend down")
    cache = _cache(client)

    with pytest.raises(RuntimeError):
        await cache.acheck(prompt="prompt")

    stats = cache.stats
    assert (stats.checks, stats.errors, stats.hits, stats.misses) == (1, 1, 0, 0)
    assert stats.hit_rate is None
    assert stats.avg_lookup_seconds is not None


def test_wrapping_keeps_the_public_signature(client):
    import inspect

    cache = _cache(client)

    wrapped = inspect.signature(LangCacheSemanticCache.check.__wrapped__)
    bound = inspect.signature(cache.check)
    assert bound.parameters.keys() == set(wrapped.parameters) - {"self"}
    assert bound.return_annotation == wrapped.return_annotation
    assert inspect.iscoroutinefunction(cache.acheck)
    assert cache.check.__doc__
