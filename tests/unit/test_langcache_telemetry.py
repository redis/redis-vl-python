"""LangCacheSemanticCache reports to OpenTelemetry like SemanticCache does.

The service does the search, so a LangCache lookup records neither a retrieval
span nor a distance point: only the operation span and its duration.
"""

from unittest.mock import MagicMock, create_autospec, patch

import pytest

from redisvl.extensions.cache.llm.langcache import LangCacheSemanticCache
from redisvl.utils import otel

pytest.importorskip("langcache")


@pytest.fixture
def cache():
    # Local import: at module scope this would break collection in an
    # environment without the optional langcache dependency.
    from langcache import LangCache

    entry = MagicMock()
    entry.model_dump.return_value = {
        "id": "entry-1",
        "prompt": "paris",
        "response": "Paris",
        "similarity": 0.95,
        "created_at": 0.0,
        "updated_at": 0.0,
        "attributes": {},
    }
    response = MagicMock()
    response.data = [entry]

    with patch.object(LangCacheSemanticCache, "_create_client") as create:
        # Autospecced against the real SDK class, as mock_langcache_client does in
        # test_langcache_semantic_cache.py: a renamed or removed method fails here
        # instead of being silently auto-vivified.
        client = create_autospec(LangCache, instance=True, spec_set=True)
        client.search.return_value = response
        client.search_async.return_value = response
        client.set.return_value = MagicMock(entry_id="entry-1")
        client.set_async.return_value = MagicMock(entry_id="entry-1")
        # A response object, not an AsyncMock: the wrapper reads model_dump off it.
        client.delete_query_async.return_value = MagicMock()
        create.return_value = client
        yield (
            LangCacheSemanticCache(
                name="langcache",
                server_url="https://api.example.com",
                cache_id="cache",
                api_key="key",
            ),
            client,
        )


def _operations(reader):
    """Map the operation-duration points to their count, by attributes."""
    points = {}
    data = reader.get_metrics_data()
    for resource_metrics in data.resource_metrics if data else []:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name != otel.SEMANTIC_CACHE_OPERATION_DURATION:
                    continue
                for point in metric.data.data_points:
                    points[frozenset(point.attributes.items())] = point.count
    return points


def _attributes(operation, result=None, error_type=None):
    attributes = {otel.CACHE_NAME: "langcache", otel.CACHE_OPERATION: operation}
    if result is not None:
        attributes[otel.CACHE_RESULT] = result
    if error_type is not None:
        attributes[otel.ERROR_TYPE] = error_type
    return frozenset(attributes.items())


async def _call(cache, method, use_async, *args, **kwargs):
    if use_async:
        return await getattr(cache, f"a{method}")(*args, **kwargs)
    return getattr(cache, method)(*args, **kwargs)


@pytest.mark.parametrize("use_async", [False, True])
async def test_a_check_records_a_hit_then_a_miss(cache, otel_sdk, use_async):
    spans, reader = otel_sdk
    cache, client = cache

    assert await _call(cache, "check", use_async, prompt="paris")

    empty = MagicMock()
    empty.data = []
    client.search.return_value = empty
    client.search_async.return_value = empty
    assert await _call(cache, "check", use_async, prompt="rome") == []

    assert [span.name for span in spans.get_finished_spans()] == [
        "semantic_cache.check",
        "semantic_cache.check",
    ]
    assert spans.get_finished_spans()[0].attributes[otel.CACHE_RESULT] == "hit"
    assert spans.get_finished_spans()[1].attributes[otel.CACHE_RESULT] == "miss"

    # Only the operation duration is recorded: no retrieval span above, and no
    # distance metric here, because the service does the search.
    data = reader.get_metrics_data()
    names = {
        metric.name
        for resource_metrics in (data.resource_metrics if data else [])
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    }
    assert names == {otel.SEMANTIC_CACHE_OPERATION_DURATION}
    assert _operations(reader) == {
        _attributes("check", "hit"): 1,
        _attributes("check", "miss"): 1,
    }


@pytest.mark.parametrize("use_async", [False, True])
async def test_a_failed_check_records_the_error_type(cache, otel_sdk, use_async):
    _, reader = otel_sdk
    cache, client = cache
    client.search.side_effect = RuntimeError("down")
    client.search_async.side_effect = RuntimeError("down")

    with pytest.raises(RuntimeError):
        await _call(cache, "check", use_async, prompt="paris")

    assert _operations(reader) == {_attributes("check", error_type="RuntimeError"): 1}


@pytest.mark.parametrize("use_async", [False, True])
async def test_store_and_delete_are_operations_too(cache, otel_sdk, use_async):
    _, reader = otel_sdk
    cache, client = cache
    drop_by_id = client.delete_by_id_async if use_async else client.delete_by_id
    drop_by_query = client.delete_query_async if use_async else client.delete_query

    assert await _call(cache, "store", use_async, "paris", "Paris") == "entry-1"
    await _call(cache, "delete_by_id", use_async, "entry-1")
    await _call(cache, "delete_by_attributes", use_async, {"city": "paris"})

    # One point per method, so a drop recorded by only one of them shows up as one.
    assert drop_by_id.call_count == 1
    assert drop_by_query.call_count == 1
    assert _operations(reader) == {
        _attributes("store"): 1,
        _attributes("drop"): 2,
    }
