"""SemanticCache metrics, read back through an in-memory OpenTelemetry reader."""

import pytest

from redisvl.extensions.cache.llm import SemanticCache
from redisvl.query.filter import Tag
from redisvl.utils import otel
from redisvl.utils.vectorize import CustomVectorizer
from tests.conftest import skip_if_no_redis_search

# "near paris" is about 0.02 from "paris" in cosine distance; "rome" and
# "bread" are 1.0 from "paris", and from each other.
VECTORS = {
    "paris": [1.0, 0.0, 0.0, 0.0],
    "near paris": [0.98, 0.2, 0.0, 0.0],
    "bread": [0.0, 1.0, 0.0, 0.0],
    "rome": [0.0, 0.0, 0.0, 1.0],
}


@pytest.fixture
def cache(client, redis_url, worker_id):
    skip_if_no_redis_search(client)
    cache = SemanticCache(
        name=f"metrics_cache_{worker_id}",
        vectorizer=CustomVectorizer(
            embed=lambda text: VECTORS.get(text, [0.0, 0.0, 1.0, 0.0])
        ),
        distance_threshold=0.1,
        filterable_fields=[{"name": "label", "type": "tag"}],
        redis_url=redis_url,
    )
    yield cache
    cache.delete()


def _points(reader, name):
    """Map each point of one histogram, by its attributes, to ``(count, sum)``,
    checking the unit and bucket advice every point carries."""
    unit, _, boundaries = otel._HISTOGRAMS[name]
    data = reader.get_metrics_data()
    points = {}
    for resource_metrics in data.resource_metrics if data else []:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == name:
                    assert metric.unit == unit
                    for point in metric.data.data_points:
                        assert tuple(point.explicit_bounds) == boundaries
                        points[frozenset(point.attributes.items())] = (
                            point.count,
                            point.sum,
                        )
    return points


async def _call(cache, method, use_async, *args, **kwargs):
    if use_async:
        return await getattr(cache, f"a{method}")(*args, **kwargs)
    return getattr(cache, method)(*args, **kwargs)


@pytest.mark.parametrize("use_async", [False, True])
async def test_operation_durations_carry_only_their_attributes(
    cache, otel_sdk, use_async
):
    _, reader = otel_sdk
    key = await _call(cache, "store", use_async, "paris", "Paris")
    await _call(cache, "check", use_async, "near paris")
    await _call(cache, "check", use_async, "bread")
    await _call(cache, "update", use_async, key, response="Paris, France")
    with pytest.raises(TypeError, match="Return fields must be a list"):
        await _call(cache, "check", use_async, "paris", return_fields="prompt")
    await _call(cache, "drop", use_async, keys=[key])

    points = _points(reader, otel.SEMANTIC_CACHE_OPERATION_DURATION)
    name = (otel.CACHE_NAME, cache.name)
    assert {attributes: count for attributes, (count, _) in points.items()} == {
        frozenset({name, (otel.CACHE_OPERATION, "store")}): 1,
        frozenset(
            {name, (otel.CACHE_OPERATION, "check"), (otel.CACHE_RESULT, "hit")}
        ): 1,
        frozenset(
            {name, (otel.CACHE_OPERATION, "check"), (otel.CACHE_RESULT, "miss")}
        ): 1,
        frozenset({name, (otel.CACHE_OPERATION, "update")}): 1,
        frozenset(
            {name, (otel.CACHE_OPERATION, "check"), (otel.ERROR_TYPE, "TypeError")}
        ): 1,
        frozenset({name, (otel.CACHE_OPERATION, "drop")}): 1,
    }


@pytest.mark.parametrize("use_async", [False, True])
async def test_nearest_distance_is_recorded_whenever_there_is_a_candidate(
    cache, otel_sdk, use_async
):
    _, reader = otel_sdk
    await _call(cache, "check", use_async, "paris")  # empty cache: no candidate
    for prompt in ("paris", "rome"):
        await _call(
            cache, "store", use_async, prompt, prompt, filters={"label": "city"}
        )
    await _call(cache, "check", use_async, "near paris", num_results=2)
    await _call(cache, "check", use_async, "bread")
    no_match = Tag("label") == "food"
    await _call(cache, "check", use_async, "paris", filter_expression=no_match)

    points = _points(reader, otel.SEMANTIC_CACHE_DISTANCE)
    name = (otel.CACHE_NAME, cache.name)
    hit = points.pop(frozenset({name, (otel.CACHE_RESULT, "hit")}))
    miss = points.pop(frozenset({name, (otel.CACHE_RESULT, "miss")}))
    assert points == {}
    assert hit == (1, pytest.approx(0.02, abs=0.005))
    assert miss == (1, pytest.approx(1.0, abs=1e-6))
