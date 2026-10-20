"""SemanticCache and vectorizer spans, read back through an in-memory exporter."""

import json

import pytest

from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.extensions.message_history import SemanticMessageHistory
from redisvl.extensions.router import Route, SemanticRouter
from redisvl.utils import otel
from redisvl.utils.vectorize import CustomVectorizer
from tests.conftest import skip_if_no_redis_search

# "near paris" is about 0.02 from "paris" in cosine distance.
VECTORS = {
    "paris": [1.0, 0.0, 0.0, 0.0],
    "near paris": [0.98, 0.2, 0.0, 0.0],
}


def _vectorizer(**kwargs):
    return CustomVectorizer(
        embed=lambda text: VECTORS.get(text, [0.0, 0.0, 1.0, 0.0]), **kwargs
    )


@pytest.fixture
def cache(client, redis_url, worker_id):
    skip_if_no_redis_search(client)
    cache = SemanticCache(
        name=f"tracing_cache_{worker_id}",
        vectorizer=_vectorizer(),
        distance_threshold=0.1,
        redis_url=redis_url,
    )
    yield cache
    cache.delete()


async def _call(cache, method, use_async, *args, **kwargs):
    if use_async:
        return await getattr(cache, f"a{method}")(*args, **kwargs)
    return getattr(cache, method)(*args, **kwargs)


def _spans(exporter):
    """Map each finished span's name to ``(parent name, kind, attributes)``."""
    finished = exporter.get_finished_spans()
    names = {span.context.span_id: span.name for span in finished}
    tree = {
        span.name: (
            names.get(span.parent.span_id) if span.parent else None,
            span.kind.name,
            dict(span.attributes),
        )
        for span in finished
    }
    assert len(tree) == len(finished), "duplicate span names"
    return tree


def _recorded(exporter):
    """Everything a span could leak text through, as one string."""
    return json.dumps(
        [
            [
                span.name,
                span.status.description,
                dict(span.attributes),
                [[event.name, dict(event.attributes)] for event in span.events],
            ]
            for span in exporter.get_finished_spans()
        ]
    )


def _points(reader, name):
    return [
        point
        for resource_metrics in reader.get_metrics_data().resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]


@pytest.mark.parametrize("use_async", [False, True])
async def test_lookup_produces_check_embeddings_and_retrieval_spans(
    cache, otel_sdk, use_async
):
    spans, _ = otel_sdk
    await _call(cache, "store", use_async, "paris", "Paris")
    spans.clear()

    await _call(cache, "check", use_async, "near paris")

    index = cache.index.name
    tree = _spans(spans)
    assert tree.keys() == {
        "semantic_cache.check",
        "embeddings custom",
        f"retrieval {index}",
    }
    assert tree["semantic_cache.check"] == (
        None,
        "INTERNAL",
        {otel.CACHE_NAME: cache.name, otel.CACHE_RESULT: "hit"},
    )
    assert tree["embeddings custom"] == (
        "semantic_cache.check",
        "CLIENT",
        {
            otel.GEN_AI_OPERATION_NAME: "embeddings",
            otel.GEN_AI_PROVIDER_NAME: "custom",
            otel.GEN_AI_REQUEST_MODEL: "custom",
            otel.GEN_AI_EMBEDDINGS_DIMENSION_COUNT: 4,
        },
    )
    parent, kind, attributes = tree[f"retrieval {index}"]
    assert (parent, kind) == ("semantic_cache.check", "CLIENT")
    assert attributes.pop(otel.CACHE_DISTANCE) == pytest.approx(0.02, abs=0.005)
    assert attributes == {
        otel.GEN_AI_OPERATION_NAME: "retrieval",
        otel.GEN_AI_DATA_SOURCE_ID: index,
        otel.GEN_AI_RETRIEVAL_TOP_K: 1,
        otel.DB_SYSTEM_NAME: "redis",
        otel.DB_COLLECTION_NAME: index,
        otel.CACHE_DISTANCE_THRESHOLD: 0.1,
    }


def test_a_precomputed_vector_skips_the_embeddings_span(cache, otel_sdk):
    spans, _ = otel_sdk
    cache.check(vector=VECTORS["paris"])
    assert _spans(spans).keys() == {
        "semantic_cache.check",
        f"retrieval {cache.index.name}",
    }


def test_a_failing_provider_marks_its_spans_without_recording_the_message(
    client, redis_url, worker_id, otel_sdk
):
    skip_if_no_redis_search(client)
    spans, reader = otel_sdk

    def embed(text):
        if text == "dimension test":  # CustomVectorizer's dimension probe
            return [1.0, 0.0, 0.0, 0.0]
        raise ValueError(f"provider rejected {text!r}")

    cache = SemanticCache(
        name=f"tracing_failing_{worker_id}",
        vectorizer=CustomVectorizer(embed=embed),
        redis_url=redis_url,
    )
    try:
        with pytest.raises(ValueError, match="paris"):
            cache.check("paris")
    finally:
        cache.delete()

    failed = [
        span
        for span in spans.get_finished_spans()
        if span.name in ("semantic_cache.check", "embeddings custom")
    ]
    assert len(failed) == 2
    for span in failed:
        assert span.status.status_code.name == "ERROR"
        assert span.status.description is None
        assert span.attributes[otel.ERROR_TYPE] == "ValueError"
        assert span.events == ()
    assert "paris" not in _recorded(spans)
    (point,) = _points(reader, otel.GEN_AI_CLIENT_OPERATION_DURATION)
    assert point.attributes[otel.ERROR_TYPE] == "ValueError"


@pytest.mark.parametrize("use_async", [False, True])
async def test_an_embeddings_cache_hit_skips_the_provider(
    client, redis_url, worker_id, otel_sdk, use_async
):
    skip_if_no_redis_search(client)
    spans, reader = otel_sdk
    embeddings_cache = EmbeddingsCache(
        name=f"tracing_embeddings_{worker_id}", redis_url=redis_url
    )
    vectorizer = _vectorizer(cache=embeddings_cache)
    try:
        await _call(vectorizer, "embed", use_async, "paris")
        await _call(vectorizer, "embed", use_async, "paris")
        await _call(vectorizer, "embed_many", use_async, ["paris", "rome"])
    finally:
        embeddings_cache.clear()

    assert [
        span.attributes[otel.EMBEDDINGS_CACHE_RESULT]
        for span in spans.get_finished_spans()
    ] == ["miss", "hit", "partial"]
    (point,) = _points(reader, otel.GEN_AI_CLIENT_OPERATION_DURATION)
    assert point.count == 2


def test_router_and_message_history_lookups_produce_embeddings_spans(
    client, redis_url, worker_id, otel_sdk
):
    skip_if_no_redis_search(client)
    spans, _ = otel_sdk
    router = SemanticRouter(
        name=f"tracing_router_{worker_id}",
        routes=[Route(name="city", references=["paris"])],
        vectorizer=_vectorizer(),
        redis_url=redis_url,
        overwrite=True,
    )
    history = SemanticMessageHistory(
        name=f"tracing_history_{worker_id}",
        vectorizer=_vectorizer(),
        redis_url=redis_url,
        overwrite=True,
    )
    try:
        spans.clear()
        router("near paris")
        history.get_relevant("near paris")
    finally:
        router.delete()
        history.delete()

    assert [span.name for span in spans.get_finished_spans()] == [
        "embeddings custom",
        "embeddings custom",
    ]


def test_no_prompt_or_response_text_is_recorded_by_default(
    cache, otel_sdk, monkeypatch
):
    spans, _ = otel_sdk
    monkeypatch.delenv(otel.CAPTURE_CONTENT_ENV_VAR, raising=False)
    cache.store("paris", "a secret response")
    cache.check("paris")

    recorded = _recorded(spans)
    assert "paris" not in recorded
    assert "secret" not in recorded


def test_span_content_capture_records_the_query_and_documents(
    cache, otel_sdk, monkeypatch
):
    spans, _ = otel_sdk
    monkeypatch.setenv(otel.CAPTURE_CONTENT_ENV_VAR, "SPAN_ONLY")
    cache.store("paris", "Paris")
    (hit,) = cache.check("near paris")

    (span,) = [
        span for span in spans.get_finished_spans() if span.name.startswith("retrieval")
    ]
    assert span.attributes[otel.GEN_AI_RETRIEVAL_QUERY_TEXT] == "near paris"
    (document,) = json.loads(span.attributes[otel.GEN_AI_RETRIEVAL_DOCUMENTS])
    assert document["id"] == hit["entry_id"]
    assert document["score"] == pytest.approx(0.02, abs=0.005)
