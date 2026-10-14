"""OpenTelemetry spans and metrics for SemanticCache. Internal.

Each operation gets a ``semantic_cache.{operation}`` span and a duration point.
A lookup's search gets a ``retrieval {index}`` child span carrying the
threshold, ``top_k`` and the nearest distance. The embeddings span comes from
the vectorizer.

Metric attributes stay low-cardinality: the cache name, the operation, the
lookup result and the error type. Prompt text and retrieved documents are
recorded only on the retrieval span, and only when content capture is enabled.
"""

import functools
import inspect
import json
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Callable, Iterator, ParamSpec, TypeVar, cast

from redisvl.extensions.constants import ENTRY_ID_FIELD_NAME
from redisvl.query import VectorQuery
from redisvl.utils import otel
from redisvl.utils.log import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def _result(hits: list[Any]) -> str:
    return "hit" if hits else "miss"


def _finish(
    cache: Any,
    operation: str,
    span: Any,
    start: float,
    hits: list[Any] | None = None,
    error: Exception | None = None,
) -> None:
    # Telemetry must never change what the caller sees, so a failure here
    # (an old opentelemetry-api installed by another package, say) is dropped.
    try:
        attributes = {otel.CACHE_NAME: cache.name, otel.CACHE_OPERATION: operation}
        if error is not None:
            attributes[otel.ERROR_TYPE] = otel.error_type(error)
        elif hits is not None:
            attributes[otel.CACHE_RESULT] = _result(hits)
            span.set_attribute(otel.CACHE_RESULT, attributes[otel.CACHE_RESULT])
        otel.histogram(otel.SEMANTIC_CACHE_OPERATION_DURATION).record(
            perf_counter() - start, attributes
        )
    except Exception:
        logger.debug("Could not record RedisVL telemetry", exc_info=True)


def operation(
    name: str, *, lookup: bool = False
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Trace and time a sync or async SemanticCache method.

    A lookup also records whether it hit, judged from the hits it returns. A
    failure records ``error.type`` instead and re-raises unchanged. A call that never completes, such as a cancelled task, records
    no duration.
    """
    span_name = f"semantic_cache.{name}"

    def decorate(method: Callable[P, R]) -> Callable[P, R]:
        if inspect.iscoroutinefunction(method):

            @functools.wraps(method)
            async def async_wrapper(cache: Any, *args: Any, **kwargs: Any) -> Any:
                start = perf_counter()
                with otel.start_span(span_name, {otel.CACHE_NAME: cache.name}) as span:
                    try:
                        result = await method(cache, *args, **kwargs)
                    except Exception as error:
                        _finish(cache, name, span, start, error=error)
                        raise
                    _finish(cache, name, span, start, result if lookup else None)
                    return result

            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(method)
        def wrapper(cache: Any, *args: Any, **kwargs: Any) -> Any:
            start = perf_counter()
            with otel.start_span(span_name, {otel.CACHE_NAME: cache.name}) as span:
                try:
                    result: Any = method(cache, *args, **kwargs)
                except Exception as error:
                    _finish(cache, name, span, start, error=error)
                    raise
                _finish(cache, name, span, start, result if lookup else None)
                return result

        return cast(Callable[P, R], wrapper)

    return decorate


@contextmanager
def retrieval(index_name: str, top_k: int, distance_threshold: float) -> Iterator[Any]:
    """Trace a lookup's vector search as a ``retrieval {index}`` client span."""
    attributes = {
        otel.GEN_AI_OPERATION_NAME: "retrieval",
        otel.GEN_AI_DATA_SOURCE_ID: index_name,
        otel.GEN_AI_RETRIEVAL_TOP_K: top_k,
        otel.DB_SYSTEM_NAME: "redis",
        otel.DB_COLLECTION_NAME: index_name,
        otel.CACHE_DISTANCE_THRESHOLD: distance_threshold,
    }
    with otel.start_span(
        f"retrieval {index_name}", attributes, otel.SpanKind.CLIENT
    ) as span:
        yield span


def record_lookup(
    span: Any,
    cache_name: str,
    candidates: list[dict[str, Any]],
    hits: list[Any],
    prompt: str | None,
) -> None:
    """Record what a lookup found, on a miss as well as a hit.

    The nearest candidate's distance goes on the retrieval span and the
    distance histogram. Nothing is recorded for it when there was no candidate
    at all: an empty cache, or a filter that excluded every entry. Candidates
    arrive nearest first.

    With content capture on, each document's ``score`` is its raw vector
    distance, so lower is closer, unlike the relevance score the GenAI
    conventions describe. No single conversion suits every distance metric.
    """
    try:
        scored = [
            candidate
            for candidate in candidates
            if candidate.get(VectorQuery.DISTANCE_ID) is not None
        ]
        if scored:
            distance = float(scored[0][VectorQuery.DISTANCE_ID])
            span.set_attribute(otel.CACHE_DISTANCE, distance)
            otel.histogram(otel.SEMANTIC_CACHE_DISTANCE).record(
                distance,
                {otel.CACHE_NAME: cache_name, otel.CACHE_RESULT: _result(hits)},
            )
        if otel.capture_content() and span.is_recording():
            if prompt is not None:
                span.set_attribute(otel.GEN_AI_RETRIEVAL_QUERY_TEXT, prompt)
            documents = [
                {
                    "id": candidate.get(ENTRY_ID_FIELD_NAME),
                    "score": float(candidate[VectorQuery.DISTANCE_ID]),
                }
                for candidate in scored
            ]
            span.set_attribute(otel.GEN_AI_RETRIEVAL_DOCUMENTS, json.dumps(documents))
    except Exception:
        logger.debug("Could not record RedisVL telemetry", exc_info=True)
