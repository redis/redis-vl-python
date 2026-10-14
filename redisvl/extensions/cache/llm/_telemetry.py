"""OpenTelemetry metrics for SemanticCache. Internal.

Attributes stay low-cardinality: the cache name, the operation, the lookup
result and the error type. Prompts, filters, entry ids and distances never
become metric attributes.
"""

import functools
import inspect
from time import perf_counter
from typing import Any, Callable, ParamSpec, TypeVar, cast

from redisvl.query import VectorQuery
from redisvl.utils import otel
from redisvl.utils.log import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def _result(hits: list[Any]) -> str:
    return "hit" if hits else "miss"


def _record_duration(
    cache: Any,
    operation: str,
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
        otel.histogram(otel.SEMANTIC_CACHE_OPERATION_DURATION).record(
            perf_counter() - start, attributes
        )
    except Exception:
        logger.debug("Could not record a SemanticCache metric", exc_info=True)


def timed(
    operation: str, *, lookup: bool = False
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Record the duration of a sync or async SemanticCache method.

    A lookup also records whether it hit, judged from the hits it returns. A
    failure records ``error.type`` instead and re-raises unchanged. A call that
    never completes, such as a cancelled task, records nothing.
    """

    def decorate(method: Callable[P, R]) -> Callable[P, R]:
        if inspect.iscoroutinefunction(method):

            @functools.wraps(method)
            async def async_wrapper(cache: Any, *args: Any, **kwargs: Any) -> Any:
                start = perf_counter()
                try:
                    result = await method(cache, *args, **kwargs)
                except Exception as error:
                    _record_duration(cache, operation, start, error=error)
                    raise
                _record_duration(
                    cache, operation, start, hits=result if lookup else None
                )
                return result

            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(method)
        def wrapper(cache: Any, *args: Any, **kwargs: Any) -> Any:
            start = perf_counter()
            try:
                result: Any = method(cache, *args, **kwargs)
            except Exception as error:
                _record_duration(cache, operation, start, error=error)
                raise
            _record_duration(cache, operation, start, hits=result if lookup else None)
            return result

        return cast(Callable[P, R], wrapper)

    return decorate


def record_nearest_distance(
    cache_name: str, candidates: list[dict[str, Any]], hits: list[Any]
) -> None:
    """Record the nearest candidate's distance, on a miss as well as a hit.

    Nothing is recorded when there was no candidate at all: an empty cache, or
    a filter that excluded every entry. Candidates arrive nearest first.
    """
    try:
        distance = next(
            (
                candidate[VectorQuery.DISTANCE_ID]
                for candidate in candidates
                if candidate.get(VectorQuery.DISTANCE_ID) is not None
            ),
            None,
        )
        if distance is not None:
            otel.histogram(otel.SEMANTIC_CACHE_DISTANCE).record(
                float(distance),
                {otel.CACHE_NAME: cache_name, otel.CACHE_RESULT: _result(hits)},
            )
    except Exception:
        logger.debug("Could not record a SemanticCache metric", exc_info=True)
