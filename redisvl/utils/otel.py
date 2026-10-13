"""OpenTelemetry plumbing for RedisVL's own instrumentation. Internal.

RedisVL depends only on the OpenTelemetry API, through the optional
``redisvl[otel]`` extra; the application owns the SDK and exporters. The tracer
and meter come from the global providers, so they do nothing until the
application configures an SDK. When the API is not installed, or
``REDISVL_OTEL_DISABLED`` is ``true``, stand-ins with the same interface are
returned instead, so instrumented code never needs to check.

The stand-in span leaves the active context alone, so write to a span through
the handle from ``start_as_current_span(...) as span``, never through
``trace.get_current_span()``, which would reach the caller's span instead.

RedisVL never constructs a provider and never initialises redis-py's own
observability; both belong to the application.
"""

import os
from contextlib import ContextDecorator
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from redisvl.version import __version__

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import SpanKind
except ImportError:  # the redisvl[otel] extra is not installed
    metrics = trace = None  # type: ignore[assignment]

    class SpanKind(Enum):  # type: ignore[no-redef]
        """Stand-in for ``opentelemetry.trace.SpanKind``."""

        INTERNAL = 0
        SERVER = 1
        CLIENT = 2
        PRODUCER = 3
        CONSUMER = 4


if TYPE_CHECKING:
    from opentelemetry.metrics import Meter
    from opentelemetry.trace import Tracer

INSTRUMENTATION_SCOPE = "redisvl"
# The semantic-conventions release the standard names below follow. Bump the
# two together.
SCHEMA_URL = "https://opentelemetry.io/schemas/1.44.0"
# Read once, on first use. Same convention as OTEL_SDK_DISABLED: only "true",
# in any case, disables.
DISABLED_ENV_VAR = "REDISVL_OTEL_DISABLED"

SEMANTIC_CACHE_OPERATION_DURATION = "redisvl.semantic_cache.operation.duration"
SEMANTIC_CACHE_DISTANCE = "redisvl.semantic_cache.distance"
GEN_AI_CLIENT_OPERATION_DURATION = "gen_ai.client.operation.duration"

# Metric and attribute names. ``redisvl.*`` names are RedisVL's own; the rest come from the
# OpenTelemetry semantic conventions, where ``gen_ai.*`` is still in
# Development and may be renamed upstream.
CACHE_NAME = "redisvl.cache.name"
CACHE_OPERATION = "redisvl.cache.operation"
CACHE_RESULT = "redisvl.cache.result"
CACHE_DISTANCE = "redisvl.cache.distance"
CACHE_DISTANCE_THRESHOLD = "redisvl.cache.distance_threshold"
CACHE_NUM_RESULTS = "redisvl.cache.num_results"
EMBEDDINGS_CACHE_RESULT = "redisvl.embeddings_cache.result"
ERROR_TYPE = "error.type"
DB_SYSTEM_NAME = "db.system.name"
DB_COLLECTION_NAME = "db.collection.name"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_EMBEDDINGS_DIMENSION_COUNT = "gen_ai.embeddings.dimension.count"
GEN_AI_DATA_SOURCE_ID = "gen_ai.data_source.id"
GEN_AI_RETRIEVAL_TOP_K = "gen_ai.retrieval.top_k"


class _NoOpSpan(ContextDecorator):
    """A span that records nothing, usable as its own context manager or
    decorator, like ``start_as_current_span``'s result."""

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        return None

    def is_recording(self) -> bool:
        return False

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: Any) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, *args: Any, **kwargs: Any) -> None:
        pass


class _NoOpTracer:
    _span = _NoOpSpan()

    def start_as_current_span(self, *args: Any, **kwargs: Any) -> _NoOpSpan:
        return self._span


class _NoOpHistogram:
    def record(self, *args: Any, **kwargs: Any) -> None:
        pass


class _NoOpMeter:
    _histogram = _NoOpHistogram()

    def create_histogram(self, *args: Any, **kwargs: Any) -> _NoOpHistogram:
        return self._histogram


def _enabled() -> bool:
    disabled = os.environ.get(DISABLED_ENV_VAR, "").strip().lower() == "true"
    return trace is not None and not disabled


@lru_cache(maxsize=None)
def get_tracer() -> "Tracer":
    """Return RedisVL's tracer, or a no-op stand-in."""
    if not _enabled():
        return _NoOpTracer()  # type: ignore[return-value]
    return trace.get_tracer(INSTRUMENTATION_SCOPE, __version__, schema_url=SCHEMA_URL)


@lru_cache(maxsize=None)
def get_meter() -> "Meter":
    """Return RedisVL's meter, or a no-op stand-in."""
    if not _enabled():
        return _NoOpMeter()  # type: ignore[return-value]
    return metrics.get_meter(INSTRUMENTATION_SCOPE, __version__, schema_url=SCHEMA_URL)
