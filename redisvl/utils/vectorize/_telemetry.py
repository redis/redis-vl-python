"""OpenTelemetry spans and metrics for vectorizers. Internal.

Every ``embed`` call gets an ``embeddings {model}`` client span following the
GenAI conventions, which covers every vectorizer, and every RedisVL component
that embeds, from one place. ``gen_ai.client.operation.duration`` is recorded
only for calls that reach the provider, not for embeddings-cache hits.
"""

from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator

from redisvl.utils import otel
from redisvl.utils.log import get_logger

logger = get_logger(__name__)

# Vectorizer types with a registered ``gen_ai.provider.name``. Others, such as
# ``hf``, ``ollama`` and ``voyageai``, report their own type.
_PROVIDERS = {
    "openai": "openai",
    "azure_openai": "azure.ai.openai",
    "cohere": "cohere",
    "mistral": "mistral_ai",
    "vertexai": "gcp.vertex_ai",
    # The generic Google value, since this vectorizer serves both the Vertex AI
    # and Gemini Developer API backends.
    "google_genai": "gcp.gen_ai",
    "bedrock": "aws.bedrock",
}


class _Embeddings:
    """The handle an embeddings span yields to the code it wraps."""

    def __init__(self, span: Any, attributes: dict[str, Any]) -> None:
        self._span = span
        self._attributes = attributes

    def cache_result(self, hits: int, total: int) -> None:
        """Record how many of ``total`` items the embeddings cache served."""
        result = "hit" if hits == total else "miss" if hits == 0 else "partial"
        try:
            self._span.set_attribute(otel.EMBEDDINGS_CACHE_RESULT, result)
        except Exception:
            logger.debug("Could not record RedisVL telemetry", exc_info=True)

    @contextmanager
    def provider_call(self) -> Iterator[None]:
        """Time a call to the embedding provider."""
        start = perf_counter()
        attributes = self._attributes
        try:
            yield
        except Exception as error:
            attributes = {**attributes, otel.ERROR_TYPE: otel.error_type(error)}
            raise
        finally:
            try:
                otel.histogram(otel.GEN_AI_CLIENT_OPERATION_DURATION).record(
                    perf_counter() - start, attributes
                )
            except Exception:
                logger.debug("Could not record RedisVL telemetry", exc_info=True)


@contextmanager
def embeddings(vectorizer: Any) -> Iterator[_Embeddings]:
    """Trace one embed call as an ``embeddings {model}`` client span."""
    metric_attributes = {
        otel.GEN_AI_OPERATION_NAME: "embeddings",
        otel.GEN_AI_PROVIDER_NAME: _PROVIDERS.get(vectorizer.type, vectorizer.type),
        otel.GEN_AI_REQUEST_MODEL: vectorizer.model,
    }
    span_attributes = dict(metric_attributes)
    if vectorizer.dims:
        span_attributes[otel.GEN_AI_EMBEDDINGS_DIMENSION_COUNT] = vectorizer.dims
    with otel.start_span(
        f"embeddings {vectorizer.model}", span_attributes, otel.SpanKind.CLIENT
    ) as span:
        yield _Embeddings(span, metric_attributes)
