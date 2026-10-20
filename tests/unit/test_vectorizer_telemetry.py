"""Vectorizer telemetry, checked without an SDK or a provider."""

import asyncio
from types import SimpleNamespace

import pytest

from redisvl.utils import otel
from redisvl.utils.vectorize import _telemetry


def test_only_completed_provider_calls_record_a_duration(monkeypatch):
    recorded = []
    histogram = SimpleNamespace(
        record=lambda value, attributes: recorded.append(attributes)
    )
    monkeypatch.setattr(otel, "histogram", lambda name: histogram)
    call = _telemetry._Embeddings(span=None, attributes={})

    with call.provider_call():
        pass
    with pytest.raises(KeyError):
        with call.provider_call():
            raise KeyError
    with pytest.raises(asyncio.CancelledError):
        with call.provider_call():
            raise asyncio.CancelledError

    assert recorded == [{}, {otel.ERROR_TYPE: "KeyError"}]


@pytest.mark.parametrize(
    "backend, provider",
    [("vertex", "gcp.vertex_ai"), ("gemini", "gcp.gemini")],
)
def test_google_genai_reports_the_backend_it_resolved(backend, provider):
    vectorizer = SimpleNamespace(
        type="google_genai", model="gemini-embedding-001", dims=None, backend=backend
    )

    with _telemetry.embeddings(vectorizer) as call:
        assert call._attributes[otel.GEN_AI_PROVIDER_NAME] == provider
