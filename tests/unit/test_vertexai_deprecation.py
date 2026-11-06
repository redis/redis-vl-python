"""Offline tests for the VertexAIVectorizer deprecation (issue #620).

These use a fake `vertexai` module so they run without google-cloud-aiplatform
or live GCP credentials.
"""

import sys
import types
import warnings

import pytest


@pytest.fixture
def fake_vertexai(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "proj")
    monkeypatch.setenv("GCP_LOCATION", "us-central1")

    vertexai_mod = types.ModuleType("vertexai")

    def init(**kwargs):
        return None

    vertexai_mod.init = init

    lang_mod = types.ModuleType("vertexai.language_models")

    class _FakeEmbedding:
        def __init__(self, values):
            self.values = values

    class FakeTextEmbeddingModel:
        @classmethod
        def from_pretrained(cls, model):
            return cls()

        def get_embeddings(self, contents, **kwargs):
            items = contents if isinstance(contents, list) else [contents]
            return [_FakeEmbedding([0.1, 0.2, 0.3]) for _ in items]

    lang_mod.TextEmbeddingModel = FakeTextEmbeddingModel
    vertexai_mod.language_models = lang_mod

    monkeypatch.setitem(sys.modules, "vertexai", vertexai_mod)
    monkeypatch.setitem(sys.modules, "vertexai.language_models", lang_mod)
    return vertexai_mod


def _deprecation_warnings(records):
    return [w for w in records if issubclass(w.category, DeprecationWarning)]


def test_vertexai_vectorizer_warns_once(fake_vertexai):
    from redisvl.utils.vectorize.vertexai import VertexAIVectorizer

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        VertexAIVectorizer(model="textembedding-gecko")

    dep = _deprecation_warnings(rec)
    assert len(dep) == 1
    assert "GoogleGenAIVectorizer" in str(dep[0].message)


def test_vertexai_text_vectorizer_warns_once(fake_vertexai):
    """The already-deprecated alias subclasses the now-deprecated parent; the
    idempotency sentinel keeps that to a single warning."""
    from redisvl.utils.vectorize.text.vertexai import VertexAITextVectorizer

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        VertexAITextVectorizer(model="textembedding-gecko")

    dep = _deprecation_warnings(rec)
    assert len(dep) == 1
    assert "GoogleGenAIVectorizer" in str(dep[0].message)
