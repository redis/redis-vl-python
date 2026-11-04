"""
The default SemanticCache vectorizer, and the warning for caches that predate it.

`redis/langcache-embed-v1` and `-v2` both emit 768-dimension vectors, so a v1
index accepts v2 query vectors and the schema check cannot catch the mismatch.
The warning is the only signal a caller gets, so these tests pin when it fires
and that it names the way to keep v1. The vectorizer is stubbed so no model is
downloaded, and the Redis client is a mock so no server is needed.
"""

import logging
from unittest.mock import ANY, MagicMock

import pytest
from redis import Redis

from redisvl.extensions.cache.llm import semantic
from redisvl.extensions.cache.llm.semantic import (
    DEFAULT_VECTORIZER_CHANGED_WARNING,
    SemanticCache,
)
from redisvl.index import SearchIndex
from redisvl.utils.vectorize import CustomVectorizer


@pytest.fixture
def default_models(monkeypatch):
    """Record the model each default vectorizer is built with."""
    models = []

    def fake_hf_vectorizer(model, **kwargs):
        models.append(model)
        return CustomVectorizer(lambda text: [0.1] * 768)

    monkeypatch.setattr(semantic, "HFTextVectorizer", fake_hf_vectorizer)
    return models


def _warned(caplog):
    return DEFAULT_VECTORIZER_CHANGED_WARNING in caplog.messages


def test_default_vectorizer_is_langcache_embed_v2(default_models):
    SemanticCache(redis_client=MagicMock(spec=Redis), create_index=False)

    assert default_models == ["redis/langcache-embed-v2"]


def test_warning_names_how_to_keep_v1():
    assert (
        "vectorizer=HFTextVectorizer(model='redis/langcache-embed-v1')"
        in DEFAULT_VECTORIZER_CHANGED_WARNING
    )


@pytest.mark.parametrize("explicit_vectorizer", [False, True])
def test_warns_on_existing_index_only_without_a_vectorizer(
    default_models, monkeypatch, caplog, explicit_vectorizer
):
    monkeypatch.setattr(SearchIndex, "exists", lambda self: True)
    monkeypatch.setattr(SearchIndex, "create", lambda self, **kwargs: None)
    # The live schema matches, exactly as a v1-era cache's would: same dims.
    existing = MagicMock()
    existing.schema.to_dict.return_value = ANY
    monkeypatch.setattr(SearchIndex, "from_existing", lambda name, **kw: existing)
    kwargs = {}
    if explicit_vectorizer:
        kwargs["vectorizer"] = CustomVectorizer(lambda text: [0.1] * 768)

    with caplog.at_level(logging.WARNING):
        SemanticCache(redis_client=MagicMock(spec=Redis), **kwargs)

    assert _warned(caplog) is not explicit_vectorizer


def test_no_warning_when_the_index_is_new(default_models, monkeypatch, caplog):
    monkeypatch.setattr(SearchIndex, "exists", lambda self: False)
    monkeypatch.setattr(SearchIndex, "create", lambda self, **kwargs: None)

    with caplog.at_level(logging.WARNING):
        SemanticCache(redis_client=MagicMock(spec=Redis))

    assert not _warned(caplog)


def test_warns_when_index_is_asserted_to_exist(default_models, caplog):
    with caplog.at_level(logging.WARNING):
        SemanticCache(redis_client=MagicMock(spec=Redis), create_index=False)

    assert _warned(caplog)
