"""Unit tests for __repr__ implementations on core RedisVL classes."""

from unittest.mock import MagicMock, patch

import pytest

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.schema import IndexSchema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_schema(name="my-index", prefix="rvl", storage_type="hash"):
    return IndexSchema.from_dict(
        {"index": {"name": name, "prefix": prefix, "storage_type": storage_type}}
    )


def _make_mock_vectorizer(dims=768, dtype="float32"):
    mock_vec = MagicMock()
    mock_vec.dims = dims
    mock_vec.dtype = dtype
    return mock_vec


# ---------------------------------------------------------------------------
# SearchIndex / AsyncSearchIndex
# ---------------------------------------------------------------------------


def test_search_index_repr_hash():
    schema = _make_schema(name="test-index", prefix="rvl", storage_type="hash")
    index = SearchIndex(schema=schema)
    assert (
        repr(index)
        == "SearchIndex(name='test-index', prefix='rvl', storage_type='hash')"
    )


def test_search_index_repr_json():
    schema = _make_schema(name="docs", prefix="doc", storage_type="json")
    index = SearchIndex(schema=schema)
    assert repr(index) == "SearchIndex(name='docs', prefix='doc', storage_type='json')"


def test_async_search_index_repr():
    schema = _make_schema(name="async-idx", prefix="data", storage_type="json")
    index = AsyncSearchIndex(schema=schema)
    assert (
        repr(index)
        == "AsyncSearchIndex(name='async-idx', prefix='data', storage_type='json')"
    )


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------


@pytest.fixture()
def patched_semantic_cache():
    """Patch out Redis and HFTextVectorizer so SemanticCache can be instantiated
    without a running Redis server or ML model download."""
    mock_vec = _make_mock_vectorizer()
    mock_idx = MagicMock()
    mock_idx.exists.return_value = False

    with (
        patch(
            "redisvl.extensions.cache.llm.semantic.HFTextVectorizer",
            return_value=mock_vec,
        ),
        patch(
            "redisvl.extensions.cache.llm.semantic.SearchIndex",
            return_value=mock_idx,
        ),
    ):
        yield


def test_semantic_cache_repr_defaults(patched_semantic_cache):
    from redisvl.extensions.cache.llm.semantic import SemanticCache

    cache = SemanticCache(name="llmcache", distance_threshold=0.1)
    assert (
        repr(cache)
        == "SemanticCache(name='llmcache', distance_threshold=0.1, ttl=None)"
    )


def test_semantic_cache_repr_with_ttl(patched_semantic_cache):
    from redisvl.extensions.cache.llm.semantic import SemanticCache

    cache = SemanticCache(name="my-cache", distance_threshold=0.2, ttl=300)
    assert (
        repr(cache) == "SemanticCache(name='my-cache', distance_threshold=0.2, ttl=300)"
    )


# ---------------------------------------------------------------------------
# SemanticRouter
# ---------------------------------------------------------------------------


def test_semantic_router_repr():
    from redisvl.extensions.router.schema import Route
    from redisvl.extensions.router.semantic import SemanticRouter

    routes = [
        Route(name="greeting", references=["hello", "hi"]),
        Route(name="farewell", references=["bye", "goodbye"]),
    ]
    # model_construct bypasses __init__ (and therefore Redis/vectorizer setup)
    # while still setting the Pydantic fields that __repr__ reads.
    router = SemanticRouter.model_construct(name="my-router", routes=routes)
    assert repr(router) == "SemanticRouter(name='my-router', routes=2)"


def test_semantic_router_repr_single_route():
    from redisvl.extensions.router.schema import Route
    from redisvl.extensions.router.semantic import SemanticRouter

    routes = [Route(name="support", references=["help", "issue"])]
    router = SemanticRouter.model_construct(name="support-router", routes=routes)
    assert repr(router) == "SemanticRouter(name='support-router', routes=1)"


# ---------------------------------------------------------------------------
# MessageHistory
# ---------------------------------------------------------------------------


@pytest.fixture()
def patched_message_history():
    """Patch SearchIndex.create so MessageHistory init doesn't need Redis."""
    with patch("redisvl.extensions.message_history.message_history.SearchIndex.create"):
        yield


def test_message_history_repr(patched_message_history):
    from redisvl.extensions.message_history.message_history import MessageHistory

    mh = MessageHistory(name="chat", session_tag="abc123")
    assert repr(mh) == "MessageHistory(name='chat', session_tag='abc123')"


def test_message_history_repr_custom_name(patched_message_history):
    from redisvl.extensions.message_history.message_history import MessageHistory

    mh = MessageHistory(name="support-chat", session_tag="sess-001")
    assert repr(mh) == "MessageHistory(name='support-chat', session_tag='sess-001')"


# ---------------------------------------------------------------------------
# SemanticMessageHistory
# ---------------------------------------------------------------------------


@pytest.fixture()
def patched_semantic_message_history():
    """Patch out Redis and HFTextVectorizer for SemanticMessageHistory."""
    mock_vec = _make_mock_vectorizer(dims=384)
    mock_idx = MagicMock()
    mock_idx.exists.return_value = False

    with (
        patch(
            "redisvl.extensions.message_history.semantic_history.HFTextVectorizer",
            return_value=mock_vec,
        ),
        patch(
            "redisvl.extensions.message_history.semantic_history.SearchIndex",
            return_value=mock_idx,
        ),
    ):
        yield


def test_semantic_message_history_repr(patched_semantic_message_history):
    from redisvl.extensions.message_history.semantic_history import (
        SemanticMessageHistory,
    )

    smh = SemanticMessageHistory(
        name="sem-chat", session_tag="sess-42", distance_threshold=0.3
    )
    assert (
        repr(smh)
        == "SemanticMessageHistory(name='sem-chat', session_tag='sess-42', distance_threshold=0.3)"
    )


def test_semantic_message_history_repr_custom_threshold(
    patched_semantic_message_history,
):
    from redisvl.extensions.message_history.semantic_history import (
        SemanticMessageHistory,
    )

    smh = SemanticMessageHistory(
        name="history", session_tag="s1", distance_threshold=0.5
    )
    assert (
        repr(smh)
        == "SemanticMessageHistory(name='history', session_tag='s1', distance_threshold=0.5)"
    )
