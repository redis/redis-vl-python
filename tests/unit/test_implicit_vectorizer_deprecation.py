"""
Constructing SemanticCache, SemanticMessageHistory or SemanticRouter without a
vectorizer is deprecated.

The fallback model is baked into every stored vector, so these tests pin that
it stays unchanged while deprecated, and that the warning names it so callers
can pass it explicitly. The vectorizer is stubbed so no model is downloaded,
and `create_index=False` with a mock client keeps Redis out of it.
"""

import warnings
from unittest.mock import MagicMock

import pytest
from redis import Redis

from redisvl.extensions.cache.llm import semantic as cache_module
from redisvl.extensions.cache.llm.semantic import SemanticCache
from redisvl.extensions.message_history import semantic_history as history_module
from redisvl.extensions.message_history.semantic_history import SemanticMessageHistory
from redisvl.extensions.router import semantic as router_module
from redisvl.extensions.router.schema import Route
from redisvl.extensions.router.semantic import SemanticRouter
from redisvl.utils.vectorize import CustomVectorizer

ROUTE = Route(name="greeting", references=["hello"], distance_threshold=0.5)

CASES = {
    "SemanticCache": (cache_module, "redis/langcache-embed-v1", SemanticCache, {}),
    "SemanticMessageHistory": (
        history_module,
        "sentence-transformers/all-mpnet-base-v2",
        SemanticMessageHistory,
        {"name": "history"},
    ),
    "SemanticRouter": (
        router_module,
        "sentence-transformers/all-mpnet-base-v2",
        SemanticRouter,
        {"name": "router", "routes": [ROUTE]},
    ),
}


def _stub_vectorizer():
    return CustomVectorizer(lambda text: [0.1, 0.2, 0.3])


@pytest.fixture(params=list(CASES))
def case(request, monkeypatch):
    module, model, cls, kwargs = CASES[request.param]
    built_with = []

    def fake_hf_vectorizer(model, **kwargs):
        built_with.append(model)
        return _stub_vectorizer()

    monkeypatch.setattr(module, "HFTextVectorizer", fake_hf_vectorizer)

    def build(**extra):
        return cls(
            redis_client=MagicMock(spec=Redis),
            create_index=False,
            **kwargs,
            **extra,
        )

    return request.param, model, build, built_with


def _deprecations(caught):
    return [w for w in caught if issubclass(w.category, DeprecationWarning)]


def test_omitting_the_vectorizer_warns_and_keeps_the_fallback(case):
    owner, model, build, built_with = case

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build()

    assert built_with == [model]
    [warning] = _deprecations(caught)
    message = str(warning.message)
    assert owner in message
    assert f"vectorizer=HFTextVectorizer(model='{model}')" in message
    # Attributed to the caller, not to RedisVL's own frames.
    assert warning.filename == __file__


def test_an_explicit_vectorizer_does_not_warn(case):
    _, _, build, built_with = case

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build(vectorizer=_stub_vectorizer())

    assert built_with == []
    assert _deprecations(caught) == []
