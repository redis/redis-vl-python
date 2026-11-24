"""SemanticCache applies its distance-threshold rule on every path.

The rule itself is tested in tests/unit/test_utils.py; these check that the
constructor, ``set_threshold``, ``check`` and ``acheck`` all apply it.
Hermetic: Redis and the vectorizer are stubbed out.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from redisvl.extensions.cache.llm import SemanticCache


@pytest.fixture
def cache():
    # Bypass __init__, which would connect to Redis and load a vectorizer, and
    # stub what a lookup touches.
    cache = SemanticCache.__new__(SemanticCache)
    cache.name = "test_threshold"
    cache.refresh_ttl_on_hit = True
    cache._distance_threshold = 0.1
    cache._index = Mock(query=Mock(return_value=[]))
    cache._get_async_index = AsyncMock(
        return_value=Mock(query=AsyncMock(return_value=[]))
    )
    cache._check_vector_dims = Mock()
    cache._build_check_query = Mock()
    cache._expire_many = Mock()
    cache._aexpire_many = AsyncMock()
    return cache


async def test_every_path_refuses_an_invalid_threshold(cache, monkeypatch):
    # The constructor must refuse before it loads the default vectorizer.
    monkeypatch.setattr(
        "redisvl.extensions.cache.llm.semantic.HFTextVectorizer",
        Mock(side_effect=AssertionError("default vectorizer loaded")),
    )
    with pytest.raises(ValueError, match="between 0 and 2"):
        SemanticCache(distance_threshold=float("nan"))

    with pytest.raises(ValueError, match="between 0 and 2"):
        cache.set_threshold(float("nan"))
    with pytest.raises(ValueError, match="between 0 and 2"):
        cache.check(vector=[0.1], distance_threshold=float("nan"))
    with pytest.raises(ValueError, match="between 0 and 2"):
        await cache.acheck(vector=[0.1], distance_threshold=float("nan"))
    assert cache.distance_threshold == 0.1
