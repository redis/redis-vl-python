"""SemanticCache applies one distance-threshold rule everywhere.

The cache-wide threshold (``__init__`` and ``set_threshold``) and a per-call
override (``check`` and ``acheck``) accept the same values and refuse the rest
with the same exception. Hermetic: an invalid override is refused before the
cache touches Redis or its vectorizer.
"""

import re
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from redisvl.extensions.cache.llm import SemanticCache

# The 0 and 2 boundaries run end to end in tests/integration/test_llmcache.py.
ACCEPTED = [np.float32(0.5)]

REFUSED = [
    pytest.param(2 + 1e-9, ValueError, id="just-above-2"),
    pytest.param(-1e-9, ValueError, id="just-below-0"),
    pytest.param(float("nan"), ValueError, id="nan"),
    pytest.param(True, TypeError, id="true"),
    pytest.param("0.5", TypeError, id="numeric-string"),
]


@pytest.fixture
def cache():
    # Bypass __init__, which would connect to Redis and load a vectorizer, and
    # stub what a lookup touches so that it returns no candidates.
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


@pytest.mark.parametrize("value", ACCEPTED)
async def test_every_path_accepts_a_numpy_scalar_as_a_float(cache, value):
    applied = []
    cache._within_threshold = (
        lambda candidates, threshold: applied.append(threshold) or candidates
    )

    cache.check(vector=[0.1], distance_threshold=value)
    await cache.acheck(vector=[0.1], distance_threshold=value)
    cache.set_threshold(value)

    assert applied == [float(value)] * 2
    assert cache.distance_threshold == float(value)
    # A numpy scalar must not reach the client-side filter or the span attribute.
    assert all(type(t) is float for t in [*applied, cache.distance_threshold])


def test_the_largest_threshold_keeps_a_rounded_opposite_vector():
    # Measured on Redis 8.10.3: FLOAT32 COSINE, 1536 dims, query against -query.
    candidates = [{"vector_distance": "2.00000023842"}]

    assert SemanticCache._within_threshold(candidates, 2.0) == candidates


@pytest.mark.parametrize("value, error", REFUSED)
async def test_every_path_refuses_alike(cache, value, error):
    with pytest.raises(error) as expected:
        cache.set_threshold(value)
    match = re.escape(str(expected.value))

    with pytest.raises(error, match=match):
        cache.check(vector=[0.1], distance_threshold=value)
    with pytest.raises(error, match=match):
        await cache.acheck(vector=[0.1], distance_threshold=value)
    assert cache.distance_threshold == 0.1
