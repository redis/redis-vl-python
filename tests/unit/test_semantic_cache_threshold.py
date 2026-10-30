"""SemanticCache applies one distance-threshold rule everywhere.

The cache-wide threshold (``__init__`` and ``set_threshold``) and a per-call
override (``check`` and ``acheck``) accept the same values and refuse the rest
with the same exception. Hermetic: an invalid override is refused before the
cache touches Redis or its vectorizer.
"""

import re
from unittest.mock import AsyncMock

import numpy as np
import pytest

from redisvl.extensions.cache.llm import SemanticCache

ACCEPTED = [0, 0.0, 2, 2.0, 0.5, np.float32(0.5)]

REFUSED = [
    pytest.param(2 + 1e-9, ValueError, id="just-above-2"),
    pytest.param(-1e-9, ValueError, id="just-below-0"),
    pytest.param(float("nan"), ValueError, id="nan"),
    pytest.param(True, TypeError, id="true"),
    pytest.param(False, TypeError, id="false"),
    pytest.param("0.5", TypeError, id="numeric-string"),
]


@pytest.fixture
def cache():
    # Bypass __init__, which would connect to Redis and load a vectorizer.
    cache = SemanticCache.__new__(SemanticCache)
    cache._distance_threshold = 0.1
    cache._get_async_index = AsyncMock()
    return cache


@pytest.mark.parametrize("value", ACCEPTED)
def test_set_threshold_accepts_the_closed_range(cache, value):
    cache.set_threshold(value)

    assert cache.distance_threshold == float(value)
    assert type(cache.distance_threshold) is float


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
