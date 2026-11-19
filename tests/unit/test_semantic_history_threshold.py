"""SemanticMessageHistory applies one distance-threshold rule everywhere.

The history-wide threshold (``__init__`` and ``set_distance_threshold``) and a
per-call override (``get_relevant``) accept the same values and refuse the rest
with the same exception, the rule ``SemanticCache`` uses. Hermetic: an invalid
threshold is refused before the history touches Redis or its vectorizer.
"""

import re
from unittest.mock import Mock

import numpy as np
import pytest

from redisvl.extensions.message_history import SemanticMessageHistory
from redisvl.query.filter import Tag

ACCEPTED = [0, 2, np.float32(0.5)]

REFUSED = [
    pytest.param(2 + 1e-9, ValueError, id="just-above-2"),
    pytest.param(-1e-9, ValueError, id="just-below-0"),
    pytest.param(float("nan"), ValueError, id="nan"),
    pytest.param(True, TypeError, id="true"),
    pytest.param(False, TypeError, id="false"),
    pytest.param("0.5", TypeError, id="numeric-string"),
]


@pytest.fixture
def history():
    # Bypass __init__, which would connect to Redis and load a vectorizer, and
    # stub what a lookup touches so that it returns no messages.
    history = SemanticMessageHistory.__new__(SemanticMessageHistory)
    history._distance_threshold = 0.1
    history._default_session_filter = Tag("session_tag") == "s1"
    history._vectorizer = Mock(embed=Mock(return_value=[0.1]), dtype="float32")
    history._index = Mock(query=Mock(return_value=[]))
    return history


@pytest.mark.parametrize("value", ACCEPTED)
def test_every_path_accepts_the_closed_range(history, value):
    history.get_relevant("prompt", distance_threshold=value)
    history.set_distance_threshold(value)

    (query,), _ = history._index.query.call_args
    assert query.distance_threshold == float(value)
    assert history.distance_threshold == float(value)
    assert type(history.distance_threshold) is float


@pytest.mark.parametrize("value, error", REFUSED)
def test_every_path_refuses_alike(history, monkeypatch, value, error):
    # The constructor must refuse before it loads the default vectorizer.
    monkeypatch.setattr(
        "redisvl.extensions.message_history.semantic_history.HFTextVectorizer",
        Mock(side_effect=AssertionError("default vectorizer loaded")),
    )

    with pytest.raises(error) as expected:
        history.set_distance_threshold(value)
    match = re.escape(str(expected.value))

    with pytest.raises(error, match=match):
        history.get_relevant("prompt", distance_threshold=value)
    with pytest.raises(error, match=match):
        history.get_relevant("prompt", top_k=0, distance_threshold=value)
    with pytest.raises(error, match=match):
        SemanticMessageHistory("history", distance_threshold=value)
    assert history.distance_threshold == 0.1
    history._vectorizer.embed.assert_not_called()
    history._index.query.assert_not_called()
