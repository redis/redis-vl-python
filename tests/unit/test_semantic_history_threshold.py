"""SemanticMessageHistory applies its distance-threshold rule on every path.

The rule itself is tested in tests/unit/test_utils.py; this checks that the
constructor, ``set_distance_threshold`` and ``get_relevant`` all apply it.
Hermetic: an invalid threshold is refused before the history touches Redis or
its vectorizer.
"""

from unittest.mock import Mock

import pytest

from redisvl.extensions.message_history import SemanticMessageHistory
from redisvl.query.filter import Tag


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


def test_the_largest_threshold_widens_the_range_radius(history):
    # The radius is applied by Redis, which can report an exactly opposite
    # vector a little above 2, so 2 must not be sent as the radius itself.
    history.get_relevant("prompt", distance_threshold=2)

    (query,), _ = history._index.query.call_args
    assert query.distance_threshold > 2.001


def test_every_path_refuses_an_invalid_threshold(history, monkeypatch):
    # The constructor must refuse before it loads the default vectorizer.
    monkeypatch.setattr(
        "redisvl.extensions.message_history.semantic_history.HFTextVectorizer",
        Mock(side_effect=AssertionError("default vectorizer loaded")),
    )
    nan = float("nan")

    with pytest.raises(ValueError, match="between 0 and 2"):
        SemanticMessageHistory("history", distance_threshold=nan)
    with pytest.raises(ValueError, match="between 0 and 2"):
        history.set_distance_threshold(nan)
    with pytest.raises(ValueError, match="between 0 and 2"):
        history.get_relevant("prompt", distance_threshold=nan)
    # Checked before the top_k == 0 early return, not skipped by it.
    with pytest.raises(ValueError, match="between 0 and 2"):
        history.get_relevant("prompt", top_k=0, distance_threshold=nan)
    assert history.distance_threshold == 0.1
    history._vectorizer.embed.assert_not_called()
    history._index.query.assert_not_called()
