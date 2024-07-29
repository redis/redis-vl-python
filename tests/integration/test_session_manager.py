import json
import time

import pytest

from redisvl.extensions.session_manager import (
    SemanticSessionManager,
    StandardSessionManager,
)
from redisvl.query.filter import Tag


@pytest.fixture
def standard_session(app_name):
    session = StandardSessionManager(app_name)
    yield session
    session.clear()


@pytest.fixture
def semantic_session(app_name):
    session = SemanticSessionManager(app_name)
    yield session
    session.clear()
    session.delete()


# test standard session manager
def test_specify_redis_client(client):
    session = StandardSessionManager(name="test_app", redis_client=client)
    assert isinstance(session._client, type(client))


def test_standard_store(standard_session):
    context = standard_session.get_recent()
    assert len(context) == 0

    standard_session.store(prompt="first prompt", response="first response")
    standard_session.store(prompt="second prompt", response="second response")
    standard_session.store(prompt="third prompt", response="third response")
    standard_session.store(prompt="fourth prompt", response="fourth response")
    standard_session.store(prompt="fifth prompt", response="fifth response")

    # test that order is maintained
    full_context = standard_session.get_recent(top_k=10)
    assert full_context == [
        {"role": "user", "content": "first prompt"},
        {"role": "llm", "content": "first response"},
        {"role": "user", "content": "second prompt"},
        {"role": "llm", "content": "second response"},
        {"role": "user", "content": "third prompt"},
        {"role": "llm", "content": "third response"},
        {"role": "user", "content": "fourth prompt"},
        {"role": "llm", "content": "fourth response"},
        {"role": "user", "content": "fifth prompt"},
        {"role": "llm", "content": "fifth response"},
    ]


def test_standard_add_and_get(standard_session):
    context = standard_session.get_recent()
    assert len(context) == 0

    standard_session.add_message({"role": "user", "content": "first prompt"})
    standard_session.add_message({"role": "llm", "content": "first response"})
    standard_session.add_message({"role": "user", "content": "second prompt"})
    standard_session.add_message({"role": "llm", "content": "second response"})
    standard_session.add_message(
        {
            "role": "tool",
            "content": "tool result 1",
            "tool_call_id": "tool call one",
        }
    )
    standard_session.add_message(
        {
            "role": "tool",
            "content": "tool result 2",
            "tool_call_id": "tool call two",
        }
    )
    standard_session.add_message({"role": "user", "content": "third prompt"})
    standard_session.add_message({"role": "llm", "content": "third response"})

    # test default context history size
    default_context = standard_session.get_recent()
    assert len(default_context) == 5  #  default is 5

    # test specified context history size
    partial_context = standard_session.get_recent(top_k=3)
    assert len(partial_context) == 3
    assert partial_context == [
        {"role": "tool", "content": "tool result 2", "tool_call_id": "tool call two"},
        {"role": "user", "content": "third prompt"},
        {"role": "llm", "content": "third response"},
    ]

    # test that order is maintained
    full_context = standard_session.get_recent(top_k=10)
    assert full_context == [
        {"role": "user", "content": "first prompt"},
        {"role": "llm", "content": "first response"},
        {"role": "user", "content": "second prompt"},
        {"role": "llm", "content": "second response"},
        {"role": "tool", "content": "tool result 1", "tool_call_id": "tool call one"},
        {"role": "tool", "content": "tool result 2", "tool_call_id": "tool call two"},
        {"role": "user", "content": "third prompt"},
        {"role": "llm", "content": "third response"},
    ]

    # test that a ValueError is raised when top_k is invalid
    with pytest.raises(ValueError):
        bad_context = standard_session.get_recent(top_k=-2)

    with pytest.raises(ValueError):
        bad_context = standard_session.get_recent(top_k=-2.0)

    with pytest.raises(ValueError):
        bad_context = standard_session.get_recent(top_k=1.3)

    with pytest.raises(ValueError):
        bad_context = standard_session.get_recent(top_k="3")


def test_standard_add_messages(standard_session):
    context = standard_session.get_recent()
    assert len(context) == 0

    standard_session.add_messages(
        [
            {"role": "user", "content": "first prompt"},
            {"role": "llm", "content": "first response"},
            {"role": "user", "content": "second prompt"},
            {"role": "llm", "content": "second response"},
            {
                "role": "tool",
                "content": "tool result 1",
                "tool_call_id": "tool call one",
            },
            {
                "role": "tool",
                "content": "tool resuilt 2",
                "tool_call_id": "tool call two",
            },
            {"role": "user", "content": "fourth prompt"},
            {"role": "llm", "content": "fourth response"},
        ]
    )

    full_context = standard_session.get_recent(top_k=10)
    assert full_context == [
        {"role": "user", "content": "first prompt"},
        {"role": "llm", "content": "first response"},
        {"role": "user", "content": "second prompt"},
        {"role": "llm", "content": "second response"},
        {"role": "tool", "content": "tool result 1", "tool_call_id": "tool call one"},
        {"role": "tool", "content": "tool resuilt 2", "tool_call_id": "tool call two"},
        {"role": "user", "content": "fourth prompt"},
        {"role": "llm", "content": "fourth response"},
    ]


def test_standard_messages_property(standard_session):
    standard_session.add_messages(
        [
            {"role": "user", "content": "first prompt"},
            {"role": "llm", "content": "first response"},
            {"role": "user", "content": "second prompt"},
            {"role": "llm", "content": "second response"},
            {"role": "user", "content": "third prompt"},
        ]
    )

    assert standard_session.messages == [
        {"role": "user", "content": "first prompt"},
        {"role": "llm", "content": "first response"},
        {"role": "user", "content": "second prompt"},
        {"role": "llm", "content": "second response"},
        {"role": "user", "content": "third prompt"},
    ]


def test_standard_scope(standard_session):
    # store entries under default session tag
    standard_session.store("some prompt", "some response")

    # test that changing session tag does indeed change access scope
    new_session = "def"
    standard_session.store(
        "new user prompt", "new user response", session_tag=new_session
    )
    session_filter = Tag("session_tag") == new_session
    context = standard_session.get_recent(tag_filter=session_filter)
    assert context == [
        {"role": "user", "content": "new user prompt"},
        {"role": "llm", "content": "new user response"},
    ]

    # test that default session data is still accessible
    context = standard_session.get_recent()
    assert context == [
        {"role": "user", "content": "some prompt"},
        {"role": "llm", "content": "some response"},
    ]

    bad_session = "xyz"
    bad_filter = Tag("session_tag") == bad_session
    no_context = standard_session.get_recent(tag_filter=bad_filter)
    assert no_context == []


def test_standard_get_text(standard_session):
    standard_session.store("first prompt", "first response")
    text = standard_session.get_recent(as_text=True)
    assert text == ["first prompt", "first response"]

    standard_session.add_message({"role": "system", "content": "system level prompt"})
    text = standard_session.get_recent(as_text=True)
    assert text == ["first prompt", "first response", "system level prompt"]


def test_standard_get_raw(standard_session):
    current_time = int(time.time())
    standard_session.store("first prompt", "first response")
    standard_session.store("second prompt", "second response")
    raw = standard_session.get_recent(raw=True)
    assert len(raw) == 4
    assert raw[0]["role"] == "user"
    assert raw[0]["content"] == "first prompt"
    assert current_time <= float(raw[0]["timestamp"]) <= time.time()
    assert raw[1]["role"] == "llm"
    assert raw[1]["content"] == "first response"


def test_standard_drop(standard_session):
    standard_session.store("first prompt", "first response")
    standard_session.store("second prompt", "second response")
    standard_session.store("third prompt", "third response")
    standard_session.store("fourth prompt", "fourth response")

    # test drop() with no arguments removes the last element
    standard_session.drop()
    context = standard_session.get_recent(top_k=3)
    assert context == [
        {"role": "user", "content": "third prompt"},
        {"role": "llm", "content": "third response"},
        {"role": "user", "content": "fourth prompt"},
    ]

    # test drop(id) removes the specified element
    context = standard_session.get_recent(top_k=10, raw=True)
    middle_id = context[3]["id_field"]
    standard_session.drop(middle_id)
    context = standard_session.get_recent(top_k=6)
    assert context == [
        {"role": "user", "content": "first prompt"},
        {"role": "llm", "content": "first response"},
        {"role": "user", "content": "second prompt"},
        {"role": "user", "content": "third prompt"},
        {"role": "llm", "content": "third response"},
        {"role": "user", "content": "fourth prompt"},
    ]


def test_standard_clear(standard_session):
    standard_session.store("some prompt", "some response")
    standard_session.clear()
    empty_context = standard_session.get_recent(top_k=10)
    assert empty_context == []


# test semantic session manager
def test_semantic_specify_client(client):
    session = SemanticSessionManager(
        name="test_app", session_tag="abc", redis_client=client
    )
    assert isinstance(session._index.client, type(client))


def test_semantic_scope(semantic_session):
    # store entries under default session tag
    semantic_session.store("some prompt", "some response")

    # test that changing session tag does indeed change access scope
    new_session = "def"
    semantic_session.store(
        "new user prompt", "new user response", session_tag=new_session
    )
    session_filter = Tag("session_tag") == new_session
    context = semantic_session.get_recent(tag_filter=session_filter)
    assert context == [
        {"role": "user", "content": "new user prompt"},
        {"role": "llm", "content": "new user response"},
    ]

    # test that previous session data is still accessible
    context = semantic_session.get_recent()
    assert context == [
        {"role": "user", "content": "some prompt"},
        {"role": "llm", "content": "some response"},
    ]

    bad_session = "xyz"
    bad_filter = Tag("session_tag") == bad_session
    no_context = semantic_session.get_recent(tag_filter=bad_filter)
    assert no_context == []


def test_semantic_store_and_get_recent(semantic_session):
    context = semantic_session.get_recent()
    assert len(context) == 0

    semantic_session.store(prompt="first prompt", response="first response")
    semantic_session.store(prompt="second prompt", response="second response")
    semantic_session.store(prompt="third prompt", response="third response")
    semantic_session.store(prompt="fourth prompt", response="fourth response")
    semantic_session.add_message(
        {"role": "tool", "content": "tool result", "tool_call_id": "tool id"}
    )
    # test default context history size
    default_context = semantic_session.get_recent()
    assert len(default_context) == 5  # 5 is default

    # test specified context history size
    partial_context = semantic_session.get_recent(top_k=4)
    assert len(partial_context) == 4

    # test larger context history returns full history
    too_large_context = semantic_session.get_recent(top_k=100)
    assert len(too_large_context) == 9

    # test that order is maintained
    full_context = semantic_session.get_recent(top_k=9)
    assert full_context == [
        {"role": "user", "content": "first prompt"},
        {"role": "llm", "content": "first response"},
        {"role": "user", "content": "second prompt"},
        {"role": "llm", "content": "second response"},
        {"role": "user", "content": "third prompt"},
        {"role": "llm", "content": "third response"},
        {"role": "user", "content": "fourth prompt"},
        {"role": "llm", "content": "fourth response"},
        {"role": "tool", "content": "tool result", "tool_call_id": "tool id"},
    ]

    # test that more recent entries are returned
    context = semantic_session.get_recent(top_k=4)
    assert context == [
        {"role": "llm", "content": "third response"},
        {"role": "user", "content": "fourth prompt"},
        {"role": "llm", "content": "fourth response"},
        {"role": "tool", "content": "tool result", "tool_call_id": "tool id"},
    ]

    # test no entries are returned and no error is raised if top_k == 0
    context = semantic_session.get_recent(top_k=0)
    assert context == []

    # test that a ValueError is raised when top_k is invalid
    with pytest.raises(ValueError):
        bad_context = semantic_session.get_recent(top_k=0.5)

    with pytest.raises(ValueError):
        bad_context = semantic_session.get_recent(top_k=-1)

    with pytest.raises(ValueError):
        bad_context = semantic_session.get_recent(top_k=-2.0)

    with pytest.raises(ValueError):
        bad_context = semantic_session.get_recent(top_k=1.3)

    with pytest.raises(ValueError):
        bad_context = semantic_session.get_recent(top_k="3")


def test_semantic_messages_property(semantic_session):
    semantic_session.add_messages(
        [
            {"role": "user", "content": "first prompt"},
            {"role": "llm", "content": "first response"},
            {
                "role": "tool",
                "content": "tool result 1",
                "tool_call_id": "tool call one",
            },
            {
                "role": "tool",
                "content": "tool result 2",
                "tool_call_id": "tool call two",
            },
            {"role": "user", "content": "second prompt"},
            {"role": "llm", "content": "second response"},
            {"role": "user", "content": "third prompt"},
        ]
    )

    assert semantic_session.messages == [
        {"role": "user", "content": "first prompt"},
        {"role": "llm", "content": "first response"},
        {"role": "tool", "content": "tool result 1", "tool_call_id": "tool call one"},
        {"role": "tool", "content": "tool result 2", "tool_call_id": "tool call two"},
        {"role": "user", "content": "second prompt"},
        {"role": "llm", "content": "second response"},
        {"role": "user", "content": "third prompt"},
    ]


def test_semantic_add_and_get_relevant(semantic_session):
    semantic_session.add_message(
        {"role": "system", "content": "discussing common fruits and vegetables"}
    )
    semantic_session.store(
        prompt="list of common fruits",
        response="apples, oranges, bananas, strawberries",
    )
    semantic_session.store(
        prompt="list of common vegetables",
        response="carrots, broccoli, onions, spinach",
    )
    semantic_session.store(
        prompt="winter sports in the olympics",
        response="downhill skiing, ice skating, luge",
    )
    semantic_session.add_message(
        {
            "role": "tool",
            "content": "skiing, skating, luge",
            "tool_call_id": "winter_sports()",
        }
    )

    # test default distance metric
    default_context = semantic_session.get_relevant(
        "set of common fruits like apples and bananas"
    )
    assert len(default_context) == 2
    assert default_context[0] == {"role": "user", "content": "list of common fruits"}
    assert default_context[1] == {
        "role": "llm",
        "content": "apples, oranges, bananas, strawberries",
    }

    # test increasing distance metric broadens results
    semantic_session.set_distance_threshold(0.5)
    default_context = semantic_session.get_relevant("list of fruits and vegetables")
    assert len(default_context) == 5  # 2 pairs of prompt:response, and system

    # test tool calls can also be returned
    context = semantic_session.get_relevant("winter sports like skiing")
    assert context == [
        {
            "role": "user",
            "content": "winter sports in the olympics",
        },
        {
            "role": "tool",
            "content": "skiing, skating, luge",
            "tool_call_id": "winter_sports()",
        },
        {
            "role": "llm",
            "content": "downhill skiing, ice skating, luge",
        },
    ]

    # test that a ValueError is raised when top_k is invalid
    with pytest.raises(ValueError):
        bad_context = semantic_session.get_relevant("test prompt", top_k=-1)

    with pytest.raises(ValueError):
        bad_context = semantic_session.get_relevant("test prompt", top_k=-2.0)

    with pytest.raises(ValueError):
        bad_context = semantic_session.get_relevant("test prompt", top_k=1.3)

    with pytest.raises(ValueError):
        bad_context = semantic_session.get_relevant("test prompt", top_k="3")


def test_semantic_get_raw(semantic_session):
    current_time = int(time.time())
    semantic_session.store("first prompt", "first response")
    semantic_session.store("second prompt", "second response")
    raw = semantic_session.get_recent(raw=True)
    assert len(raw) == 4
    assert raw[0]["content"] == "first prompt"
    assert raw[1]["content"] == "first response"
    assert current_time <= float(raw[0]["timestamp"]) <= time.time()


def test_semantic_drop(semantic_session):
    semantic_session.store("first prompt", "first response")
    semantic_session.store("second prompt", "second response")
    semantic_session.store("third prompt", "third response")
    semantic_session.store("fourth prompt", "fourth response")

    # test drop() with no arguments removes the last element
    semantic_session.drop()
    context = semantic_session.get_recent(top_k=3)
    assert context == [
        {"role": "user", "content": "third prompt"},
        {"role": "llm", "content": "third response"},
        {"role": "user", "content": "fourth prompt"},
    ]

    # test drop(id) removes the specified element
    context = semantic_session.get_recent(top_k=5, raw=True)
    middle_id = context[2]["id_field"]
    semantic_session.drop(middle_id)
    context = semantic_session.get_recent(top_k=4)
    assert context == [
        {"role": "user", "content": "second prompt"},
        {"role": "llm", "content": "second response"},
        {"role": "llm", "content": "third response"},
        {"role": "user", "content": "fourth prompt"},
    ]
