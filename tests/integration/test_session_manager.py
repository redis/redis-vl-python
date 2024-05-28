import json
import time

import pytest

from redisvl.extensions.session_manager import (
    SemanticSessionManager,
    StandardSessionManager,
)


@pytest.fixture
def standard_session(app_name, user_id, session_id):
    session = StandardSessionManager(app_name, session_id=session_id, user_id=user_id)
    yield session
    session.clear()
    session.delete()


@pytest.fixture
def semantic_session(app_name, user_id, session_id):
    session = SemanticSessionManager(app_name, session_id=session_id, user_id=user_id)
    yield session
    session.clear()
    session.delete()


# test standard session manager
def test_include_preamble():
    # test default key creation
    session = StandardSessionManager(name="test_app", session_id="123", user_id="abc")
    assert session.key == "test_app:abc:123"

    # test initializing and changing preamble
    session = StandardSessionManager(name="test_app", session_id="123", user_id="abc")
    assert session._preamble == {"role": "_preamble", "_content": ""}

    preamble = "system level instruction to llm."
    session = StandardSessionManager(
        name="test_app", session_id="123", user_id="abc", preamble=preamble
    )
    assert session._preamble == {"role": "_preamble", "_content": preamble}

    new_preamble = "new llm instruction."
    session.set_preamble(new_preamble)
    assert session._preamble == {"role": "_preamble", "_content": new_preamble}


def test_specify_redis_client(client):
    session = StandardSessionManager(
        name="test_app", session_id="abc", user_id="123", redis_client=client
    )
    assert isinstance(session._client, type(client))


def test_standard_store_and_fetch(standard_session):
    context = standard_session.fetch_recent()
    assert len(context) == 1  # preamle still present

    standard_session.store(prompt="first prompt", response="first response")
    standard_session.store(prompt="second prompt", response="second response")
    standard_session.store(prompt="third prompt", response="third response")
    standard_session.store(prompt="fourth prompt", response="fourth response")
    standard_session.store(prompt="fifth prompt", response="fifth response")

    # test default context history size
    default_context = standard_session.fetch_recent()
    assert len(default_context) == 7  # 3 pairs of prompt:response, and preamble

    # test specified context history size
    partial_context = standard_session.fetch_recent(top_k=2)
    assert len(partial_context) == 5  # 4 pairs of prompt:response, and preamble
    assert partial_context == [
        {"role": "_preamble", "_content": ""},
        {"role": "_user", "_content": "fourth prompt"},
        {"role": "_llm", "_content": "fourth response"},
        {"role": "_user", "_content": "fifth prompt"},
        {"role": "_llm", "_content": "fifth response"},
    ]
    # test larger context history returns full history
    too_large_context = standard_session.fetch_recent(top_k=10)
    assert len(too_large_context) == 11

    # test that no context is returned when top_k is zero
    no_context = standard_session.fetch_recent(top_k=0)
    assert len(no_context) == 1  # preamble is still present

    # test that the full context is returned when top_k is -1
    full_context = standard_session.fetch_recent(top_k=-1)
    assert len(full_context) == 11

    # test that order is maintained
    assert full_context == [
        {"role": "_preamble", "_content": ""},
        {"role": "_user", "_content": "first prompt"},
        {"role": "_llm", "_content": "first response"},
        {"role": "_user", "_content": "second prompt"},
        {"role": "_llm", "_content": "second response"},
        {"role": "_user", "_content": "third prompt"},
        {"role": "_llm", "_content": "third response"},
        {"role": "_user", "_content": "fourth prompt"},
        {"role": "_llm", "_content": "fourth response"},
        {"role": "_user", "_content": "fifth prompt"},
        {"role": "_llm", "_content": "fifth response"},
    ]


def test_standard_set_scope(standard_session, app_name, user_id, session_id):
    # test calling set_scope with no params does not change scope
    current_key = standard_session.key
    standard_session.set_scope()
    assert standard_session.key == current_key

    # test passing either user_id or session_id only changes corresponding value
    new_user = "def"
    standard_session.set_scope(user_id=new_user)
    assert standard_session.key == f"{app_name}:{new_user}:{session_id}"

    new_session = "456"
    standard_session.set_scope(session_id=new_session)
    assert standard_session.key == f"{app_name}:{new_user}:{new_session}"

    # test that changing user and session id does indeed change access scope
    standard_session.store("new user prompt", "new user response")

    standard_session.set_scope(session_id="789", user_id="ghi")
    no_context = standard_session.fetch_recent()
    assert no_context == [{"role": "_preamble", "_content": ""}]

    # change scope back to read previously stored entries
    standard_session.set_scope(session_id="456", user_id="def")
    previous_context = standard_session.fetch_recent()
    assert previous_context == [
        {"role": "_preamble", "_content": ""},
        {"role": "_user", "_content": "new user prompt"},
        {"role": "_llm", "_content": "new user response"},
    ]


def test_standard_fetch_recent_with_scope(standard_session, session_id):
    # test that passing user or session id to fetch_recent(...) changes scope
    standard_session.store("first prompt", "first response")

    context = standard_session.fetch_recent()
    assert context == [
        {"role": "_preamble", "_content": ""},
        {"role": "_user", "_content": "first prompt"},
        {"role": "_llm", "_content": "first response"},
    ]

    context = standard_session.fetch_recent(session_id="456")
    assert context == [{"role": "_preamble", "_content": ""}]

    # test that scope change persists after being updated via fetch_recent(...)
    standard_session.store("new session prompt", "new session response")
    context = standard_session.fetch_recent()
    assert context == [
        {"role": "_preamble", "_content": ""},
        {"role": "_user", "_content": "new session prompt"},
        {"role": "_llm", "_content": "new session response"},
    ]

    # clean up lingering sessions
    standard_session.clear()
    standard_session.set_scope(session_id=session_id)


def test_standard_fetch_text(standard_session):
    standard_session.store("first prompt", "first response")
    text = standard_session.fetch_recent(as_text=True)
    assert text == ["", "first prompt", "first response"]


def test_standard_fetch_raw(standard_session):
    current_time = int(time.time())
    standard_session.store("first prompt", "first response")
    standard_session.store("second prompt", "second response")
    raw = standard_session.fetch_recent(raw=True)
    assert len(raw) == 2
    assert raw[0].keys() == {
        "id_field",
        "prompt",
        "response",
        "timestamp",
        "token_count",
    }
    assert raw[0]["prompt"] == "first prompt"
    assert raw[0]["response"] == "first response"
    assert current_time <= raw[0]["timestamp"] <= time.time()
    assert raw[0]["token_count"] == 1


def test_standard_drop(standard_session):
    standard_session.store("first prompt", "first response")
    standard_session.store("second prompt", "second response")
    standard_session.store("third prompt", "third response")
    standard_session.store("fourth prompt", "fourth response")

    # test drop() with no arguments removes the last element
    standard_session.drop()
    context = standard_session.fetch_recent(top_k=1)
    assert context == [
        {"role": "_preamble", "_content": ""},
        {"role": "_user", "_content": "third prompt"},
        {"role": "_llm", "_content": "third response"},
    ]

    # test drop(timestamp) removes the specified element
    context = standard_session.fetch_recent(top_k=-1, raw=True)
    middle_id = context[1]["id_field"]
    standard_session.drop(middle_id)
    context = standard_session.fetch_recent(top_k=2)
    assert context == [
        {"role": "_preamble", "_content": ""},
        {"role": "_user", "_content": "first prompt"},
        {"role": "_llm", "_content": "first response"},
        {"role": "_user", "_content": "third prompt"},
        {"role": "_llm", "_content": "third response"},
    ]


def test_standard_clear(standard_session):
    standard_session.store("some prompt", "some response")
    standard_session.clear()
    empty_context = standard_session.fetch_recent(top_k=-1)
    assert empty_context == [{"role": "_preamble", "_content": ""}]


def test_standard_delete(standard_session):
    standard_session.store("some prompt", "some response")
    standard_session.delete()
    empty_context = standard_session.fetch_recent(top_k=-1)
    assert empty_context == [{"role": "_preamble", "_content": ""}]


# test semantic session manager

##def test_semantic_name_prefix():
##    assert False


def test_semantic_include_preamble():
    # test initializing and changing preamble
    session = SemanticSessionManager(name="test_app", session_id="123", user_id="abc")
    assert session._preamble == {"role": "_preamble", "_content": ""}

    preamble = "system level instruction to llm."
    session = SemanticSessionManager(
        name="test_app", session_id="123", user_id="abc", preamble=preamble
    )
    assert session._preamble == {"role": "_preamble", "_content": preamble}

    new_preamble = "new llm instruction."
    session.set_preamble(new_preamble)
    assert session._preamble == {"role": "_preamble", "_content": new_preamble}


def test_semantic_specify_client(client):
    session = SemanticSessionManager(
        name="test_app", session_id="abc", user_id="123", redis_client=client
    )
    assert isinstance(session._client, type(client))


def test_semantic_set_scope(semantic_session, app_name, user_id, session_id):
    # test calling set_scope with no params does not change scope
    semantic_session.store("some prompt", "some response")
    semantic_session.set_scope()
    context = semantic_session.fetch_recent()
    assert context == [
        {"role": "_preamble", "_content": ""},
        {"role": "_user", "_content": "some prompt"},
        {"role": "_llm", "_content": "some response"},
    ]

    # test that changing user and session id does indeed change access scope
    new_user = "def"
    semantic_session.set_scope(user_id=new_user)
    semantic_session.store("new user prompt", "new user response")
    context = semantic_session.fetch_recent()
    assert context == [
        {"role": "_preamble", "_content": ""},
        {"role": "_user", "_content": "new user prompt"},
        {"role": "_llm", "_content": "new user response"},
    ]

    # test that previous user and session data is still accessible
    previous_user = "abc"
    semantic_session.set_scope(user_id=previous_user)
    context = semantic_session.fetch_recent()
    assert context == [
        {"role": "_preamble", "_content": ""},
        {"role": "_user", "_content": "some prompt"},
        {"role": "_llm", "_content": "some response"},
    ]

    semantic_session.set_scope(session_id="789", user_id="ghi")
    no_context = semantic_session.fetch_recent()
    assert no_context == [{"role": "_preamble", "_content": ""}]


def test_semantic_store_and_fetch_recent(semantic_session):
    context = semantic_session.fetch_recent()
    assert len(context) == 1  # preamle still present

    semantic_session.store(prompt="first prompt", response="first response")
    semantic_session.store(prompt="second prompt", response="second response")
    semantic_session.store(prompt="third prompt", response="third response")
    semantic_session.store(prompt="fourth prompt", response="fourth response")
    semantic_session.store(prompt="fifth prompt", response="fifth response")

    # test default context history size
    default_context = semantic_session.fetch_recent()
    assert len(default_context) == 7  # 3 pairs of prompt:response, and preamble

    # test specified context history size
    partial_context = semantic_session.fetch_recent(top_k=4)
    assert len(partial_context) == 9  # 4 pairs of prompt:response, and preamble

    # test larger context history returns full history
    too_large_context = semantic_session.fetch_recent(top_k=10)
    assert len(too_large_context) == 11

    # test that no context is returned when top_k is zero
    no_context = semantic_session.fetch_recent(top_k=0)
    assert len(no_context) == 1  # preamble is still present

    # test that order is maintained
    full_context = semantic_session.fetch_recent(top_k=10)
    assert full_context == [
        {"role": "_preamble", "_content": ""},
        {"role": "_user", "_content": "first prompt"},
        {"role": "_llm", "_content": "first response"},
        {"role": "_user", "_content": "second prompt"},
        {"role": "_llm", "_content": "second response"},
        {"role": "_user", "_content": "third prompt"},
        {"role": "_llm", "_content": "third response"},
        {"role": "_user", "_content": "fourth prompt"},
        {"role": "_llm", "_content": "fourth response"},
        {"role": "_user", "_content": "fifth prompt"},
        {"role": "_llm", "_content": "fifth response"},
    ]

    # test that more recent entries are returned
    context = semantic_session.fetch_recent(top_k=3)
    assert context == [
        {"role": "_preamble", "_content": ""},
        {"role": "_user", "_content": "third prompt"},
        {"role": "_llm", "_content": "third response"},
        {"role": "_user", "_content": "fourth prompt"},
        {"role": "_llm", "_content": "fourth response"},
        {"role": "_user", "_content": "fifth prompt"},
        {"role": "_llm", "_content": "fifth response"},
    ]  # FAILING


def test_semantic_store_and_fetch_relevant(semantic_session):
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

    # test default distance metric
    default_context = semantic_session.fetch_relevant(
        "set of common fruits like apples and bananas"
    )
    assert len(default_context) == 3
    assert default_context[1] == {"role": "_user", "_content": "list of common fruits"}
    assert default_context[2] == {
        "role": "_llm",
        "_content": "apples, oranges, bananas, strawberries",
    }

    # test increasing distance metric broadens results
    semantic_session.set_distance_threshold(0.5)
    default_context = semantic_session.fetch_relevant("list of fruits and vegetables")
    assert len(default_context) == 5  # 2 pairs of prompt:response, and preamble

    # test that no context is returned when top_k is zero
    no_context = semantic_session.fetch_relevant("empty prompt", top_k=0)
    assert len(no_context) == 1  # preamble is still present


def test_semantic_fetch_raw(semantic_session):
    current_time = int(time.time())
    semantic_session.store("first prompt", "first response")
    semantic_session.store("second prompt", "second response")
    raw = semantic_session.fetch_recent(raw=True)
    assert len(raw) == 2
    assert raw[0]["prompt"] == "first prompt"
    assert raw[0]["response"] == "first response"
    assert current_time <= float(raw[0]["timestamp"]) <= time.time()
    assert int(raw[0]["token_count"]) == 1


def test_semantic_drop(semantic_session):
    semantic_session.store("first prompt", "first response")
    semantic_session.store("second prompt", "second response")
    semantic_session.store("third prompt", "third response")
    semantic_session.store("fourth prompt", "fourth response")

    # test drop() with no arguments removes the last element
    semantic_session.drop()
    context = semantic_session.fetch_recent(top_k=1)
    assert context == [
        {"role": "_preamble", "_content": ""},
        {"role": "_user", "_content": "third prompt"},
        {"role": "_llm", "_content": "third response"},
    ]

    # test drop(timestamp) removes the specified element
    context = semantic_session.fetch_recent(top_k=4, raw=True)
    middle_id = context[1]["id_field"]
    semantic_session.drop(middle_id)
    context = semantic_session.fetch_recent(top_k=2)
    assert context == [
        {"role": "_preamble", "_content": ""},
        {"role": "_user", "_content": "first prompt"},
        {"role": "_llm", "_content": "first response"},
        {"role": "_user", "_content": "third prompt"},
        {"role": "_llm", "_content": "third response"},
    ]
