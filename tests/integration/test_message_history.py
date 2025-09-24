import warnings

import pytest
from redis.exceptions import ConnectionError

from redisvl.extensions.constants import ID_FIELD_NAME
from redisvl.extensions.message_history import MessageHistory, SemanticMessageHistory
from tests.conftest import skip_if_no_redisearch


@pytest.fixture
def app_name():
    return "test_app"


@pytest.fixture
def standard_history(app_name, client):
    history = MessageHistory(app_name, redis_client=client)
    yield history
    history.clear()


@pytest.fixture
def semantic_history(app_name, client, hf_vectorizer):
    skip_if_no_redisearch(client)
    history = SemanticMessageHistory(
        app_name, redis_client=client, overwrite=True, vectorizer=hf_vectorizer
    )
    yield history
    history.clear()
    history.delete()


@pytest.fixture(autouse=True)
def disable_deprecation_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# test standard message history
def test_specify_redis_client(client):
    history = MessageHistory(name="test_app", redis_client=client)
    assert isinstance(history._index.client, type(client))


def test_specify_redis_url(client, redis_url):
    history = MessageHistory(
        name="test_app",
        session_tag="abc",
        redis_url=redis_url,
    )
    assert isinstance(history._index.client, type(client))


def test_standard_bad_connection_info():
    with pytest.raises(ConnectionError):
        MessageHistory(
            name="test_app",
            session_tag="abc",
            redis_url="redis://localhost:6389",  # bad url
        )


def test_standard_store(standard_history):
    context = standard_history.get_recent()
    assert len(context) == 0

    standard_history.store(prompt="first prompt", response="first response")
    standard_history.store(prompt="second prompt", response="second response")
    standard_history.store(prompt="third prompt", response="third response")
    standard_history.store(prompt="fourth prompt", response="fourth response")
    standard_history.store(prompt="fifth prompt", response="fifth response")

    # test that order is maintained
    full_context = standard_history.get_recent(top_k=10)
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


def test_standard_add_and_get(standard_history):
    context = standard_history.get_recent()
    assert len(context) == 0

    standard_history.add_message({"role": "user", "content": "first prompt"})
    standard_history.add_message({"role": "llm", "content": "first response"})
    standard_history.add_message({"role": "user", "content": "second prompt"})
    standard_history.add_message({"role": "llm", "content": "second response"})
    standard_history.add_message(
        {
            "role": "tool",
            "content": "tool result 1",
            "tool_call_id": "tool call one",
            "metadata": {"tool call params": "abc 123"},
        }
    )
    standard_history.add_message(
        {
            "role": "tool",
            "content": "tool result 2",
            "tool_call_id": "tool call two",
            "metadata": {"tool call params": "abc 456"},
        }
    )
    standard_history.add_message({"role": "user", "content": "third prompt"})
    standard_history.add_message({"role": "llm", "content": "third response"})

    # test default context history size
    default_context = standard_history.get_recent()
    assert len(default_context) == 5  #  default is 5

    # test specified context history size
    partial_context = standard_history.get_recent(top_k=3)
    assert len(partial_context) == 3
    assert partial_context == [
        {
            "role": "tool",
            "content": "tool result 2",
            "tool_call_id": "tool call two",
            "metadata": {"tool call params": "abc 456"},
        },
        {"role": "user", "content": "third prompt"},
        {"role": "llm", "content": "third response"},
    ]

    # test that order is maintained
    full_context = standard_history.get_recent(top_k=10)
    assert full_context == [
        {"role": "user", "content": "first prompt"},
        {"role": "llm", "content": "first response"},
        {"role": "user", "content": "second prompt"},
        {"role": "llm", "content": "second response"},
        {
            "role": "tool",
            "content": "tool result 1",
            "tool_call_id": "tool call one",
            "metadata": {"tool call params": "abc 123"},
        },
        {
            "role": "tool",
            "content": "tool result 2",
            "tool_call_id": "tool call two",
            "metadata": {"tool call params": "abc 456"},
        },
        {"role": "user", "content": "third prompt"},
        {"role": "llm", "content": "third response"},
    ]

    # test that a ValueError is raised when top_k is invalid
    with pytest.raises(ValueError):
        bad_context = standard_history.get_recent(top_k=-2)

    with pytest.raises(ValueError):
        bad_context = standard_history.get_recent(top_k=-2.0)

    with pytest.raises(ValueError):
        bad_context = standard_history.get_recent(top_k=1.3)

    with pytest.raises(ValueError):
        bad_context = standard_history.get_recent(top_k="3")


def test_standard_add_messages(standard_history):
    context = standard_history.get_recent()
    assert len(context) == 0

    standard_history.add_messages(
        [
            {"role": "user", "content": "first prompt"},
            {
                "role": "llm",
                "content": "first response",
                "metadata": {"llm provider": "openai"},
            },
            {"role": "user", "content": "second prompt"},
            {"role": "llm", "content": "second response"},
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
            {"role": "user", "content": "fourth prompt"},
            {"role": "llm", "content": "fourth response"},
        ]
    )

    full_context = standard_history.get_recent(top_k=10)
    assert len(full_context) == 8
    assert full_context == [
        {"role": "user", "content": "first prompt"},
        {
            "role": "llm",
            "content": "first response",
            "metadata": {"llm provider": "openai"},
        },
        {"role": "user", "content": "second prompt"},
        {"role": "llm", "content": "second response"},
        {"role": "tool", "content": "tool result 1", "tool_call_id": "tool call one"},
        {"role": "tool", "content": "tool result 2", "tool_call_id": "tool call two"},
        {"role": "user", "content": "fourth prompt"},
        {"role": "llm", "content": "fourth response"},
    ]


def test_standard_messages_property(standard_history):
    standard_history.add_messages(
        [
            {"role": "user", "content": "first prompt"},
            {"role": "llm", "content": "first response"},
            {"role": "user", "content": "second prompt"},
            {
                "role": "llm",
                "content": "second response",
                "metadata": {"params": "abc"},
            },
            {"role": "user", "content": "third prompt", "metadata": 42},
        ]
    )

    assert standard_history.messages == [
        {"role": "user", "content": "first prompt"},
        {"role": "llm", "content": "first response"},
        {"role": "user", "content": "second prompt"},
        {"role": "llm", "content": "second response", "metadata": {"params": "abc"}},
        {"role": "user", "content": "third prompt", "metadata": 42},
    ]


def test_standard_scope(standard_history):
    # store entries under default session tag
    standard_history.store("some prompt", "some response")

    # test that changing session tag does indeed change access scope
    new_session = "def"
    standard_history.store(
        "new user prompt", "new user response", session_tag=new_session
    )
    context = standard_history.get_recent(session_tag=new_session)
    assert context == [
        {"role": "user", "content": "new user prompt"},
        {"role": "llm", "content": "new user response"},
    ]

    # test that default session data is still accessible
    context = standard_history.get_recent()
    assert context == [
        {"role": "user", "content": "some prompt"},
        {"role": "llm", "content": "some response"},
    ]

    bad_session = "xyz"
    no_context = standard_history.get_recent(session_tag=bad_session)
    assert no_context == []


def test_standard_get_text(standard_history):
    standard_history.store("first prompt", "first response")
    text = standard_history.get_recent(as_text=True)
    assert text == ["first prompt", "first response"]

    standard_history.add_message({"role": "system", "content": "system level prompt"})
    text = standard_history.get_recent(as_text=True)
    assert text == ["first prompt", "first response", "system level prompt"]


def test_standard_get_raw(standard_history):
    standard_history.store("first prompt", "first response")
    standard_history.store("second prompt", "second response")
    raw = standard_history.get_recent(raw=True)
    assert len(raw) == 4
    assert raw[0]["role"] == "user"
    assert raw[0]["content"] == "first prompt"
    assert raw[1]["role"] == "llm"
    assert raw[1]["content"] == "first response"


def test_standard_drop(standard_history):
    standard_history.store("first prompt", "first response")
    standard_history.store("second prompt", "second response")
    standard_history.store("third prompt", "third response")
    standard_history.store("fourth prompt", "fourth response")

    # test drop() with no arguments removes the last element
    standard_history.drop()
    context = standard_history.get_recent(top_k=3)
    assert context == [
        {"role": "user", "content": "third prompt"},
        {"role": "llm", "content": "third response"},
        {"role": "user", "content": "fourth prompt"},
    ]

    # test drop(id) removes the specified element
    context = standard_history.get_recent(top_k=10, raw=True)
    middle_id = context[3][ID_FIELD_NAME]
    standard_history.drop(middle_id)
    context = standard_history.get_recent(top_k=6)
    assert context == [
        {"role": "user", "content": "first prompt"},
        {"role": "llm", "content": "first response"},
        {"role": "user", "content": "second prompt"},
        {"role": "user", "content": "third prompt"},
        {"role": "llm", "content": "third response"},
        {"role": "user", "content": "fourth prompt"},
    ]


def test_standard_clear(standard_history):
    standard_history.store("some prompt", "some response")
    standard_history.clear()
    empty_context = standard_history.get_recent(top_k=10)
    assert empty_context == []


# test semantic message history
def test_semantic_specify_client(client, hf_vectorizer):
    skip_if_no_redisearch(client)
    history = SemanticMessageHistory(
        name="test_app",
        session_tag="abc",
        redis_client=client,
        overwrite=True,
        vectorizer=hf_vectorizer,
    )
    assert isinstance(history._index.client, type(client))


def test_semantic_bad_connection_info(hf_vectorizer):
    with pytest.raises(ConnectionError):
        SemanticMessageHistory(
            name="test_app",
            session_tag="abc",
            redis_url="redis://localhost:6389",
            vectorizer=hf_vectorizer,
        )


def test_semantic_scope(semantic_history):
    # store entries under default session tag
    semantic_history.store("some prompt", "some response")

    # test that changing session tag does indeed change access scope
    new_session = "def"
    semantic_history.store(
        "new user prompt", "new user response", session_tag=new_session
    )
    context = semantic_history.get_recent(session_tag=new_session)
    assert context == [
        {"role": "user", "content": "new user prompt"},
        {"role": "llm", "content": "new user response"},
    ]

    # test that previous session data is still accessible
    context = semantic_history.get_recent()
    assert context == [
        {"role": "user", "content": "some prompt"},
        {"role": "llm", "content": "some response"},
    ]

    bad_session = "xyz"
    no_context = semantic_history.get_recent(session_tag=bad_session)
    assert no_context == []


def test_semantic_store_and_get_recent(semantic_history):
    context = semantic_history.get_recent()
    assert len(context) == 0

    semantic_history.store(prompt="first prompt", response="first response")
    semantic_history.store(prompt="second prompt", response="second response")
    semantic_history.store(prompt="third prompt", response="third response")
    semantic_history.store(prompt="fourth prompt", response="fourth response")
    semantic_history.add_message(
        {"role": "tool", "content": "tool result", "tool_call_id": "tool id"}
    )
    semantic_history.add_message(
        {
            "role": "tool",
            "content": "tool result",
            "tool_call_id": "tool id",
            "metadata": "return value from tool",
        }
    )  # test default context history size
    default_context = semantic_history.get_recent()
    assert len(default_context) == 5  # 5 is default

    # test specified context history size
    partial_context = semantic_history.get_recent(top_k=4)
    assert len(partial_context) == 4

    # test larger context history returns full history
    too_large_context = semantic_history.get_recent(top_k=100)
    assert len(too_large_context) == 10

    # test that order is maintained
    full_context = semantic_history.get_recent(top_k=10)
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
        {
            "role": "tool",
            "content": "tool result",
            "tool_call_id": "tool id",
            "metadata": "return value from tool",
        },
    ]

    # test that more recent entries are returned
    context = semantic_history.get_recent(top_k=4)
    assert context == [
        {"role": "user", "content": "fourth prompt"},
        {"role": "llm", "content": "fourth response"},
        {"role": "tool", "content": "tool result", "tool_call_id": "tool id"},
        {
            "role": "tool",
            "content": "tool result",
            "tool_call_id": "tool id",
            "metadata": "return value from tool",
        },
    ]

    # test no entries are returned and no error is raised if top_k == 0
    context = semantic_history.get_recent(top_k=0)
    assert context == []

    # test that a ValueError is raised when top_k is invalid
    with pytest.raises(ValueError):
        bad_context = semantic_history.get_recent(top_k=0.5)

    with pytest.raises(ValueError):
        bad_context = semantic_history.get_recent(top_k=-1)

    with pytest.raises(ValueError):
        bad_context = semantic_history.get_recent(top_k=-2.0)

    with pytest.raises(ValueError):
        bad_context = semantic_history.get_recent(top_k=1.3)

    with pytest.raises(ValueError):
        bad_context = semantic_history.get_recent(top_k="3")


def test_semantic_messages_property(semantic_history):
    semantic_history.add_messages(
        [
            {"role": "user", "content": "first prompt"},
            {"role": "llm", "content": "first response"},
            {
                "role": "tool",
                "content": "tool result 1",
                "tool_call_id": "tool call one",
                "metadata": 42,
            },
            {
                "role": "tool",
                "content": "tool result 2",
                "tool_call_id": "tool call two",
                "metadata": [1, 2, 3],
            },
            {"role": "user", "content": "second prompt"},
            {"role": "llm", "content": "second response"},
            {"role": "user", "content": "third prompt"},
        ]
    )

    assert semantic_history.messages == [
        {"role": "user", "content": "first prompt"},
        {"role": "llm", "content": "first response"},
        {
            "role": "tool",
            "content": "tool result 1",
            "tool_call_id": "tool call one",
            "metadata": 42,
        },
        {
            "role": "tool",
            "content": "tool result 2",
            "tool_call_id": "tool call two",
            "metadata": [1, 2, 3],
        },
        {"role": "user", "content": "second prompt"},
        {"role": "llm", "content": "second response"},
        {"role": "user", "content": "third prompt"},
    ]


def test_semantic_add_and_get_relevant(semantic_history):
    semantic_history.add_message(
        {"role": "system", "content": "discussing common fruits and vegetables"}
    )
    semantic_history.store(
        prompt="list of common fruits",
        response="apples, oranges, bananas, strawberries",
    )
    semantic_history.store(
        prompt="list of common vegetables",
        response="carrots, broccoli, onions, spinach",
    )
    semantic_history.store(
        prompt="winter sports in the olympics",
        response="downhill skiing, ice skating, luge",
    )
    semantic_history.add_message(
        {
            "role": "tool",
            "content": "skiing, skating, luge",
            "tool_call_id": "winter_sports()",
        }
    )

    # test default distance metric
    default_context = semantic_history.get_relevant(
        "set of common fruits like apples and bananas"
    )
    assert len(default_context) == 2
    assert default_context[0] == {"role": "user", "content": "list of common fruits"}
    assert default_context[1] == {
        "role": "llm",
        "content": "apples, oranges, bananas, strawberries",
    }

    # test increasing distance metric broadens results
    semantic_history.set_distance_threshold(0.5)
    default_context = semantic_history.get_relevant("list of fruits and vegetables")
    assert len(default_context) == 5  # 2 pairs of prompt:response, and system
    assert default_context == semantic_history.get_relevant(
        "list of fruits and vegetables", distance_threshold=0.5
    )

    # test tool calls can also be returned
    context = semantic_history.get_relevant("winter sports like skiing")
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
        bad_context = semantic_history.get_relevant("test prompt", top_k=-1)

    with pytest.raises(ValueError):
        bad_context = semantic_history.get_relevant("test prompt", top_k=-2.0)

    with pytest.raises(ValueError):
        bad_context = semantic_history.get_relevant("test prompt", top_k=1.3)

    with pytest.raises(ValueError):
        bad_context = semantic_history.get_relevant("test prompt", top_k="3")


def test_semantic_get_raw(semantic_history):
    semantic_history.store("first prompt", "first response")
    semantic_history.store("second prompt", "second response")
    raw = semantic_history.get_recent(raw=True)
    assert len(raw) == 4
    assert raw[0]["role"] == "user"
    assert raw[0]["content"] == "first prompt"
    assert raw[1]["role"] == "llm"
    assert raw[1]["content"] == "first response"


def test_semantic_drop(semantic_history):
    semantic_history.store("first prompt", "first response")
    semantic_history.store("second prompt", "second response")
    semantic_history.store("third prompt", "third response")
    semantic_history.store("fourth prompt", "fourth response")

    # test drop() with no arguments removes the last element
    semantic_history.drop()
    context = semantic_history.get_recent(top_k=3)
    assert context == [
        {"role": "user", "content": "third prompt"},
        {"role": "llm", "content": "third response"},
        {"role": "user", "content": "fourth prompt"},
    ]

    # test drop(id) removes the specified element
    context = semantic_history.get_recent(top_k=5, raw=True)
    middle_id = context[2][ID_FIELD_NAME]
    semantic_history.drop(middle_id)
    context = semantic_history.get_recent(top_k=4)
    assert context == [
        {"role": "user", "content": "second prompt"},
        {"role": "llm", "content": "second response"},
        {"role": "llm", "content": "third response"},
        {"role": "user", "content": "fourth prompt"},
    ]


def test_different_vector_dtypes(client, redis_url):
    skip_if_no_redisearch(client)
    try:
        bfloat_sess = SemanticMessageHistory(
            name="bfloat_history", dtype="bfloat16", redis_url=redis_url
        )
        bfloat_sess.add_message({"role": "user", "content": "bfloat message"})

        float16_sess = SemanticMessageHistory(
            name="float16_history", dtype="float16", redis_url=redis_url
        )
        float16_sess.add_message({"role": "user", "content": "float16 message"})

        float32_sess = SemanticMessageHistory(
            name="float32_history", dtype="float32", redis_url=redis_url
        )
        float32_sess.add_message({"role": "user", "content": "float32 message"})

        float64_sess = SemanticMessageHistory(
            name="float64_history", dtype="float64", redis_url=redis_url
        )
        float64_sess.add_message({"role": "user", "content": "float64 message"})

        for sess in [bfloat_sess, float16_sess, float32_sess, float64_sess]:
            sess.set_distance_threshold(0.7)
            assert len(sess.get_relevant("float message")) == 1
            sess.delete()  # Clean up
    except:
        pytest.skip("Required Redis modules not available or version too low")


def test_bad_dtype_connecting_to_exiting_history(client, redis_url):
    skip_if_no_redisearch(client)
    # Skip this test for Redis 6.2.x as FT.INFO doesn't return dims properly
    redis_version = client.info()["redis_version"]
    if redis_version.startswith("6.2"):
        pytest.skip(
            "Redis 6.2.x FT.INFO doesn't properly return vector dims for reconnection"
        )

    def create_history():
        return SemanticMessageHistory(
            name="float64 history", dtype="float64", redis_url=redis_url
        )

    def create_same_type():
        return SemanticMessageHistory(
            name="float64 history", dtype="float64", redis_url=redis_url
        )

    history = create_history()
    same_type = create_same_type()
    # under the hood uses from_existing

    with pytest.raises(ValueError):
        bad_type = SemanticMessageHistory(
            name="float64 history", dtype="float16", redis_url=redis_url
        )


def test_vectorizer_dtype_mismatch(client, redis_url, hf_vectorizer_float16):
    skip_if_no_redisearch(client)
    with pytest.raises(ValueError):
        SemanticMessageHistory(
            name="test_dtype_mismatch",
            dtype="float32",
            vectorizer=hf_vectorizer_float16,
            redis_url=redis_url,
            overwrite=True,
        )


def test_invalid_vectorizer(client, redis_url):
    skip_if_no_redisearch(client)
    with pytest.raises(TypeError):
        SemanticMessageHistory(
            name="test_invalid_vectorizer",
            vectorizer="invalid_vectorizer",  # type: ignore
            redis_url=redis_url,
            overwrite=True,
        )


def test_passes_through_dtype_to_default_vectorizer(client, redis_url):
    skip_if_no_redisearch(client)
    # The default is float32, so we should see float64 if we pass it in.
    cache = SemanticMessageHistory(
        name="test_pass_through_dtype",
        dtype="float64",
        redis_url=redis_url,
        overwrite=True,
    )
    assert cache._vectorizer.dtype == "float64"


def test_deprecated_dtype_argument(client, redis_url):
    skip_if_no_redisearch(client)
    with pytest.warns(DeprecationWarning):
        SemanticMessageHistory(
            name="float64 history", dtype="float64", redis_url=redis_url, overwrite=True
        )
