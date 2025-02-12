import pytest
from pydantic import ValidationError

from redisvl.extensions.session_manager.schema import ChatMessage
from redisvl.redis.utils import array_to_buffer
from redisvl.utils.utils import create_ulid, current_timestamp


def test_chat_message_creation():
    session_tag = create_ulid()
    timestamp = current_timestamp()
    content = "Hello, world!"

    chat_message = ChatMessage(
        entry_id=f"{session_tag}:{timestamp}",
        role="user",
        content=content,
        session_tag=session_tag,
        timestamp=timestamp,
    )

    assert chat_message.entry_id == f"{session_tag}:{timestamp}"
    assert chat_message.role == "user"
    assert chat_message.content == content
    assert chat_message.session_tag == session_tag
    assert chat_message.timestamp == timestamp
    assert chat_message.tool_call_id is None
    assert chat_message.vector_field is None


def test_chat_message_default_id_generation():
    session_tag = create_ulid()
    timestamp = current_timestamp()
    content = "Hello, world!"

    chat_message = ChatMessage(
        role="user",
        content=content,
        session_tag=session_tag,
        timestamp=timestamp,
    )

    assert chat_message.entry_id == f"{session_tag}:{timestamp}"


def test_chat_message_with_tool_call_id():
    session_tag = create_ulid()
    timestamp = current_timestamp()
    content = "Hello, world!"
    tool_call_id = create_ulid()

    chat_message = ChatMessage(
        entry_id=f"{session_tag}:{timestamp}",
        role="user",
        content=content,
        session_tag=session_tag,
        timestamp=timestamp,
        tool_call_id=tool_call_id,
    )

    assert chat_message.tool_call_id == tool_call_id


def test_chat_message_with_vector_field():
    session_tag = create_ulid()
    timestamp = current_timestamp()
    content = "Hello, world!"
    vector_field = [0.1, 0.2, 0.3]

    chat_message = ChatMessage(
        entry_id=f"{session_tag}:{timestamp}",
        role="user",
        content=content,
        session_tag=session_tag,
        timestamp=timestamp,
        vector_field=vector_field,
    )

    assert chat_message.vector_field == vector_field


def test_chat_message_to_dict():
    session_tag = create_ulid()
    timestamp = current_timestamp()
    content = "Hello, world!"
    vector_field = [0.1, 0.2, 0.3]

    chat_message = ChatMessage(
        entry_id=f"{session_tag}:{timestamp}",
        role="user",
        content=content,
        session_tag=session_tag,
        timestamp=timestamp,
        vector_field=vector_field,
    )

    data = chat_message.to_dict(dtype="float32")

    assert data["entry_id"] == f"{session_tag}:{timestamp}"
    assert data["role"] == "user"
    assert data["content"] == content
    assert data["session_tag"] == session_tag
    assert data["timestamp"] == timestamp
    assert data["vector_field"] == array_to_buffer(vector_field, "float32")


def test_chat_message_missing_fields():
    session_tag = create_ulid()
    timestamp = current_timestamp()
    content = "Hello, world!"

    with pytest.raises(ValidationError):
        ChatMessage(
            content=content,
            session_tag=session_tag,
            timestamp=timestamp,
        )


def test_chat_message_invalid_role():
    session_tag = create_ulid()
    timestamp = current_timestamp()
    content = "Hello, world!"

    with pytest.raises(ValidationError):
        ChatMessage(
            entry_id=f"{session_tag}:{timestamp}",
            role=[1, 2, 3],  # Invalid role type
            content=content,
            session_tag=session_tag,
            timestamp=timestamp,
        )
