import pytest
from pydantic import ValidationError

from redisvl.extensions.message_history.schema import ChatMessage, ChatRole
from redisvl.redis.utils import array_to_buffer
from redisvl.utils.utils import create_ulid, current_timestamp, serialize


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
    assert chat_message.metadata is None


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

    # ID should start with session:timestamp and have a UUID suffix for uniqueness
    assert chat_message.entry_id.startswith(f"{session_tag}:{timestamp}:")
    # Verify the UUID suffix is 8 hex characters
    parts = chat_message.entry_id.split(":")
    assert len(parts) == 3
    assert len(parts[2]) == 8
    assert all(c in "0123456789abcdef" for c in parts[2])


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


def test_chat_message_with_metadata():
    session_tag = create_ulid()
    timestamp = current_timestamp()
    content = "Hello, world!"
    metadata = {"language": "Python", "version": "3.13"}

    chat_message = ChatMessage(
        entry_id=f"{session_tag}:{timestamp}",
        role="user",
        content=content,
        session_tag=session_tag,
        timestamp=timestamp,
        metadata=serialize(metadata),
    )

    assert chat_message.metadata == serialize(metadata)

    # test that metadta need not be a dictionary
    for other_metadata in ["raw string", 42, [1, 2, 3], ["a", "b", "c"]]:
        chat_message = ChatMessage(
            entry_id=f"{session_tag}:{timestamp}",
            role="user",
            content=content,
            session_tag=session_tag,
            timestamp=timestamp,
            metadata=serialize(other_metadata),
        )
        assert chat_message.metadata == serialize(other_metadata)


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
    metadata = {"language": "Python", "version": "3.13"}

    chat_message = ChatMessage(
        entry_id=f"{session_tag}:{timestamp}",
        role="user",
        content=content,
        session_tag=session_tag,
        timestamp=timestamp,
        vector_field=vector_field,
        metadata=serialize(metadata),
    )

    data = chat_message.to_dict(dtype="float32")

    assert data["entry_id"] == f"{session_tag}:{timestamp}"
    assert data["role"] == "user"
    assert data["content"] == content
    assert data["session_tag"] == session_tag
    assert data["timestamp"] == timestamp
    assert data["vector_field"] == array_to_buffer(vector_field, "float32")
    assert data["metadata"] == serialize(metadata)


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


@pytest.mark.parametrize(
    "deprecated_role", ["potato", "llm", "admin", "", "User", "SYSTEM"]
)
def test_chat_message_deprecated_role(deprecated_role):
    """Deprecated string roles raise a DeprecationWarning but are stored as-is."""
    session_tag = create_ulid()
    timestamp = current_timestamp()
    content = "Hello, world!"
    message = None

    with pytest.warns(DeprecationWarning, match="deprecated value"):
        message = ChatMessage(
            entry_id=f"{session_tag}:{timestamp}",
            role=deprecated_role,
            content=content,
            session_tag=session_tag,
            timestamp=timestamp,
        )
    assert message is not None
    assert message.role == deprecated_role


@pytest.mark.parametrize(
    "role_input, expected",
    [
        ("user", ChatRole.USER),
        ("assistant", ChatRole.ASSISTANT),
        ("system", ChatRole.SYSTEM),
        ("tool", ChatRole.TOOL),
        (ChatRole.USER, ChatRole.USER),  # enum value passed directly
        (ChatRole.ASSISTANT, ChatRole.ASSISTANT),
    ],
)
def test_chat_message_role_coercion(role_input, expected):
    session_tag = create_ulid()
    content = "Hello, world!"
    message = ChatMessage(role=role_input, content=content, session_tag=session_tag)
    assert message.role == expected
    assert isinstance(message.role, ChatRole)


@pytest.mark.parametrize("invalid_role", [123, [1, 2, 3]])
def test_chat_message_invalid_role(invalid_role):
    session_tag = create_ulid()
    content = "Hello, world!"
    with pytest.raises(ValidationError):
        ChatMessage(role=invalid_role, content=content, session_tag=session_tag)


def test_chat_message_role_serializes_to_string():
    session_tag = create_ulid()
    role = "user"
    content = "Hello, world!"
    message = ChatMessage(role=role, content=content, session_tag=session_tag)
    data = message.to_dict()
    assert data["role"] == "user"
    assert isinstance(data["role"], str)


def test_chat_message_unique_ids_for_rapid_creation():
    """Test that rapidly created messages get unique IDs even with same timestamp."""
    session_tag = create_ulid()
    timestamp = current_timestamp()

    # Create multiple messages with the same session and timestamp
    messages = []
    for i in range(10):
        msg = ChatMessage(
            role="user",
            content=f"Message {i}",
            session_tag=session_tag,
            timestamp=timestamp,
        )
        messages.append(msg)

    # All IDs should be unique
    ids = [msg.entry_id for msg in messages]
    assert len(ids) == len(set(ids)), "All message IDs should be unique"

    # All IDs should start with the same session:timestamp prefix
    expected_prefix = f"{session_tag}:{timestamp}:"
    for msg_id in ids:
        assert msg_id.startswith(expected_prefix)
