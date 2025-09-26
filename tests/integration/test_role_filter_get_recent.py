"""Integration tests for role filtering in get_recent() method."""

import pytest

from redisvl.extensions.message_history import MessageHistory, SemanticMessageHistory


class TestMessageHistoryRoleFilter:
    """Test role filtering functionality in MessageHistory with real Redis."""

    def test_get_recent_single_role_system(self, redis_url):
        """Test get_recent with role='system' returns only system messages."""
        history = MessageHistory("test_role_system", redis_url=redis_url)

        # Clear any existing data
        history.clear()

        # Add various messages with different roles
        history.add_messages(
            [
                {"role": "system", "content": "System initialization"},
                {"role": "user", "content": "Hello"},
                {"role": "llm", "content": "Hi there"},
                {"role": "system", "content": "System configuration updated"},
                {
                    "role": "tool",
                    "content": "Function executed",
                    "tool_call_id": "call1",
                },
            ]
        )

        # Get only system messages
        result = history.get_recent(role="system", top_k=10)

        assert len(result) == 2
        assert all(msg["role"] == "system" for msg in result)
        assert result[0]["content"] == "System initialization"
        assert result[1]["content"] == "System configuration updated"

        # Cleanup
        history.delete()

    def test_get_recent_single_role_user(self, redis_url):
        """Test get_recent with role='user' returns only user messages."""
        history = MessageHistory("test_role_user", redis_url=redis_url)
        history.clear()

        history.add_messages(
            [
                {"role": "system", "content": "Welcome"},
                {"role": "user", "content": "First question"},
                {"role": "llm", "content": "First answer"},
                {"role": "user", "content": "Second question"},
                {"role": "user", "content": "Third question"},
            ]
        )

        result = history.get_recent(role="user", top_k=10)

        assert len(result) == 3
        assert all(msg["role"] == "user" for msg in result)
        assert result[0]["content"] == "First question"
        assert result[2]["content"] == "Third question"

        history.delete()

    def test_get_recent_single_role_llm(self, redis_url):
        """Test get_recent with role='llm' returns only llm messages."""
        history = MessageHistory("test_role_llm", redis_url=redis_url)
        history.clear()

        history.add_messages(
            [
                {"role": "user", "content": "Question 1"},
                {"role": "llm", "content": "Answer 1"},
                {"role": "user", "content": "Question 2"},
                {"role": "llm", "content": "Answer 2"},
                {"role": "system", "content": "System note"},
            ]
        )

        result = history.get_recent(role="llm", top_k=10)

        assert len(result) == 2
        assert all(msg["role"] == "llm" for msg in result)
        assert result[0]["content"] == "Answer 1"
        assert result[1]["content"] == "Answer 2"

        history.delete()

    def test_get_recent_single_role_tool(self, redis_url):
        """Test get_recent with role='tool' returns only tool messages."""
        history = MessageHistory("test_role_tool", redis_url=redis_url)
        history.clear()

        history.add_messages(
            [
                {"role": "user", "content": "Run function"},
                {
                    "role": "tool",
                    "content": "Function result 1",
                    "tool_call_id": "call1",
                },
                {"role": "llm", "content": "Processing"},
                {
                    "role": "tool",
                    "content": "Function result 2",
                    "tool_call_id": "call2",
                },
            ]
        )

        result = history.get_recent(role="tool", top_k=10)

        assert len(result) == 2
        assert all(msg["role"] == "tool" for msg in result)
        assert all("tool_call_id" in msg for msg in result)

        history.delete()

    def test_get_recent_multiple_roles(self, redis_url):
        """Test get_recent with multiple roles returns matching messages."""
        history = MessageHistory("test_multi_roles", redis_url=redis_url)
        history.clear()

        history.add_messages(
            [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "User message"},
                {"role": "llm", "content": "LLM message"},
                {"role": "tool", "content": "Tool message", "tool_call_id": "call1"},
            ]
        )

        # Get system and user messages only
        result = history.get_recent(role=["system", "user"], top_k=10)

        assert len(result) == 2
        assert all(msg["role"] in ["system", "user"] for msg in result)
        assert result[0]["content"] == "System message"
        assert result[1]["content"] == "User message"

        history.delete()

    def test_get_recent_no_role_filter_backward_compatibility(self, redis_url):
        """Test get_recent with role=None returns all messages (backward compatibility)."""
        history = MessageHistory("test_no_filter", redis_url=redis_url)
        history.clear()

        history.add_messages(
            [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "User"},
                {"role": "llm", "content": "LLM"},
                {"role": "tool", "content": "Tool", "tool_call_id": "call1"},
            ]
        )

        # No role filter - should return all messages
        result = history.get_recent(role=None, top_k=10)

        assert len(result) == 4
        roles = {msg["role"] for msg in result}
        assert roles == {"system", "user", "llm", "tool"}

        history.delete()

    def test_get_recent_invalid_role_raises_error(self, redis_url):
        """Test get_recent with invalid role raises ValueError."""
        history = MessageHistory("test_invalid", redis_url=redis_url)

        with pytest.raises(ValueError, match="Invalid role"):
            history.get_recent(role="invalid_role")

        history.delete()

    def test_get_recent_invalid_role_in_list_raises_error(self, redis_url):
        """Test get_recent with invalid role in list raises ValueError."""
        history = MessageHistory("test_invalid_list", redis_url=redis_url)

        with pytest.raises(ValueError, match="Invalid role"):
            history.get_recent(role=["system", "invalid_role"])

        history.delete()

    def test_get_recent_empty_role_list_raises_error(self, redis_url):
        """Test get_recent with empty role list raises ValueError."""
        history = MessageHistory("test_empty_list", redis_url=redis_url)

        with pytest.raises(ValueError, match="roles cannot be empty"):
            history.get_recent(role=[])

        history.delete()

    def test_get_recent_role_with_other_parameters(self, redis_url):
        """Test role filter works with other parameters like top_k."""
        history = MessageHistory("test_with_params", redis_url=redis_url)
        history.clear()

        # Add many system messages
        for i in range(5):
            history.add_message({"role": "system", "content": f"System message {i}"})

        # Add other messages
        history.add_message({"role": "user", "content": "User message"})
        history.add_message({"role": "llm", "content": "LLM message"})

        # Get only 2 most recent system messages
        result = history.get_recent(role="system", top_k=2)

        assert len(result) == 2
        assert all(msg["role"] == "system" for msg in result)
        # Should get most recent ones
        assert result[0]["content"] == "System message 3"
        assert result[1]["content"] == "System message 4"

        history.delete()

    def test_get_recent_role_with_session_tag(self, redis_url):
        """Test role filter works with session_tag parameter."""
        history = MessageHistory("test_session", redis_url=redis_url)
        history.clear()

        # Add messages with different session tags
        history.add_messages(
            [
                {"role": "system", "content": "System for session1"},
                {"role": "user", "content": "User for session1"},
            ],
            session_tag="session1",
        )

        history.add_messages(
            [
                {"role": "system", "content": "System for session2"},
                {"role": "llm", "content": "LLM for session2"},
            ],
            session_tag="session2",
        )

        # Get system messages from session2 only
        result = history.get_recent(role="system", session_tag="session2", top_k=10)

        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "System for session2"

        history.delete()

    def test_get_recent_role_with_raw_output(self, redis_url):
        """Test role filter works with raw=True."""
        history = MessageHistory("test_raw", redis_url=redis_url)
        history.clear()

        history.add_message({"role": "system", "content": "System message"})

        result = history.get_recent(role="system", raw=True, top_k=10)

        assert len(result) == 1
        assert result[0]["role"] == "system"
        # Raw should include additional metadata
        assert "entry_id" in result[0]
        assert "timestamp" in result[0]
        assert "session_tag" in result[0]

        history.delete()


class TestSemanticMessageHistoryRoleFilter:
    """Test role filtering functionality in SemanticMessageHistory with real Redis."""

    def test_semantic_get_recent_with_role(self, redis_url):
        """Test SemanticMessageHistory get_recent with role filter."""
        history = SemanticMessageHistory(
            "test_semantic_recent", redis_url=redis_url, overwrite=True
        )
        history.clear()

        history.add_messages(
            [
                {"role": "system", "content": "System prompt about configuration"},
                {"role": "user", "content": "User question"},
                {"role": "llm", "content": "Assistant response"},
            ]
        )

        result = history.get_recent(role="system", top_k=10)

        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert "configuration" in result[0]["content"]

        history.delete()

    def test_semantic_get_relevant_with_role(self, redis_url):
        """Test SemanticMessageHistory get_relevant with role filter."""
        history = SemanticMessageHistory(
            "test_semantic_relevant", redis_url=redis_url, overwrite=True
        )
        history.clear()

        history.add_messages(
            [
                {
                    "role": "system",
                    "content": "System instructions about fruits and vegetables",
                },
                {"role": "user", "content": "Tell me about apples"},
                {"role": "llm", "content": "Apples are a type of fruit"},
                {"role": "user", "content": "What about cars?"},
                {"role": "llm", "content": "Cars are vehicles for transportation"},
            ]
        )

        # Search for fruit-related messages but only from system role
        result = history.get_relevant("fruits", role="system", top_k=10)

        if result:  # Semantic search might not find exact matches
            assert all(msg["role"] == "system" for msg in result)

        # Search for fruit-related messages from user role
        result = history.get_relevant("apples", role="user", top_k=10)

        if result:
            assert all(msg["role"] == "user" for msg in result)

        history.delete()


class TestRoleValidation:
    """Test role validation logic."""

    def test_valid_roles_accepted(self, redis_url):
        """Test that all valid roles are accepted."""
        valid_roles = ["system", "user", "llm", "tool"]
        history = MessageHistory("test_valid_roles", redis_url=redis_url)
        history.clear()

        # Add messages with all valid roles
        for role in valid_roles:
            if role == "tool":
                history.add_message(
                    {
                        "role": role,
                        "content": f"{role} message",
                        "tool_call_id": "call1",
                    }
                )
            else:
                history.add_message({"role": role, "content": f"{role} message"})

        # Test each valid role works
        for role in valid_roles:
            result = history.get_recent(role=role, top_k=10)
            assert len(result) >= 1
            assert all(msg["role"] == role for msg in result)

        history.delete()

    def test_case_sensitive_roles(self, redis_url):
        """Test that role validation is case sensitive."""
        history = MessageHistory("test_case", redis_url=redis_url)

        with pytest.raises(ValueError):
            history.get_recent(role="SYSTEM")  # uppercase should fail

        with pytest.raises(ValueError):
            history.get_recent(role="User")  # mixed case should fail

        history.delete()
