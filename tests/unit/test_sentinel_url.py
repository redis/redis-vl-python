"""Tests for Redis Sentinel URL connection handling.

This module tests the RedisConnectionFactory's ability to create Redis clients
from Sentinel URLs (redis+sentinel://...). It verifies:

1. Correct Sentinel class selection (AsyncSentinel for async, Sentinel for sync)
2. URL parsing (hosts, ports, service name, database, authentication)
3. Proper kwargs passthrough to Sentinel and master_for()
4. Error handling for connection failures

These tests use mocking to avoid requiring a real Sentinel deployment.

Related: GitHub Issue #465 - Async Sentinel connections were incorrectly using
the sync SentinelConnectionPool, causing runtime failures.
"""

from unittest.mock import MagicMock, patch

import pytest
from redis.exceptions import ConnectionError

from redisvl.redis.connection import RedisConnectionFactory


@pytest.mark.parametrize("use_async", [False, True])
def test_sentinel_url_connection(use_async):
    sentinel_url = (
        "redis+sentinel://username:password@host1:26379,host2:26380/mymaster/0"
    )

    # Use appropriate Sentinel class based on sync/async
    sentinel_patch_target = (
        "redisvl.redis.connection.AsyncSentinel"
        if use_async
        else "redisvl.redis.connection.Sentinel"
    )

    with patch(sentinel_patch_target) as mock_sentinel:
        mock_master = MagicMock()
        mock_sentinel.return_value.master_for.return_value = mock_master

        if use_async:
            with pytest.warns(DeprecationWarning):
                client = RedisConnectionFactory.get_async_redis_connection(sentinel_url)
        else:
            client = RedisConnectionFactory.get_redis_connection(sentinel_url)

        mock_sentinel.assert_called_once()
        call_args = mock_sentinel.call_args
        assert call_args[0][0] == [("host1", 26379), ("host2", 26380)]
        assert call_args[1]["sentinel_kwargs"] == {
            "username": "username",
            "password": "password",
        }

        mock_sentinel.return_value.master_for.assert_called_once()
        master_for_args = mock_sentinel.return_value.master_for.call_args
        assert master_for_args[0][0] == "mymaster"
        assert master_for_args[1]["db"] == "0"

        assert client == mock_master


@pytest.mark.parametrize("use_async", [False, True])
def test_sentinel_url_connection_no_auth_no_db(use_async):
    sentinel_url = "redis+sentinel://host1:26379,host2:26380/mymaster"

    # Use appropriate Sentinel class based on sync/async
    sentinel_patch_target = (
        "redisvl.redis.connection.AsyncSentinel"
        if use_async
        else "redisvl.redis.connection.Sentinel"
    )

    with patch(sentinel_patch_target) as mock_sentinel:
        mock_master = MagicMock()
        mock_sentinel.return_value.master_for.return_value = mock_master

        if use_async:
            with pytest.warns(DeprecationWarning):
                client = RedisConnectionFactory.get_async_redis_connection(sentinel_url)
        else:
            client = RedisConnectionFactory.get_redis_connection(sentinel_url)

        mock_sentinel.assert_called_once()
        call_args = mock_sentinel.call_args
        assert call_args[0][0] == [("host1", 26379), ("host2", 26380)]
        assert (
            "sentinel_kwargs" not in call_args[1]
            or call_args[1]["sentinel_kwargs"] == {}
        )

        mock_sentinel.return_value.master_for.assert_called_once()
        master_for_args = mock_sentinel.return_value.master_for.call_args
        assert master_for_args[0][0] == "mymaster"
        assert "db" not in master_for_args[1]

        assert client == mock_master


@pytest.mark.parametrize("use_async", [False, True])
def test_sentinel_url_connection_error(use_async):
    sentinel_url = "redis+sentinel://host1:26379,host2:26380/mymaster"

    # Use appropriate Sentinel class based on sync/async
    sentinel_patch_target = (
        "redisvl.redis.connection.AsyncSentinel"
        if use_async
        else "redisvl.redis.connection.Sentinel"
    )

    with patch(sentinel_patch_target) as mock_sentinel:
        mock_sentinel.return_value.master_for.side_effect = ConnectionError(
            "Test connection error"
        )

        with pytest.raises(ConnectionError):
            if use_async:
                with pytest.warns(DeprecationWarning):
                    RedisConnectionFactory.get_async_redis_connection(sentinel_url)
            else:
                RedisConnectionFactory.get_redis_connection(sentinel_url)

        mock_sentinel.assert_called_once()


def test_async_sentinel_uses_async_sentinel_class():
    """Test that async connections use AsyncSentinel (fix for issue #465)."""
    sentinel_url = "redis+sentinel://host1:26379/mymaster"

    # Track which Sentinel class is called
    sync_sentinel_called = False
    async_sentinel_called = False

    def track_sync_sentinel(*args, **kwargs):
        nonlocal sync_sentinel_called
        sync_sentinel_called = True
        mock = MagicMock()
        mock.master_for.return_value = MagicMock()
        return mock

    def track_async_sentinel(*args, **kwargs):
        nonlocal async_sentinel_called
        async_sentinel_called = True
        mock = MagicMock()
        mock.master_for.return_value = MagicMock()
        return mock

    with (
        patch("redisvl.redis.connection.Sentinel", side_effect=track_sync_sentinel),
        patch(
            "redisvl.redis.connection.AsyncSentinel", side_effect=track_async_sentinel
        ),
    ):
        with pytest.warns(DeprecationWarning):
            RedisConnectionFactory.get_async_redis_connection(sentinel_url)

    # Verify AsyncSentinel was called, not sync Sentinel
    assert async_sentinel_called, "AsyncSentinel should be called for async connections"
    assert (
        not sync_sentinel_called
    ), "Sync Sentinel should NOT be called for async connections"


def test_sync_sentinel_uses_sync_sentinel_class():
    """Test that sync connections use sync Sentinel."""
    sentinel_url = "redis+sentinel://host1:26379/mymaster"

    # Track which Sentinel class is called
    sync_sentinel_called = False
    async_sentinel_called = False

    def track_sync_sentinel(*args, **kwargs):
        nonlocal sync_sentinel_called
        sync_sentinel_called = True
        mock = MagicMock()
        mock.master_for.return_value = MagicMock()
        return mock

    def track_async_sentinel(*args, **kwargs):
        nonlocal async_sentinel_called
        async_sentinel_called = True
        mock = MagicMock()
        mock.master_for.return_value = MagicMock()
        return mock

    with (
        patch("redisvl.redis.connection.Sentinel", side_effect=track_sync_sentinel),
        patch(
            "redisvl.redis.connection.AsyncSentinel", side_effect=track_async_sentinel
        ),
    ):
        RedisConnectionFactory.get_redis_connection(sentinel_url)

    # Verify sync Sentinel was called, not AsyncSentinel
    assert sync_sentinel_called, "Sync Sentinel should be called for sync connections"
    assert (
        not async_sentinel_called
    ), "AsyncSentinel should NOT be called for sync connections"


# =============================================================================
# Additional Edge Case Tests for Sentinel URL Parsing
# =============================================================================


class TestSentinelUrlParsingEdgeCases:
    """Tests for Sentinel URL parsing edge cases not covered by main tests."""

    def test_sentinel_url_default_port_when_not_specified(self):
        """Verify default port 26379 is used when port is omitted."""
        sentinel_url = "redis+sentinel://host1/mymaster"

        with patch("redisvl.redis.connection.Sentinel") as mock_sentinel:
            mock_sentinel.return_value.master_for.return_value = MagicMock()
            RedisConnectionFactory.get_redis_connection(sentinel_url)

            call_args = mock_sentinel.call_args
            assert call_args[0][0] == [("host1", 26379)]

    def test_sentinel_url_default_service_name_when_path_empty(self):
        """Verify default service name 'mymaster' when path is empty."""
        sentinel_url = "redis+sentinel://host1:26379"

        with patch("redisvl.redis.connection.Sentinel") as mock_sentinel:
            mock_sentinel.return_value.master_for.return_value = MagicMock()
            RedisConnectionFactory.get_redis_connection(sentinel_url)

            master_for_args = mock_sentinel.return_value.master_for.call_args
            assert master_for_args[0][0] == "mymaster"

    def test_sentinel_url_password_only_auth(self):
        """Verify password-only auth works (empty username)."""
        sentinel_url = "redis+sentinel://:secretpass@host1:26379/mymaster"

        with patch("redisvl.redis.connection.Sentinel") as mock_sentinel:
            mock_sentinel.return_value.master_for.return_value = MagicMock()
            RedisConnectionFactory.get_redis_connection(sentinel_url)

            call_kwargs = mock_sentinel.call_args[1]
            assert call_kwargs["sentinel_kwargs"]["password"] == "secretpass"
            assert call_kwargs["password"] == "secretpass"

    def test_sentinel_custom_kwargs_passed_to_master_for(self):
        """Verify custom kwargs are passed through to master_for call."""
        sentinel_url = "redis+sentinel://host1:26379/mymaster"

        with patch("redisvl.redis.connection.AsyncSentinel") as mock_async_sentinel:
            mock_async_sentinel.return_value.master_for.return_value = MagicMock()

            with pytest.warns(DeprecationWarning):
                RedisConnectionFactory.get_async_redis_connection(
                    sentinel_url, decode_responses=True, socket_timeout=5.0
                )

            master_for_kwargs = mock_async_sentinel.return_value.master_for.call_args[1]
            assert master_for_kwargs["decode_responses"] is True
            assert master_for_kwargs["socket_timeout"] == 5.0
