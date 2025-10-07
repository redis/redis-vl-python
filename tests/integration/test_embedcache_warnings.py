"""Test warning behavior when using sync methods with async-only client."""

import logging
from unittest.mock import patch

import pytest
from redis import Redis

from redisvl.extensions.cache.embeddings import EmbeddingsCache


@pytest.fixture(autouse=True)
def reset_warning_flag():
    """Reset the warning flag before each test to ensure test isolation."""
    EmbeddingsCache._warning_shown = False
    yield
    # Optionally reset after test as well for cleanup
    EmbeddingsCache._warning_shown = False


@pytest.mark.asyncio
async def test_sync_methods_warn_with_async_only_client(async_client, caplog):
    """Test that sync methods warn when only async client is provided."""
    # Initialize EmbeddingsCache with only async_redis_client
    cache = EmbeddingsCache(name="test_cache", async_redis_client=async_client)

    # Mock _get_redis_client to prevent actual connection attempt
    with patch.object(cache, "_get_redis_client") as mock_get_client:
        # Mock the Redis client methods that would be called
        mock_client = mock_get_client.return_value
        mock_client.hgetall.return_value = {}  # Empty result for get_by_key
        mock_client.hset.return_value = 1  # Success for set

        # Capture log warnings
        with caplog.at_level(logging.WARNING):
            # First sync method call should warn
            _ = cache.get_by_key("test_key")

            # Check warning was logged
            assert len(caplog.records) == 1
            assert (
                "initialized with async_redis_client only" in caplog.records[0].message
            )
            assert "Use async methods" in caplog.records[0].message

            # Clear captured logs
            caplog.clear()

            # Second sync method call should NOT warn (flag prevents spam)
            _ = cache.set(text="test", model_name="model", embedding=[0.1, 0.2])

            # Should not have logged another warning
            assert len(caplog.records) == 0


def test_no_warning_with_sync_client(redis_url):
    """Test that no warning is shown when sync client is provided."""
    # Create sync redis client from redis_url
    sync_client = Redis.from_url(redis_url)

    try:
        # Initialize EmbeddingsCache with sync_redis_client
        cache = EmbeddingsCache(name="test_cache", redis_client=sync_client)

        with patch("redisvl.utils.log.get_logger") as mock_logger:
            # Sync methods should not warn
            _ = cache.get_by_key("test_key")
            _ = cache.set(text="test", model_name="model", embedding=[0.1, 0.2])

            # No warnings should have been logged
            mock_logger.return_value.warning.assert_not_called()
    finally:
        sync_client.close()


@pytest.mark.asyncio
async def test_async_methods_no_warning(async_client):
    """Test that async methods don't trigger warnings."""
    # Initialize EmbeddingsCache with only async_redis_client
    cache = EmbeddingsCache(name="test_cache", async_redis_client=async_client)

    with patch("redisvl.utils.log.get_logger") as mock_logger:
        # Async methods should not warn
        _ = await cache.aget_by_key("test_key")
        _ = await cache.aset(text="test", model_name="model", embedding=[0.1, 0.2])

        # No warnings should have been logged
        mock_logger.return_value.warning.assert_not_called()
