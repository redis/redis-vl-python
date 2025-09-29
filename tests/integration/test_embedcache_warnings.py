"""Test warning behavior when using sync methods with async-only client."""

import asyncio
import logging
from unittest.mock import patch

import pytest
from redis import Redis

from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.redis.connection import RedisConnectionFactory


@pytest.mark.asyncio
async def test_sync_methods_warn_with_async_only_client(caplog):
    """Test that sync methods warn when only async client is provided."""
    # Reset the warning flag for testing
    EmbeddingsCache._warning_shown = False

    # Create async redis client using the async method
    async_client = await RedisConnectionFactory._get_aredis_connection(
        "redis://localhost:6379"
    )

    try:
        # Initialize EmbeddingsCache with only async_redis_client
        cache = EmbeddingsCache(name="test_cache", async_redis_client=async_client)

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
    finally:
        # Properly close the async client
        await async_client.aclose()


def test_no_warning_with_sync_client():
    """Test that no warning is shown when sync client is provided."""
    # Reset the warning flag for testing
    EmbeddingsCache._warning_shown = False

    # Create sync redis client
    sync_client = Redis.from_url("redis://localhost:6379")

    # Initialize EmbeddingsCache with sync_redis_client
    cache = EmbeddingsCache(name="test_cache", redis_client=sync_client)

    with patch("redisvl.utils.log.get_logger") as mock_logger:
        # Sync methods should not warn
        _ = cache.get_by_key("test_key")
        _ = cache.set(text="test", model_name="model", embedding=[0.1, 0.2])

        # No warnings should have been logged
        mock_logger.return_value.warning.assert_not_called()

    sync_client.close()


@pytest.mark.asyncio
async def test_async_methods_no_warning():
    """Test that async methods don't trigger warnings."""
    # Reset the warning flag for testing
    EmbeddingsCache._warning_shown = False

    # Create async redis client using the async method
    async_client = await RedisConnectionFactory._get_aredis_connection(
        "redis://localhost:6379"
    )

    try:
        # Initialize EmbeddingsCache with only async_redis_client
        cache = EmbeddingsCache(name="test_cache", async_redis_client=async_client)

        with patch("redisvl.utils.log.get_logger") as mock_logger:
            # Async methods should not warn
            _ = await cache.aget_by_key("test_key")
            _ = await cache.aset(text="test", model_name="model", embedding=[0.1, 0.2])

            # No warnings should have been logged
            mock_logger.return_value.warning.assert_not_called()
    finally:
        # Properly close the async client
        await async_client.aclose()
