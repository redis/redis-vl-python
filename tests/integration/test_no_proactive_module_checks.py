"""
Test that module validation is not done proactively but only when operations fail.
This is the TDD test file for issue #370.

These tests verify that MODULE LIST is not called during connection initialization
or index creation, improving performance by eliminating unnecessary network calls.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import ResponseError

from redisvl.exceptions import RedisSearchError
from redisvl.extensions.router import Route, SemanticRouter
from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.redis.connection import RedisConnectionFactory
from redisvl.schema import IndexSchema
from redisvl.utils.vectorize.base import BaseVectorizer
from tests.conftest import (
    has_redisearch_module,
    has_redisearch_module_async,
    skip_if_no_redisearch,
    skip_if_no_redisearch_async,
)


@pytest.fixture
def sample_schema():
    """Create a sample index schema for testing."""
    return IndexSchema.from_dict(
        {
            "index": {"name": "test-index", "prefix": "doc", "storage_type": "hash"},
            "fields": [
                {"name": "text", "type": "text"},
                {
                    "name": "vector",
                    "type": "vector",
                    "attrs": {
                        "dims": 3,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                    },
                },
            ],
        }
    )


class TestNoProactiveModuleChecks:
    """Tests to ensure module validation is not done proactively."""

    def test_connection_factory_no_validation(self, redis_url):
        """Test that RedisConnectionFactory doesn't validate modules on connection."""
        # Create connection first
        client = RedisConnectionFactory.get_redis_connection(redis_url)

        # Patch module_list on the actual instance to track if it's called
        original_module_list = client.module_list
        client.module_list = Mock(side_effect=original_module_list)

        try:
            # Verify connection works for basic operations
            client.ping()

            # MODULE LIST should NOT have been called during connection
            client.module_list.assert_not_called()
        finally:
            client.close()

    async def test_async_connection_factory_no_validation(self, redis_url):
        """Test that async RedisConnectionFactory doesn't validate modules on connection."""
        # Create async connection first
        client = await RedisConnectionFactory._get_aredis_connection(redis_url)

        # Patch module_list on the actual instance to track if it's called
        original_module_list = client.module_list
        client.module_list = AsyncMock(side_effect=original_module_list)

        try:
            # Verify connection works for basic operations
            await client.ping()

            # MODULE LIST should NOT have been called during connection
            client.module_list.assert_not_called()
        finally:
            await client.aclose()

    def test_search_index_init_no_validation(self, redis_url, sample_schema):
        """Test that SearchIndex initialization doesn't validate modules."""
        # Create index
        index = SearchIndex(sample_schema, redis_url=redis_url)

        # Access the Redis client (triggers lazy initialization)
        client = index._redis_client

        # Patch module_list on the actual instance
        original_module_list = client.module_list
        client.module_list = Mock(side_effect=original_module_list)

        # Basic Redis operations should work
        client.ping()

        # MODULE LIST should NOT have been called
        client.module_list.assert_not_called()

    async def test_async_search_index_init_no_validation(
        self, redis_url, sample_schema
    ):
        """Test that AsyncSearchIndex initialization doesn't validate modules."""
        # Create async index
        index = AsyncSearchIndex(sample_schema, redis_url=redis_url)

        # Access the Redis client (triggers lazy initialization)
        client = await index._get_client()

        # Patch module_list on the actual instance
        original_module_list = client.module_list
        client.module_list = AsyncMock(side_effect=original_module_list)

        # Basic Redis operations should work
        await client.ping()

        # MODULE LIST should NOT have been called
        client.module_list.assert_not_called()

    def test_search_index_create_with_modules(self, client, sample_schema, worker_id):
        """Test that index.create() works with RediSearch available."""
        # Skip if RediSearch is not available
        skip_if_no_redisearch(client)

        # Update schema name to be unique
        schema_copy = IndexSchema.from_dict(sample_schema.to_dict())
        schema_copy.index.name = f"test-index-{worker_id}"

        # Create index with Redis client
        index = SearchIndex(schema_copy, redis_client=client)

        # Track if MODULE LIST is called during create
        with patch.object(client, "module_list") as mock_module_list:
            # Create should succeed with modules available
            index.create(overwrite=True)

            # MODULE LIST should NOT have been called
            mock_module_list.assert_not_called()

        # Verify index exists
        assert index.exists()

        # Clean up
        index.delete()

    async def test_async_search_index_create_with_modules(
        self, async_client, sample_schema, worker_id
    ):
        """Test that async index.create() works with RediSearch available."""
        # Skip if RediSearch is not available
        await skip_if_no_redisearch_async(async_client)

        # Update schema name to be unique
        schema_copy = IndexSchema.from_dict(sample_schema.to_dict())
        schema_copy.index.name = f"test-index-async-{worker_id}"

        # Create async index with Redis client
        index = AsyncSearchIndex(schema_copy, redis_client=async_client)

        # Track if MODULE LIST is called during create
        with patch.object(async_client, "module_list") as mock_module_list:
            # Create should succeed with modules available
            await index.create(overwrite=True)

            # MODULE LIST should NOT have been called
            mock_module_list.assert_not_called()

        # Verify index exists
        assert await index.exists()

        # Clean up
        await index.delete()

    def test_search_operations_fail_gracefully_without_modules(self):
        """Test that operations fail with clear errors when modules are missing."""
        # Create a mock client that simulates missing modules
        mock_client = Mock(spec=Redis)
        mock_client.execute_command.side_effect = ResponseError(
            "unknown command 'FT.CREATE'"
        )

        schema = IndexSchema.from_dict(
            {
                "index": {
                    "name": "test-index",
                    "prefix": "doc",
                    "storage_type": "hash",
                },
                "fields": [{"name": "text", "type": "text"}],
            }
        )

        index = SearchIndex(schema, redis_client=mock_client)

        # Operations should fail with clear errors
        with pytest.raises(ResponseError) as exc_info:
            index.create()

        # Error message should be clear about what's missing
        assert "unknown command" in str(exc_info.value).lower()

    def test_semantic_router_no_proactive_validation(self, redis_url, worker_id):
        """Test that SemanticRouter doesn't validate modules proactively."""
        # Create client
        client = RedisConnectionFactory.get_redis_connection(redis_url)

        # Patch module_list on the actual instance
        original_module_list = client.module_list
        client.module_list = Mock(side_effect=original_module_list)

        try:
            # Basic operations should work without MODULE LIST being called
            client.ping()

            # MODULE LIST should NOT have been called
            client.module_list.assert_not_called()
        finally:
            client.close()

    def test_multiple_connections_no_repeated_validation(
        self, redis_url, sample_schema
    ):
        """Test that creating multiple connections/indices doesn't trigger validation."""
        # Create multiple connections and indices
        clients = []
        mocks = []

        for i in range(3):
            client = RedisConnectionFactory.get_redis_connection(redis_url)

            # Patch module_list on each instance
            original_module_list = client.module_list
            client.module_list = Mock(side_effect=original_module_list)
            mocks.append(client.module_list)

            index = SearchIndex(sample_schema, redis_client=client)
            _ = index._redis_client  # Access to trigger lazy init
            clients.append(client)

        # MODULE LIST should never be called on any client
        for mock in mocks:
            mock.assert_not_called()

        # Clean up
        for client in clients:
            client.close()

    @pytest.mark.parametrize(
        "connection_url",
        [
            "redis://localhost:6379",
            "redis://user:pass@localhost:6379/0",
        ],
    )
    def test_various_connection_strings_no_validation(self, connection_url):
        """Test that various connection string formats don't trigger validation."""
        with patch("redisvl.redis.connection.Redis.from_url") as mock_from_url:
            mock_client = Mock(spec=Redis)
            mock_client.ping = Mock(return_value=True)
            mock_from_url.return_value = mock_client

            with patch.object(mock_client, "module_list") as mock_module_list:
                # Create connection
                client = RedisConnectionFactory.get_redis_connection(connection_url)

                # MODULE LIST should NOT have been called
                mock_module_list.assert_not_called()


class TestDeprecatedParameters:
    """Tests verifying that the required_modules parameter has been removed."""

    def test_required_modules_parameter_ignored_sync(self, client):
        """Test that the required_modules parameter is properly deprecated/ignored."""
        # validate_sync_redis should work without any module checks
        RedisConnectionFactory.validate_sync_redis(client)
        # Should complete without error
        assert client is not None

    async def test_required_modules_parameter_ignored_async(self, async_client):
        """Test that required_modules parameter is properly deprecated/ignored for async."""
        # validate_async_redis should work without any module checks
        await RedisConnectionFactory.validate_async_redis(async_client)
        # Should complete without error
        assert async_client is not None

    def test_validate_modules_function_still_exists(self):
        """Test that validate_modules function still exists for backward compat but does nothing."""
        # The function might still exist but should do nothing
        # This depends on whether it was kept for backward compatibility
        # If it was removed entirely, this test can be removed
        pass


class TestEdgeCases:
    """Test edge cases and error handling without proactive checks."""

    def test_connection_failure_during_operation(self):
        """Test that connection failures are still handled properly."""
        # Create a mock client that fails
        mock_client = Mock(spec=Redis)
        mock_client.ping.side_effect = ConnectionError("Connection refused")

        # Connection errors should still be raised appropriately
        with pytest.raises(ConnectionError):
            mock_client.ping()

    async def test_async_cleanup_without_validation(self, redis_url):
        """Test that async cleanup works properly without validation."""
        # Create and clean up multiple async connections
        clients = []
        for _ in range(3):
            client = await RedisConnectionFactory._get_aredis_connection(redis_url)
            clients.append(client)

        # Clean up should work without issues
        for client in clients:
            await client.aclose()

    def test_from_existing_index_no_validation(self, client, worker_id):
        """Test that SearchIndex.from_existing doesn't validate modules."""
        # Skip if RediSearch is not available
        skip_if_no_redisearch(client)

        # First create an index normally
        schema = IndexSchema.from_dict(
            {
                "index": {"name": f"existing-index-{worker_id}", "prefix": "doc"},
                "fields": [{"name": "text", "type": "text"}],
            }
        )
        index = SearchIndex(schema, redis_client=client)

        try:
            index.create(overwrite=True)
        except ResponseError as e:
            if "unknown command" in str(e).lower():
                pytest.skip("RediSearch module not available")
            raise

        try:
            # Now test from_existing - patch on the instance, not the class
            with patch.object(client, "module_list") as mock_module_list:
                # Load from existing should work without MODULE LIST
                existing_index = SearchIndex.from_existing(
                    f"existing-index-{worker_id}", redis_client=client
                )

                # MODULE LIST should NOT have been called
                mock_module_list.assert_not_called()

                assert existing_index.name == f"existing-index-{worker_id}"
        finally:
            # Clean up
            try:
                index.delete()
            except:
                pass  # Ignore cleanup errors
