"""
Test that module validation is not done proactively but only when operations fail.
This is the TDD test file for issue #370.

These tests use a plain Redis container without modules to accurately test
the behavior when Redis modules are not available.
"""

import asyncio
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import ResponseError
from testcontainers.redis import RedisContainer

from redisvl.extensions.router import SemanticRouter
from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.redis.connection import RedisConnectionFactory
from redisvl.schema import IndexSchema
from redisvl.utils.vectorize.base import BaseVectorizer


@pytest.fixture(scope="module")
def plain_redis_container():
    """Start a plain Redis container without Redis Stack modules."""
    # Use the official Redis image without modules
    container = RedisContainer(image="redis:7-alpine")
    container.start()
    yield container
    container.stop()


@pytest.fixture
def plain_redis_url(plain_redis_container):
    """Get connection URL for plain Redis container."""
    host = plain_redis_container.get_container_host_ip()
    port = plain_redis_container.get_exposed_port(6379)
    return f"redis://{host}:{port}"


@pytest.fixture
def plain_redis_client(plain_redis_url):
    """Create a Redis client connected to plain Redis."""
    client = Redis.from_url(plain_redis_url)
    yield client
    client.close()


@pytest.fixture
async def plain_async_redis_client(plain_redis_url):
    """Create an async Redis client connected to plain Redis."""
    client = await AsyncRedis.from_url(plain_redis_url)
    yield client
    await client.aclose()


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

    def test_connection_factory_no_validation(self, plain_redis_url):
        """Test that RedisConnectionFactory doesn't validate modules on connection."""
        # Patch module_list to track if it's called
        with patch.object(Redis, "module_list") as mock_module_list:
            # Create connection
            client = RedisConnectionFactory.get_redis_connection(plain_redis_url)

            # Verify connection works for basic operations
            client.ping()

            # MODULE LIST should NOT have been called during connection
            mock_module_list.assert_not_called()

            client.close()

    async def test_async_connection_factory_no_validation(self, plain_redis_url):
        """Test that async RedisConnectionFactory doesn't validate modules on connection."""
        # Patch module_list to track if it's called
        with patch.object(AsyncRedis, "module_list") as mock_module_list:
            # Create async connection
            client = await RedisConnectionFactory._get_aredis_connection(
                plain_redis_url
            )

            # Verify connection works for basic operations
            await client.ping()

            # MODULE LIST should NOT have been called during connection
            mock_module_list.assert_not_called()

            await client.aclose()

    def test_search_index_init_no_validation(self, plain_redis_url, sample_schema):
        """Test that SearchIndex initialization doesn't validate modules."""
        with patch.object(Redis, "module_list") as mock_module_list:
            # Create index with plain Redis
            index = SearchIndex(sample_schema, redis_url=plain_redis_url)

            # Access the Redis client (triggers lazy initialization)
            client = index._redis_client

            # Basic Redis operations should work
            client.ping()

            # MODULE LIST should NOT have been called
            mock_module_list.assert_not_called()

    async def test_async_search_index_init_no_validation(
        self, plain_redis_url, sample_schema
    ):
        """Test that AsyncSearchIndex initialization doesn't validate modules."""
        with patch.object(AsyncRedis, "module_list") as mock_module_list:
            # Create async index with plain Redis
            index = AsyncSearchIndex(sample_schema, redis_url=plain_redis_url)

            # Access the Redis client (triggers lazy initialization)
            client = await index._get_client()

            # Basic Redis operations should work
            await client.ping()

            # MODULE LIST should NOT have been called
            mock_module_list.assert_not_called()

    def test_search_index_create_fails_without_modules(
        self, plain_redis_client, sample_schema
    ):
        """Test that index.create() fails gracefully when RediSearch is not available."""
        # Create index with plain Redis client
        index = SearchIndex(sample_schema, redis_client=plain_redis_client)

        # Attempt to create index - should fail because FT.CREATE is not available
        with pytest.raises(ResponseError) as exc_info:
            index.create()

        # Error should indicate the command is unknown (module not loaded)
        assert (
            "unknown command" in str(exc_info.value).lower()
            or "ft.create" in str(exc_info.value).lower()
        )

    async def test_async_search_index_create_fails_without_modules(
        self, plain_async_redis_client, sample_schema
    ):
        """Test that async index.create() fails gracefully when RediSearch is not available."""
        # Create async index with plain Redis client
        index = AsyncSearchIndex(sample_schema, redis_client=plain_async_redis_client)

        # Attempt to create index - should fail because FT.CREATE is not available
        with pytest.raises(ResponseError) as exc_info:
            await index.create()

        # Error should indicate the command is unknown (module not loaded)
        assert (
            "unknown command" in str(exc_info.value).lower()
            or "ft.create" in str(exc_info.value).lower()
        )

    def test_search_operations_fail_without_modules(
        self, plain_redis_client, sample_schema
    ):
        """Test that search operations fail gracefully when RediSearch is not available."""
        from redisvl.exceptions import RedisSearchError

        index = SearchIndex(sample_schema, redis_client=plain_redis_client)

        # Test that operations requiring RediSearch fail appropriately
        # exists() calls listall() which uses FT._LIST - will raise ResponseError
        with pytest.raises(ResponseError) as exc_info:
            index.exists()
        assert "unknown command" in str(exc_info.value).lower()

        # info() wraps errors in RedisSearchError
        with pytest.raises(RedisSearchError) as exc_info:
            index.info()
        assert "ft.info" in str(exc_info.value).lower()

        # search() wraps errors in RedisSearchError
        with pytest.raises(RedisSearchError) as exc_info:
            index.search("test query")
        assert "ft.search" in str(exc_info.value).lower()

        # fetch() uses HGETALL or JSON.GET - these work without modules
        # So we'll skip testing fetch as it doesn't require RediSearch

    async def test_async_search_operations_fail_without_modules(
        self, plain_async_redis_client, sample_schema
    ):
        """Test that async search operations fail gracefully when RediSearch is not available."""
        from redisvl.exceptions import RedisSearchError

        index = AsyncSearchIndex(sample_schema, redis_client=plain_async_redis_client)

        # Test that operations requiring RediSearch fail appropriately
        # exists() calls listall() which uses FT._LIST - will raise ResponseError
        with pytest.raises(ResponseError) as exc_info:
            await index.exists()
        assert "unknown command" in str(exc_info.value).lower()

        # info() wraps errors in RedisSearchError
        with pytest.raises(RedisSearchError) as exc_info:
            await index.info()
        assert "ft.info" in str(exc_info.value).lower()

        # search() wraps errors in RedisSearchError
        with pytest.raises(RedisSearchError) as exc_info:
            await index.search("test query")
        assert "ft.search" in str(exc_info.value).lower()

        # fetch() uses HGETALL or JSON.GET - these work without modules
        # So we'll skip testing fetch as it doesn't require RediSearch

    def test_semantic_router_no_validation(self, plain_redis_client):
        """Test that SemanticRouter doesn't validate modules proactively."""
        from redisvl.extensions.router import Route
        from redisvl.utils.vectorize import BaseVectorizer

        # Create a mock vectorizer that inherits from BaseVectorizer
        mock_vectorizer = Mock(spec=BaseVectorizer)
        mock_vectorizer.dims = 384  # Common embedding dimension
        mock_vectorizer.dtype = "float32"
        mock_vectorizer.embed_many = Mock(return_value=[[0.1] * 384])

        # Create a simple route
        routes = [Route(name="test_route", references=["hello world"])]

        with patch.object(Redis, "module_list") as mock_module_list:
            # Create router with plain Redis - it will try to create index
            # but we're just testing that MODULE LIST is not called
            try:
                router = SemanticRouter(
                    name="test-router",
                    routes=routes,
                    vectorizer=mock_vectorizer,
                    redis_client=plain_redis_client,
                )
            except ResponseError:
                # Expected - the router will fail when trying to create the index
                # but that's OK, we just want to verify no MODULE LIST was called
                pass

            # MODULE LIST should NOT have been called
            mock_module_list.assert_not_called()

    def test_multiple_connections_no_repeated_validation(
        self, plain_redis_url, sample_schema
    ):
        """Test that creating multiple connections/indices doesn't trigger validation."""
        with patch.object(Redis, "module_list") as mock_module_list:
            # Create multiple connections and indices
            for i in range(5):
                client = RedisConnectionFactory.get_redis_connection(plain_redis_url)
                index = SearchIndex(sample_schema, redis_client=client)
                _ = index._redis_client  # Access to trigger lazy init

            # MODULE LIST should never be called
            mock_module_list.assert_not_called()

    def test_error_message_helpful_without_proactive_check(
        self, plain_redis_client, sample_schema
    ):
        """Test that error messages are still helpful when operations fail."""
        index = SearchIndex(sample_schema, redis_client=plain_redis_client)

        # Try to create index
        with pytest.raises(ResponseError) as exc_info:
            index.create()

        # The error should be clear about what went wrong
        error_str = str(exc_info.value)

        # Should mention the actual command that failed
        assert "FT.CREATE" in error_str or "unknown command" in error_str.lower()

        # User can infer from this that RediSearch module is not installed
        # without us having done a proactive check

    @pytest.mark.parametrize(
        "redis_url",
        [
            "redis://localhost:6379",
            "redis://user:pass@localhost:6379/0",
            "rediss://localhost:6379",
        ],
    )
    def test_various_connection_strings_no_validation(self, redis_url):
        """Test that various connection string formats don't trigger validation."""
        with patch("redisvl.redis.connection.Redis.from_url") as mock_from_url:
            mock_client = Mock(spec=Redis)
            mock_from_url.return_value = mock_client

            with patch.object(mock_client, "module_list") as mock_module_list:
                # Create connection
                client = RedisConnectionFactory.get_redis_connection(redis_url)

                # MODULE LIST should NOT have been called
                mock_module_list.assert_not_called()


class TestPerformanceImprovement:
    """Tests to verify performance improvements from removing proactive checks."""

    def test_no_module_list_network_call(self, plain_redis_client, sample_schema):
        """Verify that MODULE LIST network call is eliminated."""
        # Track all Redis commands
        commands_executed = []

        original_execute = plain_redis_client.execute_command

        def track_execute(cmd, *args, **kwargs):
            commands_executed.append(cmd.upper())
            return original_execute(cmd, *args, **kwargs)

        plain_redis_client.execute_command = track_execute

        # Create index
        index = SearchIndex(sample_schema, redis_client=plain_redis_client)

        # Access client (triggers lazy init in current implementation)
        _ = index._redis_client

        # MODULE should not appear in executed commands
        module_commands = [cmd for cmd in commands_executed if "MODULE" in cmd]
        assert (
            len(module_commands) == 0
        ), f"MODULE commands were executed: {module_commands}"


class TestRemovedParameters:
    """Tests verifying that the required_modules parameter has been removed."""

    def test_no_required_modules_parameter(self, plain_redis_client):
        """Test that the required_modules parameter has been removed."""
        # validate_sync_redis should work without any module checks
        RedisConnectionFactory.validate_sync_redis(plain_redis_client)
        # Should complete without error
        assert plain_redis_client is not None

    @pytest.mark.asyncio
    async def test_no_required_modules_parameter_async(self):
        """Test that the required_modules parameter has been removed from async connection."""
        with patch("redisvl.redis.connection.AsyncRedis.from_url") as mock_from_url:
            mock_client = AsyncMock()
            mock_from_url.return_value = mock_client

            # Should work without any module checks
            client = await RedisConnectionFactory._get_aredis_connection(
                "redis://localhost:6379"
            )
            assert client is not None
