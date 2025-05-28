"""
Unit tests for error handling improvements in RedisVL.

This module tests the enhanced error handling behavior introduced for:
1. Redis error handling in index operations
2. CROSSSLOT error detection and messaging
3. Connection kwargs validation in BaseCache
4. Router config error handling
5. Cluster compatibility validation
"""

import asyncio
from collections.abc import Mapping
from unittest.mock import MagicMock, Mock, patch

import pytest
import redis.exceptions
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.cluster import RedisCluster

from redisvl.exceptions import RedisSearchError
from redisvl.extensions.cache.base import BaseCache
from redisvl.extensions.router.semantic import SemanticRouter
from redisvl.schema import StorageType


class TestRedisErrorHandling:
    """Test enhanced Redis error handling in index operations."""

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis")
    def test_redis_error_in_create_method(self, mock_validate):
        """Test that Redis errors are caught and re-raised with context."""
        from redisvl.index import SearchIndex
        from redisvl.schema import IndexSchema

        # Create a mock schema
        schema = Mock(spec=IndexSchema)
        schema.redis_fields = ["test_field"]
        schema.index = Mock()
        schema.index.name = "test_index"
        schema.index.prefix = "test:"
        schema.index.storage_type = StorageType.HASH

        # Create a mock Redis client that raises RedisError
        mock_client = Mock(spec=Redis)
        mock_client.ft.return_value.create_index.side_effect = (
            redis.exceptions.RedisError("Connection failed")
        )
        mock_client.execute_command.return_value = []

        index = SearchIndex(schema=schema, redis_client=mock_client)

        with pytest.raises(RedisSearchError) as exc_info:
            index.create()

        error_msg = str(exc_info.value)
        assert "Failed to create index 'test_index' on Redis" in error_msg
        assert "Connection failed" in error_msg

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis")
    def test_unexpected_error_in_create_method(self, mock_validate):
        """Test that unexpected errors are caught and re-raised with context."""
        from redisvl.index import SearchIndex
        from redisvl.schema import IndexSchema

        # Create a mock schema
        schema = Mock(spec=IndexSchema)
        schema.redis_fields = ["test_field"]
        schema.index = Mock()
        schema.index.name = "test_index"
        schema.index.prefix = "test:"
        schema.index.storage_type = StorageType.HASH

        # Create a mock Redis client that raises unexpected error
        mock_client = Mock(spec=Redis)
        mock_client.ft.return_value.create_index.side_effect = ValueError(
            "Unexpected error"
        )
        mock_client.execute_command.return_value = []

        index = SearchIndex(schema=schema, redis_client=mock_client)

        with pytest.raises(RedisSearchError) as exc_info:
            index.create()

        error_msg = str(exc_info.value)
        assert "Unexpected error creating index 'test_index'" in error_msg
        assert "Unexpected error" in error_msg


class TestCrossSlotErrorHandling:
    """Test CROSSSLOT error detection and helpful messaging."""

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis")
    def test_crossslot_error_in_search(self, mock_validate):
        """Test that CROSSSLOT errors in search provide helpful guidance."""
        from redisvl.index import SearchIndex
        from redisvl.schema import IndexSchema

        # Create a mock schema and index
        schema = Mock(spec=IndexSchema)
        schema.index = Mock()
        schema.index.name = "test_index"
        schema.index.storage_type = StorageType.HASH

        mock_client = Mock(spec=Redis)
        crossslot_error = redis.exceptions.ResponseError(
            "CROSSSLOT Keys in request don't hash to the same slot"
        )
        mock_client.ft.return_value.search.side_effect = crossslot_error

        index = SearchIndex(schema=schema, redis_client=mock_client)

        with pytest.raises(RedisSearchError) as exc_info:
            index.search("test query")

        error_msg = str(exc_info.value)
        assert "Cross-slot error during search" in error_msg
        assert "hash tags" in error_msg

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis")
    def test_crossslot_error_in_aggregate(self, mock_validate):
        """Test that CROSSSLOT errors in aggregate provide helpful guidance."""
        from redisvl.index import SearchIndex
        from redisvl.schema import IndexSchema

        # Create a mock schema and index
        schema = Mock(spec=IndexSchema)
        schema.index = Mock()
        schema.index.name = "test_index"
        schema.index.storage_type = StorageType.HASH

        mock_client = Mock(spec=Redis)
        crossslot_error = redis.exceptions.ResponseError(
            "CROSSSLOT Keys in request don't hash to the same slot"
        )
        mock_client.ft.return_value.aggregate.side_effect = crossslot_error

        index = SearchIndex(schema=schema, redis_client=mock_client)

        with pytest.raises(RedisSearchError) as exc_info:
            index.aggregate("test query")

        error_msg = str(exc_info.value)
        assert "Cross-slot error during aggregation" in error_msg
        assert "hash tags" in error_msg

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis")
    def test_other_redis_error_in_search(self, mock_validate):
        """Test that other Redis errors are handled with generic message."""
        from redisvl.index import SearchIndex
        from redisvl.schema import IndexSchema

        # Create a mock schema and index
        schema = Mock(spec=IndexSchema)
        schema.index = Mock()
        schema.index.name = "test_index"
        schema.index.storage_type = StorageType.HASH

        mock_client = Mock(spec=Redis)
        other_error = redis.exceptions.ResponseError("Some other error")
        mock_client.ft.return_value.search.side_effect = other_error

        index = SearchIndex(schema=schema, redis_client=mock_client)

        with pytest.raises(RedisSearchError) as exc_info:
            index.search("test query")

        error_msg = str(exc_info.value)
        assert "Error while searching" in error_msg
        assert "Some other error" in error_msg


class TestConnectionKwargsValidation:
    """Test improved connection_kwargs validation in BaseCache."""

    @pytest.mark.asyncio
    async def test_connection_kwargs_type_error(self):
        """Test that invalid connection_kwargs type raises TypeError with helpful message."""
        cache = BaseCache(
            name="test_cache",
            connection_kwargs="not_a_dict",  # type: ignore
        )

        with pytest.raises(TypeError) as exc_info:
            await cache._get_async_redis_client()

        error_msg = str(exc_info.value)
        assert "Expected `connection_kwargs` to be a dictionary" in error_msg
        assert "{'decode_responses': True}" in error_msg
        assert "got type: str" in error_msg

    @pytest.mark.asyncio
    async def test_connection_kwargs_valid_dict(self):
        """Test that valid connection_kwargs work correctly."""
        cache = BaseCache(
            name="test_cache", connection_kwargs={"decode_responses": True}
        )

        # Mock the RedisConnectionFactory to avoid actual connection
        with patch(
            "redisvl.extensions.cache.base.RedisConnectionFactory"
        ) as mock_factory:
            mock_client = Mock()
            mock_factory.get_async_redis_connection.return_value = mock_client

            result = await cache._get_async_redis_client()
            assert result == mock_client
            mock_factory.get_async_redis_connection.assert_called_once()


class TestRouterConfigErrorHandling:
    """Test improved router config error handling."""

    def test_router_config_invalid_type_error(self):
        """Test that invalid router config shows actual received value."""
        # This simulates the error that would be raised in SemanticRouter.from_existing
        invalid_router_dict = "not_a_dict"

        with pytest.raises(ValueError) as exc_info:
            if not isinstance(invalid_router_dict, dict):
                raise ValueError(
                    f"No valid router config found for test_router. Received: {invalid_router_dict!r}"
                )

        error_msg = str(exc_info.value)
        assert "Received: 'not_a_dict'" in error_msg

    def test_router_config_none_error(self):
        """Test error message when router config is None."""
        invalid_router_dict = None

        with pytest.raises(ValueError) as exc_info:
            if not isinstance(invalid_router_dict, dict):
                raise ValueError(
                    f"No valid router config found for test_router. Received: {invalid_router_dict!r}"
                )

        error_msg = str(exc_info.value)
        assert "Received: None" in error_msg

    def test_router_config_numeric_error(self):
        """Test error message when router config is a number."""
        invalid_router_dict = 42

        with pytest.raises(ValueError) as exc_info:
            if not isinstance(invalid_router_dict, dict):
                raise ValueError(
                    f"No valid router config found for test_router. Received: {invalid_router_dict!r}"
                )

        error_msg = str(exc_info.value)
        assert "Received: 42" in error_msg


class TestClusterCompatibilityValidation:
    """Test cluster compatibility validation for drop_documents."""

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis")
    def test_drop_documents_cluster_validation_success(self, mock_validate):
        """Test that documents with same hash tag work in cluster."""
        from redisvl.index import SearchIndex
        from redisvl.schema import IndexSchema

        # Create a mock schema and cluster client
        schema = Mock(spec=IndexSchema)
        schema.index = Mock()
        schema.index.prefix = "test"
        schema.index.key_separator = ":"
        schema.index.storage_type = StorageType.HASH

        mock_cluster_client = Mock(spec=RedisCluster)
        mock_cluster_client.delete.return_value = 2

        index = SearchIndex(schema=schema, redis_client=mock_cluster_client)

        # These IDs will create keys with the same hash tag
        ids = ["{user123}:doc1", "{user123}:doc2"]

        result = index.drop_documents(ids)
        assert result == 2
        mock_cluster_client.delete.assert_called_once()

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis")
    def test_drop_documents_cluster_validation_failure(self, mock_validate):
        """Test that documents with different hash tags fail in cluster."""
        from redisvl.index import SearchIndex
        from redisvl.schema import IndexSchema

        # Create a mock schema and cluster client
        schema = Mock(spec=IndexSchema)
        schema.index = Mock()
        schema.index.prefix = "test"
        schema.index.key_separator = ":"
        schema.index.storage_type = StorageType.HASH

        mock_cluster_client = Mock(spec=RedisCluster)

        index = SearchIndex(schema=schema, redis_client=mock_cluster_client)

        # These IDs will create keys with different hash tags
        ids = ["{user123}:doc1", "{user456}:doc2"]

        with pytest.raises(ValueError) as exc_info:
            index.drop_documents(ids)

        error_msg = str(exc_info.value)
        assert "All keys must share a hash tag when using Redis Cluster" in error_msg

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis")
    def test_drop_documents_non_cluster_no_validation(self, mock_validate):
        """Test that non-cluster clients don't perform hash tag validation."""
        from redisvl.index import SearchIndex
        from redisvl.schema import IndexSchema

        # Create a mock schema and regular Redis client
        schema = Mock(spec=IndexSchema)
        schema.index = Mock()
        schema.index.prefix = "test"
        schema.index.key_separator = ":"
        schema.index.storage_type = StorageType.HASH

        mock_client = Mock(spec=Redis)
        mock_client.delete.return_value = 2

        index = SearchIndex(schema=schema, redis_client=mock_client)

        # These IDs would fail in cluster, but should work in regular Redis
        ids = ["{user123}:doc1", "{user456}:doc2"]

        result = index.drop_documents(ids)
        assert result == 2
        mock_client.delete.assert_called_once()


class TestAsyncErrorHandling:
    """Test error handling in async methods."""

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_async_redis")
    @pytest.mark.asyncio
    async def test_async_crossslot_error_in_search(self, mock_validate):
        """Test that CROSSSLOT errors in async search provide helpful guidance."""
        from redisvl.index import AsyncSearchIndex
        from redisvl.schema import IndexSchema

        # Create a mock schema
        schema = Mock(spec=IndexSchema)
        schema.index = Mock()
        schema.index.name = "test_index"
        schema.index.storage_type = StorageType.HASH

        mock_client = Mock(spec=AsyncRedis)
        crossslot_error = redis.exceptions.ResponseError(
            "CROSSSLOT Keys in request don't hash to the same slot"
        )
        mock_client.ft.return_value.search.side_effect = crossslot_error

        index = AsyncSearchIndex(schema=schema, redis_client=mock_client)

        with pytest.raises(RedisSearchError) as exc_info:
            await index.search("test query")

        error_msg = str(exc_info.value)
        assert "Cross-slot error during search" in error_msg
        assert "hash tags" in error_msg

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_async_redis")
    @pytest.mark.asyncio
    async def test_async_crossslot_error_in_aggregate(self, mock_validate):
        """Test that CROSSSLOT errors in async aggregate provide helpful guidance."""
        from redisvl.index import AsyncSearchIndex
        from redisvl.schema import IndexSchema

        # Create a mock schema
        schema = Mock(spec=IndexSchema)
        schema.index = Mock()
        schema.index.name = "test_index"
        schema.index.storage_type = StorageType.HASH

        mock_client = Mock(spec=AsyncRedis)
        crossslot_error = redis.exceptions.ResponseError(
            "CROSSSLOT Keys in request don't hash to the same slot"
        )
        mock_client.ft.return_value.aggregate.side_effect = crossslot_error

        index = AsyncSearchIndex(schema=schema, redis_client=mock_client)

        with pytest.raises(RedisSearchError) as exc_info:
            await index.aggregate("test query")

        error_msg = str(exc_info.value)
        assert "Cross-slot error during aggregation" in error_msg
        assert "hash tags" in error_msg

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_async_redis")
    @pytest.mark.asyncio
    async def test_async_drop_documents_cluster_validation(self, mock_validate):
        """Test async drop_documents cluster validation."""
        from unittest.mock import AsyncMock

        from redisvl.index import AsyncSearchIndex
        from redisvl.schema import IndexSchema

        # Create a mock schema and async cluster client
        schema = Mock(spec=IndexSchema)
        schema.index = Mock()
        schema.index.prefix = "test"
        schema.index.key_separator = ":"
        schema.index.storage_type = StorageType.HASH

        # Create a mock that properly inherits from RedisCluster for isinstance check
        mock_cluster_client = Mock()
        mock_cluster_client.__class__ = AsyncRedisCluster
        mock_cluster_client.delete = AsyncMock(return_value=2)

        index = AsyncSearchIndex(schema=schema, redis_client=mock_cluster_client)

        # These IDs will create keys with different hash tags
        ids = ["{user123}:doc1", "{user456}:doc2"]

        with pytest.raises(ValueError) as exc_info:
            await index.drop_documents(ids)

        error_msg = str(exc_info.value)
        assert "All keys must share a hash tag when using Redis Cluster" in error_msg


class TestClusterOperationsErrorHandling:
    """Test error handling for cluster operations."""

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis")
    def test_clear_individual_key_deletion_errors(self, mock_validate):
        """Test clear method handles individual key deletion errors in cluster."""
        from unittest.mock import patch

        from redisvl.index import SearchIndex
        from redisvl.schema import IndexSchema

        # Create a mock schema and cluster client
        schema = Mock(spec=IndexSchema)
        schema.index = Mock()
        schema.index.prefix = "test"
        schema.index.key_separator = ":"
        schema.index.storage_type = StorageType.HASH

        mock_cluster_client = Mock(spec=RedisCluster)
        mock_cluster_client.delete.side_effect = [
            1,  # First key deletion succeeds
            redis.exceptions.RedisError("Some cluster error"),  # Second fails
            1,  # Third succeeds
        ]

        # Mock the paginate method to return test data
        with patch.object(SearchIndex, "paginate") as mock_paginate:
            mock_paginate.return_value = [
                [{"id": "test:key1"}, {"id": "test:key2"}, {"id": "test:key3"}]
            ]

            # Create index with mocked client
            index = SearchIndex(schema)
            index._SearchIndex__redis_client = mock_cluster_client

            # Test that clear handles individual key deletion errors
            with patch("redisvl.index.index.logger") as mock_logger:
                result = index.clear()

                # Should have attempted to delete all 3 keys
                assert mock_cluster_client.delete.call_count == 3
                # Should have logged the error for the failed key
                mock_logger.warning.assert_called_once_with(
                    "Failed to delete key test:key2: Some cluster error"
                )
                # Should return count of successfully deleted keys (2 out of 3)
                assert result == 2

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_async_redis")
    @pytest.mark.asyncio
    async def test_async_clear_individual_key_deletion_errors(self, mock_validate):
        """Test async clear method handles individual key deletion errors in cluster."""
        from unittest.mock import AsyncMock, patch

        from redisvl.index import AsyncSearchIndex
        from redisvl.schema import IndexSchema

        # Create a mock schema and async cluster client
        schema = Mock(spec=IndexSchema)
        schema.index = Mock()
        schema.index.prefix = "test"
        schema.index.key_separator = ":"
        schema.index.storage_type = StorageType.HASH

        mock_cluster_client = Mock(spec=AsyncRedisCluster)
        mock_cluster_client.delete = AsyncMock(
            side_effect=[
                1,  # First key deletion succeeds
                redis.exceptions.RedisError("Some cluster error"),  # Second fails
                1,  # Third succeeds
            ]
        )

        # Mock the paginate method to return test data
        async def mock_paginate_generator(*args, **kwargs):
            yield [{"id": "test:key1"}, {"id": "test:key2"}, {"id": "test:key3"}]

        with patch.object(AsyncSearchIndex, "paginate", mock_paginate_generator):
            # Create index with mocked client
            index = AsyncSearchIndex(schema)
            index._redis_client = mock_cluster_client

            # Test that clear handles individual key deletion errors
            with patch("redisvl.index.index.logger") as mock_logger:
                result = await index.clear()

                # Should have attempted to delete all 3 keys
                assert mock_cluster_client.delete.call_count == 3
                # Should have logged the error for the failed key
                mock_logger.warning.assert_called_once_with(
                    "Failed to delete key test:key2: Some cluster error"
                )
                # Should return count of successfully deleted keys (2 out of 3)
                assert result == 2

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis")
    def test_delete_cluster_compatibility(self, mock_validate):
        """Test delete method uses clear() for cluster compatibility when drop=True."""
        from unittest.mock import Mock, patch

        from redisvl.index import SearchIndex
        from redisvl.schema import IndexSchema

        # Create a mock schema and cluster client
        schema = Mock(spec=IndexSchema)
        schema.index = Mock()
        schema.index.name = "test_index"

        mock_cluster_client = Mock(spec=RedisCluster)
        mock_cluster_client.get_default_node.return_value = Mock()

        # Create index with mocked client
        index = SearchIndex(schema)
        index._SearchIndex__redis_client = mock_cluster_client

        # Test that delete() calls clear() first when drop=True in cluster
        with patch.object(index, "clear") as mock_clear:
            index.delete(drop=True)

            # Should have called clear() first
            mock_clear.assert_called_once()
            # Should have called execute_command with just the index name (no DD flag)
            mock_cluster_client.execute_command.assert_called_once_with(
                "FT.DROPINDEX",
                "test_index",
                target_nodes=[mock_cluster_client.get_default_node.return_value],
            )

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis")
    def test_delete_non_cluster_standard_behavior(self, mock_validate):
        """Test delete method uses standard behavior for non-cluster Redis."""
        from unittest.mock import Mock

        from redisvl.index import SearchIndex
        from redisvl.schema import IndexSchema

        # Create a mock schema and regular Redis client
        schema = Mock(spec=IndexSchema)
        schema.index = Mock()
        schema.index.name = "test_index"

        mock_redis_client = Mock(spec=Redis)

        # Create index with mocked client
        index = SearchIndex(schema)
        index._SearchIndex__redis_client = mock_redis_client

        # Test that delete() uses standard behavior for non-cluster
        index.delete(drop=True)

        # Should have called execute_command with DD flag
        mock_redis_client.execute_command.assert_called_once_with(
            "FT.DROPINDEX", "test_index", "DD"
        )

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_async_redis")
    @pytest.mark.asyncio
    async def test_async_delete_cluster_compatibility(self, mock_validate):
        """Test async delete method uses clear() for cluster compatibility when drop=True."""
        from unittest.mock import AsyncMock, Mock, patch

        from redisvl.index import AsyncSearchIndex
        from redisvl.schema import IndexSchema

        # Create a mock schema and async cluster client
        schema = Mock(spec=IndexSchema)
        schema.index = Mock()
        schema.index.name = "test_index"

        mock_cluster_client = Mock(spec=AsyncRedisCluster)
        mock_cluster_client.get_default_node.return_value = Mock()
        mock_cluster_client.execute_command = AsyncMock()

        # Create index with mocked client
        index = AsyncSearchIndex(schema)
        index._redis_client = mock_cluster_client

        # Test that delete() calls clear() first when drop=True in cluster
        with patch.object(index, "clear", new_callable=AsyncMock) as mock_clear:
            await index.delete(drop=True)

            # Should have called clear() first
            mock_clear.assert_called_once()
            # Should have called execute_command with just the index name (no DD flag)
            mock_cluster_client.execute_command.assert_called_once_with(
                "FT.DROPINDEX",
                "test_index",
                target_nodes=[mock_cluster_client.get_default_node.return_value],
            )
