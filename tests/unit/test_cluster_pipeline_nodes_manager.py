"""
Unit tests for issue #365: ClusterPipeline AttributeError
https://github.com/redis/redis-vl-python/issues/365

The issue reports that index.load() fails with:
AttributeError: 'ClusterPipeline' object has no attribute 'nodes_manager'
when using Redis Cluster.

The error occurs specifically in get_protocol_version() function from redis-py
when called with a ClusterPipeline object.
"""

from unittest.mock import Mock, patch

import pytest
from redis.cluster import ClusterPipeline
from redis.commands.helpers import get_protocol_version

from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema


class TestClusterPipelineNodeManagerIssue:
    """Unit tests for ClusterPipeline AttributeError issue #365."""

    def test_get_protocol_version_with_cluster_pipeline_error(self):
        """
        Test that reproduces the get_protocol_version() error with ClusterPipeline.

        This test directly calls get_protocol_version() with a ClusterPipeline object
        to reproduce the AttributeError: 'ClusterPipeline' object has no attribute 'nodes_manager'.
        """
        # Create a mock ClusterPipeline that doesn't have nodes_manager attribute
        mock_pipeline = Mock(spec=ClusterPipeline)
        # Ensure nodes_manager attribute doesn't exist
        del mock_pipeline.nodes_manager

        # This should trigger the AttributeError
        with pytest.raises(
            AttributeError, match="Mock object has no attribute 'nodes_manager'"
        ):
            get_protocol_version(mock_pipeline)

    def test_get_protocol_version_cluster_pipeline_missing_nodes_manager(self):
        """
        Test get_protocol_version() fails when ClusterPipeline lacks nodes_manager.
        """
        # Create actual ClusterPipeline mock with specific behavior
        mock_pipeline = Mock()
        mock_pipeline.__class__ = ClusterPipeline

        # Remove nodes_manager to simulate the actual issue
        if hasattr(mock_pipeline, "nodes_manager"):
            delattr(mock_pipeline, "nodes_manager")

        # The error occurs when trying to access nodes_manager.connection_kwargs
        with pytest.raises(AttributeError):
            get_protocol_version(mock_pipeline)

    def test_search_index_demonstrates_cluster_pipeline_issue_before_fix(self):
        """
        Test that demonstrates the issue before the fix was applied.
        This test shows what would happen if we still used get_protocol_version directly.
        """
        from redis.cluster import ClusterPipeline
        from redis.commands.helpers import get_protocol_version

        # Mock the ClusterPipeline without nodes_manager
        mock_client = Mock(spec=ClusterPipeline)
        if hasattr(mock_client, "nodes_manager"):
            delattr(mock_client, "nodes_manager")

        # This would fail with the original get_protocol_version
        with pytest.raises(
            AttributeError, match="Mock object has no attribute 'nodes_manager'"
        ):
            get_protocol_version(mock_client)

    def test_proposed_fix_get_protocol_version_wrapper(self):
        """
        Test a proposed fix that wraps get_protocol_version to handle ClusterPipeline safely.
        """
        from redis.cluster import ClusterPipeline

        def get_protocol_version(client):
            """
            Safe wrapper for get_protocol_version that handles ClusterPipeline.

            ClusterPipeline doesn't have nodes_manager attribute, so we need to
            get the protocol version from the underlying cluster client.
            """
            if isinstance(client, ClusterPipeline):
                # For ClusterPipeline, try to get protocol from the cluster client
                if hasattr(client, "_redis_cluster") and client._redis_cluster:
                    return get_protocol_version(client._redis_cluster)
                else:
                    # Fallback to protocol 3 if we can't determine
                    return "3"
            else:
                return get_protocol_version(client)

        # Test with regular client (should work normally)
        mock_regular_client = Mock()
        mock_regular_client.nodes_manager.connection_kwargs.get.return_value = "3"

        # Test with ClusterPipeline (should use fallback)
        mock_pipeline = Mock(spec=ClusterPipeline)
        if hasattr(mock_pipeline, "nodes_manager"):
            delattr(mock_pipeline, "nodes_manager")

        # Test the safe wrapper
        result = get_protocol_version(mock_pipeline)
        assert result == "3"  # Should fallback to protocol 3

        # Test with pipeline that has _redis_cluster - simplified test
        mock_pipeline_with_cluster = Mock(spec=ClusterPipeline)
        if hasattr(mock_pipeline_with_cluster, "nodes_manager"):
            delattr(mock_pipeline_with_cluster, "nodes_manager")
        mock_pipeline_with_cluster._redis_cluster = None  # No cluster client available

        # Should still fallback gracefully
        result = get_protocol_version(mock_pipeline_with_cluster)
        assert result == "3"  # Should fallback to protocol 3

    def test_get_protocol_version_import_and_usage(self):
        """
        Test that the get_protocol_version function can be imported and used
        as a replacement for get_protocol_version in the fixed code.
        """
        from redis.cluster import ClusterPipeline

        from redisvl.utils.redis_protocol import get_protocol_version

        # Test with ClusterPipeline mock
        mock_pipeline = Mock(spec=ClusterPipeline)
        if hasattr(mock_pipeline, "nodes_manager"):
            delattr(mock_pipeline, "nodes_manager")

        # This should not raise an exception anymore
        result = get_protocol_version(mock_pipeline)
        assert result == "3"  # Should fallback to protocol 3

    def test_get_protocol_version_works_with_mocked_cluster_pipeline(self):
        """
        Test that get_protocol_version works with a mocked ClusterPipeline.
        This demonstrates the fix working without needing SearchIndex integration.
        """
        from redisvl.utils.redis_protocol import get_protocol_version

        # Mock the redis client to be a ClusterPipeline without nodes_manager
        mock_client = Mock()
        mock_client.__class__ = ClusterPipeline
        if hasattr(mock_client, "nodes_manager"):
            delattr(mock_client, "nodes_manager")

        # Now test that get_protocol_version works without error
        protocol_version = get_protocol_version(mock_client)
        assert protocol_version == "3"  # Should fallback gracefully

        # The safe version should not raise AttributeError
        # This demonstrates the fix working
        assert protocol_version in ["2", "3"]

    def test_clusterpipeline_with_redis_cluster_returning_resp2(self):
        """
        Test ClusterPipeline with _redis_cluster that returns RESP2 protocol.
        """
        from unittest.mock import patch

        from redisvl.utils.redis_protocol import get_protocol_version

        mock_pipeline = Mock(spec=ClusterPipeline)
        mock_cluster = Mock()
        mock_pipeline._redis_cluster = mock_cluster

        # Mock redis_get_protocol_version to return "2" when called with mock_cluster
        with patch(
            "redisvl.utils.redis_protocol.redis_get_protocol_version"
        ) as mock_get:
            mock_get.return_value = "2"
            result = get_protocol_version(mock_pipeline)
            assert result == "2"
            mock_get.assert_called_once_with(mock_cluster)

    def test_async_clusterpipeline_without_nodes_manager(self):
        """
        Test AsyncClusterPipeline without nodes_manager attribute.
        """
        from redis.asyncio.cluster import ClusterPipeline as AsyncClusterPipeline

        from redisvl.utils.redis_protocol import get_protocol_version

        mock_pipeline = Mock(spec=AsyncClusterPipeline)
        if hasattr(mock_pipeline, "nodes_manager"):
            delattr(mock_pipeline, "nodes_manager")
        mock_pipeline._redis_cluster = None

        # Should fallback to "3" for async pipelines
        result = get_protocol_version(mock_pipeline)
        assert result == "3"

    def test_async_clusterpipeline_with_valid_cluster(self):
        """
        Test AsyncClusterPipeline with valid _redis_cluster attribute.
        """
        from redis.asyncio.cluster import ClusterPipeline as AsyncClusterPipeline

        from redisvl.utils.redis_protocol import get_protocol_version

        mock_pipeline = Mock(spec=AsyncClusterPipeline)
        mock_cluster = Mock()
        mock_cluster.nodes_manager.connection_kwargs.get.return_value = "3"
        mock_pipeline._redis_cluster = mock_cluster

        result = get_protocol_version(mock_pipeline)
        assert result == "3"

    def test_exception_during_protocol_detection(self):
        """
        Test that exceptions during protocol detection are handled gracefully.
        """
        from redisvl.utils.redis_protocol import get_protocol_version

        mock_pipeline = Mock(spec=ClusterPipeline)
        mock_cluster = Mock()
        # Simulate exception when accessing connection_kwargs
        mock_cluster.nodes_manager.connection_kwargs.get.side_effect = RuntimeError(
            "Connection error"
        )
        mock_pipeline._redis_cluster = mock_cluster

        # Should not raise, should fallback to "3"
        result = get_protocol_version(mock_pipeline)
        assert result == "3"

    def test_regular_redis_client_passthrough(self):
        """
        Test that regular Redis clients pass through to standard get_protocol_version.
        """
        from unittest.mock import patch

        from redis import Redis

        from redisvl.utils.redis_protocol import get_protocol_version

        mock_client = Mock(spec=Redis)

        # Mock redis_get_protocol_version to return "2" for regular clients
        with patch(
            "redisvl.utils.redis_protocol.redis_get_protocol_version"
        ) as mock_get:
            mock_get.return_value = "2"
            result = get_protocol_version(mock_client)
            assert result == "2"
            mock_get.assert_called_once_with(mock_client)

    def test_none_client_handling(self):
        """
        Test handling of None as client parameter.
        """
        from redisvl.utils.redis_protocol import get_protocol_version

        # Should handle None gracefully
        result = get_protocol_version(None)
        assert result == "3"  # Should fallback

    def test_integer_protocol_version_handling(self):
        """
        Test handling of integer protocol versions (instead of strings).
        """
        from unittest.mock import patch

        from redisvl.utils.redis_protocol import get_protocol_version

        mock_client = Mock()

        # Mock redis_get_protocol_version to return integer 3
        with patch(
            "redisvl.utils.redis_protocol.redis_get_protocol_version"
        ) as mock_get:
            mock_get.return_value = 3
            result = get_protocol_version(mock_client)
            assert result == 3  # Should handle integer return

    def test_clusterpipeline_attribute_chain_partial(self):
        """
        Test ClusterPipeline with incomplete attribute chains.
        """
        from redisvl.utils.redis_protocol import get_protocol_version

        # Test with _redis_cluster having no nodes_manager
        mock_pipeline = Mock(spec=ClusterPipeline)
        mock_cluster = Mock()
        delattr(mock_cluster, "nodes_manager")
        mock_pipeline._redis_cluster = mock_cluster

        result = get_protocol_version(mock_pipeline)
        assert result == "3"  # Should fallback

    def test_multiple_fallback_scenarios(self):
        """
        Test that multiple types of failures all fallback correctly.
        """
        from redisvl.utils.redis_protocol import get_protocol_version

        test_cases = [
            # ClusterPipeline with no _redis_cluster
            (Mock(spec=ClusterPipeline), "3"),
            # ClusterPipeline with None _redis_cluster
            (Mock(spec=ClusterPipeline, _redis_cluster=None), "3"),
            # Unknown client type
            (Mock(), "3"),
            # String client (edge case)
            ("not_a_client", "3"),
        ]

        for client, expected in test_cases:
            result = get_protocol_version(client)
            assert result == expected
