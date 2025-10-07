"""
Unit tests for the redis_protocol wrapper.
"""

from unittest.mock import Mock

import pytest
from redis.cluster import ClusterPipeline

from redisvl.utils.redis_protocol import get_protocol_version


def test_get_protocol_version_handles_missing_nodes_manager():
    """
    Test that get_protocol_version returns None when ClusterPipeline
    lacks nodes_manager attribute (issue #365).
    """
    # Create a mock ClusterPipeline without nodes_manager
    mock_pipeline = Mock(spec=ClusterPipeline)
    # Ensure nodes_manager doesn't exist
    if hasattr(mock_pipeline, "nodes_manager"):
        delattr(mock_pipeline, "nodes_manager")

    # Should return None without raising AttributeError
    result = get_protocol_version(mock_pipeline)
    assert result is None


def test_get_protocol_version_with_valid_nodes_manager():
    """
    Test that get_protocol_version works correctly when nodes_manager exists.
    """
    # Create a mock ClusterPipeline with nodes_manager
    mock_pipeline = Mock(spec=ClusterPipeline)
    mock_pipeline.nodes_manager = Mock()
    mock_pipeline.nodes_manager.connection_kwargs = {"protocol": "3"}

    # Should return the protocol version
    result = get_protocol_version(mock_pipeline)
    assert result == "3"


def test_get_protocol_version_with_none_client():
    """
    Test that get_protocol_version handles None input gracefully.
    """
    result = get_protocol_version(None)
    assert result is None


def test_protocol_version_affects_never_decode():
    """
    Test that None protocol version results in NEVER_DECODE being set.
    This is the actual behavior in redisvl code.
    """
    from redis.client import NEVER_DECODE

    mock_pipeline = Mock(spec=ClusterPipeline)
    if hasattr(mock_pipeline, "nodes_manager"):
        delattr(mock_pipeline, "nodes_manager")

    protocol = get_protocol_version(mock_pipeline)

    # This simulates the code in index.py and utils.py
    options = {}
    if protocol not in ["3", 3]:
        options[NEVER_DECODE] = True

    # When protocol is None, NEVER_DECODE should be set
    assert protocol is None
    assert NEVER_DECODE in options
