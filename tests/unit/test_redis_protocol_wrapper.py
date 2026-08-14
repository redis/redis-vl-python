"""
Unit tests for the redis_protocol wrapper.
"""

from unittest.mock import Mock

import pytest
from redis import Redis
from redis.cluster import ClusterPipeline
from redis.connection import DEFAULT_RESP_VERSION

from redisvl.utils.redis_protocol import effective_protocol, get_protocol_version


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


def _client(protocol):
    """A real client carrying (or omitting) an explicit protocol.

    Built rather than mocked because redis-py's ``get_protocol_version`` is
    ``isinstance``-gated and reads ``connection_pool``, an attribute set in
    ``__init__``. A ``Mock`` fails the first check and ``Mock(spec=Redis)``
    lacks the second, so either would test the stand-in and not the helper.
    Constructing a client opens no socket.
    """
    kwargs = {} if protocol is None else {"protocol": protocol}
    return Redis.from_url("redis://localhost:6379", **kwargs)


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (2, 2),
        # A URL query string yields a str on redis-py < 8, an int from 8.0 on,
        # so both spellings have to resolve to the same number.
        ("2", 2),
        (3, 3),
        ("3", 3),
    ],
)
def test_effective_protocol_coerces_an_explicit_setting(configured, expected):
    assert effective_protocol(_client(configured)) == expected


def test_effective_protocol_defers_to_redis_py_when_unset():
    """An absent kwarg means redis-py's default, which 8.0 flipped from 2 to 3.

    Asserted against redis-py's own constant rather than a literal, because the
    point of reading ``DEFAULT_RESP_VERSION`` is that RedisVL stops encoding a
    fact about redis-py's release history.
    """
    assert effective_protocol(_client(None)) == DEFAULT_RESP_VERSION


def test_effective_protocol_survives_an_unreadable_client():
    """A ClusterPipeline with no nodes_manager must not crash the caller."""
    mock_pipeline = Mock(spec=ClusterPipeline)
    if hasattr(mock_pipeline, "nodes_manager"):
        delattr(mock_pipeline, "nodes_manager")

    assert effective_protocol(mock_pipeline) == DEFAULT_RESP_VERSION
