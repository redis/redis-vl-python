"""
Unit tests for SVS-VAMANA capability detection.

Tests the core functionality that determines if SVS-VAMANA vector indexing
is supported on the connected Redis instance.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from redisvl.exceptions import RedisModuleVersionError
from redisvl.redis.connection import (
    VectorSupport,
    check_vector_capabilities,
    check_vector_capabilities_async,
    compare_versions,
    format_module_version,
)


def test_format_version_20810():
    """Test formatting version 20810 -> 2.8.10"""
    assert format_module_version(20810) == "2.8.10"


def test_compare_greater_version():
    """Test version comparison: greater version returns True."""
    assert compare_versions("8.2.0", "8.1.0") is True
    assert compare_versions("8.2.1", "8.2.0") is True
    assert compare_versions("9.0.0", "8.2.0") is True


def test_compare_lesser_version():
    """Test version comparison: lesser version returns False."""
    assert compare_versions("7.2.4", "8.2.0") is False
    assert compare_versions("8.1.9", "8.2.0") is False
    assert compare_versions("8.2.0", "8.2.1") is False


def test_check_vector_capabilities_supported():
    """Test check_vector_capabilities when SVS is supported."""
    mock_client = Mock()
    mock_client.info.return_value = {"redis_version": "8.2.0"}

    with patch(
        "redisvl.redis.connection.RedisConnectionFactory.get_modules"
    ) as mock_get_modules:
        mock_get_modules.return_value = {"search": 20810, "searchlight": 20810}

        caps = check_vector_capabilities(mock_client)

        assert caps.redis_version == "8.2.0"
        assert caps.search_version == 20810
        assert caps.searchlight_version == 20810
        assert caps.svs_vamana_supported is True


def test_check_vector_capabilities_old_redis():
    """Test check_vector_capabilities with old Redis version."""
    mock_client = Mock()
    mock_client.info.return_value = {"redis_version": "7.2.4"}

    with patch(
        "redisvl.redis.connection.RedisConnectionFactory.get_modules"
    ) as mock_get_modules:
        mock_get_modules.return_value = {"search": 20810, "searchlight": 20810}

        caps = check_vector_capabilities(mock_client)

        assert caps.redis_version == "7.2.4"
        assert caps.svs_vamana_supported is False


def test_check_vector_capabilities_old_modules():
    """Test check_vector_capabilities with old module versions."""
    mock_client = Mock()
    mock_client.info.return_value = {"redis_version": "8.2.0"}

    with patch(
        "redisvl.redis.connection.RedisConnectionFactory.get_modules"
    ) as mock_get_modules:
        mock_get_modules.return_value = {"search": 20612, "searchlight": 20612}

        caps = check_vector_capabilities(mock_client)

        assert caps.search_version == 20612
        assert caps.searchlight_version == 20612
        assert caps.svs_vamana_supported is False


@pytest.mark.asyncio
async def test_check_vector_capabilities_async_supported():
    """Test check_vector_capabilities_async when SVS is supported."""
    mock_client = AsyncMock()
    mock_client.info.return_value = {"redis_version": "8.2.0"}

    with patch(
        "redisvl.redis.connection.RedisConnectionFactory.get_modules_async"
    ) as mock_get_modules:
        mock_get_modules.return_value = {"search": 20810, "searchlight": 20810}

        caps = await check_vector_capabilities_async(mock_client)

        assert caps.redis_version == "8.2.0"
        assert caps.search_version == 20810
        assert caps.svs_vamana_supported is True


def test_for_svs_vamana_error_message():
    """Test RedisModuleVersionError.for_svs_vamana creates proper exception."""
    caps = VectorSupport(
        redis_version="7.2.4",
        search_version=20612,
        searchlight_version=20612,
        svs_vamana_supported=False,
    )

    error = RedisModuleVersionError.for_svs_vamana(caps, "8.2.0")

    error_msg = str(error)
    assert "SVS-VAMANA requires Redis >= 8.2.0" in error_msg
    assert "RediSearch >= 2.8.10" in error_msg
    assert "Redis 7.2.4" in error_msg
    assert "RediSearch 2.6.12" in error_msg
    assert "SearchLight 2.6.12" in error_msg
