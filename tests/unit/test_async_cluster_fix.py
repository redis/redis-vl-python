"""Unit tests for AsyncRedisCluster cluster parameter stripping fix (issue #346)."""

import pytest

from redisvl.redis.connection import _strip_cluster_from_url_and_kwargs


class TestAsyncClusterParameterStripping:
    """Test the helper function that strips cluster parameter from URLs and kwargs."""

    def test_strip_cluster_from_url_with_cluster_true(self):
        """Test stripping cluster=true from URL query string."""
        url = "redis://localhost:7001?cluster=true"
        cleaned_url, cleaned_kwargs = _strip_cluster_from_url_and_kwargs(url)

        assert cleaned_url == "redis://localhost:7001"
        assert cleaned_kwargs == {}

    def test_strip_cluster_from_url_with_other_params(self):
        """Test stripping cluster parameter while preserving other parameters."""
        url = (
            "redis://localhost:7001?cluster=true&decode_responses=true&socket_timeout=5"
        )
        cleaned_url, cleaned_kwargs = _strip_cluster_from_url_and_kwargs(
            url, some_kwarg="value"
        )

        assert "cluster" not in cleaned_url
        assert "decode_responses=true" in cleaned_url
        assert "socket_timeout=5" in cleaned_url
        assert cleaned_kwargs == {"some_kwarg": "value"}

    def test_strip_cluster_from_kwargs(self):
        """Test stripping cluster parameter from kwargs."""
        url = "redis://localhost:7001"
        kwargs = {"cluster": True, "decode_responses": True}
        cleaned_url, cleaned_kwargs = _strip_cluster_from_url_and_kwargs(url, **kwargs)

        assert cleaned_url == "redis://localhost:7001"
        assert "cluster" not in cleaned_kwargs
        assert cleaned_kwargs == {"decode_responses": True}

    def test_strip_cluster_from_both_url_and_kwargs(self):
        """Test stripping cluster parameter from both URL and kwargs."""
        url = "redis://localhost:7001?cluster=true"
        kwargs = {"cluster": True, "socket_timeout": 5}
        cleaned_url, cleaned_kwargs = _strip_cluster_from_url_and_kwargs(url, **kwargs)

        assert cleaned_url == "redis://localhost:7001"
        assert cleaned_kwargs == {"socket_timeout": 5}

    def test_no_cluster_parameter_unchanged(self):
        """Test that URLs and kwargs without cluster parameter remain unchanged."""
        url = "redis://localhost:7001?decode_responses=true"
        kwargs = {"socket_timeout": 5}
        cleaned_url, cleaned_kwargs = _strip_cluster_from_url_and_kwargs(url, **kwargs)

        assert cleaned_url == "redis://localhost:7001?decode_responses=true"
        assert cleaned_kwargs == {"socket_timeout": 5}

    def test_empty_url_query_and_kwargs(self):
        """Test handling of URL without query string and empty kwargs."""
        url = "redis://localhost:7001"
        cleaned_url, cleaned_kwargs = _strip_cluster_from_url_and_kwargs(url)

        assert cleaned_url == "redis://localhost:7001"
        assert cleaned_kwargs == {}

    def test_complex_url_with_auth_and_db(self):
        """Test complex URL with authentication and database selection."""
        url = "redis://user:password@localhost:7001/0?cluster=true&socket_timeout=5"
        cleaned_url, cleaned_kwargs = _strip_cluster_from_url_and_kwargs(url)

        assert cleaned_url == "redis://user:password@localhost:7001/0?socket_timeout=5"
        assert cleaned_kwargs == {}

    def test_cluster_false_also_stripped(self):
        """Test that cluster=false is also stripped (any cluster param should be removed)."""
        url = "redis://localhost:7001?cluster=false"
        cleaned_url, cleaned_kwargs = _strip_cluster_from_url_and_kwargs(url)

        assert cleaned_url == "redis://localhost:7001"
        assert cleaned_kwargs == {}
