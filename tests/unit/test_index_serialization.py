"""Tests for SearchIndex serialization helpers."""
import tempfile
from pathlib import Path

import pytest

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.schema import IndexSchema


@pytest.fixture
def sample_schema():
    """Create a sample schema for testing."""
    return IndexSchema.from_dict({
        "index": {
            "name": "test_index",
            "prefix": "test:",
            "storage_type": "hash",
        },
        "fields": [
            {"name": "text", "type": "text"},
            {"name": "vector", "type": "vector", "attrs": {"dims": 128, "algorithm": "flat"}},
        ]
    })


class TestSearchIndexSerialization:
    """Tests for SearchIndex serialization methods."""

    def test_to_dict_without_connection(self, sample_schema):
        """Test to_dict() excludes connection info by default."""
        index = SearchIndex(
            schema=sample_schema,
            redis_url="redis://localhost:6379",
        )
        
        config = index.to_dict()
        
        assert "index" in config
        assert config["index"]["name"] == "test_index"
        assert "_redis_url" not in config
        assert "_connection_kwargs" not in config

    def test_to_dict_with_connection(self, sample_schema):
        """Test to_dict() includes connection info when requested."""
        index = SearchIndex(
            schema=sample_schema,
            redis_url="redis://localhost:6379",
            connection_kwargs={"ssl": True, "socket_timeout": 30, "password": "secret"},
        )
        
        config = index.to_dict(include_connection=True)
        
        assert "_redis_url" in config
        assert config["_redis_url"] == "redis://localhost:6379"
        # Password should not be included (not in safe_keys)
        assert "password" not in config.get("_connection_kwargs", {})
        # Safe keys should be included
        assert config["_connection_kwargs"]["ssl"] is True
        assert config["_connection_kwargs"]["socket_timeout"] == 30

    def test_to_yaml_without_connection(self, sample_schema):
        """Test to_yaml() writes schema to YAML file."""
        index = SearchIndex(schema=sample_schema)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name
        
        try:
            index.to_yaml(path)
            
            content = Path(path).read_text()
            assert "test_index" in content
            assert "text" in content
            assert "vector" in content
        finally:
            Path(path).unlink()

    def test_to_yaml_with_connection(self, sample_schema):
        """Test to_yaml() includes connection when requested."""
        index = SearchIndex(
            schema=sample_schema,
            redis_url="redis://localhost:6379",
        )
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name
        
        try:
            index.to_yaml(path, include_connection=True)
            
            content = Path(path).read_text()
            assert "_redis_url" in content
        finally:
            Path(path).unlink()

    def test_roundtrip_dict(self, sample_schema):
        """Test that to_dict() and from_dict() roundtrip correctly."""
        original_index = SearchIndex(
            schema=sample_schema,
            connection_kwargs={"ssl": True},
        )
        
        config = original_index.to_dict()
        restored_index = SearchIndex.from_dict(config)
        
        assert restored_index.schema.index.name == original_index.schema.index.name
        assert restored_index.schema.index.prefix == original_index.schema.index.prefix


class TestAsyncSearchIndexSerialization:
    """Tests for AsyncSearchIndex serialization methods."""

    def test_to_dict_without_connection(self, sample_schema):
        """Test to_dict() excludes connection info by default."""
        index = AsyncSearchIndex(
            schema=sample_schema,
            redis_url="redis://localhost:6379",
        )
        
        config = index.to_dict()
        
        assert "index" in config
        assert config["index"]["name"] == "test_index"
        assert "_redis_url" not in config

    def test_to_dict_with_connection(self, sample_schema):
        """Test to_dict() includes connection info when requested."""
        index = AsyncSearchIndex(
            schema=sample_schema,
            redis_url="redis://localhost:6379",
            connection_kwargs={"ssl": True, "password": "secret"},
        )
        
        config = index.to_dict(include_connection=True)
        
        assert "_redis_url" in config
        # Password should not be included (not in safe_keys)
        assert "password" not in config.get("_connection_kwargs", {})

    def test_to_yaml(self, sample_schema):
        """Test to_yaml() writes schema to YAML file."""
        index = AsyncSearchIndex(schema=sample_schema)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name
        
        try:
            index.to_yaml(path)
            
            content = Path(path).read_text()
            assert "test_index" in content
        finally:
            Path(path).unlink()

    def test_roundtrip_dict(self, sample_schema):
        """Test that to_dict() and from_dict() roundtrip correctly."""
        original_index = AsyncSearchIndex(
            schema=sample_schema,
            connection_kwargs={"ssl": True},
        )
        
        config = original_index.to_dict()
        restored_index = AsyncSearchIndex.from_dict(config)
        
        assert restored_index.schema.index.name == original_index.schema.index.name
