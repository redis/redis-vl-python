import json

import pytest
from pydantic import ValidationError

from redisvl.extensions.cache.embeddings.schema import CacheEntry
from redisvl.redis.utils import hashify


def test_valid_cache_entry_creation():
    # Generate an entry_id first
    entry_id = hashify(f"What is AI?:text-embedding-ada-002")
    entry = CacheEntry(
        entry_id=entry_id,
        text="What is AI?",
        model_name="text-embedding-ada-002",
        embedding=[0.1, 0.2, 0.3],
    )
    assert entry.entry_id == entry_id
    assert entry.text == "What is AI?"
    assert entry.model_name == "text-embedding-ada-002"
    assert entry.embedding == [0.1, 0.2, 0.3]


def test_cache_entry_with_given_entry_id():
    entry = CacheEntry(
        entry_id="custom_id",
        text="What is AI?",
        model_name="text-embedding-ada-002",
        embedding=[0.1, 0.2, 0.3],
    )
    assert entry.entry_id == "custom_id"


def test_cache_entry_with_invalid_metadata():
    with pytest.raises(ValidationError):
        CacheEntry(
            entry_id="test_id",
            text="What is AI?",
            model_name="text-embedding-ada-002",
            embedding=[0.1, 0.2, 0.3],
            metadata="invalid_metadata",
        )


def test_cache_entry_to_dict():
    entry_id = hashify(f"What is AI?:text-embedding-ada-002")
    entry = CacheEntry(
        entry_id=entry_id,
        text="What is AI?",
        model_name="text-embedding-ada-002",
        embedding=[0.1, 0.2, 0.3],
        metadata={"author": "John"},
    )
    result = entry.to_dict()
    assert result["entry_id"] == entry_id
    assert result["text"] == "What is AI?"
    assert result["model_name"] == "text-embedding-ada-002"
    assert isinstance("embedding", str)
    assert isinstance("metadata", str)
    assert result["metadata"] == json.dumps({"author": "John"})


def test_cache_entry_deserialization():
    """Test that a CacheEntry properly deserializes data from Redis format."""
    serialized_data = {
        "entry_id": "test_id",
        "text": "What is AI?",
        "model_name": "text-embedding-ada-002",
        "embedding": json.dumps([0.1, 0.2, 0.3]),  # Serialized embedding
        "metadata": json.dumps({"source": "user_query"}),  # Serialized metadata
        "inserted_at": 1625819123.123,
    }

    entry = CacheEntry(**serialized_data)
    assert entry.entry_id == "test_id"
    assert entry.text == "What is AI?"
    assert entry.model_name == "text-embedding-ada-002"
    assert entry.embedding == [0.1, 0.2, 0.3]  # Should be deserialized
    assert entry.metadata == {"source": "user_query"}  # Should be deserialized
    assert entry.inserted_at == 1625819123.123


def test_cache_entry_with_empty_optional_fields():
    entry = CacheEntry(
        entry_id="test_id",
        text="What is AI?",
        model_name="text-embedding-ada-002",
        embedding=[0.1, 0.2, 0.3],
    )
    result = entry.to_dict()
    assert "metadata" not in result  # Empty metadata should be excluded


def test_cache_entry_timestamp_generation():
    """Test that inserted_at timestamp is automatically generated."""
    entry = CacheEntry(
        entry_id="test_id",
        text="What is AI?",
        model_name="text-embedding-ada-002",
        embedding=[0.1, 0.2, 0.3],
    )
    assert hasattr(entry, "inserted_at")
    assert isinstance(entry.inserted_at, float)

    # The timestamp should be included in the dict representation
    result = entry.to_dict()
    assert "inserted_at" in result
    assert isinstance(result["inserted_at"], float)
