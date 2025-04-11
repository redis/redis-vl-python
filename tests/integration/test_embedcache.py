import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import pytest
from redis.exceptions import ConnectionError

from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache
from redisvl.redis.utils import hashify


@pytest.fixture
def cache(redis_url):
    """Basic EmbeddingsCache fixture with cleanup."""
    cache_instance = EmbeddingsCache(
        name="test_embed_cache",
        redis_url=redis_url,
    )
    yield cache_instance
    # Clean up all keys with this prefix
    cache_instance.clear()


@pytest.fixture
def cache_with_ttl(redis_url):
    """EmbeddingsCache with TTL setting."""
    cache_instance = EmbeddingsCache(
        name="test_ttl_cache",
        ttl=2,  # 2 second TTL for testing expiration
        redis_url=redis_url,
    )
    yield cache_instance
    # Clean up all keys with this prefix
    cache_instance.clear()


@pytest.fixture
def cache_with_redis_client(client):
    """EmbeddingsCache with provided Redis client."""
    cache_instance = EmbeddingsCache(
        name="test_client_cache",
        redis_client=client,
    )
    yield cache_instance
    # Clean up all keys with this prefix
    cache_instance.clear()


@pytest.fixture
def sample_embedding_data():
    """Sample data for embedding cache tests."""
    return [
        {
            "text": "What is machine learning?",
            "model_name": "text-embedding-ada-002",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "metadata": {"source": "user_query", "category": "ai"},
        },
        {
            "text": "How do neural networks work?",
            "model_name": "text-embedding-ada-002",
            "embedding": [0.2, 0.3, 0.4, 0.5, 0.6],
            "metadata": {"source": "documentation", "category": "ai"},
        },
        {
            "text": "What's the weather like today?",
            "model_name": "text-embedding-ada-002",
            "embedding": [0.5, 0.6, 0.7, 0.8, 0.9],
            "metadata": {"source": "user_query", "category": "weather"},
        },
    ]


def test_cache_initialization(redis_url):
    """Test that the cache can be initialized with different parameters."""
    # Default initialization
    cache1 = EmbeddingsCache()
    assert cache1.name == "embedcache"
    assert cache1.ttl is None

    # Custom name and TTL
    cache2 = EmbeddingsCache(name="custom_cache", ttl=60, redis_url=redis_url)
    assert cache2.name == "custom_cache"
    assert cache2.ttl == 60

    # With redis client
    cache3 = EmbeddingsCache(redis_url=redis_url)
    client = cache3._get_redis_client()
    cache4 = EmbeddingsCache(redis_client=client)
    assert cache4._redis_client is client


def test_make_entry_id():
    """Test that entry IDs are generated consistently."""
    cache = EmbeddingsCache()
    text = "Hello world"
    model_name = "text-embedding-ada-002"

    # Test deterministic ID generation
    entry_id1 = cache._make_entry_id(text, model_name)
    entry_id2 = cache._make_entry_id(text, model_name)
    assert entry_id1 == entry_id2

    # Test different inputs produce different IDs
    different_id = cache._make_entry_id("Different text", model_name)
    assert entry_id1 != different_id

    # Test ID format
    assert isinstance(entry_id1, str)
    expected_id = hashify(f"{text}:{model_name}")
    assert entry_id1 == expected_id


def test_make_cache_key():
    """Test that cache keys are constructed properly."""
    cache = EmbeddingsCache(name="test_cache")
    text = "Hello world"
    model_name = "text-embedding-ada-002"

    # Test key construction
    key = cache._make_cache_key(text, model_name)
    entry_id = cache._make_entry_id(text, model_name)
    expected_key = f"test_cache:{entry_id}"
    assert key == expected_key

    # Test with different cache name
    cache2 = EmbeddingsCache(name="different_cache")
    key2 = cache2._make_cache_key(text, model_name)
    assert key2 == f"different_cache:{entry_id}"

    # Make sure keys are unique for different inputs
    different_key = cache._make_cache_key("Different text", model_name)
    assert key != different_key


def test_set_and_get(cache, sample_embedding_data):
    """Test setting and retrieving entries from the cache."""
    sample = sample_embedding_data[0]

    # Set the entry
    key = cache.set(
        text=sample["text"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
        metadata=sample["metadata"],
    )

    # Get the entry
    result = cache.get(sample["text"], sample["model_name"])

    # Verify the result
    assert result is not None
    assert result["text"] == sample["text"]
    assert result["model_name"] == sample["model_name"]
    assert "embedding" in result
    assert result["metadata"] == sample["metadata"]

    # Test get_by_key
    key_result = cache.get_by_key(key)
    assert key_result is not None
    assert key_result["text"] == sample["text"]

    # Test non-existent entry
    missing = cache.get("NonexistentText", sample["model_name"])
    assert missing is None

    # Test non-existent key
    missing_key = cache.get_by_key("nonexistent:key")
    assert missing_key is None


def test_exists(cache, sample_embedding_data):
    """Test checking if entries exist in the cache."""
    sample = sample_embedding_data[0]

    # Entry shouldn't exist yet
    assert not cache.exists(sample["text"], sample["model_name"])

    # Add the entry
    key = cache.set(
        text=sample["text"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
    )

    # Now it should exist
    assert cache.exists(sample["text"], sample["model_name"])

    # Test exists_by_key
    assert cache.exists_by_key(key)

    # Non-existent entries
    assert not cache.exists("NonexistentText", sample["model_name"])
    assert not cache.exists_by_key("nonexistent:key")


def test_drop(cache, sample_embedding_data):
    """Test removing entries from the cache."""
    sample = sample_embedding_data[0]

    # Add the entry
    key = cache.set(
        text=sample["text"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
    )

    # Verify it exists
    assert cache.exists_by_key(key)

    # Remove it
    cache.drop(sample["text"], sample["model_name"])

    # Verify it's gone
    assert not cache.exists_by_key(key)

    # Test drop_by_key
    key = cache.set(
        text=sample["text"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
    )
    cache.drop_by_key(key)
    assert not cache.exists_by_key(key)


def test_ttl_expiration(cache_with_ttl, sample_embedding_data):
    """Test that entries expire after TTL."""
    sample = sample_embedding_data[0]

    # Add the entry
    key = cache_with_ttl.set(
        text=sample["text"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
    )

    # Verify it exists
    assert cache_with_ttl.exists_by_key(key)

    # Wait for it to expire (TTL is 2 seconds)
    time.sleep(3)

    # Verify it's gone
    assert not cache_with_ttl.exists_by_key(key)


def test_custom_ttl(cache, sample_embedding_data):
    """Test setting a custom TTL for a specific entry."""
    sample = sample_embedding_data[0]

    # Add the entry with a 1 second TTL
    key = cache.set(
        text=sample["text"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
        ttl=1,
    )

    # Verify it exists
    assert cache.exists_by_key(key)

    # Wait for it to expire
    time.sleep(2)

    # Verify it's gone
    assert not cache.exists_by_key(key)


def test_multiple_entries(cache, sample_embedding_data):
    """Test storing and retrieving multiple entries."""
    # Store all samples
    keys = []
    for sample in sample_embedding_data:
        key = cache.set(
            text=sample["text"],
            model_name=sample["model_name"],
            embedding=sample["embedding"],
            metadata=sample.get("metadata"),
        )
        keys.append(key)

    # Check they all exist
    for i, key in enumerate(keys):
        assert cache.exists_by_key(key)
        result = cache.get_by_key(key)
        assert result["text"] == sample_embedding_data[i]["text"]

    # Drop one entry
    cache.drop_by_key(keys[0])
    assert not cache.exists_by_key(keys[0])
    assert cache.exists_by_key(keys[1])


@pytest.mark.asyncio
async def test_async_set_and_get(cache, sample_embedding_data):
    """Test async versions of set and get."""
    sample = sample_embedding_data[0]

    # Set the entry
    key = await cache.aset(
        text=sample["text"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
        metadata=sample["metadata"],
    )

    # Get the entry
    result = await cache.aget(sample["text"], sample["model_name"])

    # Verify the result
    assert result is not None
    assert result["text"] == sample["text"]
    assert result["model_name"] == sample["model_name"]
    assert "embedding" in result
    assert result["metadata"] == sample["metadata"]

    # Test aget_by_key
    key_result = await cache.aget_by_key(key)
    assert key_result is not None
    assert key_result["text"] == sample["text"]


@pytest.mark.asyncio
async def test_async_exists(cache, sample_embedding_data):
    """Test async version of exists."""
    sample = sample_embedding_data[0]

    # Entry shouldn't exist yet
    assert not await cache.aexists(sample["text"], sample["model_name"])

    # Add the entry
    key = await cache.aset(
        text=sample["text"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
    )

    # Now it should exist
    assert await cache.aexists(sample["text"], sample["model_name"])

    # Test aexists_by_key
    assert await cache.aexists_by_key(key)


@pytest.mark.asyncio
async def test_async_drop(cache, sample_embedding_data):
    """Test async version of drop."""
    sample = sample_embedding_data[0]

    # Add the entry
    key = await cache.aset(
        text=sample["text"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
    )

    # Verify it exists
    assert await cache.aexists_by_key(key)

    # Remove it
    await cache.adrop(sample["text"], sample["model_name"])

    # Verify it's gone
    assert not await cache.aexists_by_key(key)

    # Test adrop_by_key
    key = await cache.aset(
        text=sample["text"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
    )
    await cache.adrop_by_key(key)
    assert not await cache.aexists_by_key(key)


@pytest.mark.asyncio
async def test_async_ttl_expiration(cache_with_ttl, sample_embedding_data):
    """Test that entries expire after TTL in async mode."""
    sample = sample_embedding_data[0]

    # Add the entry
    key = await cache_with_ttl.aset(
        text=sample["text"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
    )

    # Verify it exists
    assert await cache_with_ttl.aexists_by_key(key)

    # Wait for it to expire (TTL is 2 seconds)
    await asyncio.sleep(3)

    # Verify it's gone
    assert not await cache_with_ttl.aexists_by_key(key)


def test_entry_id_consistency(cache, sample_embedding_data):
    """Test that entry IDs are consistent between operations."""
    sample = sample_embedding_data[0]

    # Generate an entry ID directly
    expected_id = cache._make_entry_id(sample["text"], sample["model_name"])

    # Set an entry and extract its ID from the key
    key = cache.set(
        text=sample["text"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
    )

    # Key should be cache_name:entry_id
    parts = key.split(":")
    actual_id = parts[1]

    # IDs should match
    assert actual_id == expected_id

    # Get the entry and check its ID
    result = cache.get_by_key(key)
    assert result["entry_id"] == expected_id


def test_redis_client_reuse(cache_with_redis_client, sample_embedding_data):
    """Test using the cache with a provided Redis client."""
    sample = sample_embedding_data[0]

    # Set and get an entry
    key = cache_with_redis_client.set(
        text=sample["text"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
    )

    result = cache_with_redis_client.get_by_key(key)
    assert result is not None
    assert result["text"] == sample["text"]
