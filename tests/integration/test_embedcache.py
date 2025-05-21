import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import pytest
from redis.exceptions import ConnectionError

from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache
from redisvl.redis.utils import hashify


@pytest.fixture
def cache(redis_url, worker_id):
    """Basic EmbeddingsCache fixture with cleanup."""
    cache_instance = EmbeddingsCache(
        name=f"test_embed_cache_{worker_id}",
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


def test_mset_and_mget(cache, sample_embedding_data):
    """Test batch setting and getting of embeddings."""
    # Prepare batch items
    batch_items = []
    for sample in sample_embedding_data:
        batch_items.append(
            {
                "text": sample["text"],
                "model_name": sample["model_name"],
                "embedding": sample["embedding"],
                "metadata": sample.get("metadata"),
            }
        )

    # Use mset to store embeddings
    keys = cache.mset(batch_items)
    assert len(keys) == len(batch_items)

    # Get texts and model name for mget
    texts = [item["text"] for item in batch_items]
    model_name = batch_items[0]["model_name"]  # Assuming same model

    # Test mget
    results = cache.mget(texts, model_name)
    assert len(results) == len(texts)

    # Verify all results are returned and in correct order
    for i, result in enumerate(results):
        assert result is not None
        assert result["text"] == texts[i]
        assert result["model_name"] == model_name


def test_mget_by_keys(cache, sample_embedding_data):
    """Test getting multiple embeddings by their keys."""
    # Set embeddings individually and collect keys
    keys = []
    for sample in sample_embedding_data:
        key = cache.set(
            text=sample["text"],
            model_name=sample["model_name"],
            embedding=sample["embedding"],
            metadata=sample.get("metadata"),
        )
        keys.append(key)

    # Test mget_by_keys
    results = cache.mget_by_keys(keys)
    assert len(results) == len(keys)

    # Verify all results match the original samples
    for i, result in enumerate(results):
        assert result is not None
        assert result["text"] == sample_embedding_data[i]["text"]
        assert result["model_name"] == sample_embedding_data[i]["model_name"]

    # Test with mix of existing and non-existing keys
    non_existent_key = "test_embed_cache:nonexistent"
    mixed_keys = keys[:1] + [non_existent_key] + keys[1:]
    mixed_results = cache.mget_by_keys(mixed_keys)

    assert len(mixed_results) == len(mixed_keys)
    assert mixed_results[0] is not None
    assert mixed_results[1] is None  # Non-existent key should return None
    assert mixed_results[2] is not None


def test_mexists_and_mexists_by_keys(cache, sample_embedding_data):
    """Test batch existence checks for embeddings."""
    # Set embeddings individually and collect data
    keys = []
    texts = []
    for sample in sample_embedding_data:
        key = cache.set(
            text=sample["text"],
            model_name=sample["model_name"],
            embedding=sample["embedding"],
        )
        keys.append(key)
        texts.append(sample["text"])

    model_name = sample_embedding_data[0]["model_name"]  # Assuming same model

    # Test mexists
    exist_results = cache.mexists(texts, model_name)
    assert len(exist_results) == len(texts)
    assert all(exist_results)  # All should exist

    # Test with mix of existing and non-existing texts
    non_existent_text = "This text does not exist"
    mixed_texts = texts[:1] + [non_existent_text] + texts[1:]
    mixed_results = cache.mexists(mixed_texts, model_name)

    assert len(mixed_results) == len(mixed_texts)
    assert mixed_results[0] is True
    assert mixed_results[1] is False  # Non-existent text should return False
    assert mixed_results[2] is True

    # Test mexists_by_keys
    key_exist_results = cache.mexists_by_keys(keys)
    assert len(key_exist_results) == len(keys)
    assert all(key_exist_results)  # All should exist

    # Test with mix of existing and non-existing keys
    non_existent_key = "test_embed_cache:nonexistent"
    mixed_keys = keys[:1] + [non_existent_key] + keys[1:]
    mixed_key_results = cache.mexists_by_keys(mixed_keys)

    assert len(mixed_key_results) == len(mixed_keys)
    assert mixed_key_results[0] is True
    assert mixed_key_results[1] is False  # Non-existent key should return False
    assert mixed_key_results[2] is True


def test_mdrop_and_mdrop_by_keys(cache, sample_embedding_data):
    """Test batch deletion of embeddings."""
    # Set embeddings and collect data
    keys = []
    texts = []
    for sample in sample_embedding_data:
        key = cache.set(
            text=sample["text"],
            model_name=sample["model_name"],
            embedding=sample["embedding"],
        )
        keys.append(key)
        texts.append(sample["text"])

    model_name = sample_embedding_data[0]["model_name"]  # Assuming same model

    # Test mdrop_by_keys with subset of keys
    subset_keys = keys[:2]
    cache.mdrop_by_keys(subset_keys)

    # Verify only selected keys were dropped
    for i, key in enumerate(keys):
        if i < 2:
            assert not cache.exists_by_key(key)  # Should be dropped
        else:
            assert cache.exists_by_key(key)  # Should still exist

    # Reset for mdrop test
    cache.clear()
    keys = []
    texts = []
    for sample in sample_embedding_data:
        key = cache.set(
            text=sample["text"],
            model_name=sample["model_name"],
            embedding=sample["embedding"],
        )
        keys.append(key)
        texts.append(sample["text"])

    # Test mdrop with subset of texts
    subset_texts = texts[:2]
    cache.mdrop(subset_texts, model_name)

    # Verify only selected texts were dropped
    for i, text in enumerate(texts):
        if i < 2:
            assert not cache.exists(text, model_name)  # Should be dropped
        else:
            assert cache.exists(text, model_name)  # Should still exist


@pytest.mark.asyncio
async def test_async_batch_operations(cache, sample_embedding_data):
    """Test async batch operations (amset, amget, amexists, amdrop)."""
    # Prepare batch items
    batch_items = []
    for sample in sample_embedding_data:
        batch_items.append(
            {
                "text": sample["text"],
                "model_name": sample["model_name"],
                "embedding": sample["embedding"],
                "metadata": sample.get("metadata"),
            }
        )

    # Use amset to store embeddings
    keys = await cache.amset(batch_items)
    assert len(keys) == len(batch_items)

    # Get texts and model name for amget
    texts = [item["text"] for item in batch_items]
    model_name = batch_items[0]["model_name"]  # Assuming same model

    # Test amget
    results = await cache.amget(texts, model_name)
    assert len(results) == len(texts)
    for i, result in enumerate(results):
        assert result is not None
        assert result["text"] == texts[i]

    # Test amget_by_keys
    key_results = await cache.amget_by_keys(keys)
    assert len(key_results) == len(keys)
    for result in key_results:
        assert result is not None

    # Test amexists
    exist_results = await cache.amexists(texts, model_name)
    assert len(exist_results) == len(texts)
    assert all(exist_results)  # All should exist

    # Test amexists_by_keys
    key_exist_results = await cache.amexists_by_keys(keys)
    assert len(key_exist_results) == len(keys)
    assert all(key_exist_results)  # All should exist

    # Test amdrop with first text
    await cache.amdrop([texts[0]], model_name)
    updated_exists = await cache.aexists(texts[0], model_name)
    assert not updated_exists  # Should be dropped

    # Test amdrop_by_keys with second key
    await cache.amdrop_by_keys([keys[1]])
    updated_key_exists = await cache.aexists_by_key(keys[1])
    assert not updated_key_exists  # Should be dropped


def test_batch_operations_with_missing_data(cache):
    """Test batch operations with empty lists and missing cache entries."""
    # Test with empty lists
    assert cache.mget_by_keys([]) == []
    assert cache.mexists_by_keys([]) == []
    cache.mdrop_by_keys([])  # Should not raise errors

    # Test mget with non-existent keys
    non_existent_keys = [
        "test_embed_cache:nonexistent1",
        "test_embed_cache:nonexistent2",
    ]
    results = cache.mget_by_keys(non_existent_keys)
    assert len(results) == 2
    assert results[0] is None
    assert results[1] is None

    # Test mexists with non-existent keys
    exist_results = cache.mexists_by_keys(non_existent_keys)
    assert len(exist_results) == 2
    assert not any(exist_results)  # None should exist

    # Test with empty model names and texts
    assert cache.mget([], "model") == []
    assert cache.mexists([], "model") == []
    cache.mdrop([], "model")  # Should not raise errors


def test_batch_with_ttl(cache_with_ttl, sample_embedding_data):
    """Test batch operations with TTL."""
    # Prepare batch items
    batch_items = []
    for sample in sample_embedding_data:
        batch_items.append(
            {
                "text": sample["text"],
                "model_name": sample["model_name"],
                "embedding": sample["embedding"],
                "metadata": sample.get("metadata"),
            }
        )

    # Store with default TTL (2 seconds from fixture)
    keys = cache_with_ttl.mset(batch_items)

    # Verify all exist initially
    exist_results = cache_with_ttl.mexists_by_keys(keys)
    assert all(exist_results)

    # Wait for TTL to expire
    time.sleep(3)

    # Verify all have expired
    exist_results_after = cache_with_ttl.mexists_by_keys(keys)
    assert not any(exist_results_after)

    # Test with custom TTL override
    keys = cache_with_ttl.mset(batch_items, ttl=5)  # 5 second TTL

    # Wait for 3 seconds (beyond default but before custom TTL)
    time.sleep(3)

    # Should still exist with custom TTL
    exist_results = cache_with_ttl.mexists_by_keys(keys)
    assert all(exist_results)


def test_large_batch_operations(cache):
    """Test operations with larger batches to ensure scalability."""
    # Create a larger batch of items
    large_batch = []
    for i in range(100):
        large_batch.append(
            {
                "text": f"Sample text {i}",
                "model_name": "test-model",
                "embedding": [float(i) / 100] * 5,
                "metadata": {"index": i},
            }
        )

    # Test storing large batch
    keys = cache.mset(large_batch)
    assert len(keys) == 100

    # Test retrieving large batch by keys
    results = cache.mget_by_keys(keys)
    assert len(results) == 100
    assert all(result is not None for result in results)

    # Get texts for batch retrieval
    texts = [item["text"] for item in large_batch]

    # Test retrieving by texts
    results = cache.mget(texts, "test-model")
    assert len(results) == 100
    assert all(result is not None for result in results)

    # Test existence checks
    exist_results = cache.mexists_by_keys(keys)
    assert len(exist_results) == 100
    assert all(exist_results)

    # Test batch deletion
    cache.mdrop_by_keys(keys[:50])  # Delete first half

    # Verify first half deleted, second half still exists
    for i, key in enumerate(keys):
        if i < 50:
            assert not cache.exists_by_key(key)
        else:
            assert cache.exists_by_key(key)
