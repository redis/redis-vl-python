import asyncio
import json
import time

import pytest
from redis.exceptions import ConnectionError

from redisvl.extensions.cache.embeddings.embeddings import EmbeddingsCache
from redisvl.extensions.cache.embeddings.schema import CacheEntry
from redisvl.redis.utils import hashify


@pytest.fixture
def cache(redis_url, redis_test_name):
    """Basic EmbeddingsCache fixture with cleanup."""
    cache_instance = EmbeddingsCache(
        name=redis_test_name("test_embed_cache"),
        redis_url=redis_url,
    )
    yield cache_instance
    # Clean up all keys with this prefix
    cache_instance.clear()


@pytest.fixture
def cache_with_ttl(redis_url, redis_test_name):
    """EmbeddingsCache with TTL setting."""
    cache_instance = EmbeddingsCache(
        name=redis_test_name("test_ttl_cache"),
        ttl=2,  # 2 second TTL for testing expiration
        redis_url=redis_url,
    )
    yield cache_instance
    # Clean up all keys with this prefix
    cache_instance.clear()


@pytest.fixture
def cache_with_redis_client(client, redis_test_name):
    """EmbeddingsCache with provided Redis client."""
    cache_instance = EmbeddingsCache(
        name=redis_test_name("test_client_cache"),
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
            "content": "What is machine learning?",
            "model_name": "text-embedding-ada-002",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "metadata": {"source": "user_query", "category": "ai"},
        },
        {
            "content": "How do neural networks work?",
            "model_name": "text-embedding-ada-002",
            "embedding": [0.2, 0.3, 0.4, 0.5, 0.6],
            "metadata": {"source": "documentation", "category": "ai"},
        },
        {
            "content": "What's the weather like today?",
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
    content = "Hello world"
    model_name = "text-embedding-ada-002"

    # Test deterministic ID generation
    entry_id1 = cache._make_entry_id(content, model_name)
    entry_id2 = cache._make_entry_id(content, model_name)
    assert entry_id1 == entry_id2

    # Test different inputs produce different IDs
    different_id = cache._make_entry_id("Different text", model_name)
    assert entry_id1 != different_id

    # Test ID format
    assert isinstance(entry_id1, str)
    expected_id = hashify(
        json.dumps(["str", content, model_name], separators=(",", ":"))
    )
    assert entry_id1 == expected_id

    # Content and model name can no longer bleed into each other
    assert cache._make_entry_id("a:b", "c") != cache._make_entry_id("a", "b:c")
    assert cache._make_entry_id(b"a", model_name) != cache._make_entry_id(
        "61", model_name
    )


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
        content=sample["content"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
        metadata=sample["metadata"],
    )

    # Get the entry
    result = cache.get(sample["content"], sample["model_name"])

    # Verify the result
    assert result is not None
    assert result["content"] == sample["content"]
    assert result["model_name"] == sample["model_name"]
    assert "embedding" in result
    assert result["metadata"] == sample["metadata"]

    # Test get_by_key
    key_result = cache.get_by_key(key)
    assert key_result is not None
    assert key_result["content"] == sample["content"]

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
    assert not cache.exists(sample["content"], sample["model_name"])

    # Add the entry
    key = cache.set(
        content=sample["content"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
    )

    # Now it should exist
    assert cache.exists(sample["content"], sample["model_name"])

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
        content=sample["content"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
    )

    # Verify it exists
    assert cache.exists_by_key(key)

    # Remove it
    cache.drop(sample["content"], sample["model_name"])

    # Verify it's gone
    assert not cache.exists_by_key(key)

    # Test drop_by_key
    key = cache.set(
        content=sample["content"],
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
        content=sample["content"],
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
        content=sample["content"],
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
            content=sample["content"],
            model_name=sample["model_name"],
            embedding=sample["embedding"],
            metadata=sample.get("metadata"),
        )
        keys.append(key)

    # Check they all exist
    for i, key in enumerate(keys):
        assert cache.exists_by_key(key)
        result = cache.get_by_key(key)
        assert result["content"] == sample_embedding_data[i]["content"]

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
        content=sample["content"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
        metadata=sample["metadata"],
    )

    # Get the entry
    result = await cache.aget(sample["content"], sample["model_name"])

    # Verify the result
    assert result is not None
    assert result["content"] == sample["content"]
    assert result["model_name"] == sample["model_name"]
    assert "embedding" in result
    assert result["metadata"] == sample["metadata"]

    # Test aget_by_key
    key_result = await cache.aget_by_key(key)
    assert key_result is not None
    assert key_result["content"] == sample["content"]


@pytest.mark.asyncio
async def test_async_exists(cache, sample_embedding_data):
    """Test async version of exists."""
    sample = sample_embedding_data[0]

    # Entry shouldn't exist yet
    assert not await cache.aexists(sample["content"], sample["model_name"])

    # Add the entry
    key = await cache.aset(
        content=sample["content"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
    )

    # Now it should exist
    assert await cache.aexists(sample["content"], sample["model_name"])

    # Test aexists_by_key
    assert await cache.aexists_by_key(key)


@pytest.mark.asyncio
async def test_async_drop(cache, sample_embedding_data):
    """Test async version of drop."""
    sample = sample_embedding_data[0]

    # Add the entry
    key = await cache.aset(
        content=sample["content"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
    )

    # Verify it exists
    assert await cache.aexists_by_key(key)

    # Remove it
    await cache.adrop(sample["content"], sample["model_name"])

    # Verify it's gone
    assert not await cache.aexists_by_key(key)

    # Test adrop_by_key
    key = await cache.aset(
        content=sample["content"],
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
        content=sample["content"],
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
    expected_id = cache._make_entry_id(sample["content"], sample["model_name"])

    # Set an entry and extract its ID from the key
    key = cache.set(
        content=sample["content"],
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
        content=sample["content"],
        model_name=sample["model_name"],
        embedding=sample["embedding"],
    )

    result = cache_with_redis_client.get_by_key(key)
    assert result is not None
    assert result["content"] == sample["content"]


def test_mset_and_mget(cache, sample_embedding_data):
    """Test batch setting and getting of embeddings."""
    # Prepare batch items
    batch_items = []
    for sample in sample_embedding_data:
        batch_items.append(
            {
                "content": sample["content"],
                "model_name": sample["model_name"],
                "embedding": sample["embedding"],
                "metadata": sample.get("metadata"),
            }
        )

    # Use mset to store embeddings
    keys = cache.mset(batch_items)
    assert len(keys) == len(batch_items)

    # Get contents and model name for mget
    contents = [item["content"] for item in batch_items]
    model_name = batch_items[0]["model_name"]  # Assuming same model

    # Test mget
    results = cache.mget(contents, model_name)
    assert len(results) == len(contents)

    # Verify all results are returned and in correct order
    for i, result in enumerate(results):
        assert result is not None
        assert result["content"] == contents[i]
        assert result["model_name"] == model_name


def test_mget_by_keys(cache, sample_embedding_data):
    """Test getting multiple embeddings by their keys."""
    # Set embeddings individually and collect keys
    keys = []
    for sample in sample_embedding_data:
        key = cache.set(
            content=sample["content"],
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
        assert result["content"] == sample_embedding_data[i]["content"]
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
    contents = []
    for sample in sample_embedding_data:
        key = cache.set(
            content=sample["content"],
            model_name=sample["model_name"],
            embedding=sample["embedding"],
        )
        keys.append(key)
        contents.append(sample["content"])

    model_name = sample_embedding_data[0]["model_name"]  # Assuming same model

    # Test mexists
    exist_results = cache.mexists(contents, model_name)
    assert len(exist_results) == len(contents)
    assert all(exist_results)  # All should exist

    # Test with mix of existing and non-existing contents
    non_existent_content = "This content does not exist"
    mixed_contents = contents[:1] + [non_existent_content] + contents[1:]
    mixed_results = cache.mexists(mixed_contents, model_name)

    assert len(mixed_results) == len(mixed_contents)
    assert mixed_results[0] is True
    assert mixed_results[1] is False  # Non-existent content should return False
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
    contents = []
    for sample in sample_embedding_data:
        key = cache.set(
            content=sample["content"],
            model_name=sample["model_name"],
            embedding=sample["embedding"],
        )
        keys.append(key)
        contents.append(sample["content"])

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
    contents = []
    for sample in sample_embedding_data:
        key = cache.set(
            content=sample["content"],
            model_name=sample["model_name"],
            embedding=sample["embedding"],
        )
        keys.append(key)
        contents.append(sample["content"])

    # Test mdrop with subset of contents
    subset_contents = contents[:2]
    cache.mdrop(subset_contents, model_name)

    # Verify only selected contents were dropped
    for i, content in enumerate(contents):
        if i < 2:
            assert not cache.exists(content, model_name)  # Should be dropped
        else:
            assert cache.exists(content, model_name)  # Should still exist


@pytest.mark.asyncio
async def test_async_batch_operations(cache, sample_embedding_data):
    """Test async batch operations (amset, amget, amexists, amdrop)."""
    # Prepare batch items
    batch_items = []
    for sample in sample_embedding_data:
        batch_items.append(
            {
                "content": sample["content"],
                "model_name": sample["model_name"],
                "embedding": sample["embedding"],
                "metadata": sample.get("metadata"),
            }
        )

    # Use amset to store embeddings
    keys = await cache.amset(batch_items)
    assert len(keys) == len(batch_items)

    # Get contents and model name for amget
    contents = [item["content"] for item in batch_items]
    model_name = batch_items[0]["model_name"]  # Assuming same model

    # Test amget
    results = await cache.amget(contents, model_name)
    assert len(results) == len(contents)
    for i, result in enumerate(results):
        assert result is not None
        assert result["content"] == contents[i]

    # Test amget_by_keys
    key_results = await cache.amget_by_keys(keys)
    assert len(key_results) == len(keys)
    for result in key_results:
        assert result is not None

    # Test amexists
    exist_results = await cache.amexists(contents, model_name)
    assert len(exist_results) == len(contents)
    assert all(exist_results)  # All should exist

    # Test amexists_by_keys
    key_exist_results = await cache.amexists_by_keys(keys)
    assert len(key_exist_results) == len(keys)
    assert all(key_exist_results)  # All should exist

    # Test amdrop with first content
    await cache.amdrop([contents[0]], model_name)
    updated_exists = await cache.aexists(contents[0], model_name)
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
                "content": sample["content"],
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


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_mget_by_keys_refreshes_every_hit_ttl(
    cache_with_ttl, sample_embedding_data, monkeypatch, use_async
):
    """Every hit's TTL is reset within the lookup pipeline; a miss is not created."""
    items = [
        {key: sample[key] for key in ("content", "model_name", "embedding")}
        for sample in sample_embedding_data
    ]
    keys = cache_with_ttl.mset(items, ttl=1000)
    missing = cache_with_ttl._make_key("missing")

    def refuse(*args, **kwargs):
        raise AssertionError("TTL refreshed outside the lookup pipeline")

    monkeypatch.setattr(cache_with_ttl, "expire", refuse)
    monkeypatch.setattr(cache_with_ttl, "aexpire", refuse)

    if use_async:
        results = await cache_with_ttl.amget_by_keys(keys + [missing])
    else:
        results = cache_with_ttl.mget_by_keys(keys + [missing])

    assert [result is not None for result in results] == [True, True, True, False]
    client = cache_with_ttl._get_redis_client()
    assert all(0 < client.ttl(key) <= 2 for key in keys)
    assert client.exists(missing) == 0


def test_large_batch_operations(cache):
    """Test operations with larger batches to ensure scalability."""
    # Create a larger batch of items
    large_batch = []
    for i in range(100):
        large_batch.append(
            {
                "content": f"Sample text {i}",
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

    # Get contents for batch retrieval
    contents = [item["content"] for item in large_batch]

    # Test retrieving by contents
    results = cache.mget(contents, "test-model")
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


@pytest.fixture
def colliding_caches(redis_url, redis_test_name):
    """Three caches whose names collide under an unescaped SCAN glob.

    ``base[ab]:*`` as a glob matches ``basea:``/``baseb:`` but not
    ``base[ab]:``, so clearing the bracketed cache used to wipe the other two.
    """
    base = redis_test_name("glob_collision")
    caches = {
        name: EmbeddingsCache(name=name, redis_url=redis_url)
        for name in (f"{base}[ab]", f"{base}a", f"{base}b")
    }
    yield f"{base}[ab]", caches
    for cache_instance in caches.values():
        cache_instance.clear()


def test_clear_does_not_delete_caches_colliding_under_glob(colliding_caches):
    target_name, caches = colliding_caches
    for name, cache_instance in caches.items():
        cache_instance.set(content=name, model_name="test-model", embedding=[0.1, 0.2])
        assert cache_instance.exists(name, "test-model")

    caches[target_name].clear()

    assert not caches[target_name].exists(target_name, "test-model")
    for name, cache_instance in caches.items():
        if name != target_name:
            assert cache_instance.exists(
                name, "test-model"
            ), f"clear() on {target_name!r} deleted entries belonging to {name!r}"


@pytest.mark.asyncio
async def test_aclear_does_not_delete_caches_colliding_under_glob(colliding_caches):
    target_name, caches = colliding_caches
    for name, cache_instance in caches.items():
        await cache_instance.aset(
            content=name, model_name="test-model", embedding=[0.1, 0.2]
        )
        assert await cache_instance.aexists(name, "test-model")

    await caches[target_name].aclear()

    assert not await caches[target_name].aexists(target_name, "test-model")
    for name, cache_instance in caches.items():
        if name != target_name:
            assert await cache_instance.aexists(
                name, "test-model"
            ), f"aclear() on {target_name!r} deleted entries belonging to {name!r}"


def write_legacy_entry(cache, content, model_name, embedding, ttl=None):
    """Store an entry the way RedisVL 0.32 did, under its legacy key."""
    entry_id = cache._make_legacy_entry_id(content, model_name)
    key = cache._make_key(entry_id)
    entry = CacheEntry(
        entry_id=entry_id, content=content, model_name=model_name, embedding=embedding
    )
    client = cache._get_redis_client()
    client.hset(key, mapping=entry.to_dict())
    if ttl:
        client.expire(key, ttl)
    return key


@pytest.mark.parametrize(
    "content", ["What is AI?", b"\x00\xffbinary"], ids=["str", "bytes"]
)
def test_get_migrates_a_legacy_entry(cache, content, caplog):
    """A legacy hit is returned, rewritten under its new key, and logged."""
    model_name = "text-embedding-ada-002"
    legacy_key = write_legacy_entry(cache, content, model_name, [0.1, 0.2])
    new_key = cache._make_cache_key(content, model_name)
    assert legacy_key != new_key

    with caplog.at_level("WARNING", logger="redisvl.extensions.cache.embeddings"):
        result = cache.get(content, model_name)

    assert result["embedding"] == [0.1, 0.2]
    assert result["entry_id"] == cache._make_entry_id(content, model_name)
    client = cache._get_redis_client()
    assert not client.exists(legacy_key)
    assert cache.get_by_key(new_key)["embedding"] == [0.1, 0.2]
    assert "RedisVL 0.32 and earlier" in caplog.text


def test_legacy_collision_is_not_returned(cache):
    """A legacy entry stored for ("a:b", "c") is not a hit for ("a", "b:c")."""
    legacy_key = write_legacy_entry(cache, "a:b", "c", [0.1])

    assert cache.get("a", "b:c") is None
    assert not cache.exists("a", "b:c")
    assert cache.mget(["a"], "b:c") == [None]
    # The rightful owner can still migrate it.
    assert cache._get_redis_client().exists(legacy_key)
    assert cache.get("a:b", "c")["embedding"] == [0.1]


def test_mget_mixes_new_legacy_and_missing_entries(cache, caplog):
    model_name = "m"
    cache.set("new", model_name, [0.1])
    write_legacy_entry(cache, "old-1", model_name, [0.2])
    write_legacy_entry(cache, "old-2", model_name, [0.3])

    with caplog.at_level("WARNING", logger="redisvl.extensions.cache.embeddings"):
        results = cache.mget(["new", "old-1", "missing", "old-2"], model_name)
        # Already migrated, so a second read neither falls back nor logs again.
        assert cache.mget(["old-1", "old-2"], model_name)[1]["embedding"] == [0.3]

    assert [r and r["embedding"] for r in results] == [[0.1], [0.2], None, [0.3]]
    assert caplog.text.count("RedisVL 0.32 and earlier") == 1


def test_exists_migrates_and_drop_removes_legacy_entries(cache):
    model_name = "m"
    first = write_legacy_entry(cache, "first", model_name, [0.1])
    second = write_legacy_entry(cache, "second", model_name, [0.2])
    client = cache._get_redis_client()

    assert cache.mexists(["first", "absent"], model_name) == [True, False]
    assert not client.exists(first)
    assert client.exists(cache._make_cache_key("first", model_name))

    cache.drop("second", model_name)
    assert not client.exists(second)
    assert not cache.exists("second", model_name)


def test_migration_keeps_the_legacy_ttl_when_the_cache_has_none(cache):
    legacy_key = write_legacy_entry(cache, "content", "m", [0.1], ttl=300)

    cache.get("content", "m")

    ttl = cache._get_redis_client().ttl(cache._make_cache_key("content", "m"))
    assert 0 < ttl <= 300
    assert not cache._get_redis_client().exists(legacy_key)


def test_migration_applies_the_cache_ttl(cache_with_ttl):
    write_legacy_entry(cache_with_ttl, "content", "m", [0.1])

    cache_with_ttl.get("content", "m")

    ttl = cache_with_ttl._get_redis_client().ttl(
        cache_with_ttl._make_cache_key("content", "m")
    )
    assert 0 < ttl <= 2


@pytest.mark.asyncio
async def test_async_reads_migrate_legacy_entries(cache):
    model_name = "m"
    write_legacy_entry(cache, "one", model_name, [0.1])
    write_legacy_entry(cache, "two", model_name, [0.2])
    write_legacy_entry(cache, "three", model_name, [0.3])
    write_legacy_entry(cache, "a:b", "c", [0.4])

    assert (await cache.aget("one", model_name))["embedding"] == [0.1]
    assert await cache.aexists("two", model_name)
    assert await cache.aget("a", "b:c") is None
    await cache.adrop("three", model_name)

    client = cache._get_redis_client()
    for content in ("one", "two"):
        assert client.exists(cache._make_cache_key(content, model_name))
        assert not client.exists(cache._make_legacy_cache_key(content, model_name))
    assert not client.exists(cache._make_legacy_cache_key("three", model_name))
    assert client.exists(cache._make_legacy_cache_key("a:b", "c"))
