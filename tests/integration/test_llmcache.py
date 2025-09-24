import asyncio
import warnings
from collections import namedtuple
from time import sleep, time

import pytest
from pydantic import ValidationError
from redis.exceptions import ConnectionError

from redisvl.extensions.cache.llm import SemanticCache
from redisvl.index.index import AsyncSearchIndex, SearchIndex
from redisvl.query.filter import Num, Tag, Text
from redisvl.utils.vectorize import HFTextVectorizer
from tests.conftest import skip_if_no_redisearch, skip_if_no_redisearch_async


@pytest.fixture(scope="session")
def vectorizer():
    return HFTextVectorizer("redis/langcache-embed-v1")


@pytest.fixture
def cache(client, vectorizer, redis_url, worker_id):
    skip_if_no_redisearch(client)
    cache_instance = SemanticCache(
        name=f"llmcache_{worker_id}",
        vectorizer=vectorizer,
        distance_threshold=0.2,
        redis_url=redis_url,
    )
    yield cache_instance
    cache_instance._index.delete(True)  # Clean up index


@pytest.fixture
def cache_with_filters(client, vectorizer, redis_url, worker_id):
    skip_if_no_redisearch(client)
    cache_instance = SemanticCache(
        name=f"llmcache_filters_{worker_id}",
        vectorizer=vectorizer,
        distance_threshold=0.2,
        filterable_fields=[{"name": "label", "type": "tag"}],
        redis_url=redis_url,
    )
    yield cache_instance
    cache_instance._index.delete(True)  # Clean up index


@pytest.fixture
def cache_no_cleanup(client, vectorizer, redis_url, worker_id):
    skip_if_no_redisearch(client)
    cache_instance = SemanticCache(
        name=f"llmcache_no_cleanup_{worker_id}",
        vectorizer=vectorizer,
        distance_threshold=0.2,
        redis_url=redis_url,
    )
    yield cache_instance


@pytest.fixture
def cache_with_ttl(client, vectorizer, redis_url, worker_id):
    skip_if_no_redisearch(client)
    cache_instance = SemanticCache(
        name=f"llmcache_ttl_{worker_id}",
        vectorizer=vectorizer,
        distance_threshold=0.2,
        ttl=2,
        redis_url=redis_url,
    )
    yield cache_instance
    cache_instance._index.delete(True)  # Clean up index


@pytest.fixture
def cache_with_redis_client(vectorizer, client, worker_id):
    skip_if_no_redisearch(client)
    cache_instance = SemanticCache(
        name=f"llmcache_client_{worker_id}",
        vectorizer=vectorizer,
        redis_client=client,
        distance_threshold=0.2,
    )
    yield cache_instance
    cache_instance.clear()  # Clear cache after each test
    cache_instance._index.delete(True)  # Clean up index


@pytest.fixture(autouse=True)
def disable_deprecation_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_llmcache_backwards_compat():
    from redisvl.extensions.llmcache import SemanticCache as DeprecatedSemanticCache

    assert DeprecatedSemanticCache == SemanticCache


def test_bad_ttl(cache):
    with pytest.raises(ValueError):
        cache.set_ttl(2.5)


def test_cache_ttl(cache_with_ttl):
    assert cache_with_ttl.ttl == 2
    cache_with_ttl.set_ttl(5)
    assert cache_with_ttl.ttl == 5


def test_set_ttl(cache):
    assert cache.ttl == None
    cache.set_ttl(5)
    assert cache.ttl == 5


def test_reset_ttl(cache):
    cache.set_ttl(4)
    cache.set_ttl()
    assert cache.ttl is None


def test_get_index(cache):
    assert isinstance(cache.index, SearchIndex)


@pytest.mark.asyncio
async def test_get_async_index(cache):
    async with cache:
        aindex = await cache._get_async_index()
        assert isinstance(aindex, AsyncSearchIndex)


@pytest.mark.asyncio
async def test_get_async_index_from_provided_client(cache_with_redis_client):
    async with cache_with_redis_client:
        aindex = await cache_with_redis_client._get_async_index()
        # Shouldn't have to do this because it already was done
        await aindex.create(overwrite=True, drop=True)
        assert await aindex.exists()
        assert isinstance(aindex, AsyncSearchIndex)
        assert aindex == cache_with_redis_client.aindex
        assert await cache_with_redis_client.aindex.exists()


def test_delete(cache_no_cleanup):
    cache_no_cleanup.delete()
    assert not cache_no_cleanup.index.exists()


@pytest.mark.asyncio
async def test_async_delete(cache_no_cleanup):
    async with cache_no_cleanup:
        await cache_no_cleanup.adelete()
        assert not cache_no_cleanup.index.exists()


def test_store_and_check(cache, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)

    cache.store(prompt, response, vector=vector)
    check_result = cache.check(vector=vector, distance_threshold=0.4)

    assert len(check_result) == 1
    print(check_result, flush=True)
    assert response == check_result[0]["response"]
    assert "metadata" not in check_result[0]


@pytest.mark.asyncio
async def test_async_store_and_check(cache, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)

    async with cache:
        await cache.astore(prompt, response, vector=vector)
        check_result = await cache.acheck(vector=vector, distance_threshold=0.4)

    assert len(check_result) == 1
    print(check_result, flush=True)
    assert response == check_result[0]["response"]
    assert "metadata" not in check_result[0]


def test_return_fields(cache, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)

    cache.store(prompt, response, vector=vector)

    # check default return fields
    check_result = cache.check(vector=vector)
    assert set(check_result[0].keys()) == {
        "key",
        "entry_id",
        "prompt",
        "response",
        "vector_distance",
        "inserted_at",
        "updated_at",
    }

    # check specific return fields
    fields = [
        "key",
        "entry_id",
        "prompt",
        "response",
        "vector_distance",
    ]
    check_result = cache.check(vector=vector, return_fields=fields)
    assert set(check_result[0].keys()) == set(fields)

    # check only some return fields
    fields = ["inserted_at", "updated_at"]
    check_result = cache.check(vector=vector, return_fields=fields)
    fields.append("key")
    assert set(check_result[0].keys()) == set(fields)


@pytest.mark.asyncio
async def test_async_return_fields(cache, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)

    async with cache:
        await cache.astore(prompt, response, vector=vector)

        # check default return fields
        check_result = await cache.acheck(vector=vector)
        assert set(check_result[0].keys()) == {
            "key",
            "entry_id",
            "prompt",
            "response",
            "vector_distance",
            "inserted_at",
            "updated_at",
        }

        # check specific return fields
        fields = [
            "key",
            "entry_id",
            "prompt",
            "response",
            "vector_distance",
        ]
        check_result = await cache.acheck(vector=vector, return_fields=fields)
        assert set(check_result[0].keys()) == set(fields)

        # check only some return fields
        fields = ["inserted_at", "updated_at"]
        check_result = await cache.acheck(vector=vector, return_fields=fields)
        fields.append("key")
        assert set(check_result[0].keys()) == set(fields)


# Test clearing the cache
def test_clear(cache, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)

    cache.store(prompt, response, vector=vector)
    cache.clear()
    check_result = cache.check(vector=vector)

    assert len(check_result) == 0


@pytest.mark.asyncio
async def test_async_clear(cache, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)

    async with cache:
        await cache.astore(prompt, response, vector=vector)
        await cache.aclear()
        check_result = await cache.acheck(vector=vector)

    assert len(check_result) == 0


# Test TTL functionality
def test_ttl_expiration(cache_with_ttl, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)

    cache_with_ttl.store(prompt, response, vector=vector)
    sleep(3)

    check_result = cache_with_ttl.check(vector=vector)
    assert len(check_result) == 0


@pytest.mark.asyncio
async def test_async_ttl_expiration(cache_with_ttl, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)

    async with cache_with_ttl:
        await cache_with_ttl.astore(prompt, response, vector=vector)
        await asyncio.sleep(3)

        check_result = await cache_with_ttl.acheck(vector=vector)
    assert len(check_result) == 0


def test_custom_ttl(cache_with_ttl, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)

    cache_with_ttl.store(prompt, response, vector=vector, ttl=5)
    sleep(3)

    check_result = cache_with_ttl.check(vector=vector)
    assert len(check_result) != 0
    assert cache_with_ttl.ttl == 2


@pytest.mark.asyncio
async def test_async_custom_ttl(cache_with_ttl, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)

    async with cache_with_ttl:
        await cache_with_ttl.astore(prompt, response, vector=vector, ttl=5)
        await asyncio.sleep(3)
        check_result = await cache_with_ttl.acheck(vector=vector)

    assert len(check_result) != 0
    assert cache_with_ttl.ttl == 2


def test_ttl_refresh(cache_with_ttl, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)

    cache_with_ttl.store(prompt, response, vector=vector)

    for _ in range(3):
        sleep(1)
        check_result = cache_with_ttl.check(vector=vector)

    assert len(check_result) == 1


@pytest.mark.asyncio
async def test_async_ttl_refresh(cache_with_ttl, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)

    async with cache_with_ttl:
        await cache_with_ttl.astore(prompt, response, vector=vector)

        for _ in range(3):
            await asyncio.sleep(1)
            check_result = await cache_with_ttl.acheck(vector=vector)

    assert len(check_result) == 1


# Test manual expiration of single document
def test_drop_document(cache, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)

    cache.store(prompt, response, vector=vector)
    check_result = cache.check(vector=vector)

    cache.drop(ids=[check_result[0]["entry_id"]])
    recheck_result = cache.check(vector=vector)
    assert len(recheck_result) == 0


@pytest.mark.asyncio
async def test_async_drop_document(cache, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)

    async with cache:
        await cache.astore(prompt, response, vector=vector)
        check_result = await cache.acheck(vector=vector)

        await cache.adrop(ids=[check_result[0]["entry_id"]])
        recheck_result = await cache.acheck(vector=vector)

    assert len(recheck_result) == 0


# Test manual expiration of multiple documents
def test_drop_documents(cache, vectorizer):
    prompts = [
        "This is a test prompt.",
        "This is also test prompt.",
        "This is another test prompt.",
    ]
    responses = [
        "This is a test response.",
        "This is also test response.",
        "This is a another test response.",
    ]
    for prompt, response in zip(prompts, responses):
        vector = vectorizer.embed(prompt)
        cache.store(prompt, response, vector=vector)

    check_result = cache.check(vector=vector, num_results=3)
    print(check_result, flush=True)
    ids = [r["entry_id"] for r in check_result[0:2]]  # drop first 2 entries
    cache.drop(ids=ids)

    recheck_result = cache.check(vector=vector, num_results=3)
    assert len(recheck_result) == 1


@pytest.mark.asyncio
async def test_async_drop_documents(cache, vectorizer):
    prompts = [
        "This is a test prompt.",
        "This is also test prompt.",
        "This is another test prompt.",
    ]
    responses = [
        "This is a test response.",
        "This is also test response.",
        "This is a another test response.",
    ]
    for prompt, response in zip(prompts, responses):
        vector = vectorizer.embed(prompt)
        await cache.astore(prompt, response, vector=vector)

    async with cache:
        check_result = await cache.acheck(vector=vector, num_results=3)
        print(check_result, flush=True)
        ids = [r["entry_id"] for r in check_result[0:2]]  # drop first 2 entries
        await cache.adrop(ids=ids)

        recheck_result = await cache.acheck(vector=vector, num_results=3)

    assert len(recheck_result) == 1


# Test updating document fields
def test_updating_document(cache):
    prompt = "This is a test prompt."
    response = "This is a test response."
    cache.store(prompt=prompt, response=response)

    check_result = cache.check(prompt=prompt, return_fields=["updated_at"])
    key = check_result[0]["key"]

    sleep(1)

    metadata = {"foo": "bar"}
    cache.update(key=key, metadata=metadata)

    updated_result = cache.check(
        prompt=prompt, return_fields=["updated_at", "metadata"]
    )
    assert updated_result[0]["metadata"] == metadata
    assert updated_result[0]["updated_at"] > check_result[0]["updated_at"]


@pytest.mark.asyncio
async def test_async_updating_document(cache):
    prompt = "This is a test prompt."
    response = "This is a test response."

    async with cache:
        await cache.astore(prompt=prompt, response=response)

        check_result = await cache.acheck(prompt=prompt, return_fields=["updated_at"])
        key = check_result[0]["key"]

        await asyncio.sleep(1)

        metadata = {"foo": "bar"}
        await cache.aupdate(key=key, metadata=metadata)

        updated_result = await cache.acheck(
            prompt=prompt, return_fields=["updated_at", "metadata"]
        )

    assert updated_result[0]["metadata"] == metadata
    assert updated_result[0]["updated_at"] > check_result[0]["updated_at"]


def test_ttl_expiration_after_update(cache_with_ttl, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)
    cache_with_ttl.set_ttl(4)

    assert cache_with_ttl.ttl == 4

    cache_with_ttl.store(prompt, response, vector=vector)
    sleep(5)

    check_result = cache_with_ttl.check(vector=vector)
    assert len(check_result) == 0


@pytest.mark.asyncio
async def test_async_ttl_expiration_after_update(cache_with_ttl, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)
    cache_with_ttl.set_ttl(4)

    assert cache_with_ttl.ttl == 4

    async with cache_with_ttl:
        await cache_with_ttl.astore(prompt, response, vector=vector)
        await asyncio.sleep(5)

        check_result = await cache_with_ttl.acheck(vector=vector)

    assert len(check_result) == 0


# Test check behavior with no match
def test_check_no_match(cache, vectorizer):
    vector = vectorizer.embed("Some random sentence.")
    check_result = cache.check(vector=vector)
    assert len(check_result) == 0


def test_check_invalid_input(cache):
    with pytest.raises(ValueError):
        cache.check()

    with pytest.raises(TypeError):
        cache.check(prompt="test", return_fields="bad value")


@pytest.mark.asyncio
async def test_async_check_invalid_input(cache):
    with pytest.raises(ValueError):
        await cache.acheck()

    with pytest.raises(TypeError):
        await cache.acheck(prompt="test", return_fields="bad value")


def test_bad_connection_info(vectorizer, worker_id):
    with pytest.raises(ConnectionError):
        SemanticCache(
            name=f"test_bad_connection_{worker_id}",
            vectorizer=vectorizer,
            distance_threshold=0.2,
            redis_url="redis://localhost:6389",
        )


def test_store_with_metadata(cache, vectorizer):
    prompt = "This is another test prompt."
    response = "This is another test response."
    metadata = {"source": "test"}
    vector = vectorizer.embed(prompt)

    cache.store(prompt, response, vector=vector, metadata=metadata)
    check_result = cache.check(vector=vector, num_results=1)

    assert len(check_result) == 1
    print(check_result, flush=True)
    assert check_result[0]["response"] == response
    assert check_result[0]["metadata"] == metadata
    assert check_result[0]["prompt"] == prompt


def test_store_with_empty_metadata(cache, vectorizer):
    prompt = "This is another test prompt."
    response = "This is another test response."
    metadata = {}
    vector = vectorizer.embed(prompt)

    cache.store(prompt, response, vector=vector, metadata=metadata)
    check_result = cache.check(vector=vector, num_results=1)

    assert len(check_result) == 1
    print(check_result, flush=True)
    assert check_result[0]["response"] == response
    assert check_result[0]["metadata"] == metadata
    assert check_result[0]["prompt"] == prompt


def test_store_with_invalid_metadata(cache, vectorizer):
    prompt = "This is another test prompt."
    response = "This is another test response."
    metadata = namedtuple("metadata", "source")(**{"source": "test"})

    vector = vectorizer.embed(prompt)

    with pytest.raises(ValidationError):
        cache.store(prompt, response, vector=vector, metadata=metadata)


def test_distance_threshold(cache):
    initial_threshold = cache.distance_threshold
    new_threshold = 0.1

    cache.set_threshold(new_threshold)
    assert cache.distance_threshold == new_threshold
    assert cache.distance_threshold != initial_threshold


def test_distance_threshold_out_of_range(cache):
    out_of_range_threshold = -1
    with pytest.raises(ValueError):
        cache.set_threshold(out_of_range_threshold)


def test_multiple_items(cache, vectorizer):
    prompts_responses = {
        "prompt1": "response1",
        "prompt2": "response2",
        "prompt3": "response3",
    }

    for prompt, response in prompts_responses.items():
        vector = vectorizer.embed(prompt)
        cache.store(prompt, response, vector=vector)

    for prompt, expected_response in prompts_responses.items():
        vector = vectorizer.embed(prompt)
        check_result = cache.check(vector=vector)
        assert len(check_result) == 1
        print(check_result, flush=True)
        assert check_result[0]["response"] == expected_response
        assert "metadata" not in check_result[0]


def test_store_and_check_with_provided_client(cache_with_redis_client, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)

    cache_with_redis_client.store(prompt, response, vector=vector)
    check_result = cache_with_redis_client.check(vector=vector)

    assert len(check_result) == 1
    print(check_result, flush=True)
    assert response == check_result[0]["response"]
    assert "metadata" not in check_result[0]


@pytest.mark.asyncio
async def test_async_store_and_check_with_provided_client(
    cache_with_redis_client, vectorizer
):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)

    await cache_with_redis_client.astore(prompt, response, vector=vector)
    check_result = await cache_with_redis_client.acheck(vector=vector)

    assert len(check_result) == 1
    print(check_result, flush=True)
    assert response == check_result[0]["response"]
    assert "metadata" not in check_result[0]


def test_vector_size(cache, vectorizer):
    prompt = "This is test prompt."
    response = "This is a test response."

    vector = vectorizer.embed(prompt)
    cache.store(prompt=prompt, response=response, vector=vector)

    # Test we can query with modified embeddings of correct size
    vector_2 = [v * 0.99 for v in vector]  # same dimensions
    check_result = cache.check(vector=vector_2)
    assert check_result[0]["prompt"] == prompt

    # Test that error is raised when we try to load wrong size vectors
    with pytest.raises(ValueError):
        cache.store(prompt=prompt, response=response, vector=vector[0:-1])

    with pytest.raises(ValueError):
        cache.store(prompt=prompt, response=response, vector=[1, 2, 3])

    # Test that error is raised when we try to query with wrong size vector
    with pytest.raises(ValueError):
        cache.check(vector=vector[0:-1])

    with pytest.raises(ValueError):
        cache.check(vector=[1, 2, 3])


def test_cache_with_filters(cache_with_filters):
    assert "label" in cache_with_filters._index.schema.fields


def test_cache_filtering(cache_with_filters):
    tag_1 = "group 0"
    tag_2 = "group 1"
    tag_3 = "group 2"
    tag_4 = "group 3"
    tags = [tag_1, tag_2, tag_3, tag_4]

    filter_1 = Tag("label") == tag_1
    filter_2 = Tag("label") == tag_2
    filter_3 = Tag("label") == tag_3

    for i in range(4):
        prompt = f"test prompt {i}"
        response = f"test response {i}"
        cache_with_filters.store(prompt, response, filters={"label": tags[i]})

    # test we can specify one specific tag
    results = cache_with_filters.check(
        "test prompt 1", filter_expression=filter_1, num_results=5
    )
    assert len(results) == 1
    assert results[0]["prompt"] == "test prompt 0"

    # test we can pass a list of tags
    combined_filter = filter_1 | filter_2 | filter_3
    results = cache_with_filters.check(
        "test prompt 1",
        filter_expression=combined_filter,
        num_results=5,
        distance_threshold=0.5,
    )
    assert len(results) == 3

    # test that default tag param searches full cache
    results = cache_with_filters.check(
        "test prompt 1", num_results=5, distance_threshold=0.6
    )
    assert len(results) == 4

    # test no results are returned if we pass a nonexistent tag
    bad_filter = Tag("label") == "bad tag"
    results = cache_with_filters.check(
        "test prompt 1", filter_expression=bad_filter, num_results=5
    )
    assert len(results) == 0


def test_cache_bad_filters(client, vectorizer, redis_url, worker_id):
    skip_if_no_redisearch(client)
    with pytest.raises(ValueError):
        cache_instance = SemanticCache(
            name=f"test_bad_filters_1_{worker_id}",
            vectorizer=vectorizer,
            distance_threshold=0.2,
            # invalid field type
            filterable_fields=[
                {"name": "label", "type": "tag"},
                {"name": "test", "type": "nothing"},
            ],
            redis_url=redis_url,
        )

    with pytest.raises(ValueError):
        cache_instance = SemanticCache(
            name=f"test_bad_filters_2_{worker_id}",
            vectorizer=vectorizer,
            distance_threshold=0.2,
            # duplicate field type
            filterable_fields=[
                {"name": "label", "type": "tag"},
                {"name": "label", "type": "tag"},
            ],
            redis_url=redis_url,
        )

    with pytest.raises(ValueError):
        cache_instance = SemanticCache(
            name=f"test_bad_filters_3_{worker_id}",
            vectorizer=vectorizer,
            distance_threshold=0.2,
            # reserved field name
            filterable_fields=[
                {"name": "label", "type": "tag"},
                {"name": "metadata", "type": "tag"},
            ],
            redis_url=redis_url,
        )


def test_complex_filters(cache_with_filters):
    cache_with_filters.store(prompt="prompt 1", response="response 1")
    cache_with_filters.store(prompt="prompt 2", response="response 2")
    sleep(1)
    current_timestamp = time()
    cache_with_filters.store(prompt="prompt 3", response="response 3")

    # test we can do range filters on inserted_at and updated_at fields
    range_filter = Num("inserted_at") < current_timestamp
    results = cache_with_filters.check(
        "prompt 1",
        filter_expression=range_filter,
        num_results=5,
        distance_threshold=0.5,
    )
    assert len(results) == 2

    # test we can combine range filters and text filters
    prompt_filter = Text("prompt") % "*pt 1"
    combined_filter = prompt_filter & range_filter

    results = cache_with_filters.check(
        "prompt 1", filter_expression=combined_filter, num_results=5
    )
    assert len(results) == 1


def test_cache_index_overwrite(client, redis_url, worker_id, hf_vectorizer):
    skip_if_no_redisearch(client)
    # Skip this test for Redis 6.2.x as FT.INFO doesn't return dims properly
    redis_version = client.info()["redis_version"]
    if redis_version.startswith("6.2"):
        pytest.skip(
            "Redis 6.2.x FT.INFO doesn't properly return vector dims for reconnection"
        )

    cache_no_tags = SemanticCache(
        name=f"test_cache_{worker_id}",
        redis_url=redis_url,
        vectorizer=hf_vectorizer,
    )

    cache_no_tags.store(
        prompt="this prompt has tags",
        response="this response has tags",
        filters={"some_tag": "abc"},
    )

    # filterable_fields not defined in schema, so no tags will match
    tag_filter = Tag("some_tag") == "abc"

    try:
        response = cache_no_tags.check(
            prompt="this prompt has a tag",
            filter_expression=tag_filter,
        )
    except Exception as e:
        # This will fail in Redis 8+ on query dialect 2
        if "Unknown field" in str(e):
            response = []
        else:
            raise

    assert response == []

    with pytest.raises((ValueError)):
        SemanticCache(
            name=f"test_cache_{worker_id}",
            redis_url=redis_url,
            vectorizer=hf_vectorizer,
            filterable_fields=[{"name": "some_tag", "type": "tag"}],
        )

    cache_overwrite = SemanticCache(
        name=f"test_cache_{worker_id}",
        redis_url=redis_url,
        vectorizer=hf_vectorizer,
        filterable_fields=[{"name": "some_tag", "type": "tag"}],
        overwrite=True,
    )

    response = cache_overwrite.check(
        prompt="this prompt has a tag",
        filter_expression=tag_filter,
    )
    assert len(response) == 1


def test_no_key_collision_on_identical_prompts(redis_url, worker_id, hf_vectorizer):
    private_cache = SemanticCache(
        name=f"private_cache_{worker_id}",
        redis_url=redis_url,
        vectorizer=hf_vectorizer,
        filterable_fields=[
            {"name": "user_id", "type": "tag"},
            {"name": "zip_code", "type": "numeric"},
        ],
    )

    private_cache.store(
        prompt="What is the phone number linked to my account?",
        response="The number on file is 123-555-0000",
        filters={"user_id": "gabs"},
    )

    private_cache.store(
        prompt="What's the phone number linked in my account?",
        response="The number on file is 123-555-9999",
        filters={"user_id": "cerioni", "zip_code": 90210},
    )

    private_cache.store(
        prompt="What's the phone number linked in my account?",
        response="The number on file is 123-555-1111",
        filters={"user_id": "bart"},
    )

    results = private_cache.check(
        "What's the phone number linked in my account?", num_results=5
    )
    assert len(results) == 3

    zip_code_filter = Num("zip_code") != 90210
    filtered_results = private_cache.check(
        "what's the phone number linked in my account?",
        num_results=5,
        filter_expression=zip_code_filter,
    )
    assert len(filtered_results) == 2


def test_create_cache_with_different_vector_types(client, worker_id, redis_url):
    skip_if_no_redisearch(client)
    try:
        bfloat_cache = SemanticCache(
            name=f"bfloat_cache_{worker_id}", dtype="bfloat16", redis_url=redis_url
        )
        bfloat_cache.store("bfloat16 prompt", "bfloat16 response")

        float16_cache = SemanticCache(
            name=f"float16_cache_{worker_id}", dtype="float16", redis_url=redis_url
        )
        float16_cache.store("float16 prompt", "float16 response")

        float32_cache = SemanticCache(
            name=f"float32_cache_{worker_id}", dtype="float32", redis_url=redis_url
        )
        float32_cache.store("float32 prompt", "float32 response")

        float64_cache = SemanticCache(
            name=f"float64_cache_{worker_id}", dtype="float64", redis_url=redis_url
        )
        float64_cache.store("float64 prompt", "float64 response")

        for cache in [bfloat_cache, float16_cache, float32_cache, float64_cache]:
            cache.set_threshold(0.6)
            assert len(cache.check("float prompt", num_results=5)) == 1
    except:
        pytest.skip("Required Redis modules not available or version too low")


def test_bad_dtype_connecting_to_existing_cache(client, redis_url, worker_id):
    skip_if_no_redisearch(client)
    # Skip this test for Redis 6.2.x as FT.INFO doesn't return dims properly
    redis_version = client.info()["redis_version"]
    if redis_version.startswith("6.2"):
        pytest.skip(
            "Redis 6.2.x FT.INFO doesn't properly return vector dims for reconnection"
        )

    def create_cache():
        return SemanticCache(
            name=f"float64_cache_{worker_id}", dtype="float64", redis_url=redis_url
        )

    def create_same_type():
        return SemanticCache(
            name=f"float64_cache_{worker_id}", dtype="float64", redis_url=redis_url
        )

    cache = create_cache()
    same_type = create_same_type()
    # under the hood uses from_existing

    with pytest.raises(ValueError):
        bad_type = SemanticCache(
            name=f"float64_cache_{worker_id}", dtype="float16", redis_url=redis_url
        )


def test_vectorizer_dtype_mismatch(redis_url, hf_vectorizer_float16, worker_id):
    with pytest.raises(ValueError):
        SemanticCache(
            name=f"test_dtype_mismatch_{worker_id}",
            dtype="float32",
            vectorizer=hf_vectorizer_float16,
            redis_url=redis_url,
            overwrite=True,
        )


def test_invalid_vectorizer(redis_url, worker_id):
    with pytest.raises(TypeError):
        SemanticCache(
            name=f"test_invalid_vectorizer_{worker_id}",
            vectorizer="invalid_vectorizer",  # type: ignore
            redis_url=redis_url,
            overwrite=True,
        )


def test_passes_through_dtype_to_default_vectorizer(redis_url, worker_id):
    # The default is float32, so we should see float64 if we pass it in.
    cache = SemanticCache(
        name=f"test_pass_through_dtype_{worker_id}",
        dtype="float64",
        redis_url=redis_url,
        overwrite=True,
    )
    assert cache._vectorizer.dtype == "float64"


def test_deprecated_dtype_argument(redis_url, worker_id):
    with pytest.warns(DeprecationWarning):
        SemanticCache(
            name=f"test_deprecated_dtype_{worker_id}",
            dtype="float32",
            redis_url=redis_url,
            overwrite=True,
        )


@pytest.mark.asyncio
async def test_cache_async_context_manager(
    async_client, redis_url, worker_id, hf_vectorizer
):
    await skip_if_no_redisearch_async(async_client)
    async with SemanticCache(
        name=f"test_cache_async_context_manager_{worker_id}",
        redis_url=redis_url,
        vectorizer=hf_vectorizer,
    ) as cache:
        await cache.astore("test prompt", "test response")
        assert cache._aindex
    assert cache._aindex is None


@pytest.mark.asyncio
async def test_cache_async_context_manager_with_exception(
    async_client, redis_url, worker_id, hf_vectorizer
):
    await skip_if_no_redisearch_async(async_client)
    try:
        async with SemanticCache(
            name=f"test_cache_async_context_manager_with_exception_{worker_id}",
            redis_url=redis_url,
            vectorizer=hf_vectorizer,
        ) as cache:
            await cache.astore("test prompt", "test response")
            raise ValueError("test")
    except ValueError:
        pass
    assert cache._aindex is None


@pytest.mark.asyncio
async def test_cache_async_disconnect(redis_url, worker_id, hf_vectorizer):
    cache = SemanticCache(
        name=f"test_cache_async_disconnect_{worker_id}",
        redis_url=redis_url,
        vectorizer=hf_vectorizer,
    )
    await cache.astore("test prompt", "test response")
    await cache.adisconnect()
    assert cache._aindex is None


def test_cache_disconnect(redis_url, worker_id, hf_vectorizer):
    cache = SemanticCache(
        name=f"test_cache_disconnect_{worker_id}",
        redis_url=redis_url,
        vectorizer=hf_vectorizer,
    )
    cache.store("test prompt", "test response")
    cache.disconnect()
    # We keep this index object around because it isn't lazily created
    assert cache._index.client is None
