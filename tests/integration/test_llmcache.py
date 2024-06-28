from collections import namedtuple
from time import sleep

import pytest

from redisvl.extensions.llmcache import SemanticCache
from redisvl.index.index import SearchIndex
from redisvl.utils.vectorize import HFTextVectorizer


@pytest.fixture
def vectorizer():
    return HFTextVectorizer("sentence-transformers/all-mpnet-base-v2")


@pytest.fixture
def cache(vectorizer, redis_url):
    cache_instance = SemanticCache(
        vectorizer=vectorizer, distance_threshold=0.2, redis_url=redis_url
    )
    yield cache_instance
    cache_instance.clear()  # Clear cache after each test
    cache_instance._index.delete(True)  # Clean up index


@pytest.fixture
def cache_no_cleanup(vectorizer, redis_url):
    cache_instance = SemanticCache(
        vectorizer=vectorizer, distance_threshold=0.2, redis_url=redis_url
    )
    yield cache_instance


@pytest.fixture
def cache_with_ttl(vectorizer, redis_url):
    cache_instance = SemanticCache(
        vectorizer=vectorizer, distance_threshold=0.2, ttl=2, redis_url=redis_url
    )
    yield cache_instance
    cache_instance.clear()  # Clear cache after each test
    cache_instance._index.delete(True)  # Clean up index


@pytest.fixture
def cache_with_redis_client(vectorizer, client, redis_url):
    cache_instance = SemanticCache(
        vectorizer=vectorizer,
        redis_client=client,
        distance_threshold=0.2,
        redis_url=redis_url,
    )
    yield cache_instance
    cache_instance.clear()  # Clear cache after each test
    cache_instance._index.delete(True)  # Clean up index


# Test basic store and check functionality
def test_store_and_check(cache, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)

    cache.store(prompt, response, vector=vector)
    check_result = cache.check(vector=vector)

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
        "id",
        "prompt",
        "response",
        "prompt_vector",
        "vector_distance",
    }

    # check all return fields
    fields = [
        "id",
        "prompt",
        "response",
        "inserted_at",
        "updated_at",
        "prompt_vector",
        "vector_distance",
    ]
    check_result = cache.check(vector=vector, return_fields=fields[:])
    assert set(check_result[0].keys()) == set(fields)

    # check only some return fields
    fields = ["inserted_at", "updated_at"]
    check_result = cache.check(vector=vector, return_fields=fields[:])
    fields.extend(["id", "vector_distance"])  # id and vector_distance always returned
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


# Test TTL functionality
def test_ttl_expiration(cache_with_ttl, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)

    cache_with_ttl.store(prompt, response, vector=vector)
    sleep(3)

    check_result = cache_with_ttl.check(vector=vector)
    assert len(check_result) == 0


# Test manual expiration of single document
def test_drop_document(cache, vectorizer):
    prompt = "This is a test prompt."
    response = "This is a test response."
    vector = vectorizer.embed(prompt)

    cache.store(prompt, response, vector=vector)
    check_result = cache.check(vector=vector)

    cache.drop(check_result[0]["id"])
    recheck_result = cache.check(vector=vector)
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
    keys = [r["id"] for r in check_result[0:2]]  # drop first 2 entries
    cache.drop(keys)

    recheck_result = cache.check(vector=vector, num_results=3)
    assert len(recheck_result) == 1


# Test updating document fields
def test_updating_document(cache):
    prompts = [
        "This is a test prompt.",
        "This is also test prompt.",
    ]
    responses = [
        "This is a test response.",
        "This is also test response.",
    ]
    for prompt, response in zip(prompts, responses):
        cache.store(prompt, response)

    check_result = cache.check(prompt=prompt, return_fields=["updated_at"])
    key = check_result[0]["id"]

    sleep(1)

    metadata = {"foo": "bar"}
    cache.update(key=key, metadata=metadata)

    updated_result = cache.check(
        prompt=prompt, return_fields=["updated_at", "metadata"]
    )
    assert updated_result[0]["id"] == check_result[0]["id"]
    assert updated_result[0]["metadata"] == metadata
    assert updated_result[0]["updated_at"] > check_result[0]["updated_at"]


# Test check behavior with no match
def test_check_no_match(cache, vectorizer):
    vector = vectorizer.embed("Some random sentence.")
    check_result = cache.check(vector=vector)
    assert len(check_result) == 0


# Test handling invalid input for check method
def test_check_invalid_input(cache):
    with pytest.raises(ValueError):
        cache.check()

    with pytest.raises(TypeError):
        cache.check(prompt="test", return_fields="bad value")


# Test handling invalid input for check method
def test_bad_ttl(cache):
    with pytest.raises(ValueError):
        cache.set_ttl(2.5)


# Test storing with metadata
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


# Test storing with invalid metadata
def test_store_with_invalid_metadata(cache, vectorizer):
    prompt = "This is another test prompt."
    response = "This is another test response."
    metadata = namedtuple("metadata", "source")(**{"source": "test"})

    vector = vectorizer.embed(prompt)

    with pytest.raises(
        TypeError, match=r"If specified, cached metadata must be a dictionary."
    ):
        cache.store(prompt, response, vector=vector, metadata=metadata)


# Test setting and getting the distance threshold
def test_distance_threshold(cache):
    initial_threshold = cache.distance_threshold
    new_threshold = 0.1

    cache.set_threshold(new_threshold)
    assert cache.distance_threshold == new_threshold
    assert cache.distance_threshold != initial_threshold


# Test out of range distance threshold
def test_distance_threshold_out_of_range(cache):
    out_of_range_threshold = -1
    with pytest.raises(ValueError):
        cache.set_threshold(out_of_range_threshold)


# Test storing and retrieving multiple items
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


# Test retrieving underlying SearchIndex for the cache.
def test_get_index(cache):
    assert isinstance(cache.index, SearchIndex)


# Test basic functionality with cache created with user-provided Redis client
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


# Test deleting the cache
def test_delete(cache_no_cleanup):
    cache_no_cleanup.delete()
    assert not cache_no_cleanup.index.exists()
