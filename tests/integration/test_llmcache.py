from time import sleep

import pytest

from redisvl.index import SearchIndex
from redisvl.llmcache.semantic import SemanticCache
from redisvl.vectorize.text import HFTextVectorizer


@pytest.fixture
def vectorizer():
    return HFTextVectorizer("sentence-transformers/all-mpnet-base-v2")


@pytest.fixture
def cache(vectorizer):
    return SemanticCache(vectorizer=vectorizer, semantic_threshold=0.8)


@pytest.fixture
def cache_with_ttl(vectorizer):
    return SemanticCache(vectorizer=vectorizer, semantic_threshold=0.8, ttl=2)


@pytest.fixture
def vector(vectorizer):
    return vectorizer.embed("This is a test sentence.")


def test_store_and_check_and_clear(cache, vector):
    # Check that we can store and retrieve a response
    prompt = "This is a test prompt."
    response = "This is a test response."
    cache.store(prompt, response, vector=vector)
    check_result = cache.check(vector=vector)
    assert len(check_result) >= 1
    assert response in check_result
    cache.clear()
    check_result = cache.check(vector=vector)
    assert len(check_result) == 0
    cache._index.delete(True)


def test_ttl(cache_with_ttl, vector):
    # Check that TTL expiration kicks in after 2 seconds
    prompt = "This is a test prompt."
    response = "This is a test response."
    cache_with_ttl.store(prompt, response, vector=vector)
    sleep(3)
    check_result = cache_with_ttl.check(vector=vector)
    assert len(check_result) == 0
    cache_with_ttl._index.delete(True)


def test_check_no_match(cache, vector):
    # Check behavior when there is no match in the cache
    # In this case, we're using a vector, but the cache is empty
    check_result = cache.check(vector=vector)
    assert len(check_result) == 0
    cache._index.delete(True)


def test_check_failure(cache):
    with pytest.raises(ValueError):
        cache.check(num_results=1)


def test_store_with_vector_and_metadata(cache, vector):
    # Test storing a response with a vector and metadata
    prompt = "This is another test prompt."
    response = "This is another test response."
    metadata = {"source": "test"}
    cache.store(prompt, response, vector=vector, metadata=metadata)
    check_result = cache.check(vector=vector)
    assert len(check_result) >= 1
    assert response in check_result
    cache._index.delete(True)


def test_set_threshold(cache):
    # Test the getter and setter for the threshold
    assert cache.semantic_threshold == 0.8
    cache.set_threshold(0.9)
    assert cache.semantic_threshold == 0.9
    cache._index.delete(True)


def test_from_index(client, vector):
    # Create customer index
    index = SearchIndex(name="test", fields=SemanticCache._default_fields)
    index.set_client(client)
    index.create(overwrite=True)

    cache = SemanticCache.from_index(index)
    assert cache._index == index

    cache.store("test", "test", vector=vector)
    check_result = cache.check(vector=vector)
    assert len(check_result) >= 1

    cache.clear()
    check_result = cache.check(vector=vector)
    assert len(check_result) == 0


def test_from_existing_cache(cache, vector, vectorizer):
    prompt = "This is another test prompt."
    response = "This is another test response."
    metadata = {"source": "test"}
    cache.store(prompt, response, vector=vector, metadata=metadata)
    # connect from existing?
    new_cache = SemanticCache(vectorizer=vectorizer, semantic_threshold=0.8)
    check_result = new_cache.check(vector=vector)
    assert len(check_result) >= 1
    assert response in check_result
    new_cache._index.delete(True)
