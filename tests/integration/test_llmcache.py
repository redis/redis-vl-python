import pytest

from time import sleep
from redisvl.llmcache.semantic import SemanticCache
from redisvl.providers import HuggingfaceProvider


@pytest.fixture
def provider():
    return HuggingfaceProvider("sentence-transformers/all-mpnet-base-v2")

@pytest.fixture
def cache(provider):
    return SemanticCache(provider=provider, threshold=0.8)

@pytest.fixture
def cache_with_ttl(provider):
    return SemanticCache(provider=provider, threshold=0.8, ttl=2)

@pytest.fixture
def vector(provider):
    return provider.embed("This is a test sentence.")


def test_store_and_check(cache, vector):
    # Check that we can store and retrieve a response
    prompt = "This is a test prompt."
    response = "This is a test response."
    cache.store(prompt, response, vector=vector)
    check_result = cache.check(vector=vector)
    assert len(check_result) >= 1
    assert response in check_result
    cache.clear()

def test_ttl(cache_with_ttl, vector):
    # Check that TTL expiration kicks in after 2 seconds
    prompt = "This is a test prompt."
    response = "This is a test response."
    cache_with_ttl.store(prompt, response, vector=vector)
    sleep(3)
    check_result = cache_with_ttl.check(vector=vector)
    assert len(check_result) == 0
    cache_with_ttl.clear()

def test_check_no_match(cache, vector):
    # Check behavior when there is no match in the cache
    # In this case, we're using a vector, but the cache is empty
    check_result = cache.check(vector=vector)
    assert len(check_result) == 0
    cache.clear()

def test_store_with_vector_and_metadata(cache, vector):
    # Test storing a response with a vector and metadata
    prompt = "This is another test prompt."
    response = "This is another test response."
    metadata = {"source": "test"}
    cache.store(prompt, response, vector=vector, metadata=metadata)
    check_result = cache.check(vector=vector)
    assert len(check_result) >= 1
    assert response in check_result
    cache.clear()

def test_set_threshold(cache):
    # Test the getter and setter for the threshold
    assert cache.threshold == 0.8
    cache.set_threshold(0.9)
    assert cache.threshold == 0.9
    cache.clear()

def test_from_existing(cache, vector, provider):
    prompt = "This is another test prompt."
    response = "This is another test response."
    metadata = {"source": "test"}
    cache.store(prompt, response, vector=vector, metadata=metadata)
    # connect from existing?
    new_cache = SemanticCache(provider=provider, threshold=0.8)
    check_result = new_cache.check(vector=vector)
    assert len(check_result) >= 1
    assert response in check_result
    new_cache.clear()