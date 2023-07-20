import pytest

from redisvl.llmcache.semantic import SemanticCache
from redisvl.providers import HuggingfaceProvider


@pytest.fixture
def provider():
    return HuggingfaceProvider("sentence-transformers/all-mpnet-base-v2")


@pytest.fixture
def cache(provider):
    return SemanticCache(provider=provider, threshold=0.8)


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
    cache.index.delete(drop=True)


def test_check_no_match(cache, vector):
    # Check behavior when there is no match in the cache
    # In this case, we're using a vector, but the cache is empty
    check_result = cache.check(vector=vector)
    assert len(check_result) == 0
    cache.index.delete(drop=True)


def test_store_with_vector_and_metadata(cache, vector):
    # Test storing a response with a vector and metadata
    prompt = "This is another test prompt."
    response = "This is another test response."
    metadata = {"source": "test"}
    cache.store(prompt, response, vector=vector, metadata=metadata)
    cache.index.delete(drop=True)


def test_set_threshold(cache):
    # Test the getter and setter for the threshold
    assert cache.threshold == 0.8
    cache.set_threshold(0.9)
    assert cache.threshold == 0.9
    cache.index.delete(drop=True)


def test_wrapper(cache):
    @cache.cache_response
    def test_function(prompt):
        return "This is a test response."

    # Check that the wrapper works
    test_function("This is a test prompt.")
    check_result = cache.check("This is a test prompt.")
    assert len(check_result) >= 1
    assert "This is a test response." in check_result
    cache.index.delete(drop=True)
