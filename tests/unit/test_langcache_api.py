import json

import pytest
from langcache.models import CacheEntryScope

from redisvl.extensions.llmcache.langcache_api import LangCache


@pytest.mark.asyncio
async def test_aclear_calls_async_delete_all(monkeypatch):
    lang_cache = LangCache()
    called = {}

    async def dummy_delete_all_async(cache_id, attributes, scope):
        called["cache_id"] = cache_id
        called["attributes"] = attributes
        called["scope"] = scope

    monkeypatch.setattr(
        lang_cache._api.entries, "delete_all_async", dummy_delete_all_async
    )

    await lang_cache.aclear()
    assert called["cache_id"] == lang_cache._cache_id
    assert called["attributes"] == {}
    # Should use a default CacheEntryScope when scope is None
    assert isinstance(called["scope"], CacheEntryScope)


@pytest.mark.asyncio
async def test_adelete_calls_async_delete_all_and_cache_delete(monkeypatch):
    lang_cache = LangCache()
    delete_all_called = False
    delete_cache_called = False
    scope_used = None

    async def dummy_delete_all_async(cache_id, attributes, scope):
        nonlocal delete_all_called, scope_used
        delete_all_called = True
        scope_used = scope
        assert cache_id == lang_cache._cache_id

    async def dummy_cache_delete_async(cache_id):
        nonlocal delete_cache_called
        delete_cache_called = True
        assert cache_id == lang_cache._cache_id

    monkeypatch.setattr(
        lang_cache._api.entries, "delete_all_async", dummy_delete_all_async
    )
    monkeypatch.setattr(lang_cache._api.cache, "delete_async", dummy_cache_delete_async)

    await lang_cache.adelete()
    assert delete_all_called and delete_cache_called
    assert isinstance(scope_used, CacheEntryScope)


@pytest.mark.asyncio
async def test_adrop_deletes_each_id(monkeypatch):
    lang_cache = LangCache()
    called_ids = []

    async def dummy_delete_async(cache_id, entry_id):
        called_ids.append((cache_id, entry_id))

    monkeypatch.setattr(lang_cache._api.entries, "delete_async", dummy_delete_async)

    await lang_cache.adrop(ids=["id1", "id2"])
    assert called_ids == [(lang_cache._cache_id, "id1"), (lang_cache._cache_id, "id2")]


@pytest.mark.asyncio
async def test_acheck_validates_input(monkeypatch):
    lang_cache = LangCache()
    with pytest.raises(ValueError):
        await lang_cache.acheck()
    with pytest.raises(TypeError):
        # Passing wrong type on purpose to validate error
        await lang_cache.acheck(prompt="p", return_fields="not a list")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_acheck_returns_formatted_results(monkeypatch):
    lang_cache = LangCache()

    class DummyMeta:
        def __init__(self):
            self.foo = "bar"

    class DummyEntry:
        def __init__(self, id, prompt, response, similarity, metadata=None):
            self.id = id
            self.prompt = prompt
            self.response = response
            self.similarity = similarity
            self.metadata = metadata

    entries = [
        DummyEntry("1", "p1", "r1", 0.1),
        DummyEntry("2", "p2", "r2", 0.2, metadata=DummyMeta()),
    ]

    async def dummy_search_async(cache_id, prompt, similarity_threshold, scope=None):
        assert cache_id == lang_cache._cache_id
        assert prompt == "test"
        # scope parameter is passed but can be None
        return entries

    monkeypatch.setattr(lang_cache._api.entries, "search_async", dummy_search_async)

    hits = await lang_cache.acheck(prompt="test", num_results=2)
    # Two results without timestamps
    assert len(hits) == 2
    assert hits[0] == {
        "key": "1",
        "entry_id": "1",
        "prompt": "p1",
        "response": "r1",
        "vector_distance": 0.1,
    }
    assert hits[1]["metadata"] == {"foo": "bar"}

    # Test return_fields filtering
    filtered = await lang_cache.acheck(
        prompt="test", num_results=2, return_fields=["response"]
    )
    assert all(set(hit.keys()) <= {"key", "response"} for hit in filtered)

    # Test with custom scope
    custom_scope = CacheEntryScope()
    filtered_scope = await lang_cache.acheck(
        prompt="test", num_results=2, entry_scope=custom_scope
    )
    assert len(filtered_scope) == 2  # Still should return our mocked entries


@pytest.mark.asyncio
async def test_astore_calls_create_and_returns_id(monkeypatch):
    lang_cache = LangCache()

    class DummyResp:
        def __init__(self, entry_id):
            self.entry_id = entry_id

    async def dummy_create_async(cache_id, prompt, response, attributes, ttl_millis):
        assert cache_id == lang_cache._cache_id
        assert prompt == "p"
        assert response == "r"
        # metadata should be JSON-serialized
        expected_attrs = {"f": "v", "metadata": json.dumps({"a": 1})}
        assert attributes == expected_attrs
        assert ttl_millis == 5000
        return DummyResp("newid")

    monkeypatch.setattr(lang_cache._api.entries, "create_async", dummy_create_async)

    ret = await lang_cache.astore(
        "p", "r", metadata={"a": 1}, filters={"f": "v"}, ttl=5
    )
    assert ret == "newid"


@pytest.mark.asyncio
async def test_astore_metadata_type_error():
    lang_cache = LangCache()
    with pytest.raises(ValueError):
        # Passing wrong metadata type on purpose to validate error
        await lang_cache.astore("p", "r", metadata="not a dict")  # type: ignore[arg-type]


def test_disconnect_closes_client(monkeypatch):
    lang_cache = LangCache()

    # Mock scenario where client exists
    class MockClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    mock_client = MockClient()

    # Use monkeypatch to patch the client object
    monkeypatch.setattr(lang_cache._api.sdk_configuration, "client", mock_client)

    # Call disconnect and verify it closed the client
    lang_cache.disconnect()
    assert mock_client.closed is True

    # Test with no client
    monkeypatch.setattr(lang_cache._api.sdk_configuration, "client", None)
    # Should not raise an exception
    lang_cache.disconnect()


@pytest.mark.asyncio
async def test_adisconnect_closes_clients(monkeypatch):
    lang_cache = LangCache()

    # Mock scenario where clients exist
    class MockSyncClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class MockAsyncClient:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    mock_sync_client = MockSyncClient()
    mock_async_client = MockAsyncClient()

    # Use monkeypatch to patch the client objects
    monkeypatch.setattr(lang_cache._api.sdk_configuration, "client", mock_sync_client)
    monkeypatch.setattr(
        lang_cache._api.sdk_configuration, "async_client", mock_async_client
    )

    # Call adisconnect and verify it closed both clients
    await lang_cache.adisconnect()
    assert mock_sync_client.closed is True
    assert mock_async_client.closed is True

    # Test with no clients
    monkeypatch.setattr(lang_cache._api.sdk_configuration, "client", None)
    monkeypatch.setattr(lang_cache._api.sdk_configuration, "async_client", None)
    # Should not raise an exception
    await lang_cache.adisconnect()


def test_context_manager_calls_api_methods(monkeypatch):
    # Create a client with a mock API
    class MockAPI:
        def __init__(self):
            self.enter_called = False
            self.exit_args = None

        def __enter__(self):
            self.enter_called = True
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exit_args = (exc_type, exc_val, exc_tb)

    mock_api = MockAPI()
    lang_cache = LangCache()
    # Use monkeypatch to replace the API
    monkeypatch.setattr(lang_cache, "_api", mock_api)

    # Test the synchronous context manager
    with lang_cache as lc:
        assert lc is lang_cache
        assert mock_api.enter_called is True

    # Verify __exit__ was called with None args (no exception)
    assert mock_api.exit_args == (None, None, None)


@pytest.mark.asyncio
async def test_async_context_manager_calls_api_methods(monkeypatch):
    # Create a client with a mock API
    class MockAPI:
        def __init__(self):
            self.enter_called = False
            self.exit_args = None

        async def __aenter__(self):
            self.enter_called = True
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            self.exit_args = (exc_type, exc_val, exc_tb)

    mock_api = MockAPI()
    lang_cache = LangCache()
    # Use monkeypatch to replace the API
    monkeypatch.setattr(lang_cache, "_api", mock_api)

    # Test the asynchronous context manager
    async with lang_cache as lc:
        assert lc is lang_cache
        assert mock_api.enter_called is True

    # Verify __aexit__ was called with None args (no exception)
    assert mock_api.exit_args == (None, None, None)
