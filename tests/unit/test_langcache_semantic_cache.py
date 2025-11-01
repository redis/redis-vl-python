"""Unit tests for LangCacheSemanticCache."""

import importlib.util
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from redisvl.extensions.cache.llm.langcache import LangCacheSemanticCache


@pytest.fixture
def mock_langcache_client():
    """Create a mock LangCache client via the wrapper factory method."""
    with patch.object(LangCacheSemanticCache, "_create_client") as mock_create_client:
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        # Mock context manager
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        yield mock_create_client, mock_client


@pytest.mark.skipif(
    importlib.util.find_spec("langcache") is None,
    reason="langcache package not installed",
)
class TestLangCacheSemanticCache:
    """Test suite for LangCacheSemanticCache."""

    def test_init_requires_cache_id(self):
        """Test that cache_id is required."""
        with pytest.raises(ValueError, match="cache_id is required"):
            LangCacheSemanticCache(
                name="test",
                server_url="https://api.example.com",
                cache_id="",
                api_key="test-key",
            )

    def test_init_requires_api_key(self):
        """Test that api_key is required."""
        with pytest.raises(ValueError, match="api_key is required"):
            LangCacheSemanticCache(
                name="test",
                server_url="https://api.example.com",
                cache_id="test-cache",
                api_key="",
            )

    def test_init_requires_at_least_one_search_strategy(self):
        """Test that at least one search strategy must be enabled."""
        with pytest.raises(ValueError, match="At least one of use_exact_search"):
            LangCacheSemanticCache(
                name="test",
                server_url="https://api.example.com",
                cache_id="test-cache",
                api_key="test-key",
                use_exact_search=False,
                use_semantic_search=False,
            )

    def test_init_success(self, mock_langcache_client):
        """Test successful initialization."""
        mock_create_client, _ = mock_langcache_client

        cache = LangCacheSemanticCache(
            name="test_cache",
            server_url="https://api.example.com",
            cache_id="test-cache-id",
            api_key="test-api-key",
            ttl=3600,
        )

        assert cache.name == "test_cache"
        assert cache._server_url == "https://api.example.com"
        assert cache._cache_id == "test-cache-id"
        assert cache._api_key == "test-api-key"
        assert cache.ttl == 3600

        # Verify client was initialized
        mock_create_client.assert_called_once()

    def test_store(self, mock_langcache_client):
        """Test storing a cache entry."""
        _, mock_client = mock_langcache_client

        # Mock the set method - returns a Pydantic model with entry_id
        mock_response = MagicMock()
        mock_response.entry_id = "entry-123"
        mock_client.set.return_value = mock_response

        cache = LangCacheSemanticCache(
            name="test",
            server_url="https://api.example.com",
            cache_id="test-cache",
            api_key="test-key",
        )

        entry_id = cache.store(
            prompt="What is Python?",
            response="Python is a programming language.",
            metadata={"topic": "programming"},
        )

        assert entry_id == "entry-123"
        mock_client.set.assert_called_once_with(
            prompt="What is Python?",
            response="Python is a programming language.",
            attributes={"topic": "programming"},
        )

    @pytest.mark.asyncio
    async def test_astore(self, mock_langcache_client):
        """Test async storing a cache entry."""
        _, mock_client = mock_langcache_client

        # Mock the async set method - returns a Pydantic model with entry_id
        mock_response = MagicMock()
        mock_response.entry_id = "entry-456"
        mock_client.set_async = AsyncMock(return_value=mock_response)

        cache = LangCacheSemanticCache(
            name="test",
            server_url="https://api.example.com",
            cache_id="test-cache",
            api_key="test-key",
        )

        entry_id = await cache.astore(
            prompt="What is Redis?",
            response="Redis is an in-memory database.",
        )

        assert entry_id == "entry-456"
        mock_client.set_async.assert_called_once()

    def test_check(self, mock_langcache_client):
        """Test checking the cache."""
        _, mock_client = mock_langcache_client

        # Mock search results - returns SearchResponse with data attribute
        mock_entry = MagicMock()
        mock_entry.model_dump.return_value = {
            "id": "entry-123",
            "prompt": "What is Python?",
            "response": "Python is a programming language.",
            "similarity": 0.95,  # LangCache uses similarity, not distance
            "created_at": 1234567890.0,
            "updated_at": 1234567890.0,
            "attributes": {"topic": "programming"},
        }

        mock_response = MagicMock()
        mock_response.data = [mock_entry]
        mock_client.search.return_value = mock_response

        cache = LangCacheSemanticCache(
            name="test",
            server_url="https://api.example.com",
            cache_id="test-cache",
            api_key="test-key",
        )

        results = cache.check(prompt="What is Python?")

        assert len(results) == 1
        assert results[0]["entry_id"] == "entry-123"
        assert results[0]["prompt"] == "What is Python?"
        assert results[0]["response"] == "Python is a programming language."
        assert (
            abs(results[0]["vector_distance"] - 0.05) < 0.001
        )  # 1.0 - 0.95 similarity

        mock_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_acheck(self, mock_langcache_client):
        """Test async checking the cache."""
        _, mock_client = mock_langcache_client

        # Mock async search results - returns SearchResponse with data attribute
        mock_entry = MagicMock()
        mock_entry.model_dump.return_value = {
            "id": "entry-789",
            "prompt": "What is Redis?",
            "response": "Redis is an in-memory database.",
            "similarity": 0.97,  # LangCache uses similarity, not distance
            "created_at": 1234567890.0,
            "updated_at": 1234567890.0,
            "attributes": {},
        }

        mock_response = MagicMock()
        mock_response.data = [mock_entry]
        mock_client.search_async = AsyncMock(return_value=mock_response)

        cache = LangCacheSemanticCache(
            name="test",
            server_url="https://api.example.com",
            cache_id="test-cache",
            api_key="test-key",
        )

        results = await cache.acheck(prompt="What is Redis?")

        assert len(results) == 1
        assert results[0]["entry_id"] == "entry-789"

        mock_client.search_async.assert_called_once()

    def test_check_with_distance_threshold(self, mock_langcache_client):
        """Test checking the cache with distance_threshold."""
        _, mock_client = mock_langcache_client

        # Mock search results
        mock_entry = MagicMock()
        mock_entry.model_dump.return_value = {
            "id": "entry-123",
            "prompt": "What is Python?",
            "response": "Python is a programming language.",
            "similarity": 0.85,
            "created_at": 1234567890.0,
            "updated_at": 1234567890.0,
            "attributes": {},
        }

        mock_response = MagicMock()
        mock_response.data = [mock_entry]
        mock_client.search.return_value = mock_response

        cache = LangCacheSemanticCache(
            name="test",
            server_url="https://api.example.com",
            cache_id="test-cache",
            api_key="test-key",
        )

        # distance_threshold=0.2 should be converted to similarity_threshold=0.8
        results = cache.check(prompt="What is Python?", distance_threshold=0.2)

        assert len(results) == 1
        # Verify similarity_threshold was passed correctly (1.0 - 0.2 = 0.8)
        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["similarity_threshold"] == 0.8

    def test_check_with_attributes(self, mock_langcache_client):
        """Test checking the cache with attributes filtering."""
        _, mock_client = mock_langcache_client

        # Mock search results
        mock_entry = MagicMock()
        mock_entry.model_dump.return_value = {
            "id": "entry-123",
            "prompt": "What is Python?",
            "response": "Python is a programming language.",
            "similarity": 0.95,
            "created_at": 1234567890.0,
            "updated_at": 1234567890.0,
            "attributes": {"language": "python", "topic": "programming"},
        }

        mock_response = MagicMock()
        mock_response.data = [mock_entry]
        mock_client.search.return_value = mock_response

        cache = LangCacheSemanticCache(
            name="test",
            server_url="https://api.example.com",
            cache_id="test-cache",
            api_key="test-key",
        )

        # Search with attributes filter
        results = cache.check(
            prompt="What is Python?",
            attributes={"language": "python", "topic": "programming"},
        )

        assert len(results) == 1
        assert results[0]["entry_id"] == "entry-123"

        # Verify attributes were passed to search
        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["attributes"] == {
            "language": "python",
            "topic": "programming",
        }

    def test_delete(self, mock_langcache_client):
        """Test deleting the entire cache."""
        _, mock_client = mock_langcache_client

        cache = LangCacheSemanticCache(
            name="test",
            server_url="https://api.example.com",
            cache_id="test-cache",
            api_key="test-key",
        )

        cache.delete()

        mock_client.delete_query.assert_called_once_with(attributes={})

    @pytest.mark.asyncio
    async def test_adelete(self, mock_langcache_client):
        """Test async deleting the entire cache."""
        _, mock_client = mock_langcache_client

        mock_client.delete_query_async = AsyncMock()

        cache = LangCacheSemanticCache(
            name="test",
            server_url="https://api.example.com",
            cache_id="test-cache",
            api_key="test-key",
        )

        await cache.adelete()

        mock_client.delete_query_async.assert_called_once_with(attributes={})

    def test_delete_by_id(self, mock_langcache_client):
        """Test deleting a single entry by ID."""
        _, mock_client = mock_langcache_client

        cache = LangCacheSemanticCache(
            name="test",
            server_url="https://api.example.com",
            cache_id="test-cache",
            api_key="test-key",
        )

        cache.delete_by_id("entry-123")

        mock_client.delete_by_id.assert_called_once_with(entry_id="entry-123")

    def test_update_not_supported(self, mock_langcache_client):
        """Test that update raises NotImplementedError."""
        cache = LangCacheSemanticCache(
            name="test",
            server_url="https://api.example.com",
            cache_id="test-cache",
            api_key="test-key",
        )

        with pytest.raises(NotImplementedError, match="does not support updating"):
            cache.update("key", response="new response")

    @pytest.mark.asyncio
    async def test_aupdate_not_supported(self, mock_langcache_client):
        """Test that async update raises NotImplementedError."""
        cache = LangCacheSemanticCache(
            name="test",
            server_url="https://api.example.com",
            cache_id="test-cache",
            api_key="test-key",
        )

        with pytest.raises(NotImplementedError, match="does not support updating"):
            await cache.aupdate("key", response="new response")


def test_import_error_when_langcache_not_installed():
    """Test that ImportError is raised when langcache is not installed."""
    # If langcache is installed in this environment, this test is not applicable
    if importlib.util.find_spec("langcache") is not None:
        pytest.skip("langcache package is installed")

    with pytest.raises(ImportError, match="langcache package is required"):
        LangCacheSemanticCache(
            name="test",
            server_url="https://api.example.com",
            cache_id="test-cache",
            api_key="test-key",
        )
