from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.router.semantic import SemanticRouter
from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query.sql import SQLQuery


def _schema_dict(name: str = "idx") -> dict:
    return {
        "index": {
            "name": name,
            "prefix": f"{name}:",
            "storage_type": "hash",
        },
        "fields": [],
    }


def _fake_sql_redis_module(command: str = "FT.SEARCH idx *"):
    translated = MagicMock()
    translated.to_command_string.return_value = command
    executor = MagicMock()
    executor._translator.translate.return_value = translated
    module = SimpleNamespace(create_executor=MagicMock(return_value=executor))
    return module


def test_search_index_from_existing_prefers_provided_client():
    """Use the provided sync Redis client instead of constructing a new one."""
    provided_client = MagicMock()

    with (
        patch(
            "redisvl.index.index.RedisConnectionFactory.validate_sync_redis"
        ) as mock_validate,
        patch(
            "redisvl.index.index.RedisConnectionFactory.get_redis_connection"
        ) as mock_get_connection,
        patch.object(SearchIndex, "_info", return_value={}) as mock_info,
        patch(
            "redisvl.index.index.convert_index_info_to_schema",
            return_value=_schema_dict("search-index"),
        ),
    ):
        index = SearchIndex.from_existing(
            "search-index",
            redis_client=provided_client,
            redis_url="redis://should-not-be-used:6379",
            lib_name="search-index-lib",
        )

    mock_validate.assert_called_once_with(provided_client, "search-index-lib")
    mock_get_connection.assert_not_called()
    mock_info.assert_called_once_with("search-index", provided_client)
    assert index.client is provided_client


def test_search_index_from_existing_owns_factory_created_client():
    """Reuse a single sync client created internally from redis_url."""
    created_client = MagicMock()

    with (
        patch(
            "redisvl.index.index.RedisConnectionFactory.get_redis_connection",
            return_value=created_client,
        ) as mock_get_connection,
        patch.object(SearchIndex, "_info", return_value={}) as mock_info,
        patch(
            "redisvl.index.index.convert_index_info_to_schema",
            return_value=_schema_dict("search-index"),
        ),
    ):
        index = SearchIndex.from_existing(
            "search-index",
            redis_url="redis://localhost:6380",
            connection_kwargs={"decode_responses": True},
            socket_timeout=5.0,
            lib_name="search-index-lib",
        )

    mock_get_connection.assert_called_once_with(
        redis_url="redis://localhost:6380",
        decode_responses=True,
        socket_timeout=5.0,
        lib_name="search-index-lib",
    )
    mock_info.assert_called_once_with("search-index", created_client)
    created_client.close.assert_not_called()
    assert index.client is created_client
    assert index._owns_redis_client is True
    assert index._redis_url == "redis://localhost:6380"
    assert index._connection_kwargs == {
        "decode_responses": True,
        "socket_timeout": 5.0,
    }
    assert index._lib_name == "search-index-lib"
    assert index._validated_client is True

    index.disconnect()

    created_client.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_async_search_index_from_existing_prefers_provided_client():
    """Use the provided async Redis client instead of constructing a new one."""
    provided_client = AsyncMock()

    with (
        patch(
            "redisvl.index.index.RedisConnectionFactory.validate_async_redis",
            new=AsyncMock(),
        ) as mock_validate,
        patch(
            "redisvl.index.index.RedisConnectionFactory._get_aredis_connection",
            new=AsyncMock(),
        ) as mock_get_connection,
        patch.object(
            AsyncSearchIndex, "_info", new=AsyncMock(return_value={})
        ) as mock_info,
        patch(
            "redisvl.index.index.convert_index_info_to_schema",
            return_value=_schema_dict("async-search-index"),
        ),
    ):
        index = await AsyncSearchIndex.from_existing(
            "async-search-index",
            redis_client=provided_client,
            redis_url="redis://should-not-be-used:6379",
            lib_name="async-search-index-lib",
        )

    mock_validate.assert_awaited_once_with(provided_client, "async-search-index-lib")
    mock_get_connection.assert_not_awaited()
    mock_info.assert_awaited_once_with("async-search-index", provided_client)
    assert index.client is provided_client


@pytest.mark.asyncio
async def test_async_search_index_from_existing_owns_factory_created_client():
    """Reuse a single async client created internally from redis_url."""
    created_client = AsyncMock()

    with (
        patch(
            "redisvl.index.index.RedisConnectionFactory._get_aredis_connection",
            new=AsyncMock(return_value=created_client),
        ) as mock_get_connection,
        patch.object(
            AsyncSearchIndex, "_info", new=AsyncMock(return_value={})
        ) as mock_info,
        patch(
            "redisvl.index.index.convert_index_info_to_schema",
            return_value=_schema_dict("async-search-index"),
        ),
    ):
        index = await AsyncSearchIndex.from_existing(
            "async-search-index",
            redis_url="redis://localhost:6380",
            connection_kwargs={"decode_responses": True},
            socket_timeout=5.0,
            lib_name="async-search-index-lib",
        )

    mock_get_connection.assert_awaited_once_with(
        redis_url="redis://localhost:6380",
        decode_responses=True,
        socket_timeout=5.0,
        lib_name="async-search-index-lib",
    )
    mock_info.assert_awaited_once_with("async-search-index", created_client)
    created_client.aclose.assert_not_awaited()
    assert index.client is created_client
    assert index._owns_redis_client is True
    assert index._redis_url == "redis://localhost:6380"
    assert index._connection_kwargs == {
        "decode_responses": True,
        "socket_timeout": 5.0,
    }
    assert index._lib_name == "async-search-index-lib"
    assert index._validated_client is True

    await index.disconnect()

    created_client.aclose.assert_awaited_once_with()


def test_semantic_router_from_existing_prefers_provided_client():
    """Reuse the provided Redis client when loading a semantic router."""
    provided_client = MagicMock()
    router_dict = {
        "name": "router",
        "routes": [],
        "vectorizer": {
            "type": "hf",
            "model": "sentence-transformers/all-mpnet-base-v2",
        },
        "routing_config": {},
    }
    provided_client.json.return_value.get.return_value = router_dict
    loaded_router = SimpleNamespace(name="router")

    with (
        patch(
            "redisvl.extensions.router.semantic.RedisConnectionFactory.validate_sync_redis"
        ) as mock_validate,
        patch(
            "redisvl.extensions.router.semantic.RedisConnectionFactory.get_redis_connection"
        ) as mock_get_connection,
        patch.object(
            SemanticRouter, "from_dict", return_value=loaded_router
        ) as mock_from_dict,
    ):
        result = SemanticRouter.from_existing(
            "router",
            redis_client=provided_client,
            redis_url="redis://should-not-be-used:6379",
        )

    mock_validate.assert_called_once_with(provided_client, None)
    mock_get_connection.assert_not_called()
    provided_client.json.return_value.get.assert_called_once_with("router:route_config")
    assert mock_from_dict.call_args.args[0] == router_dict
    assert mock_from_dict.call_args.kwargs["redis_url"] is None
    assert mock_from_dict.call_args.kwargs["redis_client"] is provided_client
    assert result is loaded_router


def test_semantic_router_from_existing_rebuilds_from_redis_url():
    """Keep internal kwargs out of the connection factory when loading a router."""
    created_client = MagicMock()
    router_dict = {
        "name": "router",
        "routes": [],
        "vectorizer": {
            "type": "hf",
            "model": "sentence-transformers/all-mpnet-base-v2",
        },
        "routing_config": {},
    }
    created_client.json.return_value.get.return_value = router_dict
    loaded_router = SimpleNamespace(name="router")

    with (
        patch(
            "redisvl.extensions.router.semantic.RedisConnectionFactory.get_redis_connection",
            return_value=created_client,
        ) as mock_get_connection,
        patch.object(
            SemanticRouter, "from_dict", return_value=loaded_router
        ) as mock_from_dict,
    ):
        result = SemanticRouter.from_existing(
            "router",
            redis_url="redis://localhost:6380",
            connection_kwargs={"decode_responses": True},
            socket_timeout=5.0,
            _internal_flag=True,
        )

    mock_get_connection.assert_called_once_with(
        redis_url="redis://localhost:6380",
        decode_responses=True,
        socket_timeout=5.0,
    )
    created_client.close.assert_not_called()
    assert mock_from_dict.call_args.args[0] == router_dict
    assert mock_from_dict.call_args.kwargs["redis_url"] == "redis://localhost:6380"
    assert mock_from_dict.call_args.kwargs["redis_client"] is created_client
    assert mock_from_dict.call_args.kwargs["connection_kwargs"] == {
        "decode_responses": True,
        "socket_timeout": 5.0,
    }
    assert mock_from_dict.call_args.kwargs["_index_kwargs"] == {
        "_internal_flag": True,
        "_client_validated": True,
        "_owns_redis_client": True,
    }
    assert result is loaded_router


def test_base_cache_sync_client_creation_uses_connection_factory():
    """Create cache Redis clients through the shared connection factory."""
    cache = EmbeddingsCache(redis_url="redis+sentinel://localhost:26379/mymaster")
    mock_client = MagicMock()

    with patch(
        "redisvl.extensions.cache.base.RedisConnectionFactory.get_redis_connection",
        return_value=mock_client,
    ) as mock_get_connection:
        client = cache._get_redis_client()

    mock_get_connection.assert_called_once_with(
        redis_url="redis+sentinel://localhost:26379/mymaster"
    )
    assert client is mock_client


def test_sql_query_uses_connection_factory_for_redis_url():
    """Build SQL query helper connections through the shared connection factory."""
    fake_sql_redis_module = _fake_sql_redis_module()
    mock_client = MagicMock()

    with (
        patch.dict("sys.modules", {"sql_redis": fake_sql_redis_module}),
        patch(
            "redisvl.query.sql.RedisConnectionFactory.get_redis_connection",
            return_value=mock_client,
        ) as mock_get_connection,
    ):
        command = SQLQuery("SELECT * FROM idx").redis_query_string(
            redis_url="redis://localhost:6379?cluster=true"
        )

    mock_get_connection.assert_called_once_with(
        redis_url="redis://localhost:6379?cluster=true"
    )
    fake_sql_redis_module.create_executor.assert_called_once_with(
        mock_client,
        schema_cache_strategy="lazy",
    )
    assert command == "FT.SEARCH idx *"


def test_sql_query_does_not_create_new_connection_when_client_provided():
    """Reuse a provided SQL query client instead of creating a new connection."""
    fake_sql_redis_module = _fake_sql_redis_module()
    provided_client = MagicMock()

    with (
        patch.dict("sys.modules", {"sql_redis": fake_sql_redis_module}),
        patch(
            "redisvl.query.sql.RedisConnectionFactory.get_redis_connection"
        ) as mock_get_connection,
    ):
        command = SQLQuery("SELECT * FROM idx").redis_query_string(
            redis_client=provided_client
        )

    mock_get_connection.assert_not_called()
    fake_sql_redis_module.create_executor.assert_called_once_with(
        provided_client,
        schema_cache_strategy="lazy",
    )
    assert command == "FT.SEARCH idx *"
