import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.router.semantic import SemanticRouter
from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query.sql import SQLQuery
from redisvl.schema import IndexSchema
from redisvl.utils.utils import assert_no_warnings


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


def test_search_index_from_existing_honours_explicit_owns_client():
    """``owns_client`` is an init kwarg, not a connection kwarg.

    Without the allow-list entry in ``_split_from_existing_kwargs`` it would
    fall through to ``connection_kwargs`` and redis-py would reject it. An
    explicit value also wins over the ownership ``from_existing`` would
    otherwise assume for a client it created itself.
    """
    created_client = MagicMock()

    with (
        patch(
            "redisvl.index.index.RedisConnectionFactory.get_redis_connection",
            return_value=created_client,
        ) as mock_get_connection,
        patch.object(SearchIndex, "_info", return_value={}),
        patch(
            "redisvl.index.index.convert_index_info_to_schema",
            return_value=_schema_dict("search-index"),
        ),
    ):
        index = SearchIndex.from_existing(
            "search-index",
            redis_url="redis://localhost:6380",
            owns_client=False,
        )

    mock_get_connection.assert_called_once_with(redis_url="redis://localhost:6380")
    assert index._owns_redis_client is False

    index.disconnect()

    created_client.close.assert_not_called()


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
        "owns_client": True,
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


@pytest.mark.asyncio
async def test_base_cache_async_client_creation_emits_no_warning():
    """Creating a cache's async client must not warn.

    ``_get_async_redis_client`` used to call ``get_async_redis_connection``,
    which warns unconditionally, so cache users saw a DeprecationWarning for
    an API they never called. A suite-wide filter in pyproject.toml hid it.
    This is the guard that replaced that filter.
    """
    cache = EmbeddingsCache(redis_url="redis://localhost:6379")
    mock_client = MagicMock()

    with patch(
        "redisvl.extensions.cache.base.RedisConnectionFactory._get_aredis_connection",
        new=AsyncMock(return_value=mock_client),
    ) as mock_get_connection:
        with assert_no_warnings():
            client = await cache._get_async_redis_client()

    mock_get_connection.assert_awaited_once_with(redis_url="redis://localhost:6379")
    assert client is mock_client


@pytest.mark.asyncio
async def test_base_cache_async_client_creation_is_serialised():
    """Concurrent callers must share one client, not orphan a connection pool.

    Building the client awaits a CLIENT SETINFO round trip, so a bare
    check-then-set would let two tasks each create one and leave the first
    unreachable and never closed.
    """
    cache = EmbeddingsCache(redis_url="redis://localhost:6379")
    created = []

    async def factory(*args, **kwargs):
        await asyncio.sleep(0)  # the suspension the real factory introduces
        client = MagicMock(name=f"client{len(created)}")
        created.append(client)
        return client

    with patch(
        "redisvl.extensions.cache.base.RedisConnectionFactory._get_aredis_connection",
        new=factory,
    ):
        first, second = await asyncio.gather(
            cache._get_async_redis_client(), cache._get_async_redis_client()
        )

    assert len(created) == 1
    assert first is second


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


@pytest.mark.parametrize(
    "factory",
    [
        lambda **kw: SearchIndex(IndexSchema.from_dict(_schema_dict()), **kw),
        lambda **kw: AsyncSearchIndex(IndexSchema.from_dict(_schema_dict()), **kw),
    ],
    ids=["sync", "async"],
)
@pytest.mark.parametrize("removed", ["connection_args", "redis_kwargs"])
def test_constructors_reject_removed_connection_kwargs(factory, removed):
    """Removed keywords must fail loudly, not be swallowed by **kwargs."""
    with pytest.raises(TypeError) as excinfo:
        factory(**{removed: {"decode_responses": True}})

    assert removed in str(excinfo.value)
    assert "connection_kwargs" in str(excinfo.value)


def test_from_existing_rejects_removed_connection_kwargs():
    """The from_existing path routed these into connection_kwargs before."""
    with pytest.raises(TypeError) as excinfo:
        SearchIndex.from_existing(
            "idx", redis_url="redis://localhost:6379", connection_args={"db": 1}
        )

    assert "connection_kwargs" in str(excinfo.value)


def test_rejection_message_never_echoes_the_value():
    """Connection kwargs carry passwords; the message names keys only."""
    with pytest.raises(TypeError) as excinfo:
        SearchIndex(
            IndexSchema.from_dict(_schema_dict()),
            connection_args={"password": "s3cr3t-do-not-log"},
        )

    assert "s3cr3t-do-not-log" not in str(excinfo.value)


@pytest.mark.parametrize(
    "index_cls, wrong_client",
    [
        (SearchIndex, AsyncRedis.from_url("redis://localhost:6379")),
        (AsyncSearchIndex, Redis.from_url("redis://localhost:6379")),
    ],
    ids=["sync-index-async-client", "async-index-sync-client"],
)
def test_constructors_reject_the_wrong_client_flavour(index_cls, wrong_client):
    """A mismatched client fails at construction, not at first use.

    For the sync index this rejection was previously reachable only through
    the removed set_client(); for the async index the removed
    _validate_client() silently converted the client instead.
    """
    with pytest.raises(TypeError) as excinfo:
        index_cls(IndexSchema.from_dict(_schema_dict()), redis_client=wrong_client)

    assert "Redis client" in str(excinfo.value)


def test_wrong_flavour_guard_catches_specced_mocks():
    """Pins the isinstance choice the guard documents.

    Mock(spec=...) sets __class__, so isinstance catches a mis-flavoured
    spec'd mock where issubclass(type(...)) would not. An unspecced mock must
    still pass, because much of the suite injects one.
    """
    schema = IndexSchema.from_dict(_schema_dict())

    with pytest.raises(TypeError):
        SearchIndex(schema, redis_client=MagicMock(spec=AsyncRedis))

    assert SearchIndex(schema, redis_client=MagicMock()) is not None
