import asyncio
from types import SimpleNamespace

import pytest

from redisvl.mcp.server import RedisVLMCPServer
from redisvl.mcp.settings import MCPSettings
from redisvl.schema import IndexSchema


def _dummy_settings() -> MCPSettings:
    return MCPSettings(config="/tmp/mcp.yaml")


def _startup_schema() -> IndexSchema:
    return IndexSchema.from_dict(
        {
            "index": {
                "name": "docs-index",
                "prefix": "doc",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "content", "type": "text"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": 3,
                        "distance_metric": "cosine",
                        "datatype": "float32",
                    },
                },
            ],
        }
    )


def _binding_namespace(
    *, requires_startup_vectorizer: bool = True, max_concurrency: int = 1
):
    return SimpleNamespace(
        redis_name="idx",
        read_only=False,
        requires_startup_vectorizer=requires_startup_vectorizer,
        runtime=SimpleNamespace(
            max_concurrency=max_concurrency,
            startup_timeout_seconds=1,
            request_timeout_seconds=1,
        ),
        vectorizer=SimpleNamespace(
            class_name="FakeVectorizer",
            to_init_kwargs=lambda: {},
        ),
        validate_search=lambda **kwargs: None,
    )


def _startup_config(indexes=None):
    return SimpleNamespace(
        server=SimpleNamespace(redis_url="redis://localhost:6379"),
        indexes=indexes or {"knowledge": _binding_namespace()},
    )


def _patch_probe(monkeypatch, value: bool = False):
    async def fake_probe(index):
        return value

    monkeypatch.setattr(
        RedisVLMCPServer, "_probe_native_hybrid_search", staticmethod(fake_probe)
    )


@pytest.mark.asyncio
async def test_server_registers_fastmcp_lifespan(monkeypatch):
    captured = {}

    def fake_fastmcp_init(self, *args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr("redisvl.mcp.server.FastMCP.__init__", fake_fastmcp_init)

    server = RedisVLMCPServer(_dummy_settings())

    assert captured["args"] == ("redisvl",)
    assert captured["kwargs"]["lifespan"] == server._fastmcp_lifespan


@pytest.mark.asyncio
async def test_server_lifespan_invokes_startup_and_shutdown(monkeypatch):
    monkeypatch.setattr(
        "redisvl.mcp.server.FastMCP.__init__", lambda self, *a, **k: None
    )
    server = RedisVLMCPServer(_dummy_settings())
    calls = []

    async def fake_startup():
        calls.append("startup")

    async def fake_shutdown():
        calls.append("shutdown")

    monkeypatch.setattr(server, "startup", fake_startup)
    monkeypatch.setattr(server, "shutdown", fake_shutdown)

    async with server._fastmcp_lifespan(server):
        assert calls == ["startup"]

    assert calls == ["startup", "shutdown"]


@pytest.mark.asyncio
async def test_run_guarded_rejects_before_startup():
    server = RedisVLMCPServer(_dummy_settings())
    future = asyncio.get_running_loop().create_future()
    future.set_result(None)

    with pytest.raises(RuntimeError, match="not running"):
        await server.run_guarded("test", future, timeout_seconds=1)


@pytest.mark.asyncio
async def test_run_guarded_rejects_after_shutdown():
    server = RedisVLMCPServer(_dummy_settings())
    server._semaphore = asyncio.Semaphore(1)

    await server.shutdown()

    future = asyncio.get_running_loop().create_future()
    future.set_result(None)
    with pytest.raises(RuntimeError, match="not running"):
        await server.run_guarded("test", future, timeout_seconds=1)


@pytest.mark.asyncio
async def test_run_guarded_uses_per_binding_timeout(monkeypatch):
    server = RedisVLMCPServer(_dummy_settings())
    server._semaphore = asyncio.Semaphore(1)
    server.config = _startup_config()
    server._lifecycle_state = server._LifecycleState.RUNNING

    captured = {}

    async def fake_wait_for(awaitable, timeout):
        captured["timeout"] = timeout
        return await awaitable

    monkeypatch.setattr("redisvl.mcp.server.asyncio.wait_for", fake_wait_for)

    future = asyncio.get_running_loop().create_future()
    future.set_result("ok")

    result = await server.run_guarded("test", future, timeout_seconds=42)

    assert result == "ok"
    assert captured["timeout"] == 42


@pytest.mark.asyncio
async def test_startup_sizes_semaphore_from_max_binding_concurrency(monkeypatch):
    monkeypatch.setattr(
        "redisvl.mcp.server.FastMCP.__init__", lambda self, *a, **k: None
    )
    config = _startup_config(
        indexes={
            "knowledge": _binding_namespace(max_concurrency=4),
            "tickets": _binding_namespace(max_concurrency=9),
        }
    )
    monkeypatch.setattr("redisvl.mcp.server.load_mcp_config", lambda path: config)

    captured = {}

    def fake_semaphore(value):
        captured["value"] = value
        return SimpleNamespace(value=value)

    monkeypatch.setattr("redisvl.mcp.server.asyncio.Semaphore", fake_semaphore)

    async def fake_initialize_binding(self, binding_id, binding):
        return SimpleNamespace(binding_id=binding_id)

    monkeypatch.setattr(
        RedisVLMCPServer, "_initialize_binding", fake_initialize_binding
    )
    monkeypatch.setattr(RedisVLMCPServer, "_register_tools", lambda self: None)

    server = RedisVLMCPServer(_dummy_settings())
    await server._initialize_runtime_resources()

    assert captured["value"] == 9


@pytest.mark.asyncio
async def test_startup_failure_leaves_server_stopped(monkeypatch):
    monkeypatch.setattr(
        "redisvl.mcp.server.FastMCP.__init__", lambda self, *a, **k: None
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.load_mcp_config",
        lambda path: _startup_config(),
    )

    async def fail_connection(**kwargs):
        raise RuntimeError("connect failed")

    monkeypatch.setattr(
        "redisvl.mcp.server.RedisConnectionFactory._get_aredis_connection",
        fail_connection,
    )

    server = RedisVLMCPServer(_dummy_settings())

    with pytest.raises(RuntimeError, match="connect failed"):
        await server.startup()

    assert server._lifecycle_state.name == "STOPPED"
    assert server.config is None
    assert server._semaphore is None
    assert server._bindings == {}


@pytest.mark.asyncio
async def test_startup_failure_before_index_initialization_closes_client(monkeypatch):
    monkeypatch.setattr(
        "redisvl.mcp.server.FastMCP.__init__", lambda self, *a, **k: None
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.load_mcp_config",
        lambda path: _startup_config(),
    )

    class FakeClient:
        def __init__(self):
            self.aclose_calls = 0

        async def aclose(self):
            self.aclose_calls += 1

    client = FakeClient()

    async def fake_connect(self, timeout):
        return client

    async def fail_load_schema(self, binding, client, timeout):
        raise RuntimeError("schema load failed")

    monkeypatch.setattr(RedisVLMCPServer, "_connect_redis_client", fake_connect)
    monkeypatch.setattr(RedisVLMCPServer, "_load_effective_schema", fail_load_schema)

    server = RedisVLMCPServer(_dummy_settings())

    with pytest.raises(RuntimeError, match="schema load failed"):
        await server.startup()

    assert client.aclose_calls == 1
    assert server._lifecycle_state.name == "STOPPED"
    assert server.config is None
    assert server._semaphore is None
    assert server._bindings == {}


@pytest.mark.asyncio
async def test_startup_failure_after_index_initialization_uses_index_teardown(
    monkeypatch,
):
    monkeypatch.setattr(
        "redisvl.mcp.server.FastMCP.__init__", lambda self, *a, **k: None
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.load_mcp_config",
        lambda path: _startup_config(),
    )

    class FakeClient:
        def __init__(self):
            self.aclose_calls = 0

        async def aclose(self):
            self.aclose_calls += 1

    client = FakeClient()
    disconnect_calls = []

    async def fake_connect(self, timeout):
        return client

    async def fake_load_schema(self, binding, client, timeout):
        return _startup_schema()

    async def fail_vectorizer(self, binding, schema, timeout):
        raise RuntimeError("vectorizer init failed")

    async def fake_disconnect(self):
        disconnect_calls.append(self)

    monkeypatch.setattr(RedisVLMCPServer, "_connect_redis_client", fake_connect)
    monkeypatch.setattr(RedisVLMCPServer, "_load_effective_schema", fake_load_schema)
    _patch_probe(monkeypatch, value=False)
    monkeypatch.setattr(RedisVLMCPServer, "_initialize_vectorizer", fail_vectorizer)
    monkeypatch.setattr(
        "redisvl.mcp.server.AsyncSearchIndex.disconnect",
        fake_disconnect,
        raising=False,
    )

    server = RedisVLMCPServer(_dummy_settings())

    with pytest.raises(RuntimeError, match="vectorizer init failed"):
        await server.startup()

    assert client.aclose_calls == 0
    assert len(disconnect_calls) == 1
    assert server._lifecycle_state.name == "STOPPED"
    assert server.config is None
    assert server._semaphore is None
    assert server._bindings == {}


@pytest.mark.asyncio
async def test_server_registers_tools_with_effective_schema(monkeypatch):
    monkeypatch.setattr(
        "redisvl.mcp.server.FastMCP.__init__", lambda self, *a, **k: None
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.load_mcp_config",
        lambda path: _startup_config(),
    )

    class FakeClient:
        async def aclose(self):
            return None

    async def fake_connect(self, timeout):
        return FakeClient()

    async def fake_load_schema(self, binding, client, timeout):
        return IndexSchema.from_dict(
            {
                "index": {
                    "name": "docs-index",
                    "prefix": "doc",
                    "storage_type": "hash",
                },
                "fields": [
                    {"name": "content", "type": "text"},
                    {"name": "category", "type": "tag"},
                    {"name": "location", "type": "geo"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "attrs": {
                            "algorithm": "flat",
                            "dims": 3,
                            "distance_metric": "cosine",
                            "datatype": "float32",
                        },
                    },
                ],
            }
        )

    async def fake_initialize_vectorizer(self, binding, schema, timeout):
        return SimpleNamespace(dims=3)

    registered_schemas = []

    def fake_register_search_tool(server, schema):
        registered_schemas.append(schema)

    async def fake_disconnect(self):
        return None

    monkeypatch.setattr(RedisVLMCPServer, "_connect_redis_client", fake_connect)
    monkeypatch.setattr(RedisVLMCPServer, "_load_effective_schema", fake_load_schema)
    _patch_probe(monkeypatch, value=False)
    monkeypatch.setattr(
        RedisVLMCPServer, "_initialize_vectorizer", fake_initialize_vectorizer
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.register_search_tool", fake_register_search_tool
    )
    monkeypatch.setattr("redisvl.mcp.server.register_upsert_tool", lambda server: None)
    monkeypatch.setattr(
        "redisvl.mcp.server.register_list_indexes_tool", lambda server: None
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.AsyncSearchIndex.disconnect",
        fake_disconnect,
        raising=False,
    )

    server = RedisVLMCPServer(_dummy_settings())

    await server.startup()

    assert len(registered_schemas) == 1
    assert list(registered_schemas[0].field_names) == [
        "content",
        "category",
        "location",
        "embedding",
    ]

    await server.shutdown()


@pytest.mark.asyncio
async def test_startup_while_running_raises(monkeypatch):
    monkeypatch.setattr(
        "redisvl.mcp.server.FastMCP.__init__", lambda self, *a, **k: None
    )
    server = RedisVLMCPServer(_dummy_settings())
    server._lifecycle_state = server._LifecycleState.RUNNING

    with pytest.raises(RuntimeError, match="already running"):
        await server.startup()


@pytest.mark.asyncio
async def test_shutdown_after_stop_is_idempotent():
    server = RedisVLMCPServer(_dummy_settings())

    await server.shutdown()
    await server.shutdown()

    assert server._lifecycle_state.name == "STOPPED"
