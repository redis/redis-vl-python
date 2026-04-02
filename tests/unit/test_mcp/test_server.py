import asyncio
from types import SimpleNamespace

import pytest

from redisvl.mcp.server import RedisVLMCPServer
from redisvl.mcp.settings import MCPSettings


def _dummy_settings() -> MCPSettings:
    return MCPSettings(config="/tmp/mcp.yaml")


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
        await server.run_guarded("test", future)


@pytest.mark.asyncio
async def test_run_guarded_rejects_after_shutdown():
    server = RedisVLMCPServer(_dummy_settings())
    server.config = SimpleNamespace(
        runtime=SimpleNamespace(request_timeout_seconds=1, max_concurrency=1)
    )
    server._semaphore = asyncio.Semaphore(1)

    await server.shutdown()

    future = asyncio.get_running_loop().create_future()
    future.set_result(None)
    with pytest.raises(RuntimeError, match="not running"):
        await server.run_guarded("test", future)


@pytest.mark.asyncio
async def test_startup_failure_leaves_server_stopped(monkeypatch):
    monkeypatch.setattr(
        "redisvl.mcp.server.FastMCP.__init__", lambda self, *a, **k: None
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.load_mcp_config",
        lambda path: SimpleNamespace(
            runtime=SimpleNamespace(max_concurrency=1, startup_timeout_seconds=1),
            server=SimpleNamespace(redis_url="redis://localhost:6379"),
            redis_name="idx",
        ),
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
