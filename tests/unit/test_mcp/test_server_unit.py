from types import SimpleNamespace

import pytest

from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.runtime import BindingRuntime
from redisvl.mcp.server import RedisVLMCPServer


class FakeClient:
    def __init__(self):
        self.info_calls = 0

    async def info(self, section: str):
        self.info_calls += 1
        assert section == "server"
        return {"redis_version": "8.4.0"}

    def ft(self, index_name: str):
        assert index_name == "docs-index"
        return SimpleNamespace(hybrid_search=object())


class FakeIndex:
    def __init__(self, client: FakeClient):
        self.schema = SimpleNamespace(index=SimpleNamespace(name="docs-index"))
        self._client = client

    async def _get_client(self):
        return self._client


@pytest.mark.asyncio
async def test_probe_native_hybrid_search_detects_support(monkeypatch):
    client = FakeClient()
    index = FakeIndex(client)

    monkeypatch.setattr("redisvl.mcp.server.redis_py_version", "7.1.0")

    assert await RedisVLMCPServer._probe_native_hybrid_search(index) is True
    assert client.info_calls == 1


@pytest.mark.asyncio
async def test_probe_native_hybrid_search_false_for_old_redis_py(monkeypatch):
    client = FakeClient()
    index = FakeIndex(client)

    monkeypatch.setattr("redisvl.mcp.server.redis_py_version", "7.0.0")

    assert await RedisVLMCPServer._probe_native_hybrid_search(index) is False
    # Old redis-py short-circuits before querying the server.
    assert client.info_calls == 0


def _binding_runtime(
    binding_id: str, *, effective_read_only: bool = False
) -> BindingRuntime:
    return BindingRuntime(
        binding_id=binding_id,
        binding=SimpleNamespace(),
        index=SimpleNamespace(),
        schema=SimpleNamespace(),
        vectorizer=None,
        supports_native_hybrid_search=False,
        effective_read_only=effective_read_only,
    )


def _server_with_bindings(*binding_ids: str) -> RedisVLMCPServer:
    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server._bindings = {bid: _binding_runtime(bid) for bid in binding_ids}
    return server


def test_resolve_binding_before_startup_raises():
    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server._bindings = {}

    with pytest.raises(RuntimeError, match="not been started"):
        server.resolve_binding(None)


def test_resolve_binding_defaults_to_sole_binding():
    server = _server_with_bindings("knowledge")

    assert server.resolve_binding(None).binding_id == "knowledge"


def test_resolve_binding_requires_index_when_multiple_configured():
    server = _server_with_bindings("knowledge", "tickets")

    with pytest.raises(RedisVLMCPError) as excinfo:
        server.resolve_binding(None)

    assert excinfo.value.code == MCPErrorCode.INVALID_REQUEST
    assert "knowledge" in str(excinfo.value)
    assert "tickets" in str(excinfo.value)


def test_resolve_binding_routes_to_named_index():
    server = _server_with_bindings("knowledge", "tickets")

    assert server.resolve_binding("tickets").binding_id == "tickets"


def test_resolve_binding_rejects_unknown_index():
    server = _server_with_bindings("knowledge", "tickets")

    with pytest.raises(RedisVLMCPError) as excinfo:
        server.resolve_binding("missing")

    assert excinfo.value.code == MCPErrorCode.INVALID_REQUEST
    assert "missing" in str(excinfo.value)


def _register_tools_with(monkeypatch, bindings: dict) -> list[str]:
    """Run _register_tools against the given bindings, returning registered names."""
    registered: list[str] = []
    monkeypatch.setattr(
        "redisvl.mcp.server.register_list_indexes_tool",
        lambda server: registered.append("list-indexes"),
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.register_search_tool",
        lambda server, schema: registered.append("search-records"),
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.register_upsert_tool",
        lambda server: registered.append("upsert-records"),
    )

    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server._bindings = bindings
    server._tools_registered = False
    server.tool = object()
    server.mcp_settings = SimpleNamespace(read_only=False)

    server._register_tools()
    return registered


def test_register_tools_exposes_upsert_when_a_binding_is_writable(monkeypatch):
    registered = _register_tools_with(
        monkeypatch,
        {
            "knowledge": _binding_runtime("knowledge", effective_read_only=False),
            "tickets": _binding_runtime("tickets", effective_read_only=True),
        },
    )

    assert "upsert-records" in registered
    assert "list-indexes" in registered
    assert "search-records" in registered


def test_register_tools_hides_upsert_when_every_binding_is_read_only(monkeypatch):
    registered = _register_tools_with(
        monkeypatch,
        {
            "knowledge": _binding_runtime("knowledge", effective_read_only=True),
            "tickets": _binding_runtime("tickets", effective_read_only=True),
        },
    )

    assert "upsert-records" not in registered
    # Read paths stay available even when writes are globally disabled.
    assert "list-indexes" in registered
    assert "search-records" in registered
