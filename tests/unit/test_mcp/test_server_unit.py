from types import SimpleNamespace

import pytest

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
async def test_supports_native_hybrid_search_caches_runtime_probe(monkeypatch):
    client = FakeClient()
    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server._index = FakeIndex(client)
    server._supports_native_hybrid_search = None

    monkeypatch.setattr("redisvl.mcp.server.redis_py_version", "7.1.0")

    assert await server.supports_native_hybrid_search() is True
    assert await server.supports_native_hybrid_search() is True
    assert client.info_calls == 1
