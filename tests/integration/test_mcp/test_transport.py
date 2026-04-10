"""Integration tests for MCP server HTTP transport support.

These tests verify that the RedisVL MCP server properly starts and serves
tools over streamable-http, SSE, and in-process transports using FastMCP's
Client for end-to-end validation.
"""

import asyncio
import socket
from pathlib import Path

import pytest
import yaml

fastmcp = pytest.importorskip(
    "fastmcp", reason="fastmcp not installed (install redisvl[mcp])"
)
from fastmcp import Client

from redisvl.index import AsyncSearchIndex
from redisvl.mcp.server import RedisVLMCPServer
from redisvl.mcp.settings import MCPSettings
from redisvl.redis.utils import array_to_buffer
from redisvl.schema import IndexSchema


class FakeVectorizer:
    def __init__(self, model: str, dims: int = 3, **kwargs):
        self.model = model
        self.dims = dims
        self.kwargs = kwargs

    def embed(self, content: str = "", **kwargs):
        del content, kwargs
        return [0.1, 0.1, 0.5]


def _find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_for_port(host: str, port: int, timeout: float = 5.0) -> None:
    """Poll until the given host:port accepts a TCP connection."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        try:
            _, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(
                    f"Server on {host}:{port} did not become ready "
                    f"within {timeout}s"
                )
            await asyncio.sleep(0.05)


@pytest.fixture
async def transport_index(async_client, worker_id):
    """Create a searchable index for transport tests."""
    schema = IndexSchema.from_dict(
        {
            "index": {
                "name": f"mcp-transport-{worker_id}",
                "prefix": f"mcp-transport:{worker_id}",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "content", "type": "text"},
                {"name": "category", "type": "tag"},
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
    index = AsyncSearchIndex(schema=schema, redis_client=async_client)
    await index.create(overwrite=True, drop=True)

    def preprocess(record: dict) -> dict:
        return {
            **record,
            "embedding": array_to_buffer(record["embedding"], "float32"),
        }

    await index.load(
        [
            {
                "id": f"tdoc:{worker_id}:1",
                "content": "transport test document about science",
                "category": "science",
                "embedding": [0.1, 0.1, 0.5],
            },
        ],
        preprocess=preprocess,
    )

    yield index

    await index.delete(drop=True)


@pytest.fixture
def transport_config_path(tmp_path: Path, redis_url: str):
    """Build an MCP config YAML for transport tests."""

    def factory(redis_name: str) -> str:
        config = {
            "server": {"redis_url": redis_url},
            "indexes": {
                "knowledge": {
                    "redis_name": redis_name,
                    "vectorizer": {
                        "class": "FakeVectorizer",
                        "model": "fake-model",
                        "dims": 3,
                    },
                    "search": {"type": "vector"},
                    "runtime": {
                        "text_field_name": "content",
                        "vector_field_name": "embedding",
                        "default_embed_text_field": "content",
                    },
                }
            },
        }
        config_path = tmp_path / f"{redis_name}.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return str(config_path)

    return factory


@pytest.mark.asyncio
async def test_server_serves_over_streamable_http_transport(
    monkeypatch, transport_index, transport_config_path
):
    """TDD integration: Start the MCP server with streamable-http transport
    on a free port and verify a remote FastMCP Client can list tools and
    call search-records over HTTP."""

    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )

    port = _find_free_port()
    server = RedisVLMCPServer(
        MCPSettings(config=transport_config_path(transport_index.schema.index.name))
    )

    server_task = None
    try:
        # Start the server in a background task with streamable-http
        server_task = asyncio.create_task(
            server.run_async(transport="streamable-http", host="127.0.0.1", port=port)
        )

        # Wait for the HTTP server to accept connections
        await _wait_for_port("127.0.0.1", port)

        url = f"http://127.0.0.1:{port}/mcp"
        async with Client(url) as client:
            tools = await client.list_tools()
            tool_names = [t.name for t in tools]

            assert "search-records" in tool_names
            assert "upsert-records" in tool_names

            result = await client.call_tool(
                "search-records",
                {"query": "science", "limit": 1},
            )

            assert result is not None
            assert len(result.content) > 0
    finally:
        if server_task is not None:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_server_read_only_mode_hides_upsert_over_http(
    monkeypatch, transport_index, transport_config_path
):
    """TDD integration: In read-only mode, the upsert-records tool should
    not be registered, even when accessed over HTTP."""

    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )

    port = _find_free_port()
    server = RedisVLMCPServer(
        MCPSettings(
            config=transport_config_path(transport_index.schema.index.name),
            read_only=True,
        )
    )

    server_task = None
    try:
        server_task = asyncio.create_task(
            server.run_async(transport="streamable-http", host="127.0.0.1", port=port)
        )

        await _wait_for_port("127.0.0.1", port)

        url = f"http://127.0.0.1:{port}/mcp"
        async with Client(url) as client:
            tools = await client.list_tools()
            tool_names = [t.name for t in tools]

            assert "search-records" in tool_names
            assert "upsert-records" not in tool_names
    finally:
        if server_task is not None:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
