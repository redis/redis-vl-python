"""Integration tests for MCP JWT auth over the streamable-http transport.

These start the server on a real port and connect a FastMCP client with and
without a bearer token, verifying that unauthenticated and mis-scoped requests
are rejected while a valid scoped token can list tools and search. Tokens are
minted with FastMCP's ``RSAKeyPair`` and validated against its static public
key, so no network JWKS endpoint is needed.
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
from fastmcp.client.auth import BearerAuth
from fastmcp.server.auth.providers.jwt import RSAKeyPair

from redisvl.index import AsyncSearchIndex
from redisvl.mcp.server import RedisVLMCPServer
from redisvl.mcp.settings import MCPSettings
from redisvl.schema import IndexSchema

ISSUER = "https://auth.redis.example/abc123/v2.0"
AUDIENCE = "api://redisvl-mcp"
READ_SCOPE = "kb.search.read"
WRITE_SCOPE = "kb.search.write"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _wait_for_port(host: str, port: int, timeout: float = 5.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        try:
            _, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(f"server on {host}:{port} not ready")
            await asyncio.sleep(0.05)


@pytest.fixture
async def auth_index(async_client, worker_id):
    """Create a small fulltext-searchable index for auth transport tests."""
    schema = IndexSchema.from_dict(
        {
            "index": {
                "name": f"mcp-auth-{worker_id}",
                "prefix": f"mcp-auth:{worker_id}",
                "storage_type": "hash",
            },
            "fields": [{"name": "content", "type": "text"}],
        }
    )
    index = AsyncSearchIndex(schema=schema, redis_client=async_client)
    await index.create(overwrite=True, drop=True)
    await index.load(
        [{"id": f"adoc:{worker_id}:1", "content": "transport test document science"}]
    )
    yield index
    await index.delete(drop=True)


@pytest.fixture
def auth_config_path(tmp_path: Path, redis_url: str):
    """Build a JWT-authenticated MCP config for a given index and public key."""

    def factory(redis_name: str, public_key: str) -> str:
        config = {
            "server": {
                "redis_url": redis_url,
                "auth": {
                    "type": "jwt",
                    "public_key": public_key,
                    "issuer": ISSUER,
                    "audience": AUDIENCE,
                    "required_scopes": [READ_SCOPE],
                    "read_scope": READ_SCOPE,
                    "write_scope": WRITE_SCOPE,
                },
            },
            "indexes": {
                "knowledge": {
                    "redis_name": redis_name,
                    "search": {"type": "fulltext"},
                    "runtime": {"text_field_name": "content"},
                }
            },
        }
        config_path = tmp_path / f"{redis_name}.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return str(config_path)

    return factory


async def test_http_transport_enforces_jwt_auth(auth_index, auth_config_path):
    key = RSAKeyPair.generate()
    server = RedisVLMCPServer(
        MCPSettings(
            config=auth_config_path(auth_index.schema.index.name, key.public_key)
        )
    )
    assert server._auth_enabled is True

    port = _find_free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    server_task = asyncio.create_task(
        server.run_async(transport="streamable-http", host="127.0.0.1", port=port)
    )
    try:
        await _wait_for_port("127.0.0.1", port)

        # No token is rejected.
        with pytest.raises(Exception):
            async with Client(url) as client:
                await client.list_tools()

        # Garbage token is rejected.
        with pytest.raises(Exception):
            async with Client(url, auth=BearerAuth("garbage")) as client:
                await client.list_tools()

        # Wrong audience is rejected.
        bad_aud = key.create_token(
            subject="nitin",
            issuer=ISSUER,
            audience="api://some-other-service",
            scopes=[READ_SCOPE],
        )
        with pytest.raises(Exception):
            async with Client(url, auth=BearerAuth(bad_aud)) as client:
                await client.list_tools()

        # Valid scoped token is accepted and can search.
        good = key.create_token(
            subject="nitin",
            issuer=ISSUER,
            audience=AUDIENCE,
            scopes=[READ_SCOPE],
        )
        async with Client(url, auth=BearerAuth(good)) as client:
            tool_names = [t.name for t in await client.list_tools()]
            assert "search-records" in tool_names
            result = await client.call_tool(
                "search-records", {"query": "science", "limit": 1}
            )
            assert result is not None
            assert len(result.content) > 0
    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
