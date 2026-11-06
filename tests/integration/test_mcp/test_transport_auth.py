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
import time

from authlib.jose import jwt as jose_jwt
from fastmcp import Client
from fastmcp.client.auth import BearerAuth
from fastmcp.exceptions import ToolError
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


async def _assert_unauthorized(url, auth=None) -> None:
    """Assert a request is rejected at the transport with HTTP 401."""
    with pytest.raises(Exception) as exc_info:
        async with Client(url, auth=auth) as client:
            await client.list_tools()
    assert "401" in str(exc_info.value), f"expected 401, got {exc_info.value!r}"


async def _wait_for_port(host: str, port: int, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        try:
            _, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            if asyncio.get_running_loop().time() >= deadline:
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
                    "search": {"type": "fulltext", "params": {"stopwords": None}},
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
        await _assert_unauthorized(url)

        # Garbage token is rejected.
        await _assert_unauthorized(url, BearerAuth("garbage"))

        # Wrong audience is rejected.
        bad_aud = key.create_token(
            subject="nitin",
            issuer=ISSUER,
            audience="api://some-other-service",
            scopes=[READ_SCOPE],
        )
        await _assert_unauthorized(url, BearerAuth(bad_aud))

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


async def test_http_transport_gates_write_by_scope(auth_index, auth_config_path):
    key = RSAKeyPair.generate()
    server = RedisVLMCPServer(
        MCPSettings(
            config=auth_config_path(auth_index.schema.index.name, key.public_key)
        )
    )
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    server_task = asyncio.create_task(
        server.run_async(transport="streamable-http", host="127.0.0.1", port=port)
    )
    try:
        await _wait_for_port("127.0.0.1", port)

        # A read-only token can search but cannot upsert.
        read_token = key.create_token(
            subject="nitin", issuer=ISSUER, audience=AUDIENCE, scopes=[READ_SCOPE]
        )
        async with Client(url, auth=BearerAuth(read_token)) as client:
            await client.call_tool("search-records", {"query": "science", "limit": 1})
            with pytest.raises(ToolError):
                await client.call_tool(
                    "upsert-records", {"records": [{"content": "blocked"}]}
                )

        # A read+write token can upsert.
        rw_token = key.create_token(
            subject="nitin",
            issuer=ISSUER,
            audience=AUDIENCE,
            scopes=[READ_SCOPE, WRITE_SCOPE],
        )
        async with Client(url, auth=BearerAuth(rw_token)) as client:
            result = await client.call_tool(
                "upsert-records", {"records": [{"content": "written by rw token"}]}
            )
            assert result is not None
    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


async def test_http_transport_gates_by_roles_claim(
    auth_index, tmp_path: Path, redis_url: str
):
    """Authorization carried in a ``roles`` claim (as Azure AD / Entra does)."""
    key = RSAKeyPair.generate()
    config = {
        "server": {
            "redis_url": redis_url,
            "auth": {
                "type": "jwt",
                "public_key": key.public_key,
                "issuer": ISSUER,
                "audience": AUDIENCE,
                "required_scopes": ["kb.read"],  # connect gate on scp
                "read_scope": READ_SCOPE,
                "write_scope": WRITE_SCOPE,
                "authorization_claim": "roles",
            },
        },
        "indexes": {
            "knowledge": {
                "redis_name": auth_index.schema.index.name,
                "search": {"type": "fulltext", "params": {"stopwords": None}},
                "runtime": {"text_field_name": "content"},
            }
        },
    }
    config_path = tmp_path / "roles.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    server = RedisVLMCPServer(MCPSettings(config=str(config_path)))
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    server_task = asyncio.create_task(
        server.run_async(transport="streamable-http", host="127.0.0.1", port=port)
    )
    try:
        await _wait_for_port("127.0.0.1", port)

        # scp grants connect; roles grants only read, so search works, upsert blocked.
        token = key.create_token(
            subject="nitin",
            issuer=ISSUER,
            audience=AUDIENCE,
            scopes=["kb.read"],
            additional_claims={"roles": [READ_SCOPE], "tid": "tenant-guid"},
        )
        async with Client(url, auth=BearerAuth(token)) as client:
            await client.call_tool("search-records", {"query": "science", "limit": 1})
            with pytest.raises(ToolError):
                await client.call_tool(
                    "upsert-records", {"records": [{"content": "blocked"}]}
                )
    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


async def test_http_transport_rejects_token_without_exp(auth_index, auth_config_path):
    """A non-expiring (no exp) token must be rejected over the live HTTP path."""
    key = RSAKeyPair.generate()
    server = RedisVLMCPServer(
        MCPSettings(
            config=auth_config_path(auth_index.schema.index.name, key.public_key)
        )
    )
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    server_task = asyncio.create_task(
        server.run_async(transport="streamable-http", host="127.0.0.1", port=port)
    )
    try:
        await _wait_for_port("127.0.0.1", port)

        # Properly signed and scoped, but no exp claim. authlib signs it; the
        # server must still reject it because exp is required.
        no_exp = jose_jwt.encode(
            {"alg": "RS256"},
            {
                "iss": ISSUER,
                "aud": AUDIENCE,
                "sub": "nitin",
                "scope": READ_SCOPE,
                "iat": int(time.time()),
            },
            key.private_key.get_secret_value(),
        )
        if isinstance(no_exp, bytes):
            no_exp = no_exp.decode()

        await _assert_unauthorized(url, BearerAuth(no_exp))
    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
