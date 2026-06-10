"""Unit tests for wiring auth into the MCP server constructor."""

from pathlib import Path

import pytest
import yaml

fastmcp = pytest.importorskip(
    "fastmcp", reason="fastmcp not installed (install redisvl[mcp])"
)
from fastmcp.server.auth.providers.jwt import JWTVerifier, RSAKeyPair

from redisvl.mcp.server import RedisVLMCPServer
from redisvl.mcp.settings import MCPSettings


def _write_config(tmp_path: Path, auth_block: dict | None) -> str:
    config = {
        "server": {"redis_url": "redis://localhost:6379"},
        "indexes": {
            "knowledge": {
                "redis_name": "docs-index",
                "search": {"type": "fulltext"},
                "runtime": {"text_field_name": "content"},
            }
        },
    }
    if auth_block is not None:
        config["server"]["auth"] = auth_block
    path = tmp_path / "mcp.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def test_server_without_auth_is_unauthenticated(tmp_path, monkeypatch):
    for var in ("REDISVL_MCP_AUTH_TYPE", "REDISVL_MCP_AUTH_AUDIENCE"):
        monkeypatch.delenv(var, raising=False)
    path = _write_config(tmp_path, auth_block=None)
    server = RedisVLMCPServer(MCPSettings.from_env(config=path))
    assert server._auth_enabled is False
    assert server.auth is None


def test_server_with_jwt_auth_attaches_verifier(tmp_path, monkeypatch):
    for var in ("REDISVL_MCP_AUTH_TYPE", "REDISVL_MCP_AUTH_AUDIENCE"):
        monkeypatch.delenv(var, raising=False)
    key = RSAKeyPair.generate()
    path = _write_config(
        tmp_path,
        auth_block={
            "type": "jwt",
            "public_key": key.public_key,
            "issuer": "https://auth.redis.example/abc123/v2.0",
            "audience": "api://redisvl-mcp",
            "required_scopes": ["kb.read"],
            "read_scope": "kb.search.read",
            "write_scope": "kb.search.write",
        },
    )
    server = RedisVLMCPServer(MCPSettings.from_env(config=path))
    assert server._auth_enabled is True
    assert isinstance(server.auth, JWTVerifier)
    assert server.auth_config is not None
    assert server.auth_config.read_scope == "kb.search.read"


def test_stale_auth_after_construction_fails_closed(tmp_path, monkeypatch):
    for var in ("REDISVL_MCP_AUTH_TYPE", "REDISVL_MCP_AUTH_AUDIENCE"):
        monkeypatch.delenv(var, raising=False)
    # Construct with no auth, so the provider is not wired.
    path = _write_config(tmp_path, auth_block=None)
    server = RedisVLMCPServer(MCPSettings.from_env(config=path))
    assert server._auth_enabled is False

    # The config now gains a JWT auth block (as if it became readable only after
    # construction). Startup must refuse rather than serve unauthenticated.
    key = RSAKeyPair.generate()
    Path(path).write_text(
        yaml.safe_dump(
            {
                "server": {
                    "redis_url": "redis://localhost:6379",
                    "auth": {
                        "type": "jwt",
                        "public_key": key.public_key,
                        "issuer": "https://auth.redis.example/abc123/v2.0",
                        "audience": "api://redisvl-mcp",
                    },
                },
                "indexes": {
                    "knowledge": {
                        "redis_name": "docs-index",
                        "search": {"type": "fulltext"},
                        "runtime": {"text_field_name": "content"},
                    }
                },
            }
        )
    )
    with pytest.raises(RuntimeError, match="stale"):
        server._verify_auth_not_stale()


def test_consistent_auth_does_not_fail(tmp_path, monkeypatch):
    for var in ("REDISVL_MCP_AUTH_TYPE", "REDISVL_MCP_AUTH_AUDIENCE"):
        monkeypatch.delenv(var, raising=False)
    path = _write_config(tmp_path, auth_block=None)
    server = RedisVLMCPServer(MCPSettings.from_env(config=path))
    # Unchanged config: construction and startup agree, so no error.
    server._verify_auth_not_stale()
