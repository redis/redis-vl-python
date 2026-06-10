"""Unit tests for resolving auth config from env (`MCPSettings`) and YAML.

Env vars take precedence over the YAML `server.auth` block.
"""

from pathlib import Path

import yaml

from redisvl.mcp.auth import resolve_auth_config
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


def test_resolves_none_when_unset(tmp_path, monkeypatch):
    for var in ("REDISVL_MCP_AUTH_TYPE", "REDISVL_MCP_AUTH_JWKS_URI"):
        monkeypatch.delenv(var, raising=False)
    path = _write_config(tmp_path, auth_block=None)
    settings = MCPSettings.from_env(config=path)
    assert resolve_auth_config(settings, path) is None


def test_resolves_jwt_from_yaml(tmp_path, monkeypatch):
    for var in ("REDISVL_MCP_AUTH_TYPE", "REDISVL_MCP_AUTH_AUDIENCE"):
        monkeypatch.delenv(var, raising=False)
    path = _write_config(
        tmp_path,
        auth_block={
            "type": "jwt",
            "jwks_uri": "https://auth.redis.example/keys",
            "issuer": "https://auth.redis.example/abc123/v2.0",
            "audience": "api://redisvl-mcp",
            "read_scope": "kb.search.read",
        },
    )
    settings = MCPSettings.from_env(config=path)
    cfg = resolve_auth_config(settings, path)
    assert cfg is not None
    assert cfg.type == "jwt"
    assert cfg.audience == "api://redisvl-mcp"
    assert cfg.read_scope == "kb.search.read"


def test_env_overrides_yaml(tmp_path, monkeypatch):
    path = _write_config(
        tmp_path,
        auth_block={
            "type": "jwt",
            "jwks_uri": "https://auth.redis.example/keys",
            "issuer": "https://auth.redis.example/abc123/v2.0",
            "audience": "api://from-yaml",
        },
    )
    monkeypatch.setenv("REDISVL_MCP_AUTH_TYPE", "jwt")
    monkeypatch.setenv("REDISVL_MCP_AUTH_JWKS_URI", "https://auth.redis.example/keys")
    monkeypatch.setenv("REDISVL_MCP_AUTH_AUDIENCE", "api://from-env")
    settings = MCPSettings.from_env(config=path)
    cfg = resolve_auth_config(settings, path)
    assert cfg is not None
    assert cfg.audience == "api://from-env"


def test_env_type_none_disables_yaml_auth(tmp_path, monkeypatch):
    # Explicit env type=none must turn auth off even when YAML defines a jwt block.
    path = _write_config(
        tmp_path,
        auth_block={
            "type": "jwt",
            "jwks_uri": "https://auth.redis.example/keys",
            "issuer": "https://auth.redis.example/abc123/v2.0",
            "audience": "api://redisvl-mcp",
        },
    )
    monkeypatch.setenv("REDISVL_MCP_AUTH_TYPE", "none")
    settings = MCPSettings.from_env(config=path)
    assert resolve_auth_config(settings, path) is None
