"""Unit tests for resolving transport-security config from env and YAML.

Env vars (`REDISVL_MCP_*`) take precedence over the YAML
`server.transport_security` block. Defaults to enabled when nothing is set.
"""

from pathlib import Path

import yaml

from redisvl.mcp.settings import MCPSettings
from redisvl.mcp.transport_security import resolve_transport_security_config

_TS_ENV_VARS = (
    "REDISVL_MCP_TRANSPORT_SECURITY_ENABLED",
    "REDISVL_MCP_ALLOWED_HOSTS",
    "REDISVL_MCP_ALLOWED_ORIGINS",
    "REDISVL_MCP_ALLOW_ANY_ORIGIN",
)


def _write_config(tmp_path: Path, block: dict | None) -> str:
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
    if block is not None:
        config["server"]["transport_security"] = block
    path = tmp_path / "mcp.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def _clear_env(monkeypatch):
    for var in _TS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_defaults_enabled_when_unset(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    path = _write_config(tmp_path, block=None)
    settings = MCPSettings.from_env(config=path)
    cfg = resolve_transport_security_config(settings, path)
    assert cfg.enabled is True
    assert cfg.allowed_hosts == []
    assert cfg.allowed_origins == []
    assert cfg.allow_any_origin is False


def test_resolves_from_yaml(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    path = _write_config(
        tmp_path,
        block={
            "allowed_hosts": ["proxy.internal", "proxy.internal:8000"],
            "allowed_origins": ["https://app.example"],
        },
    )
    settings = MCPSettings.from_env(config=path)
    cfg = resolve_transport_security_config(settings, path)
    assert cfg.enabled is True
    assert cfg.allowed_hosts == ["proxy.internal", "proxy.internal:8000"]
    assert cfg.allowed_origins == ["https://app.example"]


def test_env_overrides_yaml(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    path = _write_config(tmp_path, block={"allowed_hosts": ["from-yaml.example"]})
    monkeypatch.setenv("REDISVL_MCP_ALLOWED_HOSTS", "from-env.example, other.example")
    settings = MCPSettings.from_env(config=path)
    cfg = resolve_transport_security_config(settings, path)
    assert cfg.allowed_hosts == ["from-env.example", "other.example"]


def test_env_disables_guard(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    path = _write_config(tmp_path, block={"enabled": True})
    monkeypatch.setenv("REDISVL_MCP_TRANSPORT_SECURITY_ENABLED", "false")
    settings = MCPSettings.from_env(config=path)
    cfg = resolve_transport_security_config(settings, path)
    assert cfg.enabled is False


def test_env_allow_any_origin(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    path = _write_config(tmp_path, block=None)
    monkeypatch.setenv("REDISVL_MCP_ALLOW_ANY_ORIGIN", "true")
    settings = MCPSettings.from_env(config=path)
    cfg = resolve_transport_security_config(settings, path)
    assert cfg.allow_any_origin is True


# --- MCPSettings.transport_security_overrides() ------------------------------


def test_overrides_empty_when_unset(monkeypatch):
    _clear_env(monkeypatch)
    settings = MCPSettings.from_env(config="/tmp/mcp.yaml")
    assert settings.transport_security_overrides() == {}


def test_overrides_split_comma_separated_lists(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("REDISVL_MCP_ALLOWED_HOSTS", "a.example, b.example ,")
    monkeypatch.setenv("REDISVL_MCP_ALLOWED_ORIGINS", "https://a.example")
    monkeypatch.setenv("REDISVL_MCP_TRANSPORT_SECURITY_ENABLED", "false")
    settings = MCPSettings.from_env(config="/tmp/mcp.yaml")
    overrides = settings.transport_security_overrides()
    assert overrides["allowed_hosts"] == ["a.example", "b.example"]
    assert overrides["allowed_origins"] == ["https://a.example"]
    assert overrides["enabled"] is False
