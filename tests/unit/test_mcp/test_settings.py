from pydantic_settings import BaseSettings

from redisvl.mcp.settings import MCPSettings


def test_settings_reads_env_defaults(monkeypatch):
    monkeypatch.setenv("REDISVL_MCP_CONFIG", "/tmp/mcp.yaml")
    monkeypatch.setenv("REDISVL_MCP_READ_ONLY", "true")
    monkeypatch.setenv("REDISVL_MCP_TOOL_SEARCH_DESCRIPTION", "search docs")
    monkeypatch.setenv("REDISVL_MCP_TOOL_UPSERT_DESCRIPTION", "upsert docs")

    settings = MCPSettings()

    assert settings.config == "/tmp/mcp.yaml"
    assert settings.read_only is True
    assert settings.tool_search_description == "search docs"
    assert settings.tool_upsert_description == "upsert docs"


def test_settings_explicit_values_override_env(monkeypatch):
    monkeypatch.setenv("REDISVL_MCP_CONFIG", "/tmp/from-env.yaml")
    monkeypatch.setenv("REDISVL_MCP_READ_ONLY", "true")

    settings = MCPSettings.from_env(
        config="/tmp/from-arg.yaml",
        read_only=False,
    )

    assert settings.config == "/tmp/from-arg.yaml"
    assert settings.read_only is False


def test_settings_defaults_optional_descriptions(monkeypatch):
    monkeypatch.delenv("REDISVL_MCP_TOOL_SEARCH_DESCRIPTION", raising=False)
    monkeypatch.delenv("REDISVL_MCP_TOOL_UPSERT_DESCRIPTION", raising=False)
    monkeypatch.setenv("REDISVL_MCP_CONFIG", "/tmp/mcp.yaml")

    settings = MCPSettings.from_env()

    assert settings.tool_search_description is None
    assert settings.tool_upsert_description is None


def test_settings_uses_pydantic_base_settings():
    assert issubclass(MCPSettings, BaseSettings)
