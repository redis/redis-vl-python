from typing import Any, cast

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPSettings(BaseSettings):
    """Environment-backed settings for bootstrapping the MCP server."""

    model_config = SettingsConfigDict(
        env_prefix="REDISVL_MCP_",
        extra="ignore",
    )

    config: str = Field(..., min_length=1)
    read_only: bool = False
    tool_search_description: str | None = None
    tool_upsert_description: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        config: str | None = None,
        read_only: bool | None = None,
        tool_search_description: str | None = None,
        tool_upsert_description: str | None = None,
    ) -> "MCPSettings":
        """Build settings from explicit overrides plus `REDISVL_MCP_*` env vars."""
        overrides: dict[str, object] = {}
        if config is not None:
            overrides["config"] = config
        if read_only is not None:
            overrides["read_only"] = read_only
        if tool_search_description is not None:
            overrides["tool_search_description"] = tool_search_description
        if tool_upsert_description is not None:
            overrides["tool_upsert_description"] = tool_upsert_description

        # `BaseSettings` fills any missing fields from the configured env prefix.
        return cls(**cast(dict[str, Any], overrides))
