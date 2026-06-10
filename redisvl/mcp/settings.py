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

    # Authentication overrides (``REDISVL_MCP_AUTH_*``). When any are set they
    # take precedence over the YAML ``server.auth`` block. ``auth_type`` of
    # ``None`` means "fall back to YAML / no auth".
    auth_type: str | None = None
    auth_jwks_uri: str | None = None
    auth_public_key: str | None = None
    auth_issuer: str | None = None
    auth_audience: str | None = None
    auth_algorithm: str | None = None
    auth_required_scopes: str | None = None
    auth_read_scope: str | None = None
    auth_write_scope: str | None = None
    auth_authorization_claim: str | None = None
    auth_base_url: str | None = None

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

    def auth_overrides(self) -> dict[str, Any]:
        """Return the non-``None`` ``auth_*`` fields as an `MCPAuthConfig` mapping.

        Comma-separated `auth_required_scopes` is split into a list. Returns an
        empty dict when no auth env vars are set.
        """
        mapping = {
            "type": self.auth_type,
            "jwks_uri": self.auth_jwks_uri,
            "public_key": self.auth_public_key,
            "issuer": self.auth_issuer,
            "audience": self.auth_audience,
            "algorithm": self.auth_algorithm,
            "read_scope": self.auth_read_scope,
            "write_scope": self.auth_write_scope,
            "authorization_claim": self.auth_authorization_claim,
            "base_url": self.auth_base_url,
        }
        overrides: dict[str, Any] = {
            key: value for key, value in mapping.items() if value is not None
        }
        if self.auth_required_scopes is not None:
            overrides["required_scopes"] = [
                scope.strip()
                for scope in self.auth_required_scopes.split(",")
                if scope.strip()
            ]
        return overrides
