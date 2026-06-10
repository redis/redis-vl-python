"""Authentication wiring for the RedisVL MCP server.

Resolves an :class:`~redisvl.mcp.config.MCPAuthConfig` from environment
variables (``REDISVL_MCP_AUTH_*``) and/or the YAML ``server.auth`` block, and
builds a FastMCP auth provider from it. Env vars take precedence over YAML.

Auth applies only to HTTP transports; ``stdio`` is never authenticated. FastMCP
imports are deferred so this module stays importable without the ``mcp`` extra.
"""

from pathlib import Path
from typing import Any

import yaml

from redisvl.mcp.config import MCPAuthConfig, _substitute_env
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.settings import MCPSettings


def peek_yaml_auth(config_path: str | None) -> dict[str, Any] | None:
    """Read only the ``server.auth`` block from the YAML config, env-substituted.

    Returns ``None`` when the path is unset/missing or no auth block is present.
    This intentionally avoids the full runtime config load so auth can be wired
    at construction time, before the server lifespan runs.
    """
    if not config_path:
        return None
    path = Path(config_path).expanduser()
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except yaml.YAMLError:
        return None

    server = raw.get("server") if isinstance(raw, dict) else None
    auth = server.get("auth") if isinstance(server, dict) else None
    if not isinstance(auth, dict):
        return None
    return _substitute_env(auth)


def resolve_auth_config(
    settings: MCPSettings, config_path: str | None = None
) -> MCPAuthConfig | None:
    """Resolve the effective auth config from env (preferred) over YAML.

    Returns ``None`` when no auth is configured or the resolved type is
    ``none``.
    """
    yaml_auth = peek_yaml_auth(config_path) or {}
    env_auth = settings.auth_overrides()

    merged: dict[str, Any] = {**yaml_auth, **env_auth}
    if not merged:
        return None

    config = MCPAuthConfig.model_validate(merged)
    if config.type == "none":
        return None
    return config


def build_auth_provider(auth_config: MCPAuthConfig | None) -> Any | None:
    """Build a FastMCP auth provider from an `MCPAuthConfig`.

    Returns ``None`` for ``None`` / ``type == "none"``. For ``jwt`` returns a
    configured ``JWTVerifier``. The provider import is deferred so importing this
    module never requires the optional ``mcp`` extra.
    """
    if auth_config is None or auth_config.type == "none":
        return None

    if auth_config.type == "jwt":
        try:
            from fastmcp.server.auth.providers.jwt import JWTVerifier
        except ImportError as exc:  # pragma: no cover - exercised without extra
            raise RuntimeError(
                "JWT authentication requires the optional MCP dependencies. "
                "Install them with `pip install redisvl[mcp]`."
            ) from exc

        return JWTVerifier(
            public_key=auth_config.public_key,
            jwks_uri=auth_config.jwks_uri,
            issuer=auth_config.issuer,
            audience=auth_config.audience,
            algorithm=auth_config.algorithm,
            required_scopes=auth_config.required_scopes or None,
            base_url=auth_config.base_url,
        )

    raise ValueError(f"Unsupported auth type: {auth_config.type}")


def authorization_values(access_token: Any, authorization_claim: str = "scp") -> list:
    """Return the authorization values a token carries for the given claim.

    Standard OAuth scopes (``scp``/``scope``) are read from the verifier-parsed
    ``access_token.scopes``. Any other claim (for example ``roles``) is read
    from ``access_token.claims`` and normalized to a list, accepting either a
    list or a space-delimited string.
    """
    if authorization_claim in ("scp", "scope"):
        return list(getattr(access_token, "scopes", None) or [])

    claims = getattr(access_token, "claims", None) or {}
    raw = claims.get(authorization_claim)
    if isinstance(raw, str):
        return raw.split()
    if isinstance(raw, (list, tuple)):
        return [str(value) for value in raw]
    return []


def token_has_scope(
    access_token: Any, scope: str | None, authorization_claim: str = "scp"
) -> bool:
    """Return whether an access token carries the given scope.

    A ``None`` scope means no gate is configured, so access is allowed.
    """
    if scope is None:
        return True
    return scope in authorization_values(access_token, authorization_claim)


def ensure_tool_scope(server: Any, required_scope: str | None) -> None:
    """Raise if the current request's token lacks the required tool scope.

    No-ops when auth is disabled or no scope is configured. Otherwise reads the
    current access token and checks the configured authorization claim, raising
    a ``forbidden`` MCP error when the scope is absent.
    """
    auth_config = getattr(server, "auth_config", None)
    if (
        not getattr(server, "_auth_enabled", False)
        or auth_config is None
        or required_scope is None
    ):
        return

    from fastmcp.server.dependencies import get_access_token

    access_token = get_access_token()
    claim = getattr(auth_config, "authorization_claim", "scp")
    if not token_has_scope(access_token, required_scope, claim):
        raise RedisVLMCPError(
            f"Token is missing the required scope '{required_scope}'",
            code=MCPErrorCode.FORBIDDEN,
            retryable=False,
        )
