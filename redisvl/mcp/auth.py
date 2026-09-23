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
    env_auth = settings.auth_overrides()

    # An explicit env type=none disables auth, overriding any YAML auth block.
    if env_auth.get("type") == "none":
        return None

    yaml_auth = peek_yaml_auth(config_path) or {}
    merged: dict[str, Any] = {**yaml_auth, **env_auth}
    if not merged:
        return None

    config = MCPAuthConfig.model_validate(merged)
    if config.type == "none":
        return None
    return config


def missing_required_claims(claims: Any, required_claims: Any) -> list:
    """Return the configured claims absent from a token's claims mapping."""
    claims = claims or {}
    return [claim for claim in (required_claims or ()) if claim not in claims]


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

        required_claims = tuple(auth_config.required_claims or ())

        class _StrictClaimsJWTVerifier(
            JWTVerifier
        ):  # pylint: disable=too-few-public-methods
            """JWTVerifier that also requires specific claims to be present.

            FastMCP's verifier only rejects an ``exp`` that is present and past,
            so a token without ``exp`` would never expire. Requiring ``exp``
            (and ``iat``) closes that gap.
            """

            async def load_access_token(self, token: str):
                access = await super().load_access_token(token)
                if access is None:
                    return None
                if missing_required_claims(access.claims, required_claims):
                    return None
                return access

        return _StrictClaimsJWTVerifier(
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

    Prefer :func:`ensure_read_scope` / :func:`ensure_write_scope` at a call
    site; they resolve the scope name from the same server this reads.
    """
    if not getattr(server, "_auth_enabled", False):
        return

    auth_config = getattr(server, "auth_config", None)
    if auth_config is None:
        # Auth is wired, so its config has to be reachable. Returning here would
        # silently stop gating every tool the moment the attribute is renamed --
        # a fail-open that no test would catch -- so fail closed instead.
        raise RedisVLMCPError(
            "MCP auth is enabled but the server's auth configuration is "
            "unreachable; refusing to run an ungated tool",
            code=MCPErrorCode.INTERNAL_ERROR,
            retryable=False,
        )

    if required_scope is None:
        return

    from fastmcp.server.dependencies import get_access_token

    access_token = get_access_token()
    if access_token is None:
        # No authenticated request context (for example the local stdio
        # transport, which FastMCP never authenticates). Authenticated HTTP
        # transports reject tokenless requests before the tool runs, so a
        # missing token here means the scope gate does not apply.
        return

    claim = getattr(auth_config, "authorization_claim", "scp")
    if not token_has_scope(access_token, required_scope, claim):
        raise RedisVLMCPError(
            f"Token is missing the required scope '{required_scope}'",
            code=MCPErrorCode.FORBIDDEN,
            retryable=False,
        )


def _configured_scope(server: Any, attribute: str) -> str | None:
    """Read one configured scope name off the server's auth config.

    Deliberately unguarded on the attribute itself: a renamed field on
    ``MCPAuthConfig`` raises here rather than resolving to ``None`` and quietly
    turning the scope gate into a no-op.
    """
    auth_config = getattr(server, "auth_config", None)
    if auth_config is None:
        return None
    return getattr(auth_config, attribute)


def ensure_read_scope(server: Any) -> None:
    """Enforce the configured read scope for the current request."""
    ensure_tool_scope(server, _configured_scope(server, "read_scope"))


def ensure_write_scope(server: Any) -> None:
    """Enforce the configured write scope for the current request."""
    ensure_tool_scope(server, _configured_scope(server, "write_scope"))
