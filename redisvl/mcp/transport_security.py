"""DNS-rebinding protection for the RedisVL MCP server's HTTP transports.

Provides a pure-ASGI middleware (:class:`HostOriginValidationMiddleware`) that
validates ``Host`` and ``Origin`` against allowlists before any tool call runs,
plus config resolution mirroring :mod:`redisvl.mcp.auth`. See
:class:`redisvl.mcp.config.MCPTransportSecurityConfig` for what this defends
against.

Imports of Starlette/FastMCP are deferred so this module stays importable
without the optional ``mcp`` extra. Applies only to HTTP transports; ``stdio``
has no network surface.
"""

from pathlib import Path
from typing import Any

import yaml

from redisvl.mcp.config import MCPTransportSecurityConfig, _substitute_env
from redisvl.mcp.settings import MCPSettings

# Hosts that only ever refer to the local machine. Kept in sync with
# ``redisvl.cli.mcp.MCP._LOOPBACK_HOSTS``.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
# Bind addresses that listen on every interface and have no single canonical
# Host. For these the default allowlist covers only loopback; operators must
# declare their public host via ``allowed_hosts``.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", ""})


def _to_host_header_form(host: str) -> str:
    """Return the form a host takes inside an HTTP ``Host`` header.

    IPv6 literals are bracketed (``::1`` -> ``[::1]``); everything else is
    returned unchanged. Already-bracketed values pass through.
    """
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _host_variants(host: str, port: int) -> set[str]:
    """Return the bare and ``:port`` Host-header forms for one host, lowercased."""
    form = _to_host_header_form(host).lower()
    return {form, f"{form}:{port}"}


def default_allowed_hosts(host: str, port: int) -> set[str]:
    """Compute the default Host allowlist from the bind address.

    Loopback and wildcard binds expand to the full loopback set (localhost,
    127.0.0.1, [::1]) in both bare and ``:port`` forms. A specific non-wildcard
    bind allows only that host (bare and ``:port``); operators add any extra
    public hosts through config.
    """
    allowed: set[str] = set()
    if host in _LOOPBACK_HOSTS or host in _WILDCARD_HOSTS:
        for loopback in ("localhost", "127.0.0.1", "::1"):
            allowed |= _host_variants(loopback, port)
    else:
        allowed |= _host_variants(host, port)
    return allowed


def _strip_port(host: str) -> str:
    """Return the Host-header value with any trailing ``:port`` removed."""
    if host.startswith("["):
        # Bracketed IPv6, e.g. "[::1]" or "[::1]:8000".
        closing = host.find("]")
        return host[: closing + 1] if closing != -1 else host
    if host.count(":") == 1:
        return host.rsplit(":", 1)[0]
    return host


class HostOriginValidationMiddleware:
    """Reject HTTP requests whose Host/Origin headers are not allowlisted.

    Non-``http`` scopes pass through untouched. Missing/unknown ``Host`` yields
    ``400``; a present but disallowed cross-site ``Origin`` yields ``403``. A
    request with no ``Origin`` (typical of non-browser MCP clients) always
    passes the origin check.
    """

    def __init__(
        self,
        app: Any,
        *,
        allowed_hosts: frozenset[str],
        allowed_origins: frozenset[str],
        allow_any_origin: bool = False,
    ) -> None:
        self.app = app
        # Allowlists are pre-lowercased so per-request comparison is a plain
        # membership test.
        self.allowed_hosts = frozenset(h.lower() for h in allowed_hosts)
        self.allowed_origins = frozenset(o.lower() for o in allowed_origins)
        self.allow_any_origin = allow_any_origin

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}

        host = headers.get(b"host", b"").decode("latin-1").strip().lower()
        if not self._host_allowed(host):
            reason = "missing Host header" if not host else "Host not allowed"
            # 400 is used (rather than the RFC-pure 421 Misdirected Request) for
            # broad client compatibility.
            await self._reject(send, 400, reason)
            return

        origin = headers.get(b"origin")
        if origin is not None:
            origin_value = origin.decode("latin-1").strip().lower()
            if origin_value and not self._origin_allowed(origin_value):
                await self._reject(send, 403, "Origin not allowed")
                return

        await self.app(scope, receive, send)

    def _host_allowed(self, host: str) -> bool:
        if not host:
            return False
        return host in self.allowed_hosts or _strip_port(host) in self.allowed_hosts

    def _origin_allowed(self, origin: str) -> bool:
        if self.allow_any_origin:
            return True
        return origin in self.allowed_origins

    @staticmethod
    async def _reject(send: Any, status: int, reason: str) -> None:
        body = reason.encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("latin-1")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def peek_yaml_transport_security(config_path: str | None) -> dict[str, Any] | None:
    """Read only the ``server.transport_security`` block, env-substituted.

    Returns ``None`` when the path is unset/missing or the block is absent. Like
    :func:`redisvl.mcp.auth.peek_yaml_auth`, this avoids the full config load so
    the guard can be wired before the server lifespan runs.
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
    block = server.get("transport_security") if isinstance(server, dict) else None
    if not isinstance(block, dict):
        return None
    return _substitute_env(block)


def resolve_transport_security_config(
    settings: MCPSettings, config_path: str | None = None
) -> MCPTransportSecurityConfig:
    """Resolve the effective transport-security config from env over YAML.

    Always returns a config; when nothing is set it defaults to enabled with an
    empty operator allowlist (the bind-derived default allowlist is added later,
    when the bind address is known).
    """
    env_overrides = settings.transport_security_overrides()
    yaml_block = peek_yaml_transport_security(config_path) or {}
    merged: dict[str, Any] = {**yaml_block, **env_overrides}
    return MCPTransportSecurityConfig.model_validate(merged)


def build_host_origin_middleware(
    config: MCPTransportSecurityConfig, host: str, port: int
) -> list:
    """Build the Starlette middleware list for HTTP transport security.

    Returns an empty list when the guard is disabled. Otherwise returns a single
    ``Middleware`` wrapping :class:`HostOriginValidationMiddleware`, with the
    effective host allowlist (bind-derived defaults unioned with configured
    hosts) and configured origins.
    """
    if not config.enabled:
        return []

    from starlette.middleware import Middleware

    # Operator-supplied hosts are matched verbatim (lowercased). Supply the
    # exact Host-header value expected, e.g. "example.com", "example.com:8000",
    # or a bracketed IPv6 literal "[2001:db8::1]".
    allowed_hosts = default_allowed_hosts(host, port) | {
        entry.strip().lower() for entry in config.allowed_hosts if entry.strip()
    }
    allowed_origins = frozenset(origin.lower() for origin in config.allowed_origins)

    return [
        Middleware(
            HostOriginValidationMiddleware,
            allowed_hosts=frozenset(allowed_hosts),
            allowed_origins=allowed_origins,
            allow_any_origin=config.allow_any_origin,
        )
    ]
