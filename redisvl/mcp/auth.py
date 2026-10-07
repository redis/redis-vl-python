"""Authentication wiring for the RedisVL MCP server.

Resolves an :class:`~redisvl.mcp.config.MCPAuthConfig` from environment
variables (``REDISVL_MCP_AUTH_*``) and/or the YAML ``server.auth`` block, and
builds a FastMCP auth provider from it. Env vars take precedence over YAML.

Auth applies only to HTTP transports; ``stdio`` is never authenticated. FastMCP
imports are deferred so this module stays importable without the ``mcp`` extra.

Reading values *out* of a verified token lives here too, next to the wiring that
decides which tokens are verified at all. Two readers coexist deliberately:
``authorization_values`` normalizes widely because an absent scope only denies,
and ``resolve_injected_claim`` refuses everything but a single non-empty string
because a widened value grants. Split across modules, someone unifies them and
reintroduces the cross-tenant union.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import yaml

from redisvl.mcp.config import MCPAuthConfig, _substitute_env
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.settings import MCPSettings
from redisvl.query.filter import FilterExpression, Tag, is_match_all_filter
from redisvl.schema import IndexSchema

logger = logging.getLogger(__name__)


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


class InjectSpec(Protocol):
    """One locked injection entry: an index field and the claim that fills it.

    A protocol rather than an import of the configuration model, so the
    security core has no dependency on the shape of the YAML that reaches it.
    A v1.1 code tool assembles its own specs and gets the same guarantees.
    """

    @property
    def field(self) -> str: ...

    @property
    def claim(self) -> str: ...


def _injection_refused(claim: str, tool_name: str, reason: str) -> RedisVLMCPError:
    """Build the single refusal raised by every injection failure.

    The claim name is deliberately in the client-facing message. A JWT is
    signed, not encrypted, so a client holding a valid token can already read
    its own claim names, and an unauthenticated client never reaches this code
    because the HTTP layer rejects it first. Naming the claim and the tool is
    what turns a misspelled claim name from an opaque permanent failure into a
    one-line diagnosis.
    """
    message = (
        f"Tool '{tool_name}' requires the '{claim}' claim to scope every query, "
        f"and {reason}; refusing to run an unscoped query"
    )
    logger.warning("%s", message)
    return RedisVLMCPError(
        message,
        code=MCPErrorCode.FORBIDDEN,
        retryable=False,
    )


def resolve_injected_claim(claim: str, *, tool_name: str) -> str:
    """Resolve one token claim into a value safe to inject as a tag equality.

    Deliberately the mirror image of :func:`authorization_values`, which is
    directly above for that reason. That reader normalizes widely -- it
    space-splits a string and coerces list members -- because an absent scope
    only ever denies. Here a widened value *grants*, so every shape but a
    single non-empty string is refused.

    Unlike :func:`ensure_tool_scope`, a tokenless request raises rather than
    returning. The scope gate reads ``None`` as "stdio, so no gate applies";
    reached here, the same exit would attach no tenant clause and run the query
    against every tenant.
    """
    from fastmcp.server.dependencies import get_access_token

    access_token = get_access_token()
    if access_token is None:
        raise _injection_refused(
            claim, tool_name, "this request carries no verified token"
        )

    claims = getattr(access_token, "claims", None) or {}
    if claim not in claims:
        raise _injection_refused(claim, tool_name, "the token does not carry it")

    value = claims[claim]
    if not isinstance(value, str):
        # The load-bearing check. `_formatted_tag_value` escapes each element
        # *before* joining them with `|`, so a list claim renders as a genuine
        # cross-tenant union -- `@tenant_id:{a|b}` -- that no character scan of
        # the output can distinguish from a legitimate one, because the `|` is
        # structure rather than content. Type is the only place to catch it.
        raise _injection_refused(
            claim,
            tool_name,
            f"it holds a {type(value).__name__} rather than a single string value",
        )

    if not value.strip():
        raise _injection_refused(claim, tool_name, "it is empty")

    if value != value.strip():
        # Not normalized to the stripped value: two tenants named `acme` and
        # `acme ` would then collapse onto one. Surrounding whitespace cannot
        # be meaningful in a tag equality, so refuse it instead of guessing.
        raise _injection_refused(claim, tool_name, "it is padded with whitespace")

    if "|" in value:
        # Defence in depth. `Tag` escapes `|` inside a single value since
        # 0.27.1, so this is unreachable through the `==` path today; it is the
        # backstop for that character class changing again. `test_injected_pipe
        # _cannot_union_across_tenants` pins the rendering property itself.
        raise _injection_refused(claim, tool_name, "it contains the union operator '|'")

    return value


def _injected_tag_field(schema: IndexSchema, field_name: str) -> Any:
    """Return the schema field an injection entry names, or ``None``.

    ``None`` means "not usable for injection" for any reason -- absent, wrong
    type, or unindexed. Callers decide whether that is a startup failure or a
    request-time refusal.
    """
    field = schema.fields.get(field_name)
    if field is None or field.type != "tag":
        return None
    if getattr(field.attrs, "no_index", False):
        return None
    return field


def build_injected_filter(
    inject_specs: Sequence[InjectSpec],
    schema: IndexSchema,
    *,
    tool_name: str,
) -> FilterExpression:
    """Build the tenant-scoping expression for the current request.

    Entries AND together, and any one unusable claim refuses the whole request
    rather than narrowing by the entries that did resolve.
    """
    if not inject_specs:
        # Callers guard on truthiness before reaching here; an empty list would
        # otherwise have to return a match-all, which is the one value this
        # function exists to make unreachable.
        raise RedisVLMCPError(
            f"Tool '{tool_name}' asked for claim injection with no entries "
            "configured; refusing to run an unscoped query",
            code=MCPErrorCode.INTERNAL_ERROR,
            retryable=False,
        )

    combined: FilterExpression | None = None
    clauses: list[str] = []

    for spec in inject_specs:
        value = resolve_injected_claim(spec.claim, tool_name=tool_name)

        if _injected_tag_field(schema, spec.field) is None:
            # Startup validation already rejected this, so reaching it means the
            # bound schema changed under a registered tool. Re-checking costs a
            # dict lookup and keeps the guarantee tied to the schema actually in
            # force rather than to the one present at registration.
            raise _injection_refused(
                spec.claim,
                tool_name,
                f"its target field '{spec.field}' is no longer an indexed tag "
                "field on the bound index",
            )

        clause = Tag(spec.field) == value
        if is_match_all_filter(clause):
            # The check has to run here, on the clause alone. An intersection
            # elides a `*` operand, so `locked & match_all` renders as `locked`
            # and a check on the combined expression would pass while the
            # tenant clause had silently vanished.
            raise _injection_refused(
                spec.claim,
                tool_name,
                "it renders as a filter that matches every document",
            )

        clauses.append(str(clause))
        combined = clause if combined is None else combined & clause

    assert combined is not None  # for mypy; the empty case raised above
    rendered = str(combined)
    missing = [clause for clause in clauses if clause not in rendered]
    if missing:
        # Operands render verbatim into `(left right)`, so containment is exact
        # rather than approximate. If a future change to `format_expression`
        # ever drops one, fail the request instead of serving a query that is
        # scoped to fewer tenants' worth of clauses than were configured.
        raise RedisVLMCPError(
            f"Tool '{tool_name}' built an injected filter that lost the clauses "
            f"{missing}; refusing to run a query that may not be scoped",
            code=MCPErrorCode.INTERNAL_ERROR,
            retryable=False,
        )

    return combined


def validate_inject_against_schema(
    inject_specs: Sequence[InjectSpec],
    schema: IndexSchema,
    *,
    profile_name: str,
) -> None:
    """Fail startup when an injection entry names a field it cannot scope by.

    Configuration validation cannot do this: the schema is only known once the
    binding has been inspected at startup. Without it, a profile would register
    cleanly and then refuse every request -- or, worse for a `no_index` field,
    return nothing and look like correct scoping.
    """
    field_names = ", ".join(sorted(schema.field_names))

    for spec in inject_specs:
        field = schema.fields.get(spec.field)
        if field is None:
            raise ValueError(
                f"custom_tools '{profile_name}' lock.inject references unknown "
                f"field '{spec.field}' on index '{schema.index.name}'; "
                f"available: {field_names}"
            )
        if field.type != "tag":
            raise ValueError(
                f"custom_tools '{profile_name}' lock.inject field '{spec.field}' "
                f"is a {field.type} field; injection requires a tag field, "
                "because text equality is an exact-phrase match that tokenizes "
                "on punctuation and so matches neighbouring values too"
            )
        if getattr(field.attrs, "no_index", False):
            raise ValueError(
                f"custom_tools '{profile_name}' lock.inject field '{spec.field}' "
                "is declared NOINDEX, so a filter on it matches nothing; every "
                "call would return an empty result set that looks like correct "
                "scoping"
            )
