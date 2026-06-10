# Spec: Authentication for the RedisVL MCP Server

**Status:** Draft
**Date:** 2026-06-09
**Owner:** RedisVL MCP
**Tracking issue:** redis/redis-vl-python#625
**Related:** `redisvl/mcp/`, FastMCP auth docs (https://gofastmcp.com/servers/auth/authentication)

---

## 1. Background

RedisVL ships an MCP server (`redisvl/mcp/`) built as a `FastMCP` subclass that
exposes one configured Redis index to MCP clients for search and optional
upsert.

Today the server is constructed with **no authentication**:

```python
# redisvl/mcp/server.py:73
super().__init__("redisvl", lifespan=self._fastmcp_lifespan)
```

The only access control is the `--read-only` flag, which simply skips
registration of the `upsert-records` tool (`redisvl/mcp/server.py:195-203`). It
is **not** a transport-level control.

The CLI (`redisvl/cli/mcp.py`) can bind the server to a network port via the
`sse` and `streamable-http` transports (`--host`, `--port`). In that mode, any
client that can reach the port has full access to the index. This is the gap
this spec closes.

### Current relevant facts

- `fastmcp >= 2.0.0` is the declared optional dependency (`pyproject.toml`,
  extra `mcp`). Locally installed: **fastmcp 3.1.0**.
- FastMCP's constructor accepts an `auth=` provider (verified).
- `JWTVerifier` lives at `fastmcp.server.auth.providers.jwt.JWTVerifier` with
  signature `(public_key, jwks_uri, issuer, audience, algorithm,
  required_scopes, base_url, ssrf_safe, http_client)`.
- The same module exports `RSAKeyPair` (`.generate()`, `.create_token(...)`),
  which we use to mint real signed JWTs in tests.

---

## 2. Goals / Non-goals

### Goals

1. Require authentication for the **HTTP transports** (`sse`,
   `streamable-http`), opt-in and off by default.
2. Default mechanism: **JWT bearer-token validation** (`JWTVerifier`) against an
   existing IdP's JWKS endpoint or a static public key. No OAuth authorization
   server.
3. Validate the **audience (`aud`)** claim, not just signature and issuer, so a
   token minted for another service cannot be replayed (RFC 8707).
4. **Scope-gate read vs write**: a token's scope/role claim decides whether the
   caller can reach the upsert tool, not just whether it can connect.
5. Config-driven and import-safe: optional, never breaks installs without the
   `mcp` extra; secrets stay out of the command line.
6. Make the insecure case loud: warn (or fail closed) when an HTTP transport is
   bound with no auth.

### Non-goals

- Authenticating `stdio` (local subprocess, no network surface; FastMCP ignores
  auth there).
- **Per-tenant / fine-grained authorization** that maps identity claims (e.g. a
  tenant id or role) to a specific Redis ACL user, key-prefix fence, or injected
  query filter. See §3.4. This belongs in a gateway/policy layer.
- Running a full OAuth authorization server (`OAuthProvider`).
- Per-vendor login providers (GitHub, Google, ...). A single generic
  `oauth-proxy` option is preferred later over one branch per vendor.

---

## 3. Design

### 3.1 Auth tiers

| Tier | Provider | When |
|------|----------|------|
| **JWT validation** (default, phase 1) | `JWTVerifier` | An existing IdP/key issues JWTs; validate issuer/audience/scopes against JWKS or static key. |
| **OAuth proxy** (phase 2) | generic `oauth-proxy` over `OAuthProxy` | Interactive login through an existing provider, no class-per-vendor. |
| **None** (default) | - | No auth. Allowed, but warns/fails on HTTP transports. |

Phase 1 ships `jwt` + `none`. The `auth.type` switch is designed so
`oauth-proxy` slots in later without breaking changes.

### 3.2 Lifecycle constraint

The auth provider must be passed to `super().__init__()` (`server.py:73`), i.e.
**at construction time**, but YAML config is loaded later in `startup()` ->
`_initialize_runtime_resources()` (`server.py:300-302`). Resolution: a small
standalone helper reads auth config from `MCPSettings` (env) and, if a config
path is set, peeks only the `server.auth` block, without running full startup.

### 3.3 Transport interaction

- `stdio`: provider not attached; no warning.
- `sse` / `streamable-http`: provider attached if configured. If not, the CLI
  warns loudly, and fails closed on non-loopback (`0.0.0.0`) binds unless
  `--allow-unauthenticated` is passed.

### 3.4 Scope gating, and the per-tenant boundary

Phase 1 enforces **coarse** authorization at the server:

- A configured **read scope** is required to call `search-records`.
- A configured **write scope** is required to call `upsert-records` (in
  addition to the server not being `--read-only`).

What phase 1 deliberately does **not** do: map a token's tenant/role claims to a
*different Redis identity per request*. The server holds one Redis connection
for one index, established at startup. Re-authenticating to Redis as a
per-request ACL user derived from the token is a much larger change and is the
job of a gateway/policy layer that sits in front of the MCP server (validate
token -> look up binding table -> inject Redis credentials + query filters).
RedisVL provides the coarse token validation and scope gate; the gateway owns
fine-grained, per-tenant enforcement. This may be revisited as future work.

---

## 4. Config schema

### 4.1 Env vars (`MCPSettings`, prefix `REDISVL_MCP_`)

| Env var | Field | Notes |
|---------|-------|-------|
| `REDISVL_MCP_AUTH_TYPE` | `auth_type` | `none` (default) \| `jwt` |
| `REDISVL_MCP_AUTH_JWKS_URI` | `auth_jwks_uri` | JWKS endpoint |
| `REDISVL_MCP_AUTH_PUBLIC_KEY` | `auth_public_key` | Static PEM (alternative to JWKS) |
| `REDISVL_MCP_AUTH_ISSUER` | `auth_issuer` | Expected `iss` |
| `REDISVL_MCP_AUTH_AUDIENCE` | `auth_audience` | Expected `aud` (required for jwt) |
| `REDISVL_MCP_AUTH_ALGORITHM` | `auth_algorithm` | Default `RS256` |
| `REDISVL_MCP_AUTH_REQUIRED_SCOPES` | `auth_required_scopes` | Comma-separated connect scopes |
| `REDISVL_MCP_AUTH_READ_SCOPE` | `auth_read_scope` | Scope required for search |
| `REDISVL_MCP_AUTH_WRITE_SCOPE` | `auth_write_scope` | Scope required for upsert |
| `REDISVL_MCP_AUTH_BASE_URL` | `auth_base_url` | Public server URL |

### 4.2 YAML (optional `server.auth` block)

```yaml
server:
  redis_url: ${REDIS_URL:-redis://localhost:6379}
  auth:
    type: jwt                       # none | jwt
    jwks_uri: ${MCP_JWKS_URI}
    issuer: ${MCP_ISSUER}
    audience: api://redisvl-mcp
    algorithm: RS256
    required_scopes: [kb.read]
    read_scope: kb.search.read
    write_scope: kb.search.write
    # public_key: ${MCP_PUBLIC_KEY}  # use instead of jwks_uri for static keys
```

`${ENV}` / `${ENV:-default}` substitution already works via
`redisvl/mcp/config.py` (`_ENV_PATTERN`, line 13). Env vars override YAML.

### 4.3 Validation rules

- `type: jwt` requires **exactly one** of `jwks_uri` or `public_key`.
- `type: jwt` requires `audience` (reject unbounded audience).
- `type: jwt` should set `issuer` (warn if missing).
- Unknown `type` -> config error at load time.
- `read_scope` / `write_scope` optional; when set, gate the respective tool.

---

## 5. Implementation plan

- `redisvl/mcp/config.py`: add `MCPAuthConfig` (+ validators) and optional
  `auth: MCPAuthConfig | None` on `MCPServerConfig`.
- `redisvl/mcp/settings.py`: add `auth_*` fields, plumb through `from_env`.
- New `redisvl/mcp/auth.py`:
  - `resolve_auth_config(settings, config_path) -> MCPAuthConfig | None`
    (env over YAML peek).
  - `build_auth_provider(auth_config) -> Any | None` returning `None` for
    `none`, a configured `JWTVerifier` for `jwt`; provider imports guarded for
    the optional `mcp` extra.
  - `peek_yaml_auth(config_path) -> dict | None` reads only `server.auth` with
    env substitution.
- `redisvl/mcp/server.py`: build the provider in `__init__`, pass `auth=`;
  store `self._auth_enabled`; apply read/write scope gates at tool registration.
- `redisvl/cli/mcp.py`: warn / fail-closed on HTTP transport without auth;
  add `--allow-unauthenticated`.
- `pyproject.toml`: confirm/raise the `fastmcp` floor that guarantees
  `JWTVerifier`.

---

## 6. Test plan (TDD)

Tests are written first; implementation follows until green.

### 6.1 Unit (`tests/unit/test_mcp/`)

`test_auth_config.py`
- `none`/unset -> resolves to no auth.
- `jwt` with `jwks_uri` + `issuer` + `audience` -> valid config.
- `jwt` with both `jwks_uri` and `public_key` -> error.
- `jwt` with neither `jwks_uri` nor `public_key` -> error.
- `jwt` missing `audience` -> error.
- unknown `type` -> error.

`test_auth_provider.py`
- `build_auth_provider(none)` -> `None`.
- `build_auth_provider(jwt)` -> instance of `JWTVerifier` carrying issuer,
  audience, required scopes.
- import-safe behavior when `fastmcp` is unavailable.

`test_settings.py` (extend)
- `REDISVL_MCP_AUTH_*` env vars populate the new fields.
- explicit `from_env` args override env.

`test_auth_resolution.py`
- env overrides YAML `server.auth`.
- YAML-only `server.auth` is honored when env is unset.

### 6.2 Integration (`tests/integration/test_mcp/test_auth.py`)

Uses `RSAKeyPair` to mint real RS256 tokens and a `JWTVerifier` configured with
the key pair's public key (no network JWKS needed).

Canonical token fixture (sanitized; see §6.3):
- valid read token -> `search-records` succeeds.
- valid write token -> `upsert-records` succeeds.
- read-only token -> `upsert-records` rejected (insufficient scope).
- wrong `aud` -> rejected.
- wrong `iss` -> rejected.
- expired token -> rejected.
- no/garbage token -> rejected (401).
- no-auth server -> any request succeeds (back-compat).

### 6.3 Canonical token fixture (sanitized)

Modeled on a real enterprise OIDC access token, with all org-specific values
replaced by dummies (`nitin`, org `redis`):

```jsonc
{
  "iss":   "https://auth.redis.example/abc123/v2.0",
  "aud":   "api://redisvl-mcp",
  "oid":   "00000000-nitin-0000-000000000000",
  "upn":   "nitin@redis.example",
  "tid":   "11111111-2222-3333-4444-555555555555",
  "roles": ["kb.search.read"],
  "scp":   "kb.read"
}
```

The write-path test uses `roles: ["kb.search.write"]` / appropriate scope. The
`tid`/`oid`/`upn` claims are carried but **not** acted on by the server in phase
1 (they would drive a gateway binding table; see §3.4).

---

## 7. Docs

- Add an Authentication section to `docs/user_guide/15_mcp.ipynb`: the
  `server.auth` block, `REDISVL_MCP_AUTH_*` env vars, a worked `JWTVerifier`
  example, scope gating, and the security note that HTTP transports must not be
  exposed unauthenticated. Note `stdio` needs no auth.

---

## 8. Rollout

- Backward compatible: auth defaults to `none`; existing deployments unaffected
  except the new HTTP-without-auth warning.
- Phase 1: JWT + None + scope gating + warning. Phase 2: generic `oauth-proxy`.
