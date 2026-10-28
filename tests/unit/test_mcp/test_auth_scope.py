"""Unit tests for read/write scope gating and the configurable auth claim."""

import pytest

# These tests monkeypatch fastmcp.server.dependencies.get_access_token, which
# imports fastmcp; skip the module when the optional extra is absent.
pytest.importorskip("fastmcp", reason="fastmcp not installed (install redisvl[mcp])")

from redisvl.mcp.auth import (
    authorization_values,
    ensure_read_scope,
    ensure_tool_scope,
    ensure_write_scope,
    token_has_scope,
)
from redisvl.mcp.config import MCPAuthConfig
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError


class _AccessToken:
    def __init__(self, scopes=None, claims=None):
        self.scopes = scopes or []
        self.claims = claims or {}


class _Cfg:
    def __init__(self, authorization_claim="scp", read_scope=None, write_scope=None):
        self.authorization_claim = authorization_claim
        self.read_scope = read_scope
        self.write_scope = write_scope


class _Server:
    def __init__(
        self,
        enabled=True,
        authorization_claim="scp",
        read_scope=None,
        write_scope=None,
    ):
        self._auth_enabled = enabled
        self.auth_config = (
            _Cfg(authorization_claim, read_scope, write_scope) if enabled else None
        )


# --- claim selection -------------------------------------------------------


def test_authorization_values_reads_scopes_for_scp():
    tok = _AccessToken(scopes=["kb.read"], claims={"roles": ["kb.search.read"]})
    assert authorization_values(tok, "scp") == ["kb.read"]


def test_authorization_values_reads_named_claim_list():
    tok = _AccessToken(scopes=["kb.read"], claims={"roles": ["kb.search.read"]})
    assert authorization_values(tok, "roles") == ["kb.search.read"]


def test_authorization_values_splits_space_delimited_claim():
    tok = _AccessToken(claims={"roles": "kb.search.read kb.search.write"})
    assert authorization_values(tok, "roles") == ["kb.search.read", "kb.search.write"]


def test_authorization_values_missing_claim_is_empty():
    assert authorization_values(_AccessToken(), "roles") == []


# --- token_has_scope -------------------------------------------------------


def test_token_has_scope_uses_named_claim():
    tok = _AccessToken(scopes=["kb.read"], claims={"roles": ["kb.search.read"]})
    assert token_has_scope(tok, "kb.search.read", "roles")
    assert not token_has_scope(tok, "kb.search.write", "roles")
    # The default scp claim does not see the roles claim.
    assert not token_has_scope(tok, "kb.search.read")


# --- config ----------------------------------------------------------------


def test_authorization_claim_defaults_to_scp():
    assert MCPAuthConfig().authorization_claim == "scp"


def test_authorization_claim_can_be_roles():
    cfg = MCPAuthConfig(
        type="jwt",
        public_key="-----BEGIN PUBLIC KEY-----\nMII...\n-----END PUBLIC KEY-----",
        issuer="https://auth.redis.example/abc123/v2.0",
        audience="api://redisvl-mcp",
        authorization_claim="roles",
    )
    assert cfg.authorization_claim == "roles"


# --- ensure_tool_scope -----------------------------------------------------


def test_ensure_tool_scope_noop_when_auth_disabled():
    # No token lookup, no raise.
    ensure_tool_scope(_Server(enabled=False), "kb.search.write")


def test_ensure_tool_scope_noop_when_scope_not_configured():
    ensure_tool_scope(_Server(), None)


def test_ensure_tool_scope_allows_when_scope_present(monkeypatch):
    tok = _AccessToken(claims={"roles": ["kb.search.write"]})
    monkeypatch.setattr(
        "fastmcp.server.dependencies.get_access_token", lambda: tok, raising=False
    )
    ensure_tool_scope(_Server(authorization_claim="roles"), "kb.search.write")


def test_ensure_tool_scope_forbids_when_scope_missing(monkeypatch):
    tok = _AccessToken(claims={"roles": ["kb.search.read"]})
    monkeypatch.setattr(
        "fastmcp.server.dependencies.get_access_token", lambda: tok, raising=False
    )
    with pytest.raises(RedisVLMCPError) as exc:
        ensure_tool_scope(_Server(authorization_claim="roles"), "kb.search.write")
    assert exc.value.code == MCPErrorCode.FORBIDDEN
    assert exc.value.retryable is False


def test_ensure_tool_scope_noop_when_no_token(monkeypatch):
    # No token means no authenticated request context (for example stdio).
    # Authenticated HTTP transports reject tokenless requests before the tool
    # runs, so the gate must not fire here.
    monkeypatch.setattr(
        "fastmcp.server.dependencies.get_access_token", lambda: None, raising=False
    )
    ensure_tool_scope(_Server(), "kb.search.read")


# --- ensure_read_scope / ensure_write_scope --------------------------------


def test_scope_helpers_resolve_their_configured_scope(monkeypatch):
    # The helper reads the scope name off the server, so a wrapper never has to
    # know which auth_config field its side of the gate uses.
    tok = _AccessToken(claims={"roles": ["kb.search.read"]})
    monkeypatch.setattr(
        "fastmcp.server.dependencies.get_access_token", lambda: tok, raising=False
    )
    server = _Server(
        authorization_claim="roles",
        read_scope="kb.search.read",
        write_scope="kb.search.write",
    )

    ensure_read_scope(server)

    with pytest.raises(RedisVLMCPError) as exc:
        ensure_write_scope(server)
    assert exc.value.code == MCPErrorCode.FORBIDDEN


def test_scope_helpers_noop_when_auth_disabled():
    server = _Server(enabled=False)
    ensure_read_scope(server)
    ensure_write_scope(server)


def test_scope_gate_fails_closed_when_auth_config_is_unreachable(monkeypatch):
    # Renaming the server's auth_config attribute used to make every call site
    # resolve a None scope and return early, silently ungating every tool.
    tok = _AccessToken(claims={"roles": []})
    monkeypatch.setattr(
        "fastmcp.server.dependencies.get_access_token", lambda: tok, raising=False
    )
    server = _Server(authorization_claim="roles", read_scope="kb.search.read")
    server._renamed_auth_config = server.auth_config
    del server.auth_config

    with pytest.raises(RedisVLMCPError) as exc:
        ensure_read_scope(server)
    assert exc.value.code == MCPErrorCode.INTERNAL_ERROR
    assert exc.value.retryable is False


def test_scope_helper_raises_when_the_config_field_is_renamed():
    # An unguarded getattr, so a renamed MCPAuthConfig field is a loud failure
    # rather than a None scope that turns the gate into a no-op.
    server = _Server(read_scope="kb.search.read")
    del server.auth_config.read_scope

    with pytest.raises(AttributeError):
        ensure_read_scope(server)
