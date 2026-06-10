"""Unit tests for read/write scope gating and the configurable auth claim."""

import pytest

from redisvl.mcp.auth import authorization_values, ensure_tool_scope, token_has_scope
from redisvl.mcp.config import MCPAuthConfig
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError


class _AccessToken:
    def __init__(self, scopes=None, claims=None):
        self.scopes = scopes or []
        self.claims = claims or {}


class _Cfg:
    def __init__(self, authorization_claim="scp"):
        self.authorization_claim = authorization_claim


class _Server:
    def __init__(self, enabled=True, authorization_claim="scp"):
        self._auth_enabled = enabled
        self.auth_config = _Cfg(authorization_claim) if enabled else None


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


def test_ensure_tool_scope_forbids_when_no_token(monkeypatch):
    monkeypatch.setattr(
        "fastmcp.server.dependencies.get_access_token", lambda: None, raising=False
    )
    with pytest.raises(RedisVLMCPError) as exc:
        ensure_tool_scope(_Server(), "kb.search.read")
    assert exc.value.code == MCPErrorCode.FORBIDDEN
