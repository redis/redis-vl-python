"""Unit tests for building a FastMCP auth provider from `MCPAuthConfig`."""

import pytest

from redisvl.mcp.config import MCPAuthConfig

fastmcp = pytest.importorskip(
    "fastmcp", reason="fastmcp not installed (install redisvl[mcp])"
)
from fastmcp.server.auth.providers.jwt import JWTVerifier, RSAKeyPair

from redisvl.mcp.auth import (
    build_auth_provider,
    missing_required_claims,
    token_has_scope,
)


def test_build_returns_none_for_none_type():
    assert build_auth_provider(MCPAuthConfig(type="none")) is None


def test_build_returns_none_for_missing_config():
    assert build_auth_provider(None) is None


def test_build_returns_jwt_verifier():
    key = RSAKeyPair.generate()
    provider = build_auth_provider(
        MCPAuthConfig(
            type="jwt",
            public_key=key.public_key,
            issuer="https://auth.redis.example/abc123/v2.0",
            audience="api://redisvl-mcp",
            required_scopes=["kb.read"],
        )
    )
    assert isinstance(provider, JWTVerifier)


def test_token_has_scope_helper():
    class _AccessToken:
        def __init__(self, scopes):
            self.scopes = scopes

    assert token_has_scope(_AccessToken(["kb.search.read"]), "kb.search.read")
    assert not token_has_scope(_AccessToken(["kb.search.read"]), "kb.search.write")
    # No required scope configured means the gate is open.
    assert token_has_scope(_AccessToken([]), None)


def test_missing_required_claims():
    assert missing_required_claims({"exp": 1, "iat": 1}, ["exp", "iat"]) == []
    assert missing_required_claims({"iat": 1}, ["exp", "iat"]) == ["exp"]
    assert missing_required_claims({}, ["exp"]) == ["exp"]
    assert missing_required_claims(None, ["exp"]) == ["exp"]
    assert missing_required_claims({"exp": 1}, []) == []
