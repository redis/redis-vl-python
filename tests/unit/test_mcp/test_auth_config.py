"""Unit tests for MCP auth config validation (`MCPAuthConfig`).

These cover the configuration contract only; provider construction and token
verification are tested separately.
"""

import pytest
from pydantic import ValidationError

from redisvl.mcp.config import MCPAuthConfig


def test_auth_type_defaults_to_none():
    cfg = MCPAuthConfig()
    assert cfg.type == "none"


def test_jwt_with_jwks_uri_is_valid():
    cfg = MCPAuthConfig(
        type="jwt",
        jwks_uri="https://auth.redis.example/abc123/discovery/v2.0/keys",
        issuer="https://auth.redis.example/abc123/v2.0",
        audience="api://redisvl-mcp",
        required_scopes=["kb.read"],
        read_scope="kb.search.read",
        write_scope="kb.search.write",
    )
    assert cfg.type == "jwt"
    assert cfg.audience == "api://redisvl-mcp"
    assert cfg.read_scope == "kb.search.read"


def test_jwt_with_static_public_key_is_valid():
    cfg = MCPAuthConfig(
        type="jwt",
        public_key="-----BEGIN PUBLIC KEY-----\nMII...\n-----END PUBLIC KEY-----",
        issuer="https://auth.redis.example/abc123/v2.0",
        audience="api://redisvl-mcp",
    )
    assert cfg.public_key is not None
    assert cfg.jwks_uri is None


def test_jwt_rejects_both_jwks_uri_and_public_key():
    with pytest.raises(ValidationError, match="exactly one"):
        MCPAuthConfig(
            type="jwt",
            jwks_uri="https://auth.redis.example/keys",
            public_key="-----BEGIN PUBLIC KEY-----\nMII...\n-----END PUBLIC KEY-----",
            audience="api://redisvl-mcp",
        )


def test_jwt_rejects_neither_jwks_uri_nor_public_key():
    with pytest.raises(ValidationError, match="exactly one"):
        MCPAuthConfig(
            type="jwt",
            issuer="https://auth.redis.example/abc123/v2.0",
            audience="api://redisvl-mcp",
        )


def test_jwt_requires_audience():
    with pytest.raises(ValidationError, match="audience"):
        MCPAuthConfig(
            type="jwt",
            jwks_uri="https://auth.redis.example/keys",
            issuer="https://auth.redis.example/abc123/v2.0",
        )


def test_unknown_auth_type_is_rejected():
    with pytest.raises(ValidationError):
        MCPAuthConfig(type="kerberos")


def test_none_type_ignores_jwt_fields():
    # A `none` config should never require jwt-only fields.
    cfg = MCPAuthConfig(type="none")
    assert cfg.audience is None
    assert cfg.jwks_uri is None
