"""Integration tests for MCP JWT authentication.

These mint real RS256 tokens with FastMCP's ``RSAKeyPair`` and verify them
through a ``JWTVerifier`` built from ``MCPAuthConfig``. No network JWKS endpoint
is required: the verifier is configured with the key pair's static public key.

The token fixture is modeled on a real enterprise OIDC access token, with all
organization-specific values replaced by dummies (subject ``nitin``, org
``redis``). The ``tid``/``oid``/``upn`` claims are carried but not acted on by
the server in phase 1; they would drive a gateway binding table.
"""

import time

import pytest

fastmcp = pytest.importorskip(
    "fastmcp", reason="fastmcp not installed (install redisvl[mcp])"
)
from authlib.jose import jwt as jose_jwt
from fastmcp.server.auth.providers.jwt import RSAKeyPair

from redisvl.mcp.auth import build_auth_provider, token_has_scope
from redisvl.mcp.config import MCPAuthConfig


@pytest.fixture(scope="session", autouse=True)
def redis_container():
    # JWT validation is pure crypto; shadow the repo-wide Docker/Redis fixture so
    # these tests do not require a running Redis.
    yield None


ISSUER = "https://auth.redis.example/abc123/v2.0"
AUDIENCE = "api://redisvl-mcp"
READ_SCOPE = "kb.search.read"
WRITE_SCOPE = "kb.search.write"

# Sanitized claims modeled on a real OIDC access token.
BASE_CLAIMS = {
    "oid": "00000000-nitin-0000-000000000000",
    "upn": "nitin@redis.example",
    "tid": "11111111-2222-3333-4444-555555555555",
}


@pytest.fixture(scope="module")
def key() -> RSAKeyPair:
    return RSAKeyPair.generate()


@pytest.fixture()
def verifier(key):
    return build_auth_provider(
        MCPAuthConfig(
            type="jwt",
            public_key=key.public_key,
            issuer=ISSUER,
            audience=AUDIENCE,
            required_scopes=[READ_SCOPE],
            read_scope=READ_SCOPE,
            write_scope=WRITE_SCOPE,
        )
    )


def _mint(key, *, scopes, audience=AUDIENCE, issuer=ISSUER, expires_in=3600):
    return key.create_token(
        subject="nitin",
        issuer=issuer,
        audience=audience,
        scopes=scopes,
        expires_in_seconds=expires_in,
        additional_claims=BASE_CLAIMS,
    )


async def test_valid_read_token_is_accepted(verifier, key):
    token = _mint(key, scopes=[READ_SCOPE])
    access = await verifier.verify_token(token)
    assert access is not None
    assert READ_SCOPE in access.scopes
    assert token_has_scope(access, READ_SCOPE)
    assert not token_has_scope(access, WRITE_SCOPE)


async def test_write_token_carries_write_scope(verifier, key):
    token = _mint(key, scopes=[READ_SCOPE, WRITE_SCOPE])
    access = await verifier.verify_token(token)
    assert access is not None
    assert token_has_scope(access, WRITE_SCOPE)


async def test_wrong_audience_is_rejected(verifier, key):
    token = _mint(key, scopes=[READ_SCOPE], audience="api://some-other-service")
    assert await verifier.verify_token(token) is None


async def test_wrong_issuer_is_rejected(verifier, key):
    token = _mint(key, scopes=[READ_SCOPE], issuer="https://evil.example/v2.0")
    assert await verifier.verify_token(token) is None


async def test_expired_token_is_rejected(verifier, key):
    token = _mint(key, scopes=[READ_SCOPE], expires_in=-10)
    assert await verifier.verify_token(token) is None


async def test_missing_required_scope_is_rejected(verifier, key):
    # Required connect scope is READ_SCOPE; a token without it must fail.
    token = _mint(key, scopes=["kb.unrelated"])
    assert await verifier.verify_token(token) is None


async def test_garbage_token_is_rejected(verifier):
    assert await verifier.verify_token("not-a-jwt") is None


def _mint_raw(key, claims):
    """Sign a token with arbitrary claims (used to omit exp/iat)."""
    token = jose_jwt.encode(
        {"alg": "RS256"}, claims, key.private_key.get_secret_value()
    )
    return token.decode() if isinstance(token, bytes) else token


async def test_token_without_exp_is_rejected(verifier, key):
    # A token with no exp would never expire; it must be rejected.
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "nitin",
        "scope": READ_SCOPE,
        "iat": int(time.time()),
    }
    assert await verifier.verify_token(_mint_raw(key, claims)) is None


async def test_token_without_iat_is_rejected(verifier, key):
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "nitin",
        "scope": READ_SCOPE,
        "exp": int(time.time()) + 3600,
    }
    assert await verifier.verify_token(_mint_raw(key, claims)) is None
