"""Unit tests for the CLI unauthenticated-HTTP-bind guard."""

import pytest

from redisvl.cli.mcp import MCP


def test_stdio_never_warns():
    assert MCP._check_http_auth("stdio", "127.0.0.1", False, False) is None


def test_http_with_auth_is_silent():
    assert MCP._check_http_auth("streamable-http", "0.0.0.0", True, False) is None


def test_http_loopback_without_auth_warns():
    msg = MCP._check_http_auth("streamable-http", "127.0.0.1", False, False)
    assert msg is not None
    assert "without" in msg.lower()


def test_http_non_loopback_without_auth_fails_closed():
    with pytest.raises(RuntimeError, match="non-loopback"):
        MCP._check_http_auth("streamable-http", "0.0.0.0", False, False)


def test_http_non_loopback_allowed_with_flag_warns():
    msg = MCP._check_http_auth("streamable-http", "0.0.0.0", False, True)
    assert msg is not None
    assert "WARNING" in msg
