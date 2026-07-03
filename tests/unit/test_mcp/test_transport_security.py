"""Unit tests for the HTTP transport-security guard (DNS-rebinding defense).

Drives ``HostOriginValidationMiddleware`` directly with synthetic ASGI
scope/receive/send, and covers the bind-derived default host allowlist.
"""

import asyncio

import pytest

from redisvl.mcp.config import MCPTransportSecurityConfig
from redisvl.mcp.transport_security import (
    HostOriginValidationMiddleware,
    _strip_port,
    build_host_origin_middleware,
    default_allowed_hosts,
)


class _RecordingApp:
    """ASGI app stand-in that records whether it was invoked."""

    def __init__(self):
        self.called = False

    async def __call__(self, scope, receive, send):
        self.called = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def _http_scope(headers: dict[bytes, bytes]) -> dict:
    return {
        "type": "http",
        "headers": [(key, value) for key, value in headers.items()],
    }


def _run(middleware, scope):
    """Drive a middleware once, returning (status, downstream_called)."""
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    status = next(
        (m["status"] for m in sent if m["type"] == "http.response.start"), None
    )
    return status


def _middleware(app, *, hosts=("127.0.0.1:8000",), origins=(), allow_any_origin=False):
    return HostOriginValidationMiddleware(
        app,
        allowed_hosts=frozenset(hosts),
        allowed_origins=frozenset(origins),
        allow_any_origin=allow_any_origin,
    )


# --- Host validation ---------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    [b"127.0.0.1:8000", b"127.0.0.1", b"localhost", b"localhost:8000", b"[::1]"],
)
def test_allowed_loopback_host_passes(host):
    app = _RecordingApp()
    hosts = default_allowed_hosts("127.0.0.1", 8000)
    mw = _middleware(app, hosts=hosts)
    status = _run(mw, _http_scope({b"host": host}))
    assert app.called is True
    assert status == 200


def test_spoofed_host_rejected_and_app_not_called():
    app = _RecordingApp()
    mw = _middleware(app, hosts=default_allowed_hosts("127.0.0.1", 8000))
    status = _run(mw, _http_scope({b"host": b"evil.com"}))
    assert status == 400
    assert app.called is False


def test_missing_host_rejected():
    app = _RecordingApp()
    mw = _middleware(app)
    status = _run(mw, _http_scope({}))
    assert status == 400
    assert app.called is False


def test_host_is_case_insensitive():
    app = _RecordingApp()
    mw = _middleware(app, hosts={"localhost", "localhost:8000"})
    status = _run(mw, _http_scope({b"host": b"LOCALHOST:8000"}))
    assert status == 200
    assert app.called is True


def test_host_with_default_port_matches_bare_allowlist_entry():
    # Allowlist only carries the bare form; a client that includes a port still
    # matches via the port-stripped comparison.
    app = _RecordingApp()
    mw = _middleware(app, hosts={"example.com"})
    status = _run(mw, _http_scope({b"host": b"example.com:8000"}))
    assert status == 200


# --- Origin validation -------------------------------------------------------


def test_absent_origin_passes():
    app = _RecordingApp()
    mw = _middleware(app, hosts={"localhost:8000"})
    status = _run(mw, _http_scope({b"host": b"localhost:8000"}))
    assert status == 200
    assert app.called is True


def test_cross_site_origin_rejected():
    app = _RecordingApp()
    mw = _middleware(app, hosts={"localhost:8000"})
    status = _run(
        mw,
        _http_scope({b"host": b"localhost:8000", b"origin": b"https://evil.com"}),
    )
    assert status == 403
    assert app.called is False


def test_allowlisted_origin_passes():
    app = _RecordingApp()
    mw = _middleware(app, hosts={"localhost:8000"}, origins={"https://good.example"})
    status = _run(
        mw,
        _http_scope({b"host": b"localhost:8000", b"origin": b"https://good.example"}),
    )
    assert status == 200
    assert app.called is True


def test_allow_any_origin_passes_any_origin():
    app = _RecordingApp()
    mw = _middleware(app, hosts={"localhost:8000"}, allow_any_origin=True)
    status = _run(
        mw,
        _http_scope({b"host": b"localhost:8000", b"origin": b"https://evil.com"}),
    )
    assert status == 200
    assert app.called is True


def test_origin_is_case_insensitive():
    app = _RecordingApp()
    mw = _middleware(app, hosts={"localhost:8000"}, origins={"https://good.example"})
    status = _run(
        mw,
        _http_scope({b"host": b"localhost:8000", b"origin": b"HTTPS://GOOD.EXAMPLE"}),
    )
    assert status == 200


# --- Non-http scopes ---------------------------------------------------------


def test_non_http_scope_passes_through():
    app = _RecordingApp()
    mw = _middleware(app)

    async def receive():
        return {"type": "websocket.receive"}

    async def send(message):
        pass

    asyncio.run(mw({"type": "websocket"}, receive, send))
    assert app.called is True


# --- default_allowed_hosts ---------------------------------------------------


def test_default_allowed_hosts_loopback_expansion():
    hosts = default_allowed_hosts("127.0.0.1", 8000)
    assert {"localhost", "localhost:8000", "127.0.0.1", "127.0.0.1:8000"} <= hosts
    assert "[::1]" in hosts and "[::1]:8000" in hosts


def test_default_allowed_hosts_specific_host():
    hosts = default_allowed_hosts("192.168.1.10", 9000)
    assert hosts == {"192.168.1.10", "192.168.1.10:9000"}


def test_default_allowed_hosts_wildcard_is_loopback_only():
    hosts = default_allowed_hosts("0.0.0.0", 8000)
    # No synthesized external host; only the loopback set.
    assert hosts == default_allowed_hosts("127.0.0.1", 8000)
    assert "0.0.0.0" not in hosts


def test_default_allowed_hosts_ipv6_bind_is_bracketed():
    hosts = default_allowed_hosts("2001:db8::1", 8000)
    assert hosts == {"[2001:db8::1]", "[2001:db8::1]:8000"}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("127.0.0.1:8000", "127.0.0.1"),
        ("127.0.0.1", "127.0.0.1"),
        ("[::1]:8000", "[::1]"),
        ("[::1]", "[::1]"),
        ("localhost", "localhost"),
    ],
)
def test_strip_port(raw, expected):
    assert _strip_port(raw) == expected


# --- build_host_origin_middleware --------------------------------------------


def test_build_middleware_disabled_returns_empty():
    cfg = MCPTransportSecurityConfig(enabled=False)
    assert build_host_origin_middleware(cfg, "127.0.0.1", 8000) == []


def test_build_middleware_merges_configured_hosts():
    cfg = MCPTransportSecurityConfig(allowed_hosts=["proxy.internal:8000"])
    built = build_host_origin_middleware(cfg, "0.0.0.0", 8000)
    assert len(built) == 1
    kwargs = built[0].kwargs
    assert "proxy.internal:8000" in kwargs["allowed_hosts"]
    assert "127.0.0.1:8000" in kwargs["allowed_hosts"]
