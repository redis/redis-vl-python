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


# --- which transports get the guard -------------------------------------------


@pytest.mark.parametrize(
    "transport, configured_default, guarded",
    [
        ("streamable-http", "stdio", True),
        ("sse", "stdio", True),
        # FastMCP's own default HTTP name, served exactly like streamable-http.
        ("http", "stdio", True),
        # An omitted transport falls back to fastmcp.settings.transport, which
        # can name an HTTP transport just as an explicit argument can.
        (None, "streamable-http", True),
        (None, "http", True),
        ("stdio", "stdio", False),
        (None, "stdio", False),
    ],
)
def test_every_http_transport_is_served_behind_the_guard(
    monkeypatch, transport, configured_default, guarded
):
    fastmcp = pytest.importorskip(
        "fastmcp", reason="fastmcp not installed (install redisvl[mcp])"
    )
    from redisvl.mcp.server import RedisVLMCPServer

    served: dict = {}

    async def record_serve(self, transport=None, show_banner=None, **kwargs):
        served["middleware"] = kwargs.get("middleware")

    monkeypatch.setattr(fastmcp.FastMCP, "run_async", record_serve)
    monkeypatch.setattr(fastmcp.settings, "transport", configured_default)
    guard = object()
    monkeypatch.setattr(
        "redisvl.mcp.server.build_host_origin_middleware", lambda *args: [guard]
    )
    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server._transport_security = MCPTransportSecurityConfig()

    asyncio.run(server.run_async(transport=transport, host="0.0.0.0", port=8000))

    assert (served["middleware"] == [guard]) is guarded


def _run_with_settings(monkeypatch, settings, **run_kwargs):
    """Run `run_async` against substitute FastMCP settings, recording the guard."""
    fastmcp = pytest.importorskip(
        "fastmcp", reason="fastmcp not installed (install redisvl[mcp])"
    )
    from redisvl.mcp.server import RedisVLMCPServer

    built: list = []
    served: dict = {}

    async def record_serve(self, transport=None, show_banner=None, **kwargs):
        served["middleware"] = kwargs.get("middleware")

    monkeypatch.setattr(fastmcp.FastMCP, "run_async", record_serve)
    monkeypatch.setattr(fastmcp, "settings", settings)
    monkeypatch.setattr(
        "redisvl.mcp.server.build_host_origin_middleware",
        lambda config, host, port: built.append((host, port)) or ["guard"],
    )
    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server._transport_security = MCPTransportSecurityConfig()
    asyncio.run(server.run_async(**run_kwargs))
    return built, served.get("middleware")


def test_an_omitted_transport_serves_stdio_on_fastmcp_without_the_setting(
    monkeypatch,
):
    # `settings.transport` arrived in FastMCP 3.1; before it an omitted
    # transport was always stdio. The supported range starts at 2.0.
    from types import SimpleNamespace

    built, middleware = _run_with_settings(
        monkeypatch, SimpleNamespace(host="127.0.0.1", port=8000), transport=None
    )
    assert built == []
    assert middleware is None


def test_the_guard_allowlists_the_address_fastmcp_actually_binds(monkeypatch):
    # FastMCP binds FASTMCP_HOST / FASTMCP_PORT when no host or port is passed,
    # so a hardcoded 127.0.0.1:8000 allowlist would reject legitimate Host
    # headers on that bind.
    from types import SimpleNamespace

    settings = SimpleNamespace(transport="streamable-http", host="0.0.0.0", port=9123)

    built, middleware = _run_with_settings(monkeypatch, settings, transport=None)
    assert built == [("0.0.0.0", 9123)]
    assert middleware == ["guard"]

    # An explicit host and port still take precedence over settings.
    built, _ = _run_with_settings(
        monkeypatch, settings, transport="http", host="10.0.0.5", port=7000
    )
    assert built == [("10.0.0.5", 7000)]
