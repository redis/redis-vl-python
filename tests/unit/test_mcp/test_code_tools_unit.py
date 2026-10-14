"""Code tools: what a valid one looks like, and what wraps every call."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import _schema

fastmcp = pytest.importorskip(
    "fastmcp", reason="fastmcp not installed (install redisvl[mcp])"
)

from redisvl.mcp.config import MCPConfig
from redisvl.mcp.context import IndexHandle, ToolContext, WritableIndexHandle
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.runtime import BindingRuntime
from redisvl.mcp.server import RedisVLMCPServer
from redisvl.mcp.settings import MCPSettings
from redisvl.mcp.tools.code import build_code_tool_spec, register_code_tool


async def recent_tickets(ctx: ToolContext, query: str, limit: int = 5) -> dict:
    """Find recent support tickets matching a query."""
    return {"query": query, "limit": limit}


class FakeIndex:
    async def exists(self):
        return True


class FakeServer:
    def __init__(self, *, timeout: float = 30.0, read_only: bool = False):
        self.config = MCPConfig.model_validate(
            {
                "server": {"redis_url": "redis://localhost:6379"},
                "indexes": {
                    "tickets": {
                        "redis_name": "tickets-index",
                        "search": {"type": "fulltext"},
                        "runtime": {"text_field_name": "content"},
                    }
                },
            }
        )
        # Copied rather than validated in, so a test can use a sub-second
        # timeout the config model (whole seconds) would refuse.
        binding = self.config.indexes["tickets"]
        self.binding = binding.model_copy(
            update={
                "runtime": binding.runtime.model_copy(
                    update={"request_timeout_seconds": timeout}
                )
            }
        )
        self.read_only = read_only
        self.resolved: list[str | None] = []
        self.registered: dict[str, Any] = {}
        self.auth_config = SimpleNamespace(read_scope="kb.read", write_scope="kb.write")
        self._auth_enabled = False

    def resolve_binding(self, index_id=None):
        self.resolved.append(index_id)
        return BindingRuntime(
            binding_id="tickets",
            binding=self.binding,
            index=FakeIndex(),
            schema=_schema(),
            vectorizer=None,
            supports_native_hybrid_search=False,
            effective_read_only=self.read_only,
        )

    async def run_guarded(self, operation_name, awaitable, *, timeout_seconds):
        return await awaitable

    def tool(self, name=None, description=None, **kwargs):
        def decorator(fn):
            self.registered[name] = fn
            return fn

        return decorator


def _registered(fn, server=None, **options):
    server = server or FakeServer()
    spec = build_code_tool_spec(fn, **options)
    register_code_tool(server, spec, "tickets")
    return server.registered[spec.name]


def _as_caller(monkeypatch, scopes=None):
    token = None if scopes is None else SimpleNamespace(scopes=scopes, claims={})
    monkeypatch.setattr(
        "fastmcp.server.dependencies.get_access_token", lambda: token, raising=False
    )


# --------------------------------------------------------------------------
# What counts as a code tool
# --------------------------------------------------------------------------


def test_a_sync_function_is_refused():
    def sync_tool(ctx: ToolContext, query: str) -> dict:
        """Search."""
        return {}

    with pytest.raises(TypeError, match="must be an async function"):
        build_code_tool_spec(sync_tool)


async def _no_context(query: str) -> dict:
    """Search."""
    return {}


async def _untyped_context(ctx, query: str) -> dict:
    """Search."""
    return {}


async def _keyword_only_context(*, ctx: ToolContext, query: str) -> dict:
    """Search."""
    return {}


async def _unannotated(ctx: ToolContext, query) -> dict:
    """Search."""
    return {}


async def _varargs(ctx: ToolContext, *queries: str) -> dict:
    """Search."""
    return {}


async def _varkw(ctx: ToolContext, **options: str) -> dict:
    """Search."""
    return {}


async def _positional_only(ctx: ToolContext, query: str, /) -> dict:
    """Search."""
    return {}


async def _unresolvable(ctx: ToolContext, query: "NoSuchType") -> dict:  # noqa: F821
    """Search."""
    return {}


@pytest.mark.parametrize(
    "fn, message",
    [
        pytest.param(_no_context, "ToolContext as its first", id="no-context"),
        pytest.param(_untyped_context, "ToolContext as its first", id="untyped"),
        pytest.param(
            _keyword_only_context, "first, positional parameter", id="kw-only-context"
        ),
        pytest.param(_unannotated, "'query' needs a type", id="unannotated"),
        pytest.param(_varargs, "'queries' must be a plain", id="varargs"),
        pytest.param(_varkw, "'options' must be a plain", id="varkw"),
        pytest.param(_positional_only, "'query' must be a plain", id="pos-only"),
        pytest.param(_unresolvable, "cannot be resolved", id="unresolvable"),
    ],
)
def test_a_malformed_signature_is_refused_where_it_is_declared(fn, message):
    with pytest.raises(TypeError, match=message):
        build_code_tool_spec(fn)


def test_string_annotations_are_resolved_to_real_types():
    async def postponed(ctx: "ToolContext", query: "str") -> "dict":
        """Search."""
        return {}

    spec = build_code_tool_spec(postponed)

    assert spec.annotations == {"query": str, "return": dict}


def test_a_tool_without_a_description_is_refused():
    async def undocumented(ctx: ToolContext, query: str) -> dict:
        return {}

    with pytest.raises(ValueError, match="needs a description"):
        build_code_tool_spec(undocumented)
    with pytest.raises(ValueError, match="needs a description"):
        build_code_tool_spec(recent_tickets, description="   ")


def test_the_description_is_the_cleaned_docstring_unless_given():
    async def documented(ctx: ToolContext, query: str) -> dict:
        """
        Find tickets.

        Use for support questions.
        """
        return {}

    assert (
        build_code_tool_spec(documented).description
        == "Find tickets.\n\nUse for support questions."
    )
    assert (
        build_code_tool_spec(documented, description="Search tickets.").description
        == "Search tickets."
    )


@pytest.mark.parametrize(
    "options, expected",
    [
        ({}, "recent-tickets"),
        ({"namespace": "support"}, "support-recent-tickets"),
        ({"name": "tickets"}, "tickets"),
        ({"name": "tickets", "namespace": "support"}, "support-tickets"),
    ],
)
def test_the_name_comes_from_the_function_unless_given(options, expected):
    assert build_code_tool_spec(recent_tickets, **options).name == expected


@pytest.mark.parametrize(
    "options, message",
    [
        ({"name": "Recent.Tickets"}, "is invalid"),
        # The namespace is applied before the name is checked.
        ({"namespace": "redisvl"}, "uses a reserved prefix"),
    ],
)
def test_a_name_the_profile_rules_refuse_is_refused_here_too(options, message):
    with pytest.raises(
        ValueError, match=f"code tool .*recent_tickets name .* {message}"
    ):
        build_code_tool_spec(recent_tickets, **options)


def test_a_blank_scope_is_refused():
    with pytest.raises(ValueError, match="scope must not be blank"):
        build_code_tool_spec(recent_tickets, scope=" ")


# --------------------------------------------------------------------------
# What the model sees
# --------------------------------------------------------------------------


async def test_the_model_sees_the_arguments_but_never_the_context(monkeypatch):
    # A real FastMCP server and client, so the call below goes through FastMCP's
    # own argument validation against the synthesised signature.
    _as_caller(monkeypatch)
    fake = FakeServer()
    mcp = fastmcp.FastMCP("code-tools")
    mcp.resolve_binding = fake.resolve_binding
    mcp.run_guarded = fake.run_guarded
    register_code_tool(mcp, build_code_tool_spec(recent_tickets), "tickets")

    async with fastmcp.Client(mcp) as client:
        (tool,) = await client.list_tools()
        called = await client.call_tool("recent-tickets", {"query": "refund"})
        refused = await client.call_tool(
            "recent-tickets", {"query": "refund", "ctx": {}}, raise_on_error=False
        )

    assert tool.name == "recent-tickets"
    assert tool.description == "Find recent support tickets matching a query."
    assert set(tool.inputSchema["properties"]) == {"query", "limit"}
    assert tool.inputSchema["required"] == ["query"]
    assert called.data == {"query": "refund", "limit": 5}
    assert refused.is_error


# --------------------------------------------------------------------------
# What wraps every call
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "writes, handle_type", [(False, IndexHandle), (True, WritableIndexHandle)]
)
async def test_the_tool_runs_with_a_context_for_its_binding(
    monkeypatch, writes, handle_type
):
    _as_caller(monkeypatch)
    seen: list[ToolContext] = []

    async def inspect_context(ctx: ToolContext, query: str) -> dict:
        """Search."""
        seen.append(ctx)
        return {"exists": await ctx.index.exists()}

    server = FakeServer()
    fn = _registered(inspect_context, server, writes=writes)

    assert await fn(query="refund") == {"exists": True}
    assert server.resolved == ["tickets"]
    assert type(seen[0].index) is handle_type


async def test_a_writing_tool_is_refused_on_an_index_now_read_only(monkeypatch):
    # Startup refuses the pairing, but tools register once and the binding is
    # re-read on every start, so the call itself has to check too.
    _as_caller(monkeypatch)
    called: list[str] = []

    async def write(ctx: ToolContext, ticket_id: str) -> dict:
        """Close a ticket."""
        called.append(ticket_id)
        return {}

    fn = _registered(write, FakeServer(read_only=True), writes=True)

    with pytest.raises(RedisVLMCPError) as exc_info:
        await fn(ticket_id="t1")

    assert exc_info.value.code == MCPErrorCode.FORBIDDEN
    assert called == []


@pytest.mark.parametrize(
    "writes, scopes",
    [
        pytest.param(False, ["kb.write"], id="read-without-read-scope"),
        pytest.param(True, ["kb.read"], id="write-without-write-scope"),
        pytest.param(False, ["kb.read"], id="read-without-tool-scope"),
        pytest.param(True, ["kb.write"], id="write-without-tool-scope"),
    ],
)
async def test_a_token_missing_a_required_scope_never_reaches_the_tool(
    monkeypatch, writes, scopes
):
    called: list[str] = []

    async def tool_fn(ctx: ToolContext, query: str) -> dict:
        """Search."""
        called.append(query)
        return {}

    server = FakeServer()
    server._auth_enabled = True
    _as_caller(monkeypatch, scopes)
    fn = _registered(tool_fn, server, writes=writes, scope="tickets.read")

    with pytest.raises(RedisVLMCPError) as exc_info:
        await fn(query="refund")

    assert exc_info.value.code == MCPErrorCode.FORBIDDEN
    assert called == []


@pytest.mark.parametrize(
    "writes, scopes",
    [(False, ["kb.read", "tickets.read"]), (True, ["kb.write", "tickets.read"])],
)
async def test_the_server_scope_and_the_tool_scope_together_admit_the_caller(
    monkeypatch, writes, scopes
):
    server = FakeServer()
    server._auth_enabled = True
    _as_caller(monkeypatch, scopes)
    fn = _registered(recent_tickets, server, writes=writes, scope="tickets.read")

    assert await fn(query="refund") == {"query": "refund", "limit": 5}


async def test_a_tool_that_overruns_the_binding_timeout_is_cut_off(monkeypatch):
    _as_caller(monkeypatch)

    async def slow(ctx: ToolContext, query: str) -> dict:
        """Search slowly."""
        await asyncio.sleep(5)
        return {}

    fn = _registered(slow, FakeServer(timeout=0.05))

    with pytest.raises(RedisVLMCPError) as exc_info:
        await fn(query="refund")

    assert exc_info.value.code == MCPErrorCode.BACKEND_UNAVAILABLE
    assert exc_info.value.retryable is True
    assert "tool 'slow' did not finish within 0.05s" in str(exc_info.value)


@pytest.mark.parametrize(
    "raised, code",
    [
        (ValueError("bad sku"), MCPErrorCode.INVALID_REQUEST),
        (RuntimeError("boom"), MCPErrorCode.INTERNAL_ERROR),
    ],
)
async def test_an_exception_from_the_tool_is_mapped(monkeypatch, raised, code):
    _as_caller(monkeypatch)

    async def failing(ctx: ToolContext, query: str) -> dict:
        """Search."""
        raise raised

    with pytest.raises(RedisVLMCPError) as exc_info:
        await _registered(failing)(query="refund")

    assert exc_info.value.code == code
    assert exc_info.value.__cause__ is raised


async def test_an_mcp_error_from_the_tool_passes_through_unchanged(monkeypatch):
    _as_caller(monkeypatch)
    error = RedisVLMCPError(
        "sku not found", code=MCPErrorCode.INVALID_REQUEST, retryable=False
    )

    async def failing(ctx: ToolContext, query: str) -> dict:
        """Search."""
        raise error

    with pytest.raises(RedisVLMCPError) as exc_info:
        await _registered(failing)(query="refund")

    assert exc_info.value is error


async def test_a_timed_out_call_gives_back_its_concurrency_slot(monkeypatch):
    # Against the real run_guarded: a tool cut off mid-call must release the
    # semaphore and the active-request count, or the cap leaks one slot per
    # timeout until the server stops answering.
    _as_caller(monkeypatch)

    class HangingIndex:
        async def exists(self):
            await asyncio.sleep(5)

    server = RedisVLMCPServer(MCPSettings(config="/tmp/mcp.yaml"))
    server._semaphore = asyncio.Semaphore(1)
    server._lifecycle_state = server._LifecycleState.RUNNING
    server.config = FakeServer().config
    fake = FakeServer(timeout=0.05)
    runtime = fake.resolve_binding("tickets")
    server._bindings = {"tickets": replace(runtime, index=HangingIndex())}
    registered: dict[str, Any] = {}
    monkeypatch.setattr(
        server,
        "tool",
        lambda name=None, description=None: lambda fn: registered.setdefault(name, fn),
    )

    async def probe(ctx: ToolContext) -> dict:
        """Probe the index."""
        return {"exists": await ctx.index.exists()}

    register_code_tool(server, build_code_tool_spec(probe), "tickets")

    with pytest.raises(RedisVLMCPError):
        await registered["probe"]()

    assert server._active_requests == 0
    assert server._semaphore.locked() is False
