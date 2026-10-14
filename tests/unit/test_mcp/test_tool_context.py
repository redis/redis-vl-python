"""The handle a code tool gets on Redis: an allow-list, every call guarded."""

import inspect
from types import SimpleNamespace

import pytest
from conftest import _schema

pytest.importorskip("fastmcp", reason="fastmcp not installed (install redisvl[mcp])")

from redisvl.index import AsyncSearchIndex
from redisvl.mcp.config import MCPConfig
from redisvl.mcp.context import IndexHandle, ToolContext, WritableIndexHandle
from redisvl.mcp.runtime import BindingRuntime
from redisvl.query.sql import SQLQuery

READS = {
    "query",
    "batch_query",
    "search",
    "batch_search",
    "aggregate",
    "fetch",
    "exists",
    "info",
}
WRITES = {"load", "drop_documents", "expire_documents"}
TIMEOUT = 7.0


class RecordingIndex:
    """Answers any method with a record of how it was called."""

    def __init__(self):
        self.awaited: list[str] = []

    def key(self, id):
        return f"doc:{id}"

    def __getattr__(self, name):
        async def method(*args, **kwargs):
            self.awaited.append(name)
            return (name, args, kwargs)

        return method


class FakeServer:
    def __init__(self):
        self.guarded: list[tuple[str, float]] = []

    async def run_guarded(self, operation_name, awaitable, *, timeout_seconds):
        self.guarded.append((operation_name, timeout_seconds))
        return await awaitable


class FakeVectorizer:
    async def aembed(self, text):
        return [len(text), 0.0, 0.0]


def _runtime(vectorizer=None) -> BindingRuntime:
    config = MCPConfig.model_validate(
        {
            "server": {"redis_url": "redis://localhost:6379"},
            "indexes": {
                "knowledge": {
                    "redis_name": "docs-index",
                    "search": {"type": "fulltext"},
                    "runtime": {
                        "text_field_name": "content",
                        "request_timeout_seconds": TIMEOUT,
                    },
                }
            },
        }
    )
    return BindingRuntime(
        binding_id="knowledge",
        binding=config.indexes["knowledge"],
        index=RecordingIndex(),
        schema=_schema(),
        vectorizer=vectorizer,
        supports_native_hybrid_search=False,
        effective_read_only=False,
    )


def _as_caller(monkeypatch, token):
    monkeypatch.setattr(
        "fastmcp.server.dependencies.get_access_token", lambda: token, raising=False
    )


def _public(cls) -> set[str]:
    return {name for name in dir(cls) if not name.startswith("_")}


def test_the_read_handle_exposes_exactly_the_read_methods():
    # Pinned exactly, so a method added to the handle is a deliberate,
    # reviewed change rather than a quiet widening.
    assert _public(IndexHandle) == READS


def test_the_writable_handle_adds_exactly_the_bounded_writes():
    assert _public(WritableIndexHandle) == READS | WRITES


QUERY = object()

# (handle method, its arguments, the index method it reaches, what it forwards)
CALLS = [
    ("query", (QUERY,), {}, "query", ((QUERY,), {})),
    (
        "batch_query",
        ([QUERY],),
        {"batch_size": 3},
        "batch_query",
        (([QUERY],), {"batch_size": 3}),
    ),
    (
        "search",
        ("*",),
        {"query_params": {}},
        "search",
        (("*",), {"query_params": {}}),
    ),
    ("batch_search", ([QUERY],), {}, "batch_search", (([QUERY],), {"batch_size": 10})),
    ("aggregate", (QUERY,), {}, "aggregate", ((QUERY,), {})),
    ("fetch", ("a1",), {}, "fetch", (("a1",), {})),
    ("exists", (), {}, "exists", ((), {})),
    ("info", (), {}, "info", ((), {})),
    (
        "load",
        ([{"id": "a1"}],),
        {"id_field": "id"},
        "load",
        (
            ([{"id": "a1"}],),
            {"id_field": "id", "ttl": None, "preprocess": None, "batch_size": None},
        ),
    ),
    (
        "drop_documents",
        (["a1"],),
        {},
        "drop_documents",
        ((["a1"],), {"batch_size": 500}),
    ),
    # Addressed by id, so the key is built under the index's own prefix.
    (
        "expire_documents",
        (["a1", "a2"], 60),
        {},
        "expire_keys",
        ((["doc:a1", "doc:a2"], 60), {}),
    ),
    ("expire_documents", ("a1", 60), {}, "expire_keys", (("doc:a1", 60), {})),
]
CALL_IDS = [f"{call[0]}-{n}" for n, call in enumerate(CALLS)]


@pytest.mark.parametrize("method, args, kwargs, target, forwarded", CALLS, ids=CALL_IDS)
async def test_every_call_reaches_the_index_through_the_guard(
    method, args, kwargs, target, forwarded
):
    server = FakeServer()
    handle = WritableIndexHandle(server, RecordingIndex(), timeout_seconds=TIMEOUT)

    result = await getattr(handle, method)(*args, **kwargs)

    assert result == (target, *forwarded)
    assert server.guarded == [(method, TIMEOUT)]


@pytest.mark.parametrize("method, args, kwargs, target, forwarded", CALLS, ids=CALL_IDS)
def test_every_forwarded_call_fits_the_real_index_signature(
    method, args, kwargs, target, forwarded
):
    # The recording index accepts anything, so check what the handle forwards
    # against the method it stands in for.
    forwarded_args, forwarded_kwargs = forwarded
    signature = inspect.signature(getattr(AsyncSearchIndex, target))

    signature.bind(None, *forwarded_args, **forwarded_kwargs)


def test_the_covered_calls_are_the_whole_surface():
    assert {call[0] for call in CALLS} == READS | WRITES


async def test_a_guard_refusal_reaches_the_tool_without_touching_the_index():
    class StoppedServer:
        async def run_guarded(self, operation_name, awaitable, *, timeout_seconds):
            awaitable.close()
            raise RuntimeError("MCP server is not running")

    index = RecordingIndex()
    handle = IndexHandle(StoppedServer(), index, timeout_seconds=TIMEOUT)

    with pytest.raises(RuntimeError, match="not running"):
        await handle.fetch("a1")
    assert index.awaited == []


@pytest.mark.parametrize("method", ["query", "batch_query"])
async def test_a_sql_query_is_refused_because_it_names_its_own_index(method):
    sql = SQLQuery("SELECT * FROM tenant_docs")
    index = RecordingIndex()
    handle = IndexHandle(FakeServer(), index, timeout_seconds=TIMEOUT)

    with pytest.raises(ValueError, match="cannot run a SQLQuery"):
        await getattr(handle, method)(sql if method == "query" else [QUERY, sql])
    assert index.awaited == []


async def test_info_cannot_name_another_index():
    handle = IndexHandle(FakeServer(), RecordingIndex(), timeout_seconds=TIMEOUT)

    with pytest.raises(TypeError):
        await handle.info("other-index")


def test_a_handle_offers_no_client_and_takes_no_new_attributes():
    handle = WritableIndexHandle(FakeServer(), RecordingIndex(), timeout_seconds=1)

    with pytest.raises(AttributeError):
        handle.client
    with pytest.raises(AttributeError):
        handle.client = object()


@pytest.mark.parametrize(
    "writes, handle_type", [(False, IndexHandle), (True, WritableIndexHandle)]
)
def test_only_a_writing_tool_gets_the_writable_handle(monkeypatch, writes, handle_type):
    _as_caller(monkeypatch, None)

    ctx = ToolContext.for_request(FakeServer(), _runtime(), writes=writes)

    assert type(ctx.index) is handle_type


async def test_the_context_bounds_calls_by_the_bindings_timeout(monkeypatch):
    _as_caller(monkeypatch, None)
    server = FakeServer()
    runtime = _runtime()

    ctx = ToolContext.for_request(server, runtime, writes=False)
    await ctx.index.exists()

    assert server.guarded == [("exists", TIMEOUT)]


def test_the_context_schema_is_a_copy_a_tool_cannot_retarget_the_index_with(
    monkeypatch,
):
    _as_caller(monkeypatch, None)
    runtime = _runtime()

    ctx = ToolContext.for_request(FakeServer(), runtime, writes=False)
    ctx.schema.index.name = "tenant-index"

    assert runtime.schema.index.name == "docs-index"


def test_the_context_is_read_only(monkeypatch):
    _as_caller(monkeypatch, None)
    ctx = ToolContext.for_request(FakeServer(), _runtime(), writes=True)

    with pytest.raises(AttributeError):
        ctx.index = WritableIndexHandle(
            FakeServer(), RecordingIndex(), timeout_seconds=1
        )
    with pytest.raises(AttributeError):
        ctx.anything = object()


def test_a_tokenless_request_has_no_claims(monkeypatch):
    _as_caller(monkeypatch, None)

    assert (
        ToolContext.for_request(FakeServer(), _runtime(), writes=False).claims is None
    )


def test_a_token_without_claims_still_reads_as_a_token(monkeypatch):
    _as_caller(monkeypatch, SimpleNamespace(claims=None))

    claims = ToolContext.for_request(FakeServer(), _runtime(), writes=False).claims

    assert claims is not None and dict(claims) == {}


def test_claims_are_a_read_only_copy_all_the_way_down(monkeypatch):
    claims = {"sub": "user-42", "org": {"id": "acme", "regions": ["eu", "us"]}}
    _as_caller(monkeypatch, SimpleNamespace(claims=claims))

    seen = ToolContext.for_request(FakeServer(), _runtime(), writes=False).claims
    claims["org"]["id"] = "victim"

    assert seen["org"]["id"] == "acme"
    assert seen["org"]["regions"] == ("eu", "us")
    with pytest.raises(TypeError):
        seen["sub"] = "someone-else"
    with pytest.raises(TypeError):
        seen["org"]["id"] = "victim"


async def test_embed_uses_the_bindings_vectorizer(monkeypatch):
    _as_caller(monkeypatch, None)
    ctx = ToolContext.for_request(
        FakeServer(), _runtime(FakeVectorizer()), writes=False
    )

    assert await ctx.embed("refund") == [6, 0.0, 0.0]


async def test_embed_without_a_vectorizer_fails_clearly(monkeypatch):
    _as_caller(monkeypatch, None)
    ctx = ToolContext.for_request(FakeServer(), _runtime(), writes=False)

    with pytest.raises(RuntimeError, match="vectorizer is not configured"):
        await ctx.embed("refund")
