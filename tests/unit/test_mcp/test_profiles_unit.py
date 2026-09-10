import inspect
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Optional, get_args

import pytest
from conftest import _schema

from redisvl.mcp.config import _PROFILE_PARAM_NAMES, MCPConfig, MCPCustomToolConfig
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.filters import parse_filter
from redisvl.mcp.runtime import BindingRuntime
from redisvl.mcp.tools.profiles import (
    _PROFILE_PARAM_SPECS,
    build_profile_description,
    register_profile_tool,
    register_profile_tools,
    resolve_locked_filter,
    validate_profile_against_schema,
)
from redisvl.mcp.tools.search import merge_locked_filter

LOCKED_CATEGORY_FILTER = {"field": "category", "op": "eq", "value": "resolved"}
FILTER_HINT = "Object filter fields: content(text), category(tag), rating(numeric)."
RETURN_FIELDS_HINT = "Allowed return_fields: content, category, rating."


def _binding(search_type: str = "fulltext") -> dict[str, Any]:
    # Full-text by default so the profile path needs no vectorizer; the built-in
    # query class is monkeypatched at its import site to capture what the profile
    # builds.
    #
    # The limits are deliberately larger than, and distinct from, every
    # per-profile cap these tests declare. A default equal to a cap would let a
    # broken cap look correct, and a max_limit close to the default would hide
    # which of the two bounded a result count.
    runtime: dict[str, Any] = {
        "text_field_name": "content",
        "default_limit": 10,
        "max_limit": 20,
    }
    binding: dict[str, Any] = {
        "redis_name": "docs-index",
        "search": {"type": search_type},
        "runtime": runtime,
    }
    if search_type != "fulltext":
        # Vector and hybrid modes embed the query text, so those bindings need a
        # vector field and a vectorizer.
        runtime["vector_field_name"] = "embedding"
        runtime["default_embed_text_field"] = "content"
        binding["vectorizer"] = {"class": "FakeVectorizer", "model": "test-model"}
    return binding


def _config_with_profiles(
    *profiles: dict[str, Any],
    index_ids: tuple[str, ...] = ("knowledge",),
    search_type: str = "fulltext",
) -> MCPConfig:
    return MCPConfig.model_validate(
        {
            "server": {"redis_url": "redis://localhost:6379"},
            "indexes": {
                index_id: deepcopy(_binding(search_type)) for index_id in index_ids
            },
            "custom_tools": [deepcopy(profile) for profile in profiles],
        }
    )


def _profile(**overrides: Any) -> MCPCustomToolConfig:
    """Build one validated profile through the real config model."""
    profile: dict[str, Any] = {
        "name": "resolved-search",
        "description": "Search resolved records.",
    }
    profile.update(overrides)
    return _config_with_profiles(profile).custom_tools[0]


class FakeVectorizer:
    async def embed(self, text: str):
        return [0.1, 0.2, 0.3]


class FakeIndex:
    def __init__(self):
        self.schema = _schema()
        self.query_calls = []

    async def query(self, query):
        self.query_calls.append(query)
        return []


class FakeServer:
    def __init__(
        self,
        *profiles: dict[str, Any],
        index_ids: tuple[str, ...] = ("knowledge",),
        search_type: str = "fulltext",
    ):
        self.config = _config_with_profiles(
            *profiles, index_ids=index_ids, search_type=search_type
        )
        self.mcp_settings = SimpleNamespace(tool_search_description=None)
        self.indexes = {index_id: FakeIndex() for index_id in index_ids}
        self.vectorizer = None if search_type == "fulltext" else FakeVectorizer()
        self.native_hybrid_supported = False
        self.registered_tools: list[dict[str, Any]] = []
        self.resolved_index_ids: list[str | None] = []
        # A read scope is always declared so the wrapper's scope gate has
        # something to check. `_auth_enabled` stays off by default, mirroring the
        # unauthenticated stdio transport; a test turns it on to arm the gate.
        self.auth_config = SimpleNamespace(read_scope="kb.search.read")
        self._auth_enabled = False

    def resolve_binding(self, index_id=None):
        self.resolved_index_ids.append(index_id)
        resolved = next(iter(self.indexes)) if index_id is None else index_id
        if resolved not in self.indexes:
            raise RedisVLMCPError(
                f"Unknown index '{resolved}'; available: {', '.join(self.indexes)}",
                code=MCPErrorCode.INVALID_REQUEST,
                retryable=False,
            )
        index = self.indexes[resolved]
        return BindingRuntime(
            binding_id=resolved,
            binding=self.config.indexes[resolved],
            index=index,
            schema=index.schema,
            vectorizer=self.vectorizer,
            supports_native_hybrid_search=self.native_hybrid_supported,
            effective_read_only=False,
        )

    async def run_guarded(self, operation_name, awaitable, *, timeout_seconds=None):
        return await awaitable

    async def supports_native_hybrid_search(self):
        return self.native_hybrid_supported

    def tool(self, name=None, description=None, **kwargs):
        def decorator(fn):
            self.registered_tools.append(
                {"name": name, "description": description, "fn": fn}
            )
            return fn

        return decorator


def _capture_text_queries(monkeypatch) -> list[dict[str, Any]]:
    """Record the kwargs of every TextQuery the search tool builds."""
    built: list[dict[str, Any]] = []

    class FakeTextQuery:
        def __init__(self, **kwargs):
            built.append(kwargs)

    monkeypatch.setattr("redisvl.mcp.tools.search.TextQuery", FakeTextQuery)
    return built


def _capture_any_query(monkeypatch) -> list[tuple[str, dict[str, Any]]]:
    """Record (mode, kwargs) for every query class `_build_query` can construct.

    All four are patched at once so the recorded mode proves which construction
    branch ran, rather than the test having to trust the binding config.
    """
    built: list[tuple[str, dict[str, Any]]] = []

    def _fake_query_class(mode: str):
        class FakeQuery:
            def __init__(self, **kwargs):
                built.append((mode, kwargs))
                # Only the native hybrid branch touches this, but giving every
                # fake the attribute keeps the classes interchangeable.
                self.postprocessing_config = SimpleNamespace(apply=lambda **_: None)

        return FakeQuery

    for mode, attribute in (
        ("vector", "VectorQuery"),
        ("fulltext", "TextQuery"),
        ("hybrid-native", "HybridQuery"),
        ("hybrid-fallback", "AggregateHybridQuery"),
    ):
        monkeypatch.setattr(
            f"redisvl.mcp.tools.search.{attribute}", _fake_query_class(mode)
        )
    return built


def _registered_fns(server: FakeServer) -> dict[str, Any]:
    """Map every registered tool name to its callable."""
    return {tool["name"]: tool["fn"] for tool in server.registered_tools}


def _register(server: FakeServer, profile: MCPCustomToolConfig, binding_id="knowledge"):
    register_profile_tool(server, profile, binding_id, _schema())
    return server.registered_tools[0]["fn"]


# --------------------------------------------------------------------------
# Registration and the generated signature
# --------------------------------------------------------------------------


def test_profile_param_specs_cover_exactly_the_params_config_accepts():
    # Two independent sources of truth for the same argument list: config
    # validation rejects `params` keys outside _PROFILE_PARAM_NAMES, while the
    # wrapper's signature is generated from _PROFILE_PARAM_SPECS. Divergence is
    # silent in both directions -- an argument accepted in config but never
    # rendered into the signature is quietly ignored, and one rendered but not
    # allowed cannot be configured at all -- so assert they agree.
    assert {name for name, _, _ in _PROFILE_PARAM_SPECS} == _PROFILE_PARAM_NAMES


def test_register_profile_tool_registers_under_the_configured_name():
    profile_config = {"name": "resolved-search", "description": "Search resolved."}
    server = FakeServer(profile_config)

    register_profile_tool(server, server.config.custom_tools[0], "knowledge", _schema())

    assert server.registered_tools[0]["name"] == "resolved-search"
    # FastMCP reads the callable's __name__ in places; hyphens are not valid in a
    # Python identifier, so the wrapper sanitizes it while the tool name keeps them.
    assert server.registered_tools[0]["fn"].__name__ == "resolved_search"
    assert server.registered_tools[0]["fn"].__doc__ == "Search resolved."


def test_register_profile_tool_exposes_every_search_param_by_default():
    server = FakeServer()
    fn = _register(server, _profile())

    signature = inspect.signature(fn)

    # Order matches the built-in's: required query first, then narrowing args.
    assert list(signature.parameters) == [
        "query",
        "limit",
        "offset",
        "filter",
        "return_fields",
    ]
    # Keyword-only so the generated schema is unambiguous for the model.
    assert all(
        param.kind is inspect.Parameter.KEYWORD_ONLY
        for param in signature.parameters.values()
    )
    # A profile pins its index at registration, so the model never routes.
    assert "index" not in signature.parameters


def test_register_profile_tool_annotates_filter_as_an_object_never_a_string():
    server = FakeServer()
    fn = _register(server, _profile())

    annotation = fn.__annotations__["filter"]

    # A raw string filter bypasses the DSL's field validation and cannot be
    # safely combined with a locked filter, so `str` is never advertised as an
    # accepted filter type -- the schema refuses it before any runtime guard.
    assert annotation == Optional[dict[str, Any]]
    assert str not in get_args(annotation)
    # The signature FastMCP derives the schema from must agree.
    assert inspect.signature(fn).parameters["filter"].annotation == annotation


def test_register_profile_tool_hides_params_the_author_does_not_expose():
    server = FakeServer()
    fn = _register(
        server,
        _profile(
            params={
                "offset": {"expose": False},
                "filter": {"expose": False},
                "return_fields": {"expose": False},
            }
        ),
    )

    signature = inspect.signature(fn)

    assert list(signature.parameters) == ["query", "limit"]
    # A hidden argument is absent from the annotations too, which is what
    # FastMCP derives the advertised schema from.
    assert set(fn.__annotations__) == {"query", "limit", "return"}


def test_register_profile_tool_omits_locked_return_fields_from_the_signature():
    server = FakeServer()
    fn = _register(server, _profile(lock={"return_fields": ["content"]}))

    signature = inspect.signature(fn)

    # Locking a projection implies the model cannot choose one, so the argument
    # is not offered rather than merely overridden.
    assert "return_fields" not in signature.parameters
    assert "return_fields" not in fn.__annotations__


def test_register_profile_tool_keeps_filter_exposed_when_a_filter_is_locked():
    server = FakeServer()
    fn = _register(server, _profile(lock={"filter": LOCKED_CATEGORY_FILTER}))

    # Unlike return_fields, a locked filter is the narrowing case: the caller may
    # still add a filter, which AND-combines with the locked one.
    assert "filter" in inspect.signature(fn).parameters


# --------------------------------------------------------------------------
# Locked filter merging -- the security-critical behavior
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_profile_tool_applies_locked_filter_when_caller_supplies_none(
    monkeypatch,
):
    built = _capture_text_queries(monkeypatch)
    server = FakeServer()
    fn = _register(server, _profile(lock={"filter": LOCKED_CATEGORY_FILTER}))

    await fn(query="jam")

    assert str(built[0]["filter_expression"]) == "@category:{resolved}"


@pytest.mark.asyncio
async def test_profile_tool_and_combines_locked_filter_with_caller_filter(monkeypatch):
    built = _capture_text_queries(monkeypatch)
    server = FakeServer()
    fn = _register(server, _profile(lock={"filter": LOCKED_CATEGORY_FILTER}))

    await fn(query="jam", filter={"field": "rating", "op": "gte", "value": 4})

    assert (
        str(built[0]["filter_expression"]) == "(@category:{resolved} @rating:[4 +inf])"
    )


@pytest.mark.asyncio
async def test_profile_tool_keeps_caller_or_nested_inside_the_locked_filter(
    monkeypatch,
):
    built = _capture_text_queries(monkeypatch)
    server = FakeServer()
    fn = _register(server, _profile(lock={"filter": LOCKED_CATEGORY_FILTER}))

    await fn(
        query="jam",
        filter={
            "or": [
                {"field": "rating", "op": "gte", "value": 4},
                {"field": "content", "op": "like", "value": "jam*"},
            ]
        },
    )

    # The key scoping guarantee: both sides are fully parenthesized, so a
    # caller's `or` cannot be hoisted to the top level and widen past the lock.
    assert str(built[0]["filter_expression"]) == (
        "(@category:{resolved} (@rating:[4 +inf] | @content:(jam*)))"
    )


@pytest.mark.asyncio
async def test_profile_tool_fails_closed_when_caller_contradicts_the_locked_field(
    monkeypatch,
):
    built = _capture_text_queries(monkeypatch)
    server = FakeServer()
    fn = _register(server, _profile(lock={"filter": LOCKED_CATEGORY_FILTER}))

    await fn(query="jam", filter={"field": "category", "op": "eq", "value": "open"})

    # Both tag clauses are ANDed, which is unsatisfiable rather than a silent
    # override -- the caller cannot swap the locked category for their own.
    assert (
        str(built[0]["filter_expression"]) == "(@category:{resolved} @category:{open})"
    )


@pytest.mark.asyncio
async def test_profile_tool_passes_caller_filter_through_when_nothing_is_locked(
    monkeypatch,
):
    built = _capture_text_queries(monkeypatch)
    server = FakeServer()
    fn = _register(server, _profile())

    await fn(query="jam", filter={"field": "rating", "op": "gte", "value": 4})

    assert str(built[0]["filter_expression"]) == "@rating:[4 +inf]"


def test_merge_locked_filter_rejects_a_raw_string_caller_filter_against_a_lock():
    locked = parse_filter(LOCKED_CATEGORY_FILTER, _schema())

    with pytest.raises(RedisVLMCPError) as exc_info:
        merge_locked_filter(locked, "@category:{open}")

    # A string has no safe composition with an expression: concatenating could
    # close the locked group and escape the scope entirely.
    assert exc_info.value.code == MCPErrorCode.INVALID_FILTER
    assert exc_info.value.retryable is False


# --------------------------------------------------------------------------
# Locked return_fields
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_profile_tool_sends_locked_return_fields_to_the_query(monkeypatch):
    built = _capture_text_queries(monkeypatch)
    server = FakeServer()
    fn = _register(server, _profile(lock={"return_fields": ["content", "category"]}))

    await fn(query="jam")

    assert built[0]["return_fields"] == ["content", "category"]


@pytest.mark.asyncio
async def test_profile_tool_ignores_caller_return_fields_when_locked(monkeypatch):
    built = _capture_text_queries(monkeypatch)
    server = FakeServer()
    fn = _register(server, _profile(lock={"return_fields": ["content"]}))

    # The argument is absent from the advertised schema, but a client that sends
    # it anyway must still not widen the projection.
    await fn(query="jam", return_fields=["category", "rating"])

    assert built[0]["return_fields"] == ["content"]


@pytest.mark.asyncio
async def test_profile_tool_forwards_caller_return_fields_when_not_locked(monkeypatch):
    built = _capture_text_queries(monkeypatch)
    server = FakeServer()
    fn = _register(server, _profile())

    await fn(query="jam", return_fields=["category"])

    assert built[0]["return_fields"] == ["category"]


# --------------------------------------------------------------------------
# Limit policy
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_profile_tool_rejects_a_limit_above_the_declared_cap(monkeypatch):
    _capture_text_queries(monkeypatch)
    server = FakeServer()
    fn = _register(server, _profile(params={"limit": {"max": 3}}))

    with pytest.raises(
        RedisVLMCPError, match="limit must be less than or equal to 3"
    ) as exc_info:
        await fn(query="jam", limit=4)

    assert exc_info.value.code == MCPErrorCode.INVALID_REQUEST


@pytest.mark.asyncio
async def test_profile_tool_allows_a_limit_at_the_declared_cap(monkeypatch):
    built = _capture_text_queries(monkeypatch)
    server = FakeServer()
    fn = _register(server, _profile(params={"limit": {"max": 3}}))

    await fn(query="jam", limit=3)

    assert built[0]["num_results"] == 3


@pytest.mark.asyncio
async def test_profile_tool_uses_the_cap_as_a_fixed_limit_when_limit_is_hidden(
    monkeypatch,
):
    built = _capture_text_queries(monkeypatch)
    server = FakeServer()
    fn = _register(server, _profile(params={"limit": {"expose": False, "max": 3}}))

    assert "limit" not in inspect.signature(fn).parameters

    await fn(query="jam")

    # With the argument hidden the declared cap becomes the fixed result count.
    assert built[0]["num_results"] == 3


@pytest.mark.asyncio
async def test_profile_tool_falls_back_to_binding_default_when_limit_is_hidden_uncapped(
    monkeypatch,
):
    built = _capture_text_queries(monkeypatch)
    server = FakeServer()
    fn = _register(server, _profile(params={"limit": {"expose": False}}))

    await fn(query="jam")

    # No cap declared, so the binding's runtime.default_limit applies.
    assert built[0]["num_results"] == 10


@pytest.mark.asyncio
async def test_profile_tool_adds_offset_to_the_requested_result_window(monkeypatch):
    built = _capture_text_queries(monkeypatch)
    server = FakeServer()
    fn = _register(server, _profile())

    response = await fn(query="jam", limit=2, offset=1)

    assert built[0]["num_results"] == 3
    assert response["offset"] == 1
    assert response["limit"] == 2


# --------------------------------------------------------------------------
# Description building
# --------------------------------------------------------------------------


def test_build_profile_description_appends_both_hints_when_both_args_are_exposed():
    description = build_profile_description(_profile(), _schema())

    assert description == f"Search resolved records. {FILTER_HINT} {RETURN_FIELDS_HINT}"


def test_build_profile_description_returns_verbatim_when_hints_are_suppressed():
    profile = _profile(
        suppress_schema_hints=True, description="  Search resolved records.  "
    )

    description = build_profile_description(profile, _schema())

    # Suppression is total, but the authored text is still normalized.
    assert description == "Search resolved records."


def test_build_profile_description_omits_the_return_fields_hint_when_locked():
    profile = _profile(lock={"return_fields": ["content"]})

    description = build_profile_description(profile, _schema())

    # Enumerating returnable fields is misleading once the projection is frozen.
    assert description == f"Search resolved records. {FILTER_HINT}"
    assert RETURN_FIELDS_HINT not in description


def test_build_profile_description_omits_the_filter_hint_when_filter_is_hidden():
    profile = _profile(params={"filter": {"expose": False}})

    description = build_profile_description(profile, _schema())

    assert description == f"Search resolved records. {RETURN_FIELDS_HINT}"
    assert "Object filter fields" not in description


def test_register_profile_tool_advertises_the_built_description():
    server = FakeServer()
    profile = _profile(lock={"return_fields": ["content"]})

    register_profile_tool(server, profile, "knowledge", _schema())

    assert server.registered_tools[0]["description"] == (
        f"Search resolved records. {FILTER_HINT}"
    )


# --------------------------------------------------------------------------
# Startup validation against the bound schema
# --------------------------------------------------------------------------


def test_validate_profile_against_schema_accepts_fields_the_index_has():
    validate_profile_against_schema(
        _profile(
            lock={
                "return_fields": ["content", "rating"],
                "filter": LOCKED_CATEGORY_FILTER,
            }
        ),
        _schema(),
    )


def test_validate_profile_against_schema_rejects_unknown_locked_return_field():
    profile = _profile(lock={"return_fields": ["missing"]})

    with pytest.raises(ValueError, match="references unknown field"):
        validate_profile_against_schema(profile, _schema())


def test_validate_profile_against_schema_rejects_locked_vector_return_field():
    profile = _profile(lock={"return_fields": ["embedding"]})

    with pytest.raises(ValueError, match="cannot return vector field"):
        validate_profile_against_schema(profile, _schema())


def test_validate_profile_against_schema_rejects_locked_filter_on_unknown_field():
    profile = _profile(
        lock={"filter": {"field": "missing", "op": "eq", "value": "resolved"}}
    )

    with pytest.raises(RedisVLMCPError) as exc_info:
        validate_profile_against_schema(profile, _schema())

    # Caught at startup rather than silently matching nothing per request.
    assert exc_info.value.code == MCPErrorCode.INVALID_FILTER


def test_resolve_locked_filter_returns_none_when_no_filter_is_locked():
    assert resolve_locked_filter(_profile(), _schema()) is None


def test_resolve_locked_filter_parses_the_locked_dsl_object():
    resolved = resolve_locked_filter(
        _profile(lock={"filter": LOCKED_CATEGORY_FILTER}), _schema()
    )

    assert str(resolved) == "@category:{resolved}"


# --------------------------------------------------------------------------
# register_profile_tools
# --------------------------------------------------------------------------


def test_register_profile_tools_returns_nothing_when_no_profiles_are_configured():
    server = FakeServer()

    assert register_profile_tools(server) == []
    assert server.registered_tools == []


def test_register_profile_tools_returns_nothing_when_the_server_has_no_config():
    assert register_profile_tools(SimpleNamespace()) == []


def test_register_profile_tools_registers_every_configured_profile():
    server = FakeServer(
        {"name": "resolved-search", "description": "Search resolved."},
        {"name": "open-search", "description": "Search open."},
    )

    registered = register_profile_tools(server)

    assert registered == ["resolved-search", "open-search"]
    assert [tool["name"] for tool in server.registered_tools] == [
        "resolved-search",
        "open-search",
    ]


def test_register_profile_tools_pins_each_profile_to_its_configured_binding():
    server = FakeServer(
        {
            "name": "tickets-search",
            "description": "Search tickets.",
            "index": "tickets",
        },
        index_ids=("knowledge", "tickets"),
    )

    register_profile_tools(server)

    # The binding is resolved once at registration and frozen into the wrapper.
    assert server.resolved_index_ids == ["tickets"]


# --------------------------------------------------------------------------
# Closure isolation between profiles registered on the same server
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_profile_tools_gives_each_profile_its_own_locked_filter(
    monkeypatch,
):
    built = _capture_text_queries(monkeypatch)
    server = FakeServer(
        {
            "name": "a-search",
            "description": "Search science.",
            "lock": {"filter": {"field": "category", "op": "eq", "value": "science"}},
        },
        {
            "name": "b-search",
            "description": "Search health.",
            "lock": {"filter": {"field": "category", "op": "eq", "value": "health"}},
        },
    )

    register_profile_tools(server)
    fns = _registered_fns(server)
    await fns["a-search"](query="jam")
    await fns["b-search"](query="jam")

    # Each wrapper must close over the lock parsed for *its* profile. If the
    # registration loop ever hoisted the parsed expression into a variable shared
    # across iterations, every tool would carry whichever profile was registered
    # last -- and the only visible symptom would be cross-tenant leakage.
    assert str(built[0]["filter_expression"]) == "@category:{science}"
    assert str(built[1]["filter_expression"]) == "@category:{health}"
    # Stated the other way round, so a merge that accumulated both locks would
    # also fail rather than pass the equality above by containing extra clauses.
    assert "health" not in str(built[0]["filter_expression"])
    assert "science" not in str(built[1]["filter_expression"])


@pytest.mark.asyncio
async def test_register_profile_tools_isolates_locked_filters_across_bindings(
    monkeypatch,
):
    built = _capture_text_queries(monkeypatch)
    server = FakeServer(
        {
            "name": "a-search",
            "description": "Search knowledge.",
            "index": "knowledge",
            "lock": {"filter": {"field": "category", "op": "eq", "value": "science"}},
        },
        {
            "name": "b-search",
            "description": "Search tickets.",
            "index": "tickets",
            "lock": {"filter": {"field": "category", "op": "eq", "value": "health"}},
        },
        index_ids=("knowledge", "tickets"),
    )

    register_profile_tools(server)
    fns = _registered_fns(server)
    a_response = await fns["a-search"](query="jam")
    b_response = await fns["b-search"](query="jam")

    # The pinned binding and the locked filter are captured by the same closure,
    # so they have to stay paired: a tool routed to one index while carrying the
    # other index's lock is the exact multi-tenant failure this guards.
    assert a_response["index"] == "knowledge"
    assert b_response["index"] == "tickets"
    assert str(built[0]["filter_expression"]) == "@category:{science}"
    assert str(built[1]["filter_expression"]) == "@category:{health}"
    assert len(server.indexes["knowledge"].query_calls) == 1
    assert len(server.indexes["tickets"].query_calls) == 1


# --------------------------------------------------------------------------
# Hidden arguments are ignored, not merely unadvertised
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registered_profile_tool_ignores_a_hidden_filter_argument(monkeypatch):
    built = _capture_text_queries(monkeypatch)
    server = FakeServer()
    fn = _register(
        server,
        _profile(
            lock={"filter": {"field": "category", "op": "eq", "value": "resolved"}},
            params={"filter": {"expose": False}},
        ),
    )

    # A compliant client cannot send `filter` at all, since it is absent from the
    # advertised schema. Ignoring it here as well means the lock does not depend
    # on the client or the schema layer holding up.
    await fn(query="jam", filter={"field": "category", "op": "eq", "value": "open"})

    assert str(built[0]["filter_expression"]) == "@category:{resolved}"


@pytest.mark.asyncio
async def test_registered_profile_tool_ignores_a_hidden_offset_argument(monkeypatch):
    _capture_text_queries(monkeypatch)
    server = FakeServer()
    fn = _register(server, _profile(params={"offset": {"expose": False}}))

    response = await fn(query="jam", offset=25)

    assert response["offset"] == 0


@pytest.mark.asyncio
async def test_registered_profile_tool_ignores_a_hidden_return_fields_argument(
    monkeypatch,
):
    built = _capture_text_queries(monkeypatch)
    server = FakeServer()
    fn = _register(server, _profile(params={"return_fields": {"expose": False}}))

    await fn(query="jam", return_fields=["category"])

    # With no lock to substitute, a hidden projection falls back to the
    # binding's default rather than honoring the caller.
    assert built[0]["return_fields"] == ["content", "category", "rating"]


@pytest.mark.asyncio
async def test_registered_profile_tool_caps_an_omitted_limit(monkeypatch):
    built = _capture_text_queries(monkeypatch)
    server = FakeServer()
    fn = _register(server, _profile(params={"limit": {"expose": True, "max": 1}}))

    response = await fn(query="jam")

    # The cap has to bound the binding default too. Capping only an explicitly
    # supplied limit would let the common case -- the model omitting it -- sail
    # straight past the declared ceiling.
    assert response["limit"] == 1
    assert built[0]["num_results"] == 1


@pytest.mark.asyncio
async def test_registered_profile_tool_rejects_a_caller_filter_that_could_escape(
    monkeypatch,
):
    _capture_text_queries(monkeypatch)
    server = FakeServer()
    fn = _register(
        server,
        _profile(
            lock={"filter": {"field": "category", "op": "eq", "value": "resolved"}},
            params={"filter": {"expose": True}},
        ),
    )

    # Text values are escaped at the DSL boundary, so this payload is a literal
    # rather than syntax. Assert the end state a caller can observe: the query
    # still runs and nothing widened the locked scope.
    response = await fn(
        query="jam",
        filter={
            "field": "content",
            "op": "like",
            "value": "zzz) | (-@category:{resolved}",
        },
    )

    assert response["results"] == []


# --------------------------------------------------------------------------
# The wrapper's auth scope gate
# --------------------------------------------------------------------------


def _patch_access_token(monkeypatch, scopes: list[str]) -> None:
    """Make the current request look authenticated with the given scopes."""
    pytest.importorskip(
        "fastmcp", reason="fastmcp not installed (install redisvl[mcp])"
    )
    monkeypatch.setattr(
        "fastmcp.server.dependencies.get_access_token",
        lambda: SimpleNamespace(scopes=scopes, claims={}),
        raising=False,
    )


@pytest.mark.asyncio
async def test_profile_tool_forbids_a_token_missing_the_configured_read_scope(
    monkeypatch,
):
    _capture_text_queries(monkeypatch)
    server = FakeServer()
    server._auth_enabled = True
    _patch_access_token(monkeypatch, ["kb.search.write"])
    fn = _register(server, _profile(lock={"filter": LOCKED_CATEGORY_FILTER}))

    with pytest.raises(RedisVLMCPError) as exc_info:
        await fn(query="jam")

    assert exc_info.value.code == MCPErrorCode.FORBIDDEN
    assert exc_info.value.retryable is False
    # A profile inherits the built-in's scoping rather than restating it, so the
    # gate has to fire before any binding work. Reaching resolve_binding would
    # mean an unauthorized call had already selected and touched an index.
    assert server.resolved_index_ids == []


# --------------------------------------------------------------------------
# Locked filters reach every search mode's query class
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("search_type", "supports_native_hybrid", "expected_mode"),
    [
        ("vector", False, "vector"),
        ("fulltext", False, "fulltext"),
        ("hybrid", True, "hybrid-native"),
        ("hybrid", False, "hybrid-fallback"),
    ],
)
async def test_profile_tool_threads_the_locked_filter_into_every_search_mode(
    monkeypatch, search_type, supports_native_hybrid, expected_mode
):
    built = _capture_any_query(monkeypatch)
    server = FakeServer(search_type=search_type)
    server.native_hybrid_supported = supports_native_hybrid
    fn = _register(server, _profile(lock={"filter": LOCKED_CATEGORY_FILTER}))

    await fn(query="jam", filter={"field": "rating", "op": "gte", "value": 4})

    mode, kwargs = built[0]
    # `_build_query` constructs a different class per search mode, each with its
    # own kwargs assembly. A locked filter that only threaded into the full-text
    # branch would leave vector and hybrid profiles silently unscoped.
    assert mode == expected_mode
    assert str(kwargs["filter_expression"]) == "(@category:{resolved} @rating:[4 +inf])"
