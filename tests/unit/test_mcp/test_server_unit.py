import logging
from types import SimpleNamespace

import pytest

from redisvl.mcp.config import MCPConfig, builtin_tool_names
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.runtime import BindingRuntime
from redisvl.mcp.server import RedisVLMCPServer


class FakeClient:
    def __init__(self):
        self.info_calls = 0

    async def info(self, section: str):
        self.info_calls += 1
        assert section == "server"
        return {"redis_version": "8.4.0"}

    def ft(self, index_name: str):
        assert index_name == "docs-index"
        return SimpleNamespace(hybrid_search=object())


class FakeIndex:
    def __init__(self, client: FakeClient):
        self.schema = SimpleNamespace(index=SimpleNamespace(name="docs-index"))
        self._client = client

    async def _get_client(self):
        return self._client


@pytest.mark.asyncio
async def test_probe_native_hybrid_search_detects_support(monkeypatch):
    client = FakeClient()
    index = FakeIndex(client)

    monkeypatch.setattr("redisvl.mcp.server.redis_py_version", "7.1.0")

    assert await RedisVLMCPServer._probe_native_hybrid_search(index) is True
    assert client.info_calls == 1


@pytest.mark.asyncio
async def test_probe_native_hybrid_search_false_for_old_redis_py(monkeypatch):
    client = FakeClient()
    index = FakeIndex(client)

    monkeypatch.setattr("redisvl.mcp.server.redis_py_version", "7.0.0")

    assert await RedisVLMCPServer._probe_native_hybrid_search(index) is False
    # Old redis-py short-circuits before querying the server.
    assert client.info_calls == 0


def _binding_runtime(
    binding_id: str, *, effective_read_only: bool = False
) -> BindingRuntime:
    return BindingRuntime(
        binding_id=binding_id,
        binding=SimpleNamespace(),
        index=SimpleNamespace(),
        schema=SimpleNamespace(),
        vectorizer=None,
        supports_native_hybrid_search=False,
        effective_read_only=effective_read_only,
    )


def _server_with_bindings(*binding_ids: str) -> RedisVLMCPServer:
    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server._bindings = {bid: _binding_runtime(bid) for bid in binding_ids}
    return server


def test_resolve_binding_before_startup_raises():
    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server._bindings = {}

    with pytest.raises(RuntimeError, match="not been started"):
        server.resolve_binding(None)


def test_resolve_binding_defaults_to_sole_binding():
    server = _server_with_bindings("knowledge")

    assert server.resolve_binding(None).binding_id == "knowledge"


def test_resolve_binding_requires_index_when_multiple_configured():
    server = _server_with_bindings("knowledge", "tickets")

    with pytest.raises(RedisVLMCPError) as excinfo:
        server.resolve_binding(None)

    assert excinfo.value.code == MCPErrorCode.INVALID_REQUEST
    assert "knowledge" in str(excinfo.value)
    assert "tickets" in str(excinfo.value)


def test_resolve_binding_routes_to_named_index():
    server = _server_with_bindings("knowledge", "tickets")

    assert server.resolve_binding("tickets").binding_id == "tickets"


def test_resolve_binding_rejects_unknown_index():
    server = _server_with_bindings("knowledge", "tickets")

    with pytest.raises(RedisVLMCPError) as excinfo:
        server.resolve_binding("missing")

    assert excinfo.value.code == MCPErrorCode.INVALID_REQUEST
    assert "missing" in str(excinfo.value)


@pytest.mark.asyncio
async def test_teardown_continues_when_a_binding_fails_to_close(monkeypatch):
    """A failed close on one binding must not leak the remaining bindings."""
    server = _server_with_bindings("knowledge", "tickets")
    server.config = SimpleNamespace()
    server._semaphore = SimpleNamespace()
    server._tools_registered = True

    closed: list[str] = []

    async def fake_close_resources(self, *, index, vectorizer):
        # Fail on the first binding; the loop must still reach the second.
        if not closed:
            closed.append("knowledge")
            raise RuntimeError("disconnect failed")
        closed.append("tickets")

    monkeypatch.setattr(RedisVLMCPServer, "_close_resources", fake_close_resources)

    await server._teardown_runtime()

    # Both bindings were attempted despite the first one raising.
    assert closed == ["knowledge", "tickets"]
    # Binding state is cleared...
    assert server._bindings == {}
    # ...but tool registration is instance-level and must survive teardown, so a
    # stop/start does not re-register the same tool names on the FastMCP object.
    assert server._tools_registered is True


def _register_tools_with(monkeypatch, bindings: dict, *, config=None) -> list[str]:
    """Run _register_tools against the given bindings, returning registered names."""
    registered: list[str] = []
    monkeypatch.setattr(
        "redisvl.mcp.server.register_list_indexes_tool",
        lambda server: registered.append("list-indexes"),
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.register_search_tool",
        lambda server, schema, index_ids=None: registered.append("search-records"),
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.register_upsert_tool",
        lambda server, index_ids=None: registered.append("upsert-records"),
    )

    def fake_register_profile_tools(server):
        registered.append("register-profile-tools")
        server_config = getattr(server, "config", None)
        names = (
            []
            if server_config is None
            else [profile.name for profile in server_config.custom_tools]
        )
        registered.extend(names)
        return names

    monkeypatch.setattr(
        "redisvl.mcp.server.register_profile_tools", fake_register_profile_tools
    )

    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server._bindings = bindings
    server._tools_registered = False
    server._registered_tool_fingerprint = ""
    server._registered_tools_inject = False
    server.tool = object()
    server.config = config
    server.mcp_settings = SimpleNamespace(read_only=False)

    server._register_tools()
    return registered


def _config_with(*, builtin_tools=None, custom_tools=None) -> MCPConfig:
    """Build a real validated config so the gating logic sees the real methods."""
    server_config: dict = {"redis_url": "redis://localhost:6379"}
    if builtin_tools is not None:
        server_config["builtin_tools"] = builtin_tools
    return MCPConfig.model_validate(
        {
            "server": server_config,
            "indexes": {
                "knowledge": {
                    "redis_name": "docs-index",
                    "search": {"type": "fulltext"},
                    "runtime": {"text_field_name": "content"},
                }
            },
            "custom_tools": custom_tools or [],
        }
    )


def test_register_tools_exposes_upsert_when_a_binding_is_writable(monkeypatch):
    registered = _register_tools_with(
        monkeypatch,
        {
            "knowledge": _binding_runtime("knowledge", effective_read_only=False),
            "tickets": _binding_runtime("tickets", effective_read_only=True),
        },
    )

    assert "upsert-records" in registered
    assert "list-indexes" in registered
    assert "search-records" in registered


def test_register_tools_hides_upsert_when_every_binding_is_read_only(monkeypatch):
    registered = _register_tools_with(
        monkeypatch,
        {
            "knowledge": _binding_runtime("knowledge", effective_read_only=True),
            "tickets": _binding_runtime("tickets", effective_read_only=True),
        },
    )

    assert "upsert-records" not in registered
    # Read paths stay available even when writes are globally disabled.
    assert "list-indexes" in registered
    assert "search-records" in registered


def test_register_tools_registers_every_builtin_when_no_config_is_attached(monkeypatch):
    registered = _register_tools_with(
        monkeypatch, {"knowledge": _binding_runtime("knowledge")}
    )

    assert registered == [
        "list-indexes",
        "search-records",
        "upsert-records",
        "register-profile-tools",
    ]


@pytest.mark.parametrize(
    "disabled_tool", ["list-indexes", "search-records", "upsert-records"]
)
def test_register_tools_skips_a_builtin_the_operator_disabled(
    monkeypatch, disabled_tool
):
    registered = _register_tools_with(
        monkeypatch,
        {"knowledge": _binding_runtime("knowledge")},
        config=_config_with(builtin_tools={disabled_tool: "disabled"}),
    )

    assert disabled_tool not in registered
    # Disabling one built-in must not take the others with it.
    for other in {"list-indexes", "search-records", "upsert-records"} - {disabled_tool}:
        assert other in registered


def test_register_tools_warns_when_the_whole_tool_surface_is_empty(monkeypatch, caplog):
    """Every built-in disabled is valid config but a dead server."""
    with caplog.at_level(logging.WARNING, logger="redisvl.mcp.server"):
        registered = _register_tools_with(
            monkeypatch,
            {"knowledge": _binding_runtime("knowledge")},
            config=_config_with(
                builtin_tools={name: "disabled" for name in builtin_tool_names()}
            ),
        )

    # The surface really is empty, so the warning is not passing for some other
    # reason.
    # Only the profile-registration call itself ran, and it produced no names,
    # so the surface really is empty and the warning is not passing for another
    # reason.
    assert registered == ["register-profile-tools"]
    # A client sees a server that connects and then offers nothing, which is
    # indistinguishable from a broken deployment unless the operator is told.
    assert [
        record.message
        for record in caplog.records
        if "registered no tools" in record.message
    ]


def test_register_tools_warns_when_discovery_is_disabled_on_a_multi_index_server(
    monkeypatch, caplog
):
    """search-records needs logical index ids that only list-indexes reveals."""
    with caplog.at_level(logging.WARNING, logger="redisvl.mcp.server"):
        registered = _register_tools_with(
            monkeypatch,
            {
                "knowledge": _binding_runtime("knowledge"),
                "tickets": _binding_runtime("tickets"),
            },
            config=_config_with(builtin_tools={"list-indexes": "disabled"}),
        )

    assert "search-records" in registered and "list-indexes" not in registered
    assert [
        record.message
        for record in caplog.records
        if "cannot discover" in record.message
    ]


def test_register_tools_stays_quiet_when_discovery_is_disabled_on_one_index(
    monkeypatch, caplog
):
    """With a sole binding the index argument defaults, so discovery is optional."""
    with caplog.at_level(logging.WARNING, logger="redisvl.mcp.server"):
        _register_tools_with(
            monkeypatch,
            {"knowledge": _binding_runtime("knowledge")},
            config=_config_with(builtin_tools={"list-indexes": "disabled"}),
        )

    assert not [
        record.message
        for record in caplog.records
        if "cannot discover" in record.message
    ]


def test_register_tools_names_index_ids_when_discovery_is_disabled(monkeypatch):
    """A multi-index description must not point at a tool the server withholds."""
    captured: dict = {}
    monkeypatch.setattr(
        "redisvl.mcp.server.register_list_indexes_tool", lambda server: None
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.register_search_tool",
        lambda server, schema, index_ids=None: captured.update(index_ids=index_ids),
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.register_upsert_tool", lambda server, index_ids=None: None
    )

    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server._bindings = {
        "knowledge": _binding_runtime("knowledge"),
        "tickets": _binding_runtime("tickets"),
    }
    server._tools_registered = False
    server._registered_tool_fingerprint = ""
    server._registered_tools_inject = False
    server.tool = object()
    server.config = _config_with(builtin_tools={"list-indexes": "disabled"})
    server.mcp_settings = SimpleNamespace(read_only=False)

    server._register_tools()

    # Without discovery these ids are otherwise unlearnable, and `index` is
    # required on a multi-index server.
    assert captured["index_ids"] == ["knowledge", "tickets"]


def test_register_tools_omits_index_ids_when_discovery_is_available(monkeypatch):
    """With list-indexes published, the description should defer to it as before."""
    captured: dict = {}
    monkeypatch.setattr(
        "redisvl.mcp.server.register_list_indexes_tool", lambda server: None
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.register_search_tool",
        lambda server, schema, index_ids=None: captured.update(index_ids=index_ids),
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.register_upsert_tool", lambda server, index_ids=None: None
    )

    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server._bindings = {
        "knowledge": _binding_runtime("knowledge"),
        "tickets": _binding_runtime("tickets"),
    }
    server._tools_registered = False
    server._registered_tool_fingerprint = ""
    server._registered_tools_inject = False
    server.tool = object()
    server.config = None
    server.mcp_settings = SimpleNamespace(read_only=False)

    server._register_tools()

    assert captured["index_ids"] is None


def test_register_tools_warns_when_builtin_config_changed_after_registration(
    monkeypatch, caplog
):
    """Tools register once per process, so an edited config cannot take effect."""
    registered = _register_tools_with(
        monkeypatch,
        {"knowledge": _binding_runtime("knowledge")},
        config=_config_with(),
    )
    assert "upsert-records" in registered

    # Simulate a stop/start that reloaded a config which now disables upsert.
    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server._bindings = {"knowledge": _binding_runtime("knowledge")}
    server.tool = object()
    server._tools_registered = True
    server._registered_tool_fingerprint = ""
    server._registered_tools_inject = False
    server.config = _config_with(builtin_tools={"upsert-records": "disabled"})

    with caplog.at_level(logging.WARNING, logger="redisvl.mcp.server"):
        server._register_tools()

    # The dangerous direction: an operator disables a tool, restarts, and believes
    # it is gone while the old tool set is still what clients see.
    assert [
        r.message
        for r in caplog.records
        if "changed since tools were registered" in r.message
    ]


def test_register_tools_gives_upsert_the_same_index_ids_as_search(monkeypatch):
    """Both tools require `index`, so both need the ids when discovery is off."""
    captured: dict = {}
    monkeypatch.setattr(
        "redisvl.mcp.server.register_list_indexes_tool", lambda server: None
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.register_search_tool",
        lambda server, schema, index_ids=None: captured.update(search=index_ids),
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.register_upsert_tool",
        lambda server, index_ids=None: captured.update(upsert=index_ids),
    )

    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server._bindings = {
        "knowledge": _binding_runtime("knowledge"),
        "tickets": _binding_runtime("tickets"),
    }
    server._tools_registered = False
    server._registered_tool_fingerprint = ""
    server._registered_tools_inject = False
    server.tool = object()
    server.config = _config_with(builtin_tools={"list-indexes": "disabled"})
    server.mcp_settings = SimpleNamespace(read_only=False)

    server._register_tools()

    # Writes need the ids exactly as much as reads do.
    assert captured["upsert"] == ["knowledge", "tickets"]
    assert captured["upsert"] == captured["search"]


def test_register_tools_warns_when_discovery_is_disabled_on_a_write_only_surface(
    monkeypatch, caplog
):
    """A write-only surface loses discovery too, and must not warn silently."""
    with caplog.at_level(logging.WARNING, logger="redisvl.mcp.server"):
        registered = _register_tools_with(
            monkeypatch,
            {
                "knowledge": _binding_runtime("knowledge"),
                "tickets": _binding_runtime("tickets"),
            },
            config=_config_with(
                builtin_tools={
                    "list-indexes": "disabled",
                    "search-records": "disabled",
                }
            ),
        )

    # Only upsert is published, so a check keyed on search-records would miss it.
    assert registered == ["upsert-records", "register-profile-tools"]
    messages = [r.message for r in caplog.records if "cannot discover" in r.message]
    assert messages
    assert "upsert-records" in messages[0]


def test_register_tools_registers_configured_profiles(monkeypatch):
    registered = _register_tools_with(
        monkeypatch,
        {"knowledge": _binding_runtime("knowledge")},
        config=_config_with(
            custom_tools=[
                {"name": "resolved-search", "description": "Search resolved."},
                {"name": "open-search", "description": "Search open."},
            ]
        ),
    )

    assert registered[-2:] == ["resolved-search", "open-search"]


def test_register_tools_is_idempotent(monkeypatch):
    """A second call must not re-register the same names on the FastMCP object."""
    registered: list[str] = []
    monkeypatch.setattr(
        "redisvl.mcp.server.register_list_indexes_tool",
        lambda server: registered.append("list-indexes"),
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.register_search_tool",
        lambda server, schema, index_ids=None: registered.append("search-records"),
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.register_upsert_tool",
        lambda server, index_ids=None: registered.append("upsert-records"),
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.register_profile_tools",
        lambda server: registered.append("register-profile-tools") or [],
    )

    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server._bindings = {"knowledge": _binding_runtime("knowledge")}
    server._tools_registered = False
    server.tool = object()
    server.config = None
    server.mcp_settings = SimpleNamespace(read_only=False)

    server._register_tools()
    server._register_tools()

    assert registered.count("register-profile-tools") == 1


def test_validate_custom_tools_checks_each_profile_against_its_bound_schema(
    monkeypatch,
):
    validated: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "redisvl.mcp.server.validate_profile_against_schema",
        lambda profile, schema: validated.append((profile.name, schema.marker)),
    )

    config = MCPConfig.model_validate(
        {
            "server": {"redis_url": "redis://localhost:6379"},
            "indexes": {
                "knowledge": {
                    "redis_name": "docs-index",
                    "search": {"type": "fulltext"},
                    "runtime": {"text_field_name": "content"},
                },
                "tickets": {
                    "redis_name": "tickets-index",
                    "search": {"type": "fulltext"},
                    "runtime": {"text_field_name": "content"},
                },
            },
            "custom_tools": [
                {
                    "name": "resolved-search",
                    "description": "Search resolved.",
                    "index": "tickets",
                }
            ],
        }
    )

    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server.config = config
    server._bindings = {
        "knowledge": _binding_runtime("knowledge"),
        "tickets": _binding_runtime("tickets"),
    }
    server._bindings["knowledge"].schema.marker = "knowledge-schema"
    server._bindings["tickets"].schema.marker = "tickets-schema"

    server._validate_custom_tools_against_schema()

    # Each profile is validated against the schema of the binding it is pinned to.
    assert validated == [("resolved-search", "tickets-schema")]


def test_register_tools_warns_when_profile_config_changed_after_registration(
    monkeypatch, caplog
):
    """Profiles bake their lock in at registration, so a reload cannot retighten it."""
    registered = _register_tools_with(
        monkeypatch,
        {"knowledge": _binding_runtime("knowledge")},
        config=_config_with(
            custom_tools=[{"name": "open-search", "description": "Search open."}]
        ),
    )
    assert "open-search" in registered

    # Simulate a restart that reloaded a *tightened* config: same server object,
    # tools already registered, different profile set.
    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server._bindings = {"knowledge": _binding_runtime("knowledge")}
    server.tool = object()
    server._tools_registered = True
    server._registered_tool_fingerprint = ""
    server._registered_tools_inject = False
    server.config = _config_with(
        custom_tools=[
            {
                "name": "open-search",
                "description": "Search open.",
                "lock": {"filter": {"field": "category", "op": "eq", "value": "safe"}},
            }
        ]
    )

    with caplog.at_level(logging.WARNING, logger="redisvl.mcp.server"):
        server._register_tools()

    # The dangerous direction: an operator tightens a lock, restarts, and believes
    # it took effect while the old profile is still the one enforcing.
    assert [
        record.message
        for record in caplog.records
        if "changed since tools were registered" in record.message
    ]


# --------------------------------------------------------------------------
# Claim injection: refusing a profile that can never read a token
# --------------------------------------------------------------------------

_INJECTING = {
    "name": "tenant-search",
    "description": "Search this tenant.",
    "lock": {"inject": [{"field": "category", "from": "claim", "claim": "org"}]},
}


def _injection_server(*, auth_enabled, transport, custom_tools=None):
    """A server shell holding a validated injecting config."""
    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server.config = _config_with(custom_tools=custom_tools or [_INJECTING])
    server._bindings = {"knowledge": _binding_runtime("knowledge")}
    server._auth_enabled = auth_enabled
    server._transport = transport
    return server


@pytest.mark.parametrize("transport", ["stdio", "streamable-http", "sse", None])
def test_injection_refuses_to_start_without_auth_on_any_transport(
    monkeypatch, transport
):
    # Keyed off "auth is enabled", never off the transport name alone, so an
    # unauthenticated HTTP bind -- loopback or --allow-unauthenticated -- is
    # refused exactly as stdio is.
    server = _injection_server(auth_enabled=False, transport=transport)
    with pytest.raises(ValueError, match="authentication is not enabled"):
        server._verify_injection_has_a_token(server.config.custom_tools)


def test_injection_refuses_to_start_with_auth_configured_under_stdio(monkeypatch):
    # FastMCP never authenticates stdio, so the verifier exists and is never
    # consulted: every call would be refused at request time.
    server = _injection_server(auth_enabled=True, transport="stdio")
    with pytest.raises(ValueError, match="running over stdio"):
        server._verify_injection_has_a_token(server.config.custom_tools)


@pytest.mark.parametrize("transport", ["streamable-http", "sse", None])
def test_injection_starts_with_auth_over_http_or_an_unnamed_transport(
    monkeypatch, transport
):
    # None is an embedder that never named a transport; it is left to the
    # request-time refusal, which still fails closed.
    server = _injection_server(auth_enabled=True, transport=transport)
    server._verify_injection_has_a_token(server.config.custom_tools)


def test_a_server_without_injection_needs_no_auth(monkeypatch):
    server = _injection_server(
        auth_enabled=False,
        transport="stdio",
        custom_tools=[{"name": "open-search", "description": "Search open."}],
    )
    server._verify_injection_has_a_token(server.config.custom_tools)


@pytest.mark.asyncio
async def test_an_injection_profile_without_auth_fails_before_redis_is_touched(
    monkeypatch,
):
    # The refusal depends only on the loaded config, so an unworkable one must
    # not first need a reachable Redis and an existing index to be reported.
    connected: list[str] = []

    async def no_connect(self, binding_id, binding):
        connected.append(binding_id)
        raise AssertionError("a binding was initialized before the refusal")

    monkeypatch.setattr(
        "redisvl.mcp.server.load_mcp_config",
        lambda path: _config_with(custom_tools=[_INJECTING]),
    )
    monkeypatch.setattr(RedisVLMCPServer, "_verify_auth_not_stale", lambda self: None)
    monkeypatch.setattr(RedisVLMCPServer, "_initialize_binding", no_connect)
    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server._config_path = "unused.yaml"
    server._auth_enabled = False
    server._transport = None

    with pytest.raises(ValueError, match="authentication is not enabled"):
        await server._initialize_runtime_resources()
    assert connected == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transport, recorded",
    [
        ("streamable-http", "streamable-http"),
        # FastMCP resolves an omitted transport to its configured default, so
        # `run_async()` must be recorded as the stdio it will actually serve.
        (None, "stdio"),
    ],
)
async def test_run_async_records_the_transport_it_will_serve(
    monkeypatch, transport, recorded
):
    fastmcp = pytest.importorskip(
        "fastmcp", reason="fastmcp not installed (install redisvl[mcp])"
    )
    monkeypatch.setattr(fastmcp.settings, "transport", "stdio")

    async def no_serve(self, transport=None, show_banner=None, **kwargs):
        return None

    monkeypatch.setattr(fastmcp.FastMCP, "run_async", no_serve)
    monkeypatch.setattr(
        "redisvl.mcp.server.build_host_origin_middleware", lambda *args: []
    )
    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server._transport_security = None

    await server.run_async(transport=transport)

    assert server._transport == recorded


def _reregister_after_change(*, registered_with_inject, new_custom_tools):
    """Simulate a stop/start against an edited config on an already-registered server."""
    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server._bindings = {"knowledge": _binding_runtime("knowledge")}
    server.tool = object()
    server._tools_registered = True
    server._registered_tool_fingerprint = "the-config-at-registration"
    server._registered_tools_inject = registered_with_inject
    server.config = _config_with(custom_tools=new_custom_tools)
    server._register_tools()


@pytest.mark.parametrize(
    "registered_with_inject, new_custom_tools",
    [
        # Tightening or loosening an injection that is already live.
        pytest.param(
            True, [{"name": "tenant-search", "description": "x"}], id="removed"
        ),
        # The dangerous direction: injection added to a profile registered without
        # it would keep serving every tenant while the config says otherwise.
        pytest.param(False, [_INJECTING], id="added"),
    ],
)
def test_a_changed_tool_surface_is_fatal_when_injection_is_on_either_side(
    registered_with_inject, new_custom_tools
):
    with pytest.raises(RuntimeError, match="claim injection is configured"):
        _reregister_after_change(
            registered_with_inject=registered_with_inject,
            new_custom_tools=new_custom_tools,
        )


def test_registration_remembers_that_it_installed_injection(monkeypatch):
    # The flag the fatal check reads has to be set by registration itself. A
    # first registration with injection, then a reload that drops it, is the
    # case where only that remembered flag knows the live tools are scoped.
    for target in (
        "register_list_indexes_tool",
        "register_search_tool",
        "register_upsert_tool",
    ):
        monkeypatch.setattr(f"redisvl.mcp.server.{target}", lambda *a, **k: None)
    monkeypatch.setattr(
        "redisvl.mcp.server.register_profile_tools", lambda server: ["tenant-search"]
    )
    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server._bindings = {"knowledge": _binding_runtime("knowledge")}
    server._tools_registered = False
    server._registered_tool_fingerprint = ""
    server._registered_tools_inject = False
    server.tool = object()
    server.mcp_settings = SimpleNamespace(read_only=False)
    server.config = _config_with(custom_tools=[_INJECTING])
    server._register_tools()

    server.config = _config_with(
        custom_tools=[{"name": "tenant-search", "description": "x"}]
    )
    with pytest.raises(RuntimeError, match="claim injection is configured"):
        server._register_tools()


def test_a_changed_tool_surface_without_injection_still_only_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="redisvl.mcp.server"):
        _reregister_after_change(
            registered_with_inject=False,
            new_custom_tools=[{"name": "open-search", "description": "Search."}],
        )
    assert any(
        "changed since tools were registered" in record.message
        for record in caplog.records
    )


# --------------------------------------------------------------------------
# Claim injection: refusing an unscoped route to a tenant-scoped index
# --------------------------------------------------------------------------

_BUILTINS_OFF = {"search-records": "disabled", "upsert-records": "disabled"}


def _route_check(*, builtin_tools, custom_tools=None, read_only=False, indexes=None):
    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server.mcp_settings = SimpleNamespace(read_only=read_only)
    if indexes is None:
        config = _config_with(
            builtin_tools=builtin_tools, custom_tools=custom_tools or [_INJECTING]
        )
    else:
        config = MCPConfig.model_validate(
            {
                "server": {
                    "redis_url": "redis://localhost:6379",
                    "builtin_tools": builtin_tools,
                },
                "indexes": {
                    index_id: {
                        "redis_name": f"{index_id}-index",
                        "search": {"type": "fulltext"},
                        "runtime": {"text_field_name": "content"},
                    }
                    for index_id in indexes
                },
                "custom_tools": custom_tools,
            }
        )
    server._verify_no_unscoped_route_to_injected_indexes(config)


@pytest.mark.parametrize(
    "builtin_tools, custom_tools, expected",
    [
        pytest.param(
            {"upsert-records": "disabled"}, None, "through search-records", id="search"
        ),
        # A write can retag another tenant's document as the writer's own.
        pytest.param(
            {"search-records": "disabled"}, None, "through upsert-records", id="upsert"
        ),
        pytest.param(
            _BUILTINS_OFF,
            [_INJECTING, {"name": "open-search", "description": "Search."}],
            "'open-search' injects nothing",
            id="unscoped-profile",
        ),
        # Injecting *something* is not the tenant scope. Scoped by another
        # field, the second tool reads across the tenants the first separates.
        pytest.param(
            _BUILTINS_OFF,
            [
                _INJECTING,
                {
                    "name": "region-search",
                    "description": "Search.",
                    "lock": {
                        "inject": [{"field": "rating", "from": "claim", "claim": "r"}]
                    },
                },
            ],
            "'region-search' injects rating from claim 'r'",
            id="different-field",
        ),
        # The same field read from another claim scopes by a different value.
        pytest.param(
            _BUILTINS_OFF,
            [
                _INJECTING,
                {
                    "name": "other-org-search",
                    "description": "Search.",
                    "lock": {
                        "inject": [
                            {"field": "category", "from": "claim", "claim": "alt"}
                        ]
                    },
                },
            ],
            "'other-org-search' injects category from claim 'alt'",
            id="different-claim",
        ),
    ],
)
def test_an_unscoped_route_to_an_injected_index_is_refused(
    builtin_tools, custom_tools, expected
):
    with pytest.raises(ValueError, match=expected):
        _route_check(builtin_tools=builtin_tools, custom_tools=custom_tools)


def test_a_fully_scoped_surface_starts():
    _route_check(builtin_tools=_BUILTINS_OFF)


@pytest.mark.parametrize("server_read_only", [False, True])
def test_upsert_is_no_route_to_a_read_only_index(server_read_only):
    # Server-wide read-only, or the binding's own flag, both refuse writes per
    # call, so upsert-records cannot reach the index either way.
    server = RedisVLMCPServer.__new__(RedisVLMCPServer)
    server.mcp_settings = SimpleNamespace(read_only=server_read_only)
    raw = {
        "server": {
            "redis_url": "redis://localhost:6379",
            "builtin_tools": {"search-records": "disabled"},
        },
        "indexes": {
            "knowledge": {
                "redis_name": "docs-index",
                "read_only": not server_read_only,
                "search": {"type": "fulltext"},
                "runtime": {"text_field_name": "content"},
            }
        },
        "custom_tools": [_INJECTING],
    }
    server._verify_no_unscoped_route_to_injected_indexes(MCPConfig.model_validate(raw))


def test_an_unscoped_profile_on_another_index_is_no_route():
    _route_check(
        builtin_tools=_BUILTINS_OFF,
        indexes=("knowledge", "public"),
        custom_tools=[
            {**_INJECTING, "index": "knowledge"},
            {"name": "public-search", "description": "Search.", "index": "public"},
        ],
    )


def test_a_server_without_injection_keeps_every_route():
    _route_check(
        builtin_tools={},
        custom_tools=[{"name": "open-search", "description": "Search open."}],
    )


def test_tools_injecting_the_same_scope_share_an_index():
    # Entry order does not matter: the scope is the set of (field, claim) pairs.
    two_entry = [
        {"field": "category", "from": "claim", "claim": "org"},
        {"field": "rating", "from": "claim", "claim": "tier"},
    ]
    _route_check(
        builtin_tools=_BUILTINS_OFF,
        custom_tools=[
            {
                "name": "tenant-search",
                "description": "Search.",
                "lock": {"inject": two_entry},
            },
            {
                "name": "tenant-recent",
                "description": "Recent.",
                "lock": {"inject": list(reversed(two_entry))},
            },
        ],
    )
