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
        lambda server, schema: registered.append("search-records"),
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.register_upsert_tool",
        lambda server: registered.append("upsert-records"),
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
    # Profiles are the reason to disable a built-in, so they still register.
    assert "register-profile-tools" in registered


def test_register_tools_warns_when_the_whole_tool_surface_is_empty(monkeypatch, caplog):
    """Every built-in disabled with no profiles is valid config but a dead server."""
    with caplog.at_level(logging.WARNING, logger="redisvl.mcp.server"):
        registered = _register_tools_with(
            monkeypatch,
            {"knowledge": _binding_runtime("knowledge")},
            config=_config_with(
                builtin_tools={name: "disabled" for name in builtin_tool_names()}
            ),
        )

    # The surface really is empty -- only the profile-registration call itself ran,
    # and it produced no names -- so the warning is not passing for another reason.
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
        lambda server, schema: registered.append("search-records"),
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.register_upsert_tool",
        lambda server: registered.append("upsert-records"),
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

    server._validate_custom_tools()

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
