from typing import TYPE_CHECKING, Any

from redisvl.mcp.auth import ensure_tool_scope
from redisvl.mcp.runtime import BindingRuntime

if TYPE_CHECKING:
    from redisvl.mcp.server import RedisVLMCPServer

DEFAULT_LIST_INDEXES_DESCRIPTION = (
    "List the logical indexes configured on this server. Each entry reports the "
    "index id, an optional description, whether upsert is available, the "
    "filterable fields discovered from the index, and any explicitly configured "
    "limits. Call this first on a multi-index server to choose the correct "
    "index for search-records or upsert-records."
)

# Runtime limits surfaced to clients, included only when explicitly configured.
_LIMIT_FIELDS = ("max_limit", "max_upsert_records")


def _binding_fields(binding_runtime: BindingRuntime) -> list[dict[str, str]]:
    """Return a binding's shared filterable fields from its inspected schema.

    The vector field and the configured default embed-source text field are
    omitted: they are implementation inputs, not fields a client filters on.
    """
    embed_source = binding_runtime.binding.runtime.default_embed_text_field
    fields: list[dict[str, str]] = []
    for field in binding_runtime.schema.fields.values():
        field_type = str(getattr(field.type, "value", field.type))
        if field_type.lower() == "vector":
            continue
        if field.name == embed_source:
            continue
        fields.append({"name": field.name, "type": field_type})
    return fields


def _binding_limits(binding_runtime: BindingRuntime) -> dict[str, int]:
    """Return runtime limits that were explicitly configured for the binding.

    Defaults are intentionally excluded so the output reflects deliberate
    overrides rather than implementation defaults.
    """
    runtime = binding_runtime.binding.runtime
    configured = runtime.model_fields_set
    return {
        name: getattr(runtime, name) for name in _LIMIT_FIELDS if name in configured
    }


def _describe_binding(binding_runtime: BindingRuntime) -> dict[str, Any]:
    """Build the deterministic discovery payload for a single binding."""
    entry: dict[str, Any] = {"id": binding_runtime.binding_id}
    if binding_runtime.binding.description is not None:
        entry["description"] = binding_runtime.binding.description
    # Reflects both global read-only and the per-index read_only policy.
    entry["upsert_available"] = not binding_runtime.effective_read_only
    entry["fields"] = _binding_fields(binding_runtime)
    limits = _binding_limits(binding_runtime)
    if limits:
        entry["limits"] = limits
    return entry


def list_indexes(server: "RedisVLMCPServer") -> dict[str, Any]:
    """Return the discovery payload for every configured binding.

    The Redis index name (``redis_name``) is intentionally never exposed.
    """
    # Mirror resolve_binding: with no bindings the server is not started (or has
    # been torn down), so fail loudly rather than return an empty list that a
    # client could misread as "no indexes configured".
    if not server._bindings:
        raise RuntimeError("MCP server has not been started")
    return {
        "indexes": [
            _describe_binding(binding_runtime)
            for binding_runtime in server._bindings.values()
        ],
    }


def register_list_indexes_tool(server: "RedisVLMCPServer") -> None:
    """Register the always-available, read-only `list-indexes` MCP tool."""

    async def list_indexes_tool():
        """FastMCP wrapper for the `list-indexes` tool."""
        auth_config = getattr(server, "auth_config", None)
        read_scope = auth_config.read_scope if auth_config is not None else None
        ensure_tool_scope(server, read_scope)
        return list_indexes(server)

    server.tool(name="list-indexes", description=DEFAULT_LIST_INDEXES_DESCRIPTION)(
        list_indexes_tool
    )
