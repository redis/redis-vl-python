from typing import Any

from redisvl.mcp.auth import ensure_tool_scope

DEFAULT_LIST_INDEXES_DESCRIPTION = (
    "List the logical indexes configured on this server. Each entry reports the "
    "index id, an optional description, whether upsert is available, the "
    "filterable fields discovered from the index, and any explicitly configured "
    "limits. Call this first on a multi-index server to choose the correct "
    "index for search-records or upsert-records."
)

# Runtime limits surfaced to clients, included only when explicitly configured.
_LIMIT_FIELDS = ("max_limit", "max_upsert_records")


def _binding_fields(rt: Any) -> list[dict[str, str]]:
    """Return a binding's shared filterable fields from its inspected schema.

    The vector field and the configured default embed-source text field are
    omitted: they are implementation inputs, not fields a client filters on.
    """
    embed_source = rt.binding.runtime.default_embed_text_field
    fields: list[dict[str, str]] = []
    for field in rt.schema.fields.values():
        field_type = str(getattr(field.type, "value", field.type))
        if field_type.lower() == "vector":
            continue
        if field.name == embed_source:
            continue
        fields.append({"name": field.name, "type": field_type})
    return fields


def _binding_limits(rt: Any) -> dict[str, int]:
    """Return runtime limits that were explicitly configured for the binding.

    Defaults are intentionally excluded so the output reflects deliberate
    overrides rather than implementation defaults.
    """
    runtime = rt.binding.runtime
    configured = runtime.model_fields_set
    return {
        name: getattr(runtime, name) for name in _LIMIT_FIELDS if name in configured
    }


def _describe_binding(rt: Any) -> dict[str, Any]:
    """Build the deterministic discovery payload for a single binding."""
    entry: dict[str, Any] = {"id": rt.binding_id}
    if rt.binding.description is not None:
        entry["description"] = rt.binding.description
    # Reflects both global read-only and the per-index read_only policy.
    entry["upsert_available"] = not rt.effective_read_only
    entry["fields"] = _binding_fields(rt)
    limits = _binding_limits(rt)
    if limits:
        entry["limits"] = limits
    return entry


def list_indexes(server: Any) -> dict[str, Any]:
    """Return the discovery payload for every configured binding.

    The Redis index name (``redis_name``) is intentionally never exposed.
    """
    return {
        "indexes": [_describe_binding(rt) for rt in server._bindings.values()],
    }


def register_list_indexes_tool(server: Any) -> None:
    """Register the always-available, read-only `list-indexes` MCP tool."""
    description = (
        getattr(server.mcp_settings, "tool_list_indexes_description", None)
        or DEFAULT_LIST_INDEXES_DESCRIPTION
    )

    async def list_indexes_tool():
        """FastMCP wrapper for the `list-indexes` tool."""
        read_scope = getattr(getattr(server, "auth_config", None), "read_scope", None)
        ensure_tool_scope(server, read_scope)
        return list_indexes(server)

    server.tool(name="list-indexes", description=description)(list_indexes_tool)
