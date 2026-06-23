from dataclasses import dataclass
from typing import Any

from redisvl.index import AsyncSearchIndex
from redisvl.mcp.config import MCPIndexBindingConfig
from redisvl.schema import IndexSchema


@dataclass(frozen=True)
class BindingRuntime:
    """Immutable per-binding runtime state assembled once at server startup.

    Each configured logical index becomes one ``BindingRuntime`` bundling the
    binding config with the resources a tool call needs: the connected index,
    its effective (inspected + overridden) schema, an optional vectorizer, the
    resolved native-hybrid-search capability, and the effective write policy.

    Tools resolve a binding once via ``server.resolve_binding(index)`` and then
    read these attributes directly instead of calling back into the server.
    """

    binding_id: str
    binding: MCPIndexBindingConfig
    index: AsyncSearchIndex
    schema: IndexSchema
    vectorizer: Any | None
    supports_native_hybrid_search: bool
    effective_read_only: bool
