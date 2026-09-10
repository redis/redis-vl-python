"""Declarative custom tool profiles layered over the built-in MCP tools.

A profile is the built-in named by ``based_on`` with some of its arguments
pre-filled and frozen by the author and the rest still exposed to the model. It
resolves to a built-in call and nothing more, so it inherits that built-in's
concurrency cap, request timeout, read-only policy, auth scoping, and error
mapping without restating any of it here.

The frozen arguments are omitted from the wrapper's signature, which is what
FastMCP derives the advertised schema from -- so a locked argument is not merely
ignored when supplied, it is not offered to the model at all.
"""

import inspect
from typing import Annotated, Any, Optional

from pydantic import Field

from redisvl.mcp.auth import ensure_tool_scope
from redisvl.mcp.config import MCPCustomToolConfig
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.filters import parse_filter
from redisvl.mcp.tools.search import (
    _build_filter_hint,
    _build_return_fields_hint,
    search_records,
)
from redisvl.query.filter import FilterExpression
from redisvl.schema import IndexSchema

# Ordered so the generated signature reads like the built-in's: required query
# first, then the optional narrowing arguments. Real type objects rather than
# string annotations, so schema generation never depends on resolving a forward
# reference in this module's namespace.
_PROFILE_PARAM_SPECS: tuple[tuple[str, Any, Any], ...] = (
    ("query", str, inspect.Parameter.empty),
    ("limit", Optional[int], None),
    ("offset", int, 0),
    # Deliberately `dict` and not `str | dict`: the DSL object form is the only
    # caller filter a profile accepts, so a raw string is refused by the
    # advertised schema rather than relying on a runtime guard alone.
    ("filter", Optional[dict[str, Any]], None),
    ("return_fields", Optional[list[str]], None),
)


def build_profile_description(profile: MCPCustomToolConfig, schema: IndexSchema) -> str:
    """Build a profile's model-facing description.

    Schema hints are appended only for arguments the model can actually use, and
    are dropped entirely when the author suppresses them -- the built-in's hints
    enumerate filterable and returnable fields, which is noise or, worse,
    misleading once those arguments are locked.
    """
    description = profile.description.strip()
    if profile.suppress_schema_hints:
        return description

    parts = [description]
    if profile.param_exposed("filter"):
        parts.append(_build_filter_hint(schema))
    if profile.param_exposed("return_fields"):
        parts.append(_build_return_fields_hint(schema))
    return " ".join(parts)


def resolve_locked_filter(
    profile: MCPCustomToolConfig, schema: IndexSchema
) -> FilterExpression | None:
    """Parse a profile's locked filter against the bound schema.

    Parsing at registration rather than per call means an unknown or wrongly
    typed field fails startup instead of every request, and the resulting
    expression is reused for the life of the tool.
    """
    if profile.lock.filter is None:
        return None

    try:
        parsed = parse_filter(profile.lock.filter, schema)
    except RedisVLMCPError as exc:
        # parse_filter reports the offending field but not which profile owns it,
        # which is useless with several profiles configured.
        raise RedisVLMCPError(
            f"custom_tools '{profile.name}' lock.filter is invalid: {exc}",
            code=exc.code,
            retryable=False,
        ) from exc

    if not isinstance(parsed, FilterExpression):
        # Unreachable through config validation, which requires an object, but
        # keep the invariant explicit: a locked filter must be composable.
        raise ValueError(
            f"custom_tools '{profile.name}' lock.filter must be an object in the "
            "JSON filter DSL"
        )
    return parsed


def validate_profile_against_schema(
    profile: MCPCustomToolConfig, schema: IndexSchema
) -> None:
    """Fail fast when a profile names fields the bound index does not have.

    Config validation cannot do this: the schema is only known once the binding
    has been inspected at startup. Without it a locked projection or filter would
    silently match nothing, which looks like scoping but is not.
    """
    field_names = set(schema.field_names)
    vector_field_names = {
        field_name
        for field_name, field in schema.fields.items()
        if field.type == "vector"
    }

    for field_name in profile.lock.return_fields or []:
        if field_name not in field_names:
            raise ValueError(
                f"custom_tools '{profile.name}' lock.return_fields references "
                f"unknown field '{field_name}' on index "
                f"'{schema.index.name}'; available: "
                f"{', '.join(sorted(field_names))}"
            )
        if field_name in vector_field_names:
            raise ValueError(
                f"custom_tools '{profile.name}' lock.return_fields cannot return "
                f"vector field '{field_name}'"
            )

    _validate_locked_exists_fields(profile, schema, vector_field_names)

    # Raises RedisVLMCPError(INVALID_FILTER) naming the offending field.
    resolve_locked_filter(profile, schema)


def _iter_filter_clauses(node: Any) -> Any:
    """Yield every leaf clause of a JSON filter DSL expression."""
    if not isinstance(node, dict):
        return
    for operator in ("and", "or"):
        if operator in node:
            for child in node[operator] or []:
                yield from _iter_filter_clauses(child)
            return
    if "not" in node:
        yield from _iter_filter_clauses(node["not"])
        return
    yield node


def _validate_locked_exists_fields(
    profile: MCPCustomToolConfig,
    schema: IndexSchema,
    vector_field_names: set[str],
) -> None:
    """Reject a locked ``exists`` clause the index cannot actually answer.

    ``exists`` compiles to ``ismissing``, which Redis refuses unless the field was
    declared with ``INDEXMISSING``. The DSL also lets ``exists`` through for vector
    fields, which no other operator allows. Either way the profile parses fine and
    then fails every single call as a *retryable* backend error, so a model retries
    a permanently broken tool forever. Catching it here turns that into a startup
    failure the operator sees once.
    """
    if profile.lock.filter is None:
        return

    for clause in _iter_filter_clauses(profile.lock.filter):
        if str(clause.get("op", "")).lower() != "exists":
            continue
        field_name = clause.get("field")
        field = schema.fields.get(field_name)
        if field is None:
            continue  # resolve_locked_filter reports unknown fields.

        if field_name in vector_field_names:
            raise ValueError(
                f"custom_tools '{profile.name}' lock.filter cannot use 'exists' on "
                f"vector field '{field_name}'"
            )
        if not getattr(field.attrs, "index_missing", False):
            raise ValueError(
                f"custom_tools '{profile.name}' lock.filter uses 'exists' on field "
                f"'{field_name}', which is not indexed with INDEXMISSING; every "
                "call would fail. Declare the field with INDEXMISSING or drop the "
                "'exists' clause."
            )


def _build_signature(
    profile: MCPCustomToolConfig,
) -> tuple[inspect.Signature, dict[str, Any]]:
    """Build the wrapper signature from the arguments a profile exposes."""
    limit_cap = profile.param_max("limit")

    parameters = []
    annotations: dict[str, Any] = {}
    for name, annotation, default in _PROFILE_PARAM_SPECS:
        if not profile.param_exposed(name):
            continue
        if name == "limit" and limit_cap is not None:
            # Publish the cap in the schema rather than only enforcing it. Left
            # off, the model's only way to discover the ceiling is to exceed it
            # and read the error.
            annotation = Annotated[Optional[int], Field(le=limit_cap)]
        parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
        )
        annotations[name] = annotation
    return inspect.Signature(parameters), annotations


def register_profile_tool(
    server: Any,
    profile: MCPCustomToolConfig,
    binding_id: str,
    schema: IndexSchema,
) -> None:
    """Register one declarative profile as an MCP tool.

    Three things are settled here rather than per call: the locked filter is
    parsed once against the bound schema, the wrapper's signature is narrowed to
    the exposed arguments (which is what FastMCP advertises), and a hidden
    ``limit`` collapses its cap into a fixed result count. The wrapper then
    ignores anything the profile does not expose, so a lock holds even if a
    caller reaches it without schema validation.
    """
    locked_filter = resolve_locked_filter(profile, schema)
    locked_return_fields = profile.lock.return_fields
    limit_cap = profile.param_max("limit")
    exposes_limit = profile.param_exposed("limit")
    exposed = {
        name for name, _, _ in _PROFILE_PARAM_SPECS if profile.param_exposed(name)
    }
    signature, annotations = _build_signature(profile)

    async def profile_tool(**kwargs: Any) -> dict[str, Any]:
        auth_config = getattr(server, "auth_config", None)
        read_scope = auth_config.read_scope if auth_config is not None else None
        ensure_tool_scope(server, read_scope)

        # A hidden argument is already absent from the advertised schema, so a
        # compliant client cannot send one. Ignoring it here too means the lock
        # does not depend on the client or the schema layer holding up.
        def supplied(name: str, default: Any = None) -> Any:
            return kwargs.get(name, default) if name in exposed else default

        caller_filter = supplied("filter")
        if caller_filter is not None and not isinstance(caller_filter, dict):
            # A profile advertises `filter` as an object, so a string can only
            # arrive from a client that ignored the schema. Refuse it here rather
            # than passing raw RediSearch syntax through: with no locked filter
            # `merge_locked_filter` has nothing to combine and would forward it
            # verbatim.
            raise RedisVLMCPError(
                "filter must be an object; this tool does not accept raw string "
                "filters",
                code=MCPErrorCode.INVALID_FILTER,
                retryable=False,
            )

        limit = supplied("limit")
        if not exposes_limit:
            # With the argument hidden, a declared cap becomes the fixed result
            # count; without one, the binding's default applies.
            limit = limit_cap

        return await search_records(
            server,
            query=supplied("query", ""),
            index=binding_id,
            limit=limit,
            offset=supplied("offset", 0),
            filter=caller_filter,
            return_fields=(
                locked_return_fields
                if locked_return_fields is not None
                else supplied("return_fields")
            ),
            locked_filter=locked_filter,
            # Passed down rather than checked here: an omitted limit resolves to
            # the binding default inside search_records, so that is the only
            # place able to bound both paths.
            limit_cap=limit_cap,
        )

    profile_tool.__name__ = profile.name.replace("-", "_")
    profile_tool.__doc__ = profile.description.strip()
    profile_tool.__signature__ = signature  # type: ignore[attr-defined]
    profile_tool.__annotations__ = {**annotations, "return": dict[str, Any]}

    server.tool(
        name=profile.name,
        description=build_profile_description(profile, schema),
    )(profile_tool)


def register_profile_tools(server: Any) -> list[str]:
    """Register every configured profile, returning the names registered."""
    config = getattr(server, "config", None)
    if config is None or not config.custom_tools:
        return []

    registered: list[str] = []
    for profile in config.custom_tools:
        binding_id = config.resolved_profile_index(profile)
        runtime = server.resolve_binding(binding_id)
        register_profile_tool(server, profile, binding_id, runtime.schema)
        registered.append(profile.name)
    return registered
