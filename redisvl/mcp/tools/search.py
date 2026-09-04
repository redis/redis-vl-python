import asyncio
import inspect
from typing import Any

from redisvl.mcp.auth import ensure_tool_scope
from redisvl.mcp.config import reserved_score_metadata_field_names
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError, map_exception
from redisvl.mcp.filters import parse_filter
from redisvl.query import AggregateHybridQuery, HybridQuery, TextQuery, VectorQuery
from redisvl.query.filter import FilterExpression
from redisvl.schema import IndexSchema

DEFAULT_SEARCH_DESCRIPTION = "Search records in the configured Redis index."
_DSL_FILTER_FIELD_TYPES = frozenset({"tag", "text", "numeric"})

_NATIVE_HYBRID_DEFAULTS = {
    "combination_method": "LINEAR",
    "linear_text_weight": 0.3,
}
_FALLBACK_HYBRID_UNSUPPORTED_PARAMS = frozenset(
    {
        "vector_search_method",
        "knn_ef_runtime",
        "range_radius",
        "range_epsilon",
        "rrf_window",
        "rrf_constant",
    }
)


def _build_filter_hint(schema: IndexSchema) -> str:
    """Describe fields with typed operator support in the JSON filter DSL."""
    filter_fields = [
        f"{field.name}({getattr(field.type, 'value', field.type)})"
        for field in schema.fields.values()
        if field.type in _DSL_FILTER_FIELD_TYPES
    ]
    if not filter_fields:
        return "Object filter fields: none."
    return "Object filter fields: " + ", ".join(filter_fields) + "."


def _build_return_fields_hint(schema: IndexSchema) -> str:
    """Describe all fields that callers can request in `return_fields`."""
    returnable_fields = [
        field.name for field in schema.fields.values() if field.type != "vector"
    ]
    if not returnable_fields:
        return "Allowed return_fields: none."
    return "Allowed return_fields: " + ", ".join(returnable_fields) + "."


def _build_search_tool_description(
    schema: IndexSchema | None,
    base_description: str | None = None,
    *,
    index_ids: list[str] | None = None,
) -> str:
    """Build the `search-records` description from static text plus schema hints.

    With multiple bindings configured the schema is ambiguous (the caller picks
    an index per call), so per-field hints are omitted and a routing note is
    appended instead.

    ``index_ids`` is supplied only when discovery is unavailable -- an operator
    can disable ``list-indexes``, and pointing clients at a tool the server does
    not publish would leave them unable to satisfy the required ``index``
    argument at all. Naming the ids inline is the only way they can learn them.
    """
    description = (base_description or DEFAULT_SEARCH_DESCRIPTION).strip()
    if schema is None:
        if index_ids:
            return (
                description + " Multiple indexes are configured and discovery is "
                "disabled: pass one of these index ids as the `index` argument: "
                + ", ".join(index_ids)
                + "."
            )
        return (
            description + " Multiple indexes are configured: call list-indexes "
            "first, then pass the chosen index id as the `index` argument."
        )

    # `exists` is currently accepted for any schema field in the MCP object filter.
    exists_fields = [field.name for field in schema.fields.values()]
    if exists_fields:
        exists_hint = "Object filter exists support: " + ", ".join(exists_fields) + "."
    else:
        exists_hint = "Object filter exists support: none."

    return " ".join(
        [
            description,
            _build_filter_hint(schema),
            exists_hint,
            _build_return_fields_hint(schema),
        ]
    )


def _validate_request(
    *,
    query: str,
    limit: int | None,
    offset: int,
    return_fields: list[str] | None,
    runtime: Any,
    schema: Any,
    limit_cap: int | None = None,
) -> tuple[int, list[str]]:
    """Validate a `search-records` request and resolve default projection.

    The MCP caller can only supply query text, pagination, filters, and return
    fields. Search mode and tuning are sourced from the selected binding's
    config, so this validation step focuses only on the public request contract.

    ``limit_cap`` is a server-supplied ceiling from a custom tool profile. It has
    to be applied here rather than in the caller, because an omitted ``limit``
    resolves to the binding default at this point -- capping only the explicit
    value would let the default sail past the cap.
    """

    if not isinstance(query, str) or not query.strip():
        raise RedisVLMCPError(
            "query must be a non-empty string",
            code=MCPErrorCode.INVALID_REQUEST,
            retryable=False,
        )

    if limit is None:
        # An unspecified limit is bounded silently: the caller never named a
        # number, so there is nothing to reject.
        effective_limit = runtime.default_limit
        if limit_cap is not None:
            effective_limit = min(effective_limit, limit_cap)
    else:
        effective_limit = limit
        if limit_cap is not None and effective_limit > limit_cap:
            raise RedisVLMCPError(
                f"limit must be less than or equal to {limit_cap}",
                code=MCPErrorCode.INVALID_REQUEST,
                retryable=False,
            )
    if not isinstance(effective_limit, int) or effective_limit <= 0:
        raise RedisVLMCPError(
            "limit must be greater than 0",
            code=MCPErrorCode.INVALID_REQUEST,
            retryable=False,
        )
    if effective_limit > runtime.max_limit:
        raise RedisVLMCPError(
            f"limit must be less than or equal to {runtime.max_limit}",
            code=MCPErrorCode.INVALID_REQUEST,
            retryable=False,
        )
    if not isinstance(offset, int) or offset < 0:
        raise RedisVLMCPError(
            "offset must be greater than or equal to 0",
            code=MCPErrorCode.INVALID_REQUEST,
            retryable=False,
        )
    if offset + effective_limit > runtime.max_result_window:
        raise RedisVLMCPError(
            "offset + limit must be less than or equal to "
            f"{runtime.max_result_window}",
            code=MCPErrorCode.INVALID_REQUEST,
            retryable=False,
        )

    schema_fields = set(schema.field_names)
    vector_field_names = {
        field_name
        for field_name, field in schema.fields.items()
        if field.type == "vector"
    }

    if return_fields is None:
        fields = [
            field_name
            for field_name in schema.field_names
            if field_name not in vector_field_names
        ]
    else:
        if not isinstance(return_fields, list):
            raise RedisVLMCPError(
                "return_fields must be a list of field names",
                code=MCPErrorCode.INVALID_REQUEST,
                retryable=False,
            )
        if not return_fields:
            # An empty projection reaches Redis as no RETURN clause at all, which
            # returns *every* field -- including the vector this function refuses
            # a few lines down. "Omit the argument" is the way to ask for the
            # default projection.
            raise RedisVLMCPError(
                "return_fields must not be empty; omit it to use the default "
                "projection",
                code=MCPErrorCode.INVALID_REQUEST,
                retryable=False,
            )
        fields = []
        for field_name in return_fields:
            if not isinstance(field_name, str) or not field_name:
                raise RedisVLMCPError(
                    "return_fields must contain non-empty strings",
                    code=MCPErrorCode.INVALID_REQUEST,
                    retryable=False,
                )
            if field_name not in schema_fields:
                raise RedisVLMCPError(
                    f"Unknown return field '{field_name}'",
                    code=MCPErrorCode.INVALID_REQUEST,
                    retryable=False,
                )
            if field_name in vector_field_names:
                raise RedisVLMCPError(
                    f"Vector field '{field_name}' cannot be returned",
                    code=MCPErrorCode.INVALID_REQUEST,
                    retryable=False,
                )
            fields.append(field_name)

    return effective_limit, fields


def _normalize_record(
    result: dict[str, Any],
    score_field: str,
    score_type: str,
) -> dict[str, Any]:
    """Convert one RedisVL result into the stable MCP result shape."""
    score = result.get(score_field)
    if score_field == "score" and "__score" in result:
        score = result["__score"]
    if score is None:
        raise RedisVLMCPError(
            f"Search result missing expected score field '{score_field}'",
            code=MCPErrorCode.INTERNAL_ERROR,
            retryable=False,
        )

    record = dict(result)
    doc_id = record.pop("id", None)
    if doc_id is None:
        doc_id = record.pop("__key", None)
    if doc_id is None:
        doc_id = record.pop("key", None)
    if doc_id is None:
        raise RedisVLMCPError(
            "Search result missing id",
            code=MCPErrorCode.INTERNAL_ERROR,
            retryable=False,
        )

    for field_name in reserved_score_metadata_field_names():
        record.pop(field_name, None)

    return {
        "id": doc_id,
        "score": float(score),
        "score_type": score_type,
        "record": record,
    }


async def _embed_query(vectorizer: Any, query: str) -> Any:
    """Embed the query text, tolerating vectorizers without real async support."""
    aembed = getattr(vectorizer, "aembed", None)
    if callable(aembed):
        try:
            return await aembed(query)
        except NotImplementedError:
            pass
    embed = getattr(vectorizer, "embed")
    if inspect.iscoroutinefunction(embed):
        return await embed(query)
    return await asyncio.to_thread(embed, query)


def _get_configured_search(rt: Any) -> tuple[str, dict[str, Any]]:
    """Return the binding's configured search mode and normalized query params."""
    search_config = rt.binding.search
    return search_config.type, search_config.to_query_params()


def _require_vectorizer(rt: Any) -> Any:
    """Return the binding's vectorizer or fail when it is not configured."""
    if rt.vectorizer is None:
        raise RuntimeError("MCP server vectorizer is not configured")
    return rt.vectorizer


def _build_native_hybrid_kwargs(
    *,
    query: str,
    embedding: Any,
    runtime: Any,
    filter_expression: Any,
    return_fields: list[str],
    num_results: int,
    search_params: dict[str, Any],
) -> dict[str, Any]:
    """Build native `HybridQuery` kwargs from MCP config-owned hybrid params."""
    if runtime.text_field_name is None or runtime.vector_field_name is None:
        raise RuntimeError("Hybrid search requires configured text and vector fields")

    params = dict(search_params)
    combination_method = params.setdefault(
        "combination_method",
        _NATIVE_HYBRID_DEFAULTS["combination_method"],
    )
    if combination_method == "LINEAR":
        linear_text_weight = params.pop(
            "linear_text_weight",
            _NATIVE_HYBRID_DEFAULTS["linear_text_weight"],
        )
        params["linear_alpha"] = linear_text_weight
    else:
        params.pop("linear_text_weight", None)

    return {
        "text": query,
        "text_field_name": runtime.text_field_name,
        "vector": embedding,
        "vector_field_name": runtime.vector_field_name,
        "filter_expression": filter_expression,
        "return_fields": ["__key", *return_fields],
        "num_results": num_results,
        "yield_text_score_as": "text_score",
        "yield_vsim_score_as": "vector_similarity",
        "yield_combined_score_as": "hybrid_score",
        **params,
    }


def _build_fallback_hybrid_kwargs(
    *,
    query: str,
    embedding: Any,
    runtime: Any,
    filter_expression: Any,
    return_fields: list[str],
    num_results: int,
    search_params: dict[str, Any],
) -> dict[str, Any]:
    """Build aggregate fallback kwargs while preserving MCP fusion semantics."""
    if runtime.text_field_name is None or runtime.vector_field_name is None:
        raise RuntimeError("Hybrid search requires configured text and vector fields")

    params = dict(search_params)
    linear_text_weight = params.pop(
        "linear_text_weight",
        _NATIVE_HYBRID_DEFAULTS["linear_text_weight"],
    )
    params.pop("combination_method", None)
    for key in _FALLBACK_HYBRID_UNSUPPORTED_PARAMS:
        params.pop(key, None)
    params["alpha"] = 1 - linear_text_weight

    return {
        "text": query,
        "text_field_name": runtime.text_field_name,
        "vector": embedding,
        "vector_field_name": runtime.vector_field_name,
        "filter_expression": filter_expression,
        "return_fields": ["__key", *return_fields],
        "num_results": num_results,
        **params,
    }


def _find_unescaped(
    rendered: str, position: int, character: str, end: int | None = None
) -> int | None:
    """Return the index of the next unescaped `character` before `end`, or None.

    `end` bounds the scan. Without it a caller inside a span pays for the whole
    remaining string, which makes the enclosing walk quadratic in the number of
    spans.
    """
    limit = len(rendered) if end is None else min(end, len(rendered))
    while position < limit:
        if rendered[position] == "\\":
            position += 2
            continue
        if rendered[position] == character:
            return position
        position += 1
    return None


def _reject_escapable_filter(caller: FilterExpression) -> None:
    """Refuse a caller filter whose rendering could break out of a locked AND.

    Outside a quoted phrase every value reaches the rendering escaped or
    type-checked -- by the DSL, or at this server's filter boundary for ``like``
    patterns, which the library leaves raw. So a rejection means one of those
    failed. It is a backstop, not the primary defense -- see
    ``merge_locked_filter`` -- and it runs only when a lock exists, so an
    unlocked tool relies on those two layers alone.

    The quoted-phrase skip below assumes a rendering's quotes arrive in
    delimiting pairs, which holds for every value the DSL can render: ``==`` and
    ``!=`` quote the value and take any quote it carried out first, and ``Tag``
    and ``like`` values arrive with theirs escaped, outside any phrase. If that
    ever stops holding, the skip finds no closing quote and refuses the filter,
    so the failure is loud rather than silent.
    """
    # Braces scope as much as parens do: a tag clause holds its alternatives in
    # braces, so `@category:{sports|health}` is one scoped clause rather than a
    # union. Counting parens alone would reject the most ordinary narrowing
    # filter a caller can send.
    openers = {"(", "{"}
    closers = {")", "}"}

    rendered = str(caller)
    position = 0
    depth = 0
    escaped = False

    while position < len(rendered):
        character = rendered[position]

        if character == "\\":
            # Consume the pair. Checking whether the *previous* character was a
            # backslash instead would misread `\\` -- a literal backslash -- as
            # protecting whatever follows it, and a real delimiter after one
            # would slip through unnoticed.
            position += 2
            continue

        if character == '"':
            # A quoted phrase is a literal, so nothing inside it is structure and
            # a `|` inside it is a separator rather than a union. Unlike the range
            # span below, the whole phrase can therefore be skipped.
            #
            # The scan deliberately does not treat `\` as escaping the closing
            # quote, because RediSearch does not either: `@f:("x\")` closes the
            # phrase and yields the term `x\`. Honouring the escape would read
            # that as unterminated and refuse an ordinary value ending in a
            # backslash, a Windows path among them. Nothing is given up, since a
            # value cannot contribute a quote of its own -- `Text` replaces it,
            # and `Tag` and `like` escape theirs outside any phrase.
            end = rendered.find('"', position + 1)
            if end == -1:
                # Unterminated phrase: the remainder is unparsable.
                escaped = True
                break
            position = end + 1
            continue

        if character == "[":
            # A numeric range is bounds, not structure: an exclusive bound
            # renders as `[(5 +inf]`, where `(` is a marker rather than a group.
            # Skipping the span keeps those parens out of the depth count -- but
            # suppressing paren depth is the *only* reason to skip, so `|`
            # detection has to stay active inside it. A legitimate numeric or geo
            # range holds numbers, so a `|` in here means a value reached the
            # query string raw, which is exactly the case this backstop exists
            # to catch; skipping past it would let a union hide behind brackets.
            span_end = _find_unescaped(rendered, position + 1, "]")
            if span_end is None:
                escaped = True
                break
            if _find_unescaped(rendered, position + 1, "|", span_end) is not None:
                escaped = True
                break
            position = span_end + 1
            continue

        if character in openers:
            depth += 1
        elif character in closers:
            depth -= 1
            if depth < 0:
                # Closed more than was opened, which would terminate the locked
                # group early and leave the rest as bare syntax.
                escaped = True
                break
        elif character == "]":
            # Only reachable outside a range span, so the brackets are unbalanced.
            escaped = True
            break
        elif character == "|" and depth == 0:
            # Under DIALECT 2, `|` binds looser than the implicit intersection,
            # so one at the top level turns `locked AND caller` into a union and
            # the lock stops constraining anything.
            escaped = True
            break

        position += 1

    # Leftover depth means an unclosed group, which would swallow whatever the
    # AND appends after it.
    if escaped or depth != 0:
        raise RedisVLMCPError(
            "filter could not be safely combined with this tool's locked filter",
            code=MCPErrorCode.INVALID_FILTER,
            retryable=False,
        )


def merge_locked_filter(
    locked: FilterExpression | None,
    caller: str | FilterExpression | None,
) -> str | FilterExpression | None:
    """AND-combine an author-locked filter with a caller-supplied one.

    The locked expression always applies, so a caller can only narrow within it
    and never widen past it. That rests on two things: the caller's expression
    rendering nested inside the AND, and every value staying inside its own
    clause. ``Text`` removes the quote that delimits a phrase, ``Tag`` escapes
    the braces that delimit a tag clause, numeric values are type-checked, and
    ``like`` patterns are escaped at the filter boundary.
    """
    if locked is None:
        return caller
    if caller is None:
        return locked

    if not isinstance(caller, FilterExpression):
        # Strings skip the DSL's field validation and have no safe composition
        # with an expression -- combining them means concatenation, where a
        # crafted value can close the locked group and escape it.
        raise RedisVLMCPError(
            "filter must be an object when this tool locks a filter; "
            "raw string filters cannot be combined with a locked filter",
            code=MCPErrorCode.INVALID_FILTER,
            retryable=False,
        )

    # Structure is necessary but not sufficient, so check the rendering too.
    _reject_escapable_filter(caller)

    # `__and__` wraps the pair and parenthesizes each compound side, so a
    # caller's `or`/`not` cannot hoist itself to the top level. (Single clauses
    # render bare; only compound sides could otherwise escape.)
    return locked & caller


async def _build_query(
    *,
    rt: Any,
    query: str,
    limit: int,
    offset: int,
    filter_value: str | dict[str, Any] | None,
    return_fields: list[str],
    locked_filter: FilterExpression | None = None,
) -> tuple[Any, str, str, str]:
    """Build the RedisVL query object from the binding's search mode and params.

    Returns the query instance, the raw score field to read from RedisVL
    results, the public MCP `score_type`, and the configured `search_type`.
    """
    runtime = rt.binding.runtime
    search_type, search_params = _get_configured_search(rt)
    num_results = limit + offset
    filter_expression = merge_locked_filter(
        locked_filter, parse_filter(filter_value, rt.schema)
    )

    if search_type == "vector":
        if runtime.vector_field_name is None:
            raise RuntimeError("Vector search requires a configured vector field")
        embedding = await _embed_query(_require_vectorizer(rt), query)
        vector_kwargs = {
            "vector": embedding,
            "vector_field_name": runtime.vector_field_name,
            "filter_expression": filter_expression,
            "return_fields": return_fields,
            "num_results": num_results,
            **search_params,
        }
        if "normalize_vector_distance" not in vector_kwargs:
            vector_kwargs["normalize_vector_distance"] = True
        normalize_vector_distance = vector_kwargs["normalize_vector_distance"]
        return (
            VectorQuery(**vector_kwargs),
            "vector_distance",
            (
                "vector_distance_normalized"
                if normalize_vector_distance
                else "vector_distance"
            ),
            search_type,
        )

    if search_type == "fulltext":
        if runtime.text_field_name is None:
            raise RuntimeError("Full-text search requires a configured text field")
        return (
            TextQuery(
                text=query,
                text_field_name=runtime.text_field_name,
                filter_expression=filter_expression,
                return_fields=return_fields,
                num_results=num_results,
                **search_params,
            ),
            "score",
            "text_score",
            search_type,
        )

    embedding = await _embed_query(_require_vectorizer(rt), query)
    if rt.supports_native_hybrid_search:
        native_query = HybridQuery(
            **_build_native_hybrid_kwargs(
                query=query,
                embedding=embedding,
                runtime=runtime,
                filter_expression=filter_expression,
                return_fields=return_fields,
                num_results=num_results,
                search_params=search_params,
            )
        )
        native_query.postprocessing_config.apply(__key="@__key")
        return (
            native_query,
            "hybrid_score",
            "hybrid_score",
            search_type,
        )

    fallback_query = AggregateHybridQuery(
        **_build_fallback_hybrid_kwargs(
            query=query,
            embedding=embedding,
            runtime=runtime,
            filter_expression=filter_expression,
            return_fields=return_fields,
            num_results=num_results,
            search_params=search_params,
        )
    )
    return (
        fallback_query,
        "hybrid_score",
        "hybrid_score",
        search_type,
    )


async def search_records(
    server: Any,
    *,
    query: str,
    index: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    filter: str | dict[str, Any] | None = None,
    return_fields: list[str] | None = None,
    locked_filter: FilterExpression | None = None,
    limit_cap: int | None = None,
) -> dict[str, Any]:
    """Execute `search-records` against the selected Redis index binding.

    ``index`` names the logical binding to query. It is optional when exactly
    one binding is configured (preserving single-index behavior) and required
    when multiple bindings exist. The resolved logical id is echoed back in the
    response so multi-index clients can confirm routing.

    ``locked_filter`` and ``limit_cap`` are server-supplied and never reach the
    model: custom tool profiles pass an author-locked expression and result
    ceiling here. The filter is AND-combined with ``filter`` so a caller can only
    narrow within it, and the cap bounds the effective limit whether the caller
    named one or fell through to the binding default.
    """
    try:
        rt = server.resolve_binding(index)
        effective_limit, effective_return_fields = _validate_request(
            query=query,
            limit=limit,
            offset=offset,
            return_fields=return_fields,
            runtime=rt.binding.runtime,
            schema=rt.schema,
            limit_cap=limit_cap,
        )
        built_query, score_field, score_type, search_type = await _build_query(
            rt=rt,
            query=query.strip(),
            limit=effective_limit,
            offset=offset,
            filter_value=filter,
            return_fields=effective_return_fields,
            locked_filter=locked_filter,
        )
        raw_results = await server.run_guarded(
            "search-records",
            rt.index.query(built_query),
            timeout_seconds=rt.binding.runtime.request_timeout_seconds,
        )
        sliced_results = raw_results[offset : offset + effective_limit]
        return {
            "index": rt.binding_id,
            "search_type": search_type,
            "offset": offset,
            "limit": effective_limit,
            "results": [
                _normalize_record(
                    result,
                    score_field,
                    score_type,
                )
                for result in sliced_results
            ],
        }
    except RedisVLMCPError:
        raise
    except Exception as exc:
        raise map_exception(exc) from exc


def register_search_tool(
    server: Any, schema: IndexSchema | None, *, index_ids: list[str] | None = None
) -> None:
    """Register the MCP `search-records` tool with its config-owned contract."""
    description = _build_search_tool_description(
        index_ids=index_ids,
        schema=schema,
        base_description=server.mcp_settings.tool_search_description,
    )

    async def search_records_tool(
        query: str,
        index: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        filter: str | dict[str, Any] | None = None,
        return_fields: list[str] | None = None,
    ):
        """FastMCP wrapper for the `search-records` tool."""
        auth_config = getattr(server, "auth_config", None)
        read_scope = auth_config.read_scope if auth_config is not None else None
        ensure_tool_scope(server, read_scope)
        return await search_records(
            server,
            query=query,
            index=index,
            limit=limit,
            offset=offset,
            filter=filter,
            return_fields=return_fields,
        )

    server.tool(name="search-records", description=description)(search_records_tool)
