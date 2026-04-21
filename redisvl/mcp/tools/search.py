import asyncio
import inspect
from typing import Any

from redisvl.mcp.config import reserved_score_metadata_field_names
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError, map_exception
from redisvl.mcp.filters import parse_filter
from redisvl.query import AggregateHybridQuery, HybridQuery, TextQuery, VectorQuery

DEFAULT_SEARCH_DESCRIPTION = "Search records in the configured Redis index."

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


def _validate_request(
    *,
    query: str,
    limit: int | None,
    offset: int,
    return_fields: list[str] | None,
    server: Any,
    index: Any,
) -> tuple[int, list[str]]:
    """Validate a `search-records` request and resolve default projection.

    The MCP caller can only supply query text, pagination, filters, and return
    fields. Search mode and tuning are sourced from config, so this validation
    step focuses only on the public request contract.
    """

    runtime = server.config.runtime

    if not isinstance(query, str) or not query.strip():
        raise RedisVLMCPError(
            "query must be a non-empty string",
            code=MCPErrorCode.INVALID_REQUEST,
            retryable=False,
        )

    effective_limit = runtime.default_limit if limit is None else limit
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

    schema_fields = set(index.schema.field_names)
    vector_field_name = runtime.vector_field_name

    if return_fields is None:
        fields = [
            field_name
            for field_name in index.schema.field_names
            if field_name != vector_field_name
        ]
    else:
        if not isinstance(return_fields, list):
            raise RedisVLMCPError(
                "return_fields must be a list of field names",
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
            if field_name == vector_field_name:
                raise RedisVLMCPError(
                    f"Vector field '{vector_field_name}' cannot be returned",
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


def _get_configured_search(server: Any) -> tuple[str, dict[str, Any]]:
    """Return the configured search mode and normalized query params."""
    search_config = server.config.search
    return search_config.type, search_config.to_query_params()


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


async def _build_query(
    *,
    server: Any,
    index: Any,
    query: str,
    limit: int,
    offset: int,
    filter_value: str | dict[str, Any] | None,
    return_fields: list[str],
) -> tuple[Any, str, str, str]:
    """Build the RedisVL query object from configured search mode and params.

    Returns the query instance, the raw score field to read from RedisVL
    results, the public MCP `score_type`, and the configured `search_type`.
    """
    runtime = server.config.runtime
    search_type, search_params = _get_configured_search(server)
    num_results = limit + offset
    filter_expression = parse_filter(filter_value, index.schema)

    if search_type == "vector":
        vectorizer = await server.get_vectorizer()
        embedding = await _embed_query(vectorizer, query)
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

    vectorizer = await server.get_vectorizer()
    embedding = await _embed_query(vectorizer, query)
    if await server.supports_native_hybrid_search():
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
    limit: int | None = None,
    offset: int = 0,
    filter: str | dict[str, Any] | None = None,
    return_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Execute `search-records` against the configured Redis index binding."""
    try:
        index = await server.get_index()
        effective_limit, effective_return_fields = _validate_request(
            query=query,
            limit=limit,
            offset=offset,
            return_fields=return_fields,
            server=server,
            index=index,
        )
        built_query, score_field, score_type, search_type = await _build_query(
            server=server,
            index=index,
            query=query.strip(),
            limit=effective_limit,
            offset=offset,
            filter_value=filter,
            return_fields=effective_return_fields,
        )
        raw_results = await server.run_guarded(
            "search-records",
            index.query(built_query),
        )
        sliced_results = raw_results[offset : offset + effective_limit]
        return {
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


def register_search_tool(server: Any) -> None:
    """Register the MCP `search-records` tool with its config-owned contract."""
    description = (
        server.mcp_settings.tool_search_description or DEFAULT_SEARCH_DESCRIPTION
    )

    async def search_records_tool(
        query: str,
        limit: int | None = None,
        offset: int = 0,
        filter: str | dict[str, Any] | None = None,
        return_fields: list[str] | None = None,
    ):
        """FastMCP wrapper for the `search-records` tool."""
        return await search_records(
            server,
            query=query,
            limit=limit,
            offset=offset,
            filter=filter,
            return_fields=return_fields,
        )

    server.tool(name="search-records", description=description)(search_records_tool)
