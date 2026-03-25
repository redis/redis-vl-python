import asyncio
import inspect
from typing import Any, Optional

from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError, map_exception
from redisvl.mcp.filters import parse_filter
from redisvl.query import AggregateHybridQuery, HybridQuery, TextQuery, VectorQuery

DEFAULT_SEARCH_DESCRIPTION = "Search records in the configured Redis index."


def _validate_request(
    *,
    query: str,
    search_type: str,
    limit: Optional[int],
    offset: int,
    return_fields: Optional[list[str]],
    server: Any,
    index: Any,
) -> tuple[int, list[str]]:
    """Validate the MCP search request and resolve effective request defaults.

    This function enforces the public MCP contract for `search-records` before
    any RedisVL query objects are constructed. It also derives the default
    return-field projection from the effective index schema.
    """
    runtime = server.config.runtime

    if not isinstance(query, str) or not query.strip():
        raise RedisVLMCPError(
            "query must be a non-empty string",
            code=MCPErrorCode.INVALID_REQUEST,
            retryable=False,
        )
    if search_type not in {"vector", "fulltext", "hybrid"}:
        raise RedisVLMCPError(
            "search_type must be one of: vector, fulltext, hybrid",
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
    result: dict[str, Any], score_field: str, score_type: str
) -> dict[str, Any]:
    """Convert one RedisVL search result into the stable MCP result shape.

    RedisVL and redis-py expose scores and document identifiers under slightly
    different field names depending on the query type, so normalization happens
    here before the MCP response is returned.
    """
    score = result.get(score_field)
    if score is None and score_field == "score":
        score = result.get("__score")
    if score is None:
        raise RedisVLMCPError(
            f"Search result missing expected score field '{score_field}'",
            code=MCPErrorCode.INVALID_REQUEST,
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
            code=MCPErrorCode.INVALID_REQUEST,
            retryable=False,
        )

    for field_name in (
        "vector_distance",
        "score",
        "__score",
        "text_score",
        "vector_similarity",
        "hybrid_score",
    ):
        record.pop(field_name, None)

    return {
        "id": doc_id,
        "score": float(score),
        "score_type": score_type,
        "record": record,
    }


async def _embed_query(vectorizer: Any, query: str) -> Any:
    """Embed the user query through either an async or sync vectorizer API."""
    if hasattr(vectorizer, "aembed"):
        return await vectorizer.aembed(query)
    embed = getattr(vectorizer, "embed")
    if inspect.iscoroutinefunction(embed):
        return await embed(query)
    return await asyncio.to_thread(embed, query)


async def _build_query(
    *,
    server: Any,
    index: Any,
    query: str,
    search_type: str,
    limit: int,
    offset: int,
    filter_value: str | dict[str, Any] | None,
    return_fields: list[str],
) -> tuple[Any, str, str]:
    """Build the RedisVL query object and score metadata for one search mode.

    Returns the constructed query object along with the raw score field name and
    the stable MCP `score_type` label that the response should expose.
    """
    runtime = server.config.runtime
    num_results = limit + offset
    filter_expression = parse_filter(filter_value, index.schema)

    if search_type == "vector":
        vectorizer = await server.get_vectorizer()
        embedding = await _embed_query(vectorizer, query)
        return (
            VectorQuery(
                vector=embedding,
                vector_field_name=runtime.vector_field_name,
                filter_expression=filter_expression,
                return_fields=return_fields,
                num_results=num_results,
                normalize_vector_distance=True,
            ),
            "vector_distance",
            "vector_distance_normalized",
        )

    if search_type == "fulltext":
        return (
            TextQuery(
                text=query,
                text_field_name=runtime.text_field_name,
                filter_expression=filter_expression,
                return_fields=return_fields,
                num_results=num_results,
                stopwords=None,
            ),
            "score",
            "text_score",
        )

    vectorizer = await server.get_vectorizer()
    embedding = await _embed_query(vectorizer, query)
    if await server.supports_native_hybrid_search():
        native_query = HybridQuery(
            text=query,
            text_field_name=runtime.text_field_name,
            vector=embedding,
            vector_field_name=runtime.vector_field_name,
            filter_expression=filter_expression,
            return_fields=["__key", *return_fields],
            num_results=num_results,
            stopwords=None,
            combination_method="LINEAR",
            linear_alpha=0.7,
            yield_text_score_as="text_score",
            yield_vsim_score_as="vector_similarity",
            yield_combined_score_as="hybrid_score",
        )
        native_query.postprocessing_config.apply(__key="@__key")
        return (
            native_query,
            "hybrid_score",
            "hybrid_score",
        )

    fallback_query = AggregateHybridQuery(
        text=query,
        text_field_name=runtime.text_field_name,
        vector=embedding,
        vector_field_name=runtime.vector_field_name,
        filter_expression=filter_expression,
        return_fields=["__key", *return_fields],
        num_results=num_results,
        stopwords=None,
    )
    return (
        fallback_query,
        "hybrid_score",
        "hybrid_score",
    )


async def search_records(
    server: Any,
    *,
    query: str,
    search_type: str = "vector",
    limit: Optional[int] = None,
    offset: int = 0,
    filter: str | dict[str, Any] | None = None,
    return_fields: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Execute `search-records` against the server's configured Redis index."""
    try:
        index = await server.get_index()
        effective_limit, effective_return_fields = _validate_request(
            query=query,
            search_type=search_type,
            limit=limit,
            offset=offset,
            return_fields=return_fields,
            server=server,
            index=index,
        )
        built_query, score_field, score_type = await _build_query(
            server=server,
            index=index,
            query=query.strip(),
            search_type=search_type,
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
                _normalize_record(result, score_field, score_type)
                for result in sliced_results
            ],
        }
    except RedisVLMCPError:
        raise
    except Exception as exc:
        raise map_exception(exc) from exc


def register_search_tool(server: Any) -> None:
    """Register the MCP search tool on a server-like object."""
    description = (
        server.mcp_settings.tool_search_description or DEFAULT_SEARCH_DESCRIPTION
    )

    async def search_records_tool(
        query: str,
        search_type: str = "vector",
        limit: Optional[int] = None,
        offset: int = 0,
        filter: str | dict[str, Any] | None = None,
        return_fields: Optional[list[str]] = None,
    ):
        """FastMCP wrapper for the `search-records` tool."""
        return await search_records(
            server,
            query=query,
            search_type=search_type,
            limit=limit,
            offset=offset,
            filter=filter,
            return_fields=return_fields,
        )

    server.tool(name="search-records", description=description)(search_records_tool)
