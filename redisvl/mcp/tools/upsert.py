import asyncio
import inspect
from typing import Any, Dict, List, Optional

from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError, map_exception
from redisvl.redis.utils import array_to_buffer
from redisvl.schema.schema import StorageType
from redisvl.schema.validation import validate_object

DEFAULT_UPSERT_DESCRIPTION = "Upsert records in the configured Redis index."


def _validate_request(
    *,
    server: Any,
    records: List[Dict[str, Any]],
    id_field: Optional[str],
    skip_embedding_if_present: Optional[bool],
) -> bool:
    """Validate the public upsert request contract and resolve defaults."""
    runtime = server.config.runtime

    if not isinstance(records, list) or not records:
        raise RedisVLMCPError(
            "records must be a non-empty list",
            code=MCPErrorCode.INVALID_REQUEST,
            retryable=False,
        )
    if len(records) > runtime.max_upsert_records:
        raise RedisVLMCPError(
            "records length must be less than or equal to "
            f"{runtime.max_upsert_records}",
            code=MCPErrorCode.INVALID_REQUEST,
            retryable=False,
        )
    if id_field is not None and (not isinstance(id_field, str) or not id_field):
        raise RedisVLMCPError(
            "id_field must be a non-empty string when provided",
            code=MCPErrorCode.INVALID_REQUEST,
            retryable=False,
        )

    effective_skip_embedding = runtime.skip_embedding_if_present
    if skip_embedding_if_present is not None:
        if not isinstance(skip_embedding_if_present, bool):
            raise RedisVLMCPError(
                "skip_embedding_if_present must be a boolean when provided",
                code=MCPErrorCode.INVALID_REQUEST,
                retryable=False,
            )
        effective_skip_embedding = skip_embedding_if_present

    for record in records:
        if not isinstance(record, dict):
            raise RedisVLMCPError(
                "records must contain only objects",
                code=MCPErrorCode.INVALID_REQUEST,
                retryable=False,
            )
        if id_field is not None and id_field not in record:
            raise RedisVLMCPError(
                "id_field '{id_field}' must exist in every record".format(
                    id_field=id_field
                ),
                code=MCPErrorCode.INVALID_REQUEST,
                retryable=False,
            )

    return effective_skip_embedding


def _record_needs_embedding(
    record: Dict[str, Any],
    *,
    vector_field_name: str,
    skip_embedding_if_present: bool,
) -> bool:
    """Determine whether a record requires server-side embedding."""
    return (
        not skip_embedding_if_present
        or vector_field_name not in record
        or record[vector_field_name] is None
    )


def _validate_embed_sources(
    records: List[Dict[str, Any]],
    *,
    embed_text_field: str,
    vector_field_name: str,
    skip_embedding_if_present: bool,
) -> List[str]:
    """Collect embed sources for records that require embedding."""
    contents = []
    for record in records:
        if not _record_needs_embedding(
            record,
            vector_field_name=vector_field_name,
            skip_embedding_if_present=skip_embedding_if_present,
        ):
            continue

        content = record.get(embed_text_field)
        if not isinstance(content, str) or not content.strip():
            raise RedisVLMCPError(
                "records requiring embedding must include a non-empty "
                "'{field}' field".format(field=embed_text_field),
                code=MCPErrorCode.INVALID_REQUEST,
                retryable=False,
            )
        contents.append(content)

    return contents


async def _embed_one(vectorizer: Any, content: str) -> List[float]:
    """Embed one record, falling back from async to sync implementations."""
    aembed = getattr(vectorizer, "aembed", None)
    if callable(aembed):
        try:
            return await aembed(content)
        except NotImplementedError:
            pass

    embed = getattr(vectorizer, "embed", None)
    if embed is None:
        raise AttributeError("Configured vectorizer does not support embed()")
    if inspect.iscoroutinefunction(embed):
        return await embed(content)
    return await asyncio.to_thread(embed, content)


async def _embed_many(vectorizer: Any, contents: List[str]) -> List[List[float]]:
    """Embed multiple records with batch-first fallbacks."""
    if not contents:
        return []

    aembed_many = getattr(vectorizer, "aembed_many", None)
    if callable(aembed_many):
        try:
            return await aembed_many(contents)
        except NotImplementedError:
            pass

    embed_many = getattr(vectorizer, "embed_many", None)
    if callable(embed_many):
        if inspect.iscoroutinefunction(embed_many):
            return await embed_many(contents)
        return await asyncio.to_thread(embed_many, contents)

    embeddings = []
    for content in contents:
        embeddings.append(await _embed_one(vectorizer, content))
    return embeddings


def _vector_dtype(server: Any, index: Any) -> str:
    """Resolve the configured vector field datatype as a lowercase string."""
    field = server.config.get_vector_field(index.schema)
    datatype = getattr(field.attrs.datatype, "value", field.attrs.datatype)
    return str(datatype).lower()


def _validation_schema_for_record(
    index: Any,
    *,
    vector_field_name: str,
    record: Dict[str, Any],
) -> Any:
    """Use a JSON-shaped schema when validating list vectors for HASH storage."""
    if index.schema.index.storage_type == StorageType.HASH and isinstance(
        record.get(vector_field_name), list
    ):
        schema = index.schema.model_copy(deep=True)
        schema.index.storage_type = StorageType.JSON
        return schema
    return index.schema


def _validate_record(
    record: Dict[str, Any], *, index: Any, vector_field_name: str
) -> None:
    """Validate one record against the schema, allowing HASH list vectors."""
    validate_object(
        _validation_schema_for_record(
            index,
            vector_field_name=vector_field_name,
            record=record,
        ),
        record,
    )


def _prepare_record_for_storage(
    record: Dict[str, Any],
    *,
    server: Any,
    index: Any,
) -> Dict[str, Any]:
    """Validate records before serializing HASH vectors for storage."""
    prepared = dict(record)
    vector_field_name = server.config.runtime.vector_field_name
    _validate_record(prepared, index=index, vector_field_name=vector_field_name)

    vector_value = prepared.get(vector_field_name)

    if index.schema.index.storage_type == StorageType.HASH:
        if isinstance(vector_value, list):
            prepared[vector_field_name] = array_to_buffer(
                vector_value,
                _vector_dtype(server, index),
            )
    return prepared


async def upsert_records(
    server: Any,
    *,
    records: List[Dict[str, Any]],
    id_field: Optional[str] = None,
    skip_embedding_if_present: Optional[bool] = None,
) -> Dict[str, Any]:
    """Execute `upsert-records` against the configured Redis index."""
    try:
        index = await server.get_index()
        effective_skip_embedding = _validate_request(
            server=server,
            records=records,
            id_field=id_field,
            skip_embedding_if_present=skip_embedding_if_present,
        )
        # Copy caller-provided records before enriching them with embeddings or
        # storage-specific serialization so the MCP tool does not mutate inputs.
        prepared_records = [record.copy() for record in records]
        runtime = server.config.runtime
        for record in prepared_records:
            _validate_record(
                record,
                index=index,
                vector_field_name=runtime.vector_field_name,
            )
        embed_contents = _validate_embed_sources(
            prepared_records,
            embed_text_field=runtime.default_embed_text_field,
            vector_field_name=runtime.vector_field_name,
            skip_embedding_if_present=effective_skip_embedding,
        )

        if embed_contents:
            vectorizer = await server.get_vectorizer()
            embeddings = await _embed_many(vectorizer, embed_contents)
            # Tracks position in the compact embeddings list, which only contains
            # vectors for records that still need server-side embedding.
            embedding_index = 0
            for record in prepared_records:
                if _record_needs_embedding(
                    record,
                    vector_field_name=runtime.vector_field_name,
                    skip_embedding_if_present=effective_skip_embedding,
                ):
                    record[runtime.vector_field_name] = embeddings[embedding_index]
                    embedding_index += 1

        loadable_records = [
            _prepare_record_for_storage(record, server=server, index=index)
            for record in prepared_records
        ]

        try:
            keys = await server.run_guarded(
                "upsert-records",
                index.load(loadable_records, id_field=id_field),
            )
        except Exception as exc:
            mapped = map_exception(exc)
            mapped.metadata["partial_write_possible"] = True
            raise mapped

        return {
            "status": "success",
            "keys_upserted": len(keys),
            "keys": keys,
        }
    except RedisVLMCPError:
        raise
    except Exception as exc:
        raise map_exception(exc)


def register_upsert_tool(server: Any) -> None:
    """Register the MCP upsert tool on a server-like object."""
    description = (
        server.mcp_settings.tool_upsert_description or DEFAULT_UPSERT_DESCRIPTION
    )

    async def upsert_records_tool(
        records: List[Dict[str, Any]],
        id_field: Optional[str] = None,
        skip_embedding_if_present: Optional[bool] = None,
    ):
        """FastMCP wrapper for the `upsert-records` tool."""
        return await upsert_records(
            server,
            records=records,
            id_field=id_field,
            skip_embedding_if_present=skip_embedding_if_present,
        )

    server.tool(name="upsert-records", description=description)(upsert_records_tool)
