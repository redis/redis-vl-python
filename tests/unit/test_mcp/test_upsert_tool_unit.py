from types import SimpleNamespace
from typing import Any, List, Optional

import pytest
from redis.exceptions import RedisError

from redisvl.mcp.config import MCPConfig
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.tools.upsert import register_upsert_tool, upsert_records
from redisvl.redis.utils import array_to_buffer
from redisvl.schema import IndexSchema


def _schema(storage_type: str = "hash") -> IndexSchema:
    return IndexSchema.from_dict(
        {
            "index": {
                "name": "docs-index",
                "prefix": "doc",
                "storage_type": storage_type,
            },
            "fields": [
                {"name": "content", "type": "text"},
                {"name": "category", "type": "tag"},
                {
                    "name": "embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "flat",
                        "dims": 3,
                        "distance_metric": "cosine",
                        "datatype": "float32",
                    },
                },
            ],
        }
    )


def _config(
    storage_type: str = "hash",
    *,
    max_upsert_records: int = 5,
    skip_embedding_if_present: bool = True,
) -> MCPConfig:
    return MCPConfig.model_validate(
        {
            "server": {"redis_url": "redis://localhost:6379"},
            "indexes": {
                "knowledge": {
                    "redis_name": "docs-index",
                    "vectorizer": {"class": "FakeVectorizer", "model": "test-model"},
                    "search": {"type": "vector"},
                    "runtime": {
                        "text_field_name": "content",
                        "vector_field_name": "embedding",
                        "default_embed_text_field": "content",
                        "default_limit": 2,
                        "max_limit": 5,
                        "max_upsert_records": max_upsert_records,
                        "skip_embedding_if_present": skip_embedding_if_present,
                    },
                }
            },
        }
    )


class FakeVectorizer:
    def __init__(self):
        self.aembed_many_calls = []
        self.embed_many_calls = []
        self.aembed_calls = []
        self.embed_calls = []

    async def aembed_many(self, contents: List[str], **kwargs):
        self.aembed_many_calls.append((contents, kwargs))
        return [
            [float(index), float(index), float(index)]
            for index, _ in enumerate(contents, start=1)
        ]

    def embed_many(self, contents: List[str], **kwargs):
        self.embed_many_calls.append((contents, kwargs))
        return [[9.0, 9.0, 9.0] for _ in contents]

    async def aembed(self, content: str, **kwargs):
        self.aembed_calls.append((content, kwargs))
        return [8.0, 8.0, 8.0]

    def embed(self, content: str, **kwargs):
        self.embed_calls.append((content, kwargs))
        return [7.0, 7.0, 7.0]


class FallbackBatchVectorizer(FakeVectorizer):
    async def aembed_many(self, contents: List[str], **kwargs):
        raise NotImplementedError


class FakeIndex:
    def __init__(self, storage_type: str = "hash"):
        self.schema = _schema(storage_type)
        self.load_calls = []
        self.keys_to_return = ["doc:1"]
        self.load_exception = None

    async def load(self, data, id_field=None, **kwargs):
        materialized = list(data)
        self.load_calls.append(
            {
                "data": materialized,
                "id_field": id_field,
                "kwargs": kwargs,
            }
        )
        if self.load_exception is not None:
            raise self.load_exception
        return self.keys_to_return


class FakeServer:
    def __init__(
        self,
        *,
        storage_type: str = "hash",
        max_upsert_records: int = 5,
        skip_embedding_if_present: bool = True,
        vectorizer: Optional[FakeVectorizer] = None,
    ):
        self.config = _config(
            storage_type,
            max_upsert_records=max_upsert_records,
            skip_embedding_if_present=skip_embedding_if_present,
        )
        self.mcp_settings = SimpleNamespace(tool_upsert_description=None)
        self.index = FakeIndex(storage_type)
        self.vectorizer = vectorizer or FakeVectorizer()
        self.registered_tools = []

    async def get_index(self):
        return self.index

    async def get_vectorizer(self):
        return self.vectorizer

    async def run_guarded(self, operation_name: str, awaitable: Any):
        return await awaitable

    def tool(self, name=None, description=None, **kwargs):
        def decorator(fn):
            self.registered_tools.append(
                {
                    "name": name,
                    "description": description,
                    "fn": fn,
                }
            )
            return fn

        return decorator


@pytest.mark.asyncio
async def test_upsert_records_generates_missing_vectors_and_serializes_hash_vectors():
    server = FakeServer(storage_type="hash")
    server.index.keys_to_return = ["doc:alpha", "doc:beta"]

    response = await upsert_records(
        server,
        records=[
            {"id": "alpha", "content": "alpha doc", "category": "science"},
            {"id": "beta", "content": "beta doc", "category": "health"},
        ],
        id_field="id",
    )

    assert response == {
        "status": "success",
        "keys_upserted": 2,
        "keys": ["doc:alpha", "doc:beta"],
    }
    assert server.vectorizer.aembed_many_calls == [(["alpha doc", "beta doc"], {})]
    assert len(server.index.load_calls) == 1
    loaded_records = server.index.load_calls[0]["data"]
    assert loaded_records[0]["embedding"] == array_to_buffer([1.0, 1.0, 1.0], "float32")
    assert loaded_records[1]["embedding"] == array_to_buffer([2.0, 2.0, 2.0], "float32")
    assert server.index.load_calls[0]["id_field"] == "id"


@pytest.mark.asyncio
async def test_upsert_records_preserves_supplied_vectors_when_skip_embedding_if_present():
    server = FakeServer(storage_type="hash", skip_embedding_if_present=True)

    existing_vector = [0.1, 0.2, 0.3]
    await upsert_records(
        server,
        records=[
            {"id": "alpha", "content": "alpha doc", "embedding": existing_vector},
            {"id": "beta", "content": "beta doc"},
        ],
        id_field="id",
    )

    loaded_records = server.index.load_calls[0]["data"]
    assert loaded_records[0]["embedding"] == array_to_buffer(existing_vector, "float32")
    assert loaded_records[1]["embedding"] == array_to_buffer([1.0, 1.0, 1.0], "float32")
    assert server.vectorizer.aembed_many_calls == [(["beta doc"], {})]


@pytest.mark.asyncio
async def test_upsert_records_deep_copies_nested_values_before_loading():
    server = FakeServer(storage_type="json", skip_embedding_if_present=True)
    original_embedding = [0.1, 0.2, 0.3]
    records = [{"id": "alpha", "content": "alpha doc", "embedding": original_embedding}]

    await upsert_records(server, records=records, id_field="id")

    loaded_record = server.index.load_calls[0]["data"][0]
    assert loaded_record["embedding"] == original_embedding
    assert loaded_record["embedding"] is not original_embedding

    loaded_record["embedding"][0] = 9.9
    assert records[0]["embedding"] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_upsert_records_rejects_invalid_hash_vector_dimensions_before_serializing():
    server = FakeServer(storage_type="hash", skip_embedding_if_present=True)

    with pytest.raises(
        RedisVLMCPError, match="must have 3 dimensions, got 2"
    ) as exc_info:
        await upsert_records(
            server,
            records=[{"id": "alpha", "content": "alpha doc", "embedding": [0.1, 0.2]}],
            id_field="id",
        )

    assert exc_info.value.code == MCPErrorCode.INVALID_REQUEST
    assert server.index.load_calls == []
    assert server.vectorizer.aembed_many_calls == []


@pytest.mark.asyncio
async def test_upsert_records_overwrites_supplied_vectors_when_skip_embedding_if_present_false():
    server = FakeServer(storage_type="hash", skip_embedding_if_present=True)

    await upsert_records(
        server,
        records=[{"id": "alpha", "content": "alpha doc", "embedding": [0.1, 0.2, 0.3]}],
        id_field="id",
        skip_embedding_if_present=False,
    )

    loaded_record = server.index.load_calls[0]["data"][0]
    assert loaded_record["embedding"] == array_to_buffer([1.0, 1.0, 1.0], "float32")
    assert server.vectorizer.aembed_many_calls == [(["alpha doc"], {})]


@pytest.mark.asyncio
async def test_upsert_records_uses_batch_fallback_when_aembed_many_is_not_implemented():
    server = FakeServer(vectorizer=FallbackBatchVectorizer())

    await upsert_records(
        server,
        records=[{"content": "alpha doc"}],
    )

    loaded_record = server.index.load_calls[0]["data"][0]
    assert loaded_record["embedding"] == array_to_buffer([9.0, 9.0, 9.0], "float32")
    assert server.vectorizer.embed_many_calls == [(["alpha doc"], {})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("records", "id_field", "message"),
    [
        ([], None, "records must be a non-empty list"),
        ("bad", None, "records must be a non-empty list"),
        ([1], None, "records must contain only objects"),
        ([{"content": "alpha"}], "id", "id_field 'id' must exist"),
    ],
)
async def test_upsert_records_rejects_invalid_request_shapes(
    records, id_field, message
):
    server = FakeServer()

    with pytest.raises(RedisVLMCPError, match=message) as exc_info:
        await upsert_records(server, records=records, id_field=id_field)

    assert exc_info.value.code == MCPErrorCode.INVALID_REQUEST


@pytest.mark.asyncio
async def test_upsert_records_rejects_batches_above_runtime_limit():
    server = FakeServer(max_upsert_records=1)

    with pytest.raises(
        RedisVLMCPError, match="must be less than or equal to 1"
    ) as exc_info:
        await upsert_records(
            server,
            records=[{"content": "alpha"}, {"content": "beta"}],
        )

    assert exc_info.value.code == MCPErrorCode.INVALID_REQUEST


@pytest.mark.asyncio
async def test_upsert_records_requires_configured_embed_source_when_embedding_needed():
    server = FakeServer()

    with pytest.raises(RedisVLMCPError, match="content") as exc_info:
        await upsert_records(
            server,
            records=[{"category": "science"}],
        )

    assert exc_info.value.code == MCPErrorCode.INVALID_REQUEST


@pytest.mark.asyncio
async def test_upsert_records_validates_non_vector_fields_before_embedding():
    server = FakeServer()

    with pytest.raises(RedisVLMCPError, match="category") as exc_info:
        await upsert_records(
            server,
            records=[{"content": "alpha doc", "category": ["science"]}],
        )

    assert exc_info.value.code == MCPErrorCode.INVALID_REQUEST
    assert server.vectorizer.aembed_many_calls == []
    assert server.index.load_calls == []


@pytest.mark.asyncio
async def test_upsert_records_surfaces_partial_write_possible_on_backend_failures():
    server = FakeServer()
    server.index.load_exception = RedisError("boom")

    with pytest.raises(RedisVLMCPError) as exc_info:
        await upsert_records(server, records=[{"content": "alpha doc"}])

    assert exc_info.value.code == MCPErrorCode.BACKEND_UNAVAILABLE
    assert exc_info.value.metadata["partial_write_possible"] is True
    assert isinstance(exc_info.value.__cause__, RedisError)


def test_register_upsert_tool_uses_default_and_override_descriptions():
    default_server = FakeServer()
    register_upsert_tool(default_server)

    assert default_server.registered_tools[0]["name"] == "upsert-records"
    assert "Upsert records" in default_server.registered_tools[0]["description"]

    custom_server = FakeServer()
    custom_server.mcp_settings.tool_upsert_description = "Custom upsert description"
    register_upsert_tool(custom_server)

    assert (
        custom_server.registered_tools[0]["description"] == "Custom upsert description"
    )


@pytest.mark.asyncio
async def test_registered_upsert_tool_rejects_deprecated_embed_text_field_argument():
    server = FakeServer()
    register_upsert_tool(server)

    tool_fn = server.registered_tools[0]["fn"]

    with pytest.raises(TypeError):
        await tool_fn(records=[{"content": "alpha doc"}], embed_text_field="content")
