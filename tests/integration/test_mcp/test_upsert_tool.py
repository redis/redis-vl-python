from pathlib import Path
from typing import Any

import pytest
import yaml

from redisvl.index import AsyncSearchIndex
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.server import RedisVLMCPServer
from redisvl.mcp.settings import MCPSettings
from redisvl.mcp.tools.upsert import upsert_records
from redisvl.schema import IndexSchema


class RecordingVectorizer:
    def __init__(self, model: str, dims: int = 3, **kwargs: Any) -> None:
        self.model = model
        self.dims = dims
        self.kwargs = kwargs
        self.aembed_many_inputs: list[list[str]] = []
        self.embed_many_inputs: list[list[str]] = []
        self.aembed_inputs: list[str] = []
        self.embed_inputs: list[str] = []

    @staticmethod
    def _vector_for(text: str) -> list[float]:
        base = float(len(text))
        return [base, base + 0.1, base + 0.2]

    async def aembed(self, content: str = "", **kwargs: Any) -> list[float]:
        del kwargs
        self.aembed_inputs.append(content)
        return self._vector_for(content)

    def embed(self, content: str = "", **kwargs: Any) -> list[float]:
        del kwargs
        self.embed_inputs.append(content)
        return self._vector_for(content)

    async def aembed_many(
        self,
        contents: list[str] | None = None,
        texts: list[str] | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        del kwargs
        items = contents or texts or []
        self.aembed_many_inputs.append(list(items))
        return [self._vector_for(text) for text in items]

    def embed_many(
        self,
        contents: list[str] | None = None,
        texts: list[str] | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        del kwargs
        items = contents or texts or []
        self.embed_many_inputs.append(list(items))
        return [self._vector_for(text) for text in items]


@pytest.fixture
async def upsertable_index(async_client, worker_id):
    schema = IndexSchema.from_dict(
        {
            "index": {
                "name": f"mcp-upsert-{worker_id}",
                "prefix": f"mcp-upsert:{worker_id}",
                "storage_type": "hash",
            },
            "fields": [
                {"name": "content", "type": "text"},
                {"name": "category", "type": "tag"},
                {"name": "rating", "type": "numeric"},
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
    index = AsyncSearchIndex(schema=schema, redis_client=async_client)
    await index.create(overwrite=True, drop=True)

    yield index

    await index.delete(drop=True)


@pytest.fixture
def mcp_config_path(tmp_path: Path, redis_url: str):
    def factory(
        *,
        redis_name: str,
        read_only: bool = False,
        runtime_overrides: dict[str, Any] | None = None,
    ) -> str:
        runtime = {
            "text_field_name": "content",
            "vector_field_name": "embedding",
            "default_embed_text_field": "content",
            "default_limit": 2,
            "max_limit": 5,
            "max_upsert_records": 64,
            "skip_embedding_if_present": True,
        }
        if runtime_overrides:
            runtime.update(runtime_overrides)

        config = {
            "server": {"redis_url": redis_url},
            "indexes": {
                "knowledge": {
                    "redis_name": redis_name,
                    "vectorizer": {
                        "class": "RecordingVectorizer",
                        "model": "fake-model",
                        "dims": 3,
                    },
                    "search": {"type": "vector"},
                    "runtime": runtime,
                }
            },
        }
        config_path = tmp_path / (
            f"{redis_name}-{'readonly' if read_only else 'readwrite'}.yaml"
        )
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return str(config_path)

    return factory


@pytest.fixture
async def started_server(monkeypatch, upsertable_index, mcp_config_path):
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: RecordingVectorizer,
    )

    servers: list[RedisVLMCPServer] = []

    async def factory(
        *,
        read_only: bool = False,
        runtime_overrides: dict[str, Any] | None = None,
    ) -> RedisVLMCPServer:
        server = RedisVLMCPServer(
            MCPSettings(
                config=mcp_config_path(
                    redis_name=upsertable_index.schema.index.name,
                    read_only=read_only,
                    runtime_overrides=runtime_overrides,
                )
            )
        )
        await server.startup()
        servers.append(server)
        return server

    yield factory

    for server in servers:
        await server.shutdown()


def _record_id_from_key(key: str) -> str:
    return key.rsplit(":", 1)[-1]


@pytest.mark.asyncio
async def test_upsert_records_inserts_rows_into_hash_index(
    started_server, upsertable_index
):
    server = await started_server()

    records = [
        {"content": "first upserted document", "category": "science", "rating": 5},
        {"content": "second upserted document", "category": "health", "rating": 4},
    ]

    response = await upsert_records(server, records=records)

    assert response["status"] == "success"
    assert response["keys_upserted"] == 2
    assert len(response["keys"]) == 2

    vectorizer = await server.get_vectorizer()
    assert vectorizer.aembed_many_inputs == [
        ["first upserted document", "second upserted document"]
    ]

    stored = await upsertable_index.fetch(_record_id_from_key(response["keys"][0]))
    assert stored is not None
    assert stored["content"] == "first upserted document"
    assert stored["category"] == "science"


@pytest.mark.asyncio
async def test_upsert_records_updates_existing_row_with_id_field(
    started_server, upsertable_index
):
    server = await started_server()

    first_response = await upsert_records(
        server,
        records=[
            {
                "doc_id": "doc-1",
                "content": "original content",
                "category": "science",
                "rating": 3,
            }
        ],
        id_field="doc_id",
    )

    second_response = await upsert_records(
        server,
        records=[
            {
                "doc_id": "doc-1",
                "content": "updated content",
                "category": "engineering",
                "rating": 5,
            }
        ],
        id_field="doc_id",
    )

    assert first_response["keys"] == second_response["keys"]
    assert second_response["keys_upserted"] == 1

    stored = await upsertable_index.fetch(
        _record_id_from_key(second_response["keys"][0])
    )
    assert stored is not None
    assert stored["content"] == "updated content"
    assert stored["category"] == "engineering"
    assert int(stored["rating"]) == 5


@pytest.mark.asyncio
async def test_upsert_records_rejects_invalid_records_before_write(
    monkeypatch, started_server
):
    server = await started_server()

    called = False

    async def fail_load(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        nonlocal called
        called = True
        raise AssertionError("load should not be called for invalid records")

    monkeypatch.setattr(
        "redisvl.index.index.AsyncSearchIndex.load",
        fail_load,
    )

    with pytest.raises(RedisVLMCPError) as exc_info:
        await upsert_records(
            server,
            records=[{"category": "science"}],
        )

    assert exc_info.value.code == MCPErrorCode.INVALID_REQUEST
    assert called is False


@pytest.mark.asyncio
async def test_read_only_mode_excludes_upsert_tool(
    monkeypatch, upsertable_index, mcp_config_path
):
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: RecordingVectorizer,
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.register_search_tool",
        lambda server, schema: None,
    )

    def fake_tool(*args: Any, **kwargs: Any):
        del args, kwargs

        def decorator(func: Any) -> Any:
            return func

        return decorator

    monkeypatch.setattr(RedisVLMCPServer, "tool", fake_tool, raising=False)

    called: list[bool] = []

    def fake_register_upsert_tool(server: Any) -> None:
        called.append(server.mcp_settings.read_only)

    monkeypatch.setattr(
        "redisvl.mcp.server.register_upsert_tool",
        fake_register_upsert_tool,
        raising=False,
    )

    writeable_server = RedisVLMCPServer(
        MCPSettings(
            config=mcp_config_path(
                redis_name=upsertable_index.schema.index.name,
            )
        )
    )
    await writeable_server.startup()
    try:
        assert called == [False]
    finally:
        await writeable_server.shutdown()

    read_only_server = RedisVLMCPServer(
        MCPSettings(
            config=mcp_config_path(
                redis_name=upsertable_index.schema.index.name,
                read_only=True,
            ),
            read_only=True,
        )
    )

    await read_only_server.startup()
    try:
        assert called == [False]
    finally:
        await read_only_server.shutdown()
