from pathlib import Path
from typing import Optional

import pytest
import yaml

from redisvl.index import AsyncSearchIndex
from redisvl.mcp.server import RedisVLMCPServer
from redisvl.mcp.settings import MCPSettings
from redisvl.redis.connection import is_version_gte
from redisvl.schema import IndexSchema
from tests.conftest import get_redis_version_async


class FakeVectorizer:
    def __init__(self, model: str, dims: int = 3, **kwargs):
        self.model = model
        self.dims = dims
        self.kwargs = kwargs


class FailingAsyncCloseVectorizer(FakeVectorizer):
    async def aclose(self):
        raise RuntimeError("vectorizer close failed")


@pytest.fixture
async def existing_index(async_client, worker_id):
    created_indexes = []

    async def factory(
        *,
        index_name: str,
        storage_type: str = "hash",
        vector_path: Optional[str] = None,
    ) -> AsyncSearchIndex:
        fields = [{"name": "content", "type": "text"}]
        vector_field = {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "algorithm": "flat",
                "dims": 3,
                "distance_metric": "cosine",
                "datatype": "float32",
            },
        }
        if storage_type == "json":
            fields[0]["path"] = "$.content"
            vector_field["path"] = vector_path or "$.embedding"

        fields.append(vector_field)
        schema = IndexSchema.from_dict(
            {
                "index": {
                    "name": f"{index_name}-{worker_id}",
                    "prefix": f"{index_name}:{worker_id}",
                    "storage_type": storage_type,
                },
                "fields": fields,
            }
        )
        index = AsyncSearchIndex(schema=schema, redis_client=async_client)
        await index.create(overwrite=True, drop=True)
        created_indexes.append(index)
        return index

    yield factory

    for index in created_indexes:
        try:
            await index.delete(drop=True)
        except Exception:
            pass


@pytest.fixture
def mcp_config_path(tmp_path: Path, redis_url: str):
    def factory(
        *,
        redis_name: str,
        vector_dims: int = 3,
        schema_overrides: Optional[dict] = None,
        runtime_overrides: Optional[dict] = None,
        search: Optional[dict] = None,
    ) -> str:
        runtime = {
            "text_field_name": "content",
            "vector_field_name": "embedding",
            "default_embed_text_field": "content",
        }
        if runtime_overrides:
            runtime.update(runtime_overrides)

        config = {
            "server": {"redis_url": redis_url},
            "indexes": {
                "knowledge": {
                    "redis_name": redis_name,
                    "vectorizer": {
                        "class": "FakeVectorizer",
                        "model": "fake-model",
                        "dims": vector_dims,
                    },
                    "search": search or {"type": "vector"},
                    "runtime": runtime,
                }
            },
        }
        if schema_overrides is not None:
            config["indexes"]["knowledge"]["schema_overrides"] = schema_overrides

        config_path = tmp_path / f"{redis_name}.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return str(config_path)

    return factory


@pytest.mark.asyncio
async def test_server_startup_success(monkeypatch, existing_index, mcp_config_path):
    index = await existing_index(index_name="mcp-startup")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(config=mcp_config_path(redis_name=index.name))
    )

    await server.startup()

    started_index = await server.get_index()
    vectorizer = await server.get_vectorizer()

    assert await started_index.exists() is True
    assert started_index.schema.index.name == index.name
    assert vectorizer.dims == 3

    await server.shutdown()


@pytest.mark.asyncio
async def test_server_fails_when_hybrid_config_requires_native_runtime(
    monkeypatch, existing_index, mcp_config_path, async_client
):
    redis_version = await get_redis_version_async(async_client)
    if is_version_gte(redis_version, "8.4.0"):
        pytest.skip(f"Redis version {redis_version} supports native hybrid search")

    index = await existing_index(index_name="mcp-native-required")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(
            config=mcp_config_path(
                redis_name=index.name,
                search={
                    "type": "hybrid",
                    "params": {
                        "vector_search_method": "KNN",
                        "knn_ef_runtime": 150,
                    },
                },
            )
        )
    )

    with pytest.raises(ValueError, match="knn_ef_runtime"):
        await server.startup()


@pytest.mark.asyncio
async def test_server_fails_when_configured_index_is_missing(
    monkeypatch, mcp_config_path, worker_id
):
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(config=mcp_config_path(redis_name=f"missing-{worker_id}"))
    )

    with pytest.raises(ValueError, match="does not exist"):
        await server.startup()


@pytest.mark.asyncio
async def test_server_uses_schema_overrides_when_inspection_is_incomplete(
    monkeypatch, existing_index, mcp_config_path
):
    index = await existing_index(index_name="mcp-overrides")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    original_info = AsyncSearchIndex._info

    async def incomplete_info(name, redis_client):
        info = await original_info(name, redis_client)
        for field in info["attributes"]:
            if "VECTOR" in field:
                del field[6:]
        return info

    monkeypatch.setattr(
        "redisvl.mcp.server.AsyncSearchIndex._info",
        staticmethod(incomplete_info),
    )
    server = RedisVLMCPServer(
        MCPSettings(
            config=mcp_config_path(
                redis_name=index.name,
                schema_overrides={
                    "fields": [
                        {
                            "name": "embedding",
                            "type": "vector",
                            "attrs": {
                                "algorithm": "flat",
                                "dims": 3,
                                "datatype": "float32",
                                "distance_metric": "cosine",
                            },
                        }
                    ]
                },
            )
        )
    )

    await server.startup()

    started_index = await server.get_index()
    assert started_index.schema.fields["embedding"].attrs.dims == 3

    await server.shutdown()


@pytest.mark.asyncio
async def test_server_fails_on_conflicting_schema_override(
    monkeypatch, existing_index, mcp_config_path
):
    index = await existing_index(
        index_name="mcp-conflict",
        storage_type="json",
        vector_path="$.embedding",
    )
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(
            config=mcp_config_path(
                redis_name=index.name,
                schema_overrides={
                    "fields": [
                        {
                            "name": "embedding",
                            "type": "vector",
                            "path": "$.other_embedding",
                        }
                    ]
                },
            )
        )
    )

    with pytest.raises(ValueError, match="cannot change discovered field path"):
        await server.startup()


@pytest.mark.asyncio
async def test_server_fails_fast_on_vector_dimension_mismatch(
    monkeypatch, existing_index, mcp_config_path
):
    index = await existing_index(index_name="mcp-dims")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(config=mcp_config_path(redis_name=index.name, vector_dims=8))
    )

    with pytest.raises(ValueError, match="Vectorizer dims"):
        await server.startup()


@pytest.mark.asyncio
async def test_server_startup_failure_disconnects_index(
    monkeypatch, existing_index, mcp_config_path
):
    index = await existing_index(index_name="mcp-startup-failure")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    original_disconnect = AsyncSearchIndex.disconnect
    disconnect_called = False

    async def tracked_disconnect(self):
        nonlocal disconnect_called
        disconnect_called = True
        await original_disconnect(self)

    monkeypatch.setattr(
        "redisvl.mcp.server.AsyncSearchIndex.disconnect",
        tracked_disconnect,
    )
    server = RedisVLMCPServer(
        MCPSettings(config=mcp_config_path(redis_name=index.name, vector_dims=8))
    )

    with pytest.raises(ValueError, match="Vectorizer dims"):
        await server.startup()

    assert disconnect_called is True


@pytest.mark.asyncio
async def test_server_shutdown_disconnects_owned_client(
    monkeypatch, existing_index, mcp_config_path
):
    index = await existing_index(index_name="mcp-shutdown")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(config=mcp_config_path(redis_name=index.name))
    )

    await server.startup()
    started_index = await server.get_index()

    assert started_index.client is not None

    await server.shutdown()

    assert started_index.client is None


@pytest.mark.asyncio
async def test_server_get_index_fails_after_shutdown(
    monkeypatch, existing_index, mcp_config_path
):
    index = await existing_index(index_name="mcp-get-index-after-shutdown")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(config=mcp_config_path(redis_name=index.name))
    )

    await server.startup()
    await server.shutdown()

    with pytest.raises(RuntimeError, match="has not been started"):
        await server.get_index()


@pytest.mark.asyncio
async def test_server_shutdown_disconnects_index_when_vectorizer_close_fails(
    monkeypatch, existing_index, mcp_config_path
):
    index = await existing_index(index_name="mcp-shutdown-failure")
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FailingAsyncCloseVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(config=mcp_config_path(redis_name=index.name))
    )

    await server.startup()
    started_index = await server.get_index()

    with pytest.raises(RuntimeError, match="vectorizer close failed"):
        await server.shutdown()

    assert started_index.client is None

    with pytest.raises(RuntimeError, match="has not been started"):
        await server.get_vectorizer()
