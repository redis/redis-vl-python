from pathlib import Path

import pytest
import yaml

from redisvl.index import AsyncSearchIndex
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.server import RedisVLMCPServer
from redisvl.mcp.settings import MCPSettings
from redisvl.mcp.tools.search import search_records
from redisvl.redis.connection import is_version_gte
from redisvl.redis.utils import array_to_buffer
from redisvl.schema import IndexSchema
from tests.conftest import get_redis_version_async, skip_if_redis_version_below_async


class FakeVectorizer:
    def __init__(self, model: str, dims: int = 3, **kwargs):
        self.model = model
        self.dims = dims
        self.kwargs = kwargs

    def embed(self, content: str = "", **kwargs):
        del content, kwargs
        return [0.1, 0.1, 0.5]


@pytest.fixture
async def searchable_index(async_client, worker_id):
    schema = IndexSchema.from_dict(
        {
            "index": {
                "name": f"mcp-search-{worker_id}",
                "prefix": f"mcp-search:{worker_id}",
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

    def preprocess(record: dict) -> dict:
        return {
            **record,
            "embedding": array_to_buffer(record["embedding"], "float32"),
        }

    await index.load(
        [
            {
                "id": f"doc:{worker_id}:1",
                "content": "science article about planets",
                "category": "science",
                "rating": 5,
                "embedding": [0.1, 0.1, 0.5],
            },
            {
                "id": f"doc:{worker_id}:2",
                "content": "medical science and health",
                "category": "health",
                "rating": 4,
                "embedding": [0.1, 0.1, 0.4],
            },
            {
                "id": f"doc:{worker_id}:3",
                "content": "sports update and scores",
                "category": "sports",
                "rating": 3,
                "embedding": [-0.2, 0.1, 0.0],
            },
        ],
        preprocess=preprocess,
    )

    yield index

    await index.delete(drop=True)


@pytest.fixture
def mcp_config_path(tmp_path: Path, redis_url: str):
    def factory(redis_name: str) -> str:
        config = {
            "server": {"redis_url": redis_url},
            "indexes": {
                "knowledge": {
                    "redis_name": redis_name,
                    "vectorizer": {
                        "class": "FakeVectorizer",
                        "model": "fake-model",
                        "dims": 3,
                    },
                    "runtime": {
                        "text_field_name": "content",
                        "vector_field_name": "embedding",
                        "default_embed_text_field": "content",
                        "default_limit": 2,
                        "max_limit": 5,
                    },
                }
            },
        }
        config_path = tmp_path / f"{redis_name}.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return str(config_path)

    return factory


@pytest.fixture
async def started_server(monkeypatch, searchable_index, mcp_config_path):
    monkeypatch.setattr(
        "redisvl.mcp.server.resolve_vectorizer_class",
        lambda class_name: FakeVectorizer,
    )
    server = RedisVLMCPServer(
        MCPSettings(config=mcp_config_path(searchable_index.schema.index.name))
    )
    await server.startup()
    yield server
    await server.shutdown()


@pytest.mark.asyncio
async def test_search_records_vector_success_with_pagination_and_projection(
    started_server,
):
    response = await search_records(
        started_server,
        query="science",
        limit=1,
        offset=1,
        return_fields=["content", "category"],
    )

    assert response["search_type"] == "vector"
    assert response["offset"] == 1
    assert response["limit"] == 1
    assert len(response["results"]) == 1
    assert response["results"][0]["score_type"] == "vector_distance_normalized"
    assert set(response["results"][0]["record"]) == {"content", "category"}


@pytest.mark.asyncio
async def test_search_records_fulltext_success(started_server):
    response = await search_records(
        started_server,
        query="science",
        search_type="fulltext",
        return_fields=["content", "category"],
    )

    assert response["search_type"] == "fulltext"
    assert response["results"]
    assert response["results"][0]["score_type"] == "text_score"
    assert response["results"][0]["score"] is not None
    assert "science" in response["results"][0]["record"]["content"]


@pytest.mark.asyncio
async def test_search_records_respects_raw_string_filter(started_server):
    response = await search_records(
        started_server,
        query="science",
        filter="@category:{science}",
        return_fields=["content", "category"],
    )

    assert response["results"]
    assert all(
        result["record"]["category"] == "science" for result in response["results"]
    )


@pytest.mark.asyncio
async def test_search_records_respects_dsl_filter(started_server):
    response = await search_records(
        started_server,
        query="science",
        filter={"field": "rating", "op": "gte", "value": 4.5},
        return_fields=["content", "category", "rating"],
    )

    assert response["results"]
    assert all(
        float(result["record"]["rating"]) >= 4.5 for result in response["results"]
    )


@pytest.mark.asyncio
async def test_search_records_invalid_filter_returns_invalid_filter(started_server):
    with pytest.raises(RedisVLMCPError) as exc_info:
        await search_records(
            started_server,
            query="science",
            filter={"field": "missing", "op": "eq", "value": "science"},
        )

    assert exc_info.value.code == MCPErrorCode.INVALID_FILTER


@pytest.mark.asyncio
async def test_search_records_native_hybrid_success(started_server, async_client):
    await skip_if_redis_version_below_async(async_client, "8.4.0")

    response = await search_records(
        started_server,
        query="science",
        search_type="hybrid",
        return_fields=["content", "category"],
    )

    assert response["search_type"] == "hybrid"
    assert response["results"]
    assert response["results"][0]["score_type"] == "hybrid_score"
    assert response["results"][0]["score"] is not None


@pytest.mark.asyncio
async def test_search_records_fallback_hybrid_success(started_server, async_client):
    redis_version = await get_redis_version_async(async_client)
    if is_version_gte(redis_version, "8.4.0"):
        pytest.skip(f"Redis version {redis_version} uses native hybrid search")

    response = await search_records(
        started_server,
        query="science",
        search_type="hybrid",
        return_fields=["content", "category"],
    )

    assert response["search_type"] == "hybrid"
    assert response["results"]
    assert response["results"][0]["score_type"] == "hybrid_score"
    assert response["results"][0]["score"] is not None
