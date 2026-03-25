from types import SimpleNamespace
from typing import Optional

import pytest

from redisvl.mcp.config import MCPConfig
from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.mcp.tools.search import _embed_query, register_search_tool, search_records
from redisvl.schema import IndexSchema


def _schema() -> IndexSchema:
    return IndexSchema.from_dict(
        {
            "index": {
                "name": "docs-index",
                "prefix": "doc",
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


def _config_with_search(search_type: str, params: Optional[dict] = None) -> MCPConfig:
    return MCPConfig.model_validate(
        {
            "server": {"redis_url": "redis://localhost:6379"},
            "indexes": {
                "knowledge": {
                    "redis_name": "docs-index",
                    "vectorizer": {"class": "FakeVectorizer", "model": "test-model"},
                    "search": {"type": search_type, "params": params or {}},
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
    )


class FakeVectorizer:
    async def embed(self, text: str):
        return [0.1, 0.2, 0.3]


class FakeIndex:
    def __init__(self):
        self.schema = _schema()
        self.query_calls = []

    async def query(self, query):
        self.query_calls.append(query)
        return []


class FakeServer:
    def __init__(
        self,
        *,
        search_type: str = "vector",
        search_params: Optional[dict] = None,
    ):
        self.config = _config_with_search(search_type, search_params)
        self.mcp_settings = SimpleNamespace(tool_search_description=None)
        self.index = FakeIndex()
        self.vectorizer = FakeVectorizer()
        self.registered_tools = []
        self.native_hybrid_supported = False

    async def get_index(self):
        return self.index

    async def get_vectorizer(self):
        return self.vectorizer

    async def run_guarded(self, operation_name, awaitable):
        return await awaitable

    async def supports_native_hybrid_search(self):
        return self.native_hybrid_supported

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


class FakeQuery:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.mark.asyncio
async def test_embed_query_falls_back_to_sync_embed_when_aembed_is_not_implemented():
    class FallbackVectorizer:
        async def aembed(self, text: str):
            raise NotImplementedError

        def embed(self, text: str):
            return [0.4, 0.5, 0.6]

    embedding = await _embed_query(FallbackVectorizer(), "science")

    assert embedding == [0.4, 0.5, 0.6]


@pytest.mark.asyncio
async def test_search_records_rejects_blank_query():
    server = FakeServer()

    with pytest.raises(RedisVLMCPError) as exc_info:
        await search_records(server, query="   ")

    assert exc_info.value.code == MCPErrorCode.INVALID_REQUEST


@pytest.mark.asyncio
async def test_search_records_rejects_invalid_limit_and_offset():
    server = FakeServer()

    with pytest.raises(RedisVLMCPError) as limit_exc:
        await search_records(server, query="science", limit=0)

    with pytest.raises(RedisVLMCPError) as offset_exc:
        await search_records(server, query="science", offset=-1)

    assert limit_exc.value.code == MCPErrorCode.INVALID_REQUEST
    assert offset_exc.value.code == MCPErrorCode.INVALID_REQUEST


@pytest.mark.asyncio
async def test_search_records_rejects_unknown_or_vector_return_fields():
    server = FakeServer()

    with pytest.raises(RedisVLMCPError) as unknown_exc:
        await search_records(server, query="science", return_fields=["missing"])

    with pytest.raises(RedisVLMCPError) as vector_exc:
        await search_records(server, query="science", return_fields=["embedding"])

    assert unknown_exc.value.code == MCPErrorCode.INVALID_REQUEST
    assert vector_exc.value.code == MCPErrorCode.INVALID_REQUEST


@pytest.mark.asyncio
async def test_search_records_builds_vector_query_and_normalizes_results(monkeypatch):
    server = FakeServer(
        search_type="vector",
        search_params={"normalize_vector_distance": False, "ef_runtime": 42},
    )
    built_queries = []

    class FakeVectorQuery(FakeQuery):
        def __init__(self, **kwargs):
            built_queries.append(kwargs)
            super().__init__(**kwargs)

    async def fake_query(query):
        server.index.query_calls.append(query)
        return [
            {
                "id": "doc:1",
                "content": "science doc",
                "category": "science",
                "vector_distance": "0.93",
            }
        ]

    monkeypatch.setattr("redisvl.mcp.tools.search.VectorQuery", FakeVectorQuery)
    server.index.query = fake_query

    response = await search_records(server, query="science")

    assert built_queries[0]["vector"] == [0.1, 0.2, 0.3]
    assert built_queries[0]["vector_field_name"] == "embedding"
    assert built_queries[0]["return_fields"] == ["content", "category", "rating"]
    assert built_queries[0]["num_results"] == 2
    assert built_queries[0]["normalize_vector_distance"] is False
    assert built_queries[0]["ef_runtime"] == 42
    assert response == {
        "search_type": "vector",
        "offset": 0,
        "limit": 2,
        "results": [
            {
                "id": "doc:1",
                "score": 0.93,
                "score_type": "vector_distance",
                "record": {
                    "content": "science doc",
                    "category": "science",
                },
            }
        ],
    }


@pytest.mark.asyncio
async def test_search_records_builds_fulltext_query(monkeypatch):
    server = FakeServer(
        search_type="fulltext",
        search_params={
            "text_scorer": "BM25STD.NORM",
            "stopwords": None,
            "text_weights": {"medical": 2.5},
        },
    )
    built_queries = []

    class FakeTextQuery(FakeQuery):
        def __init__(self, **kwargs):
            built_queries.append(kwargs)
            super().__init__(**kwargs)

    async def fake_query(query):
        server.index.query_calls.append(query)
        return [
            {
                "id": "doc:2",
                "content": "medical science",
                "category": "health",
                "__score": "1.5",
            }
        ]

    monkeypatch.setattr("redisvl.mcp.tools.search.TextQuery", FakeTextQuery)
    server.index.query = fake_query

    response = await search_records(
        server,
        query="medical science",
        limit=1,
        return_fields=["content", "category"],
    )

    assert built_queries[0]["text"] == "medical science"
    assert built_queries[0]["text_field_name"] == "content"
    assert built_queries[0]["num_results"] == 1
    assert built_queries[0]["text_scorer"] == "BM25STD.NORM"
    assert built_queries[0]["stopwords"] is None
    assert built_queries[0]["text_weights"] == {"medical": 2.5}
    assert response["search_type"] == "fulltext"
    assert response["results"][0]["score"] == 1.5
    assert response["results"][0]["score_type"] == "text_score"


@pytest.mark.asyncio
async def test_search_records_builds_hybrid_query_for_native_runtime(monkeypatch):
    server = FakeServer(
        search_type="hybrid",
        search_params={
            "text_scorer": "TFIDF",
            "stopwords": None,
            "text_weights": {"hybrid": 2.0},
            "vector_search_method": "KNN",
            "knn_ef_runtime": 77,
            "combination_method": "LINEAR",
            "linear_text_weight": 0.2,
        },
    )
    server.native_hybrid_supported = True
    built_queries = []

    class FakePostProcessingConfig:
        def __init__(self):
            self.apply_calls = []

        def apply(self, **kwargs):
            self.apply_calls.append(kwargs)

    class FakeHybridQuery(FakeQuery):
        def __init__(self, **kwargs):
            self.postprocessing_config = FakePostProcessingConfig()
            built_queries.append(("native", kwargs, self.postprocessing_config))
            super().__init__(**kwargs)

    class FakeAggregateHybridQuery(FakeQuery):
        def __init__(self, **kwargs):
            built_queries.append(("fallback", kwargs))
            super().__init__(**kwargs)

    async def fake_query(query):
        server.index.query_calls.append(query)
        return [
            {
                "id": "doc:3",
                "content": "hybrid doc",
                "hybrid_score": "2.5",
            }
        ]

    monkeypatch.setattr("redisvl.mcp.tools.search.HybridQuery", FakeHybridQuery)
    monkeypatch.setattr(
        "redisvl.mcp.tools.search.AggregateHybridQuery", FakeAggregateHybridQuery
    )
    server.index.query = fake_query

    response = await search_records(server, query="hybrid")

    assert built_queries[0][0] == "native"
    assert built_queries[0][1]["vector"] == [0.1, 0.2, 0.3]
    assert built_queries[0][1]["text_scorer"] == "TFIDF"
    assert built_queries[0][1]["stopwords"] is None
    assert built_queries[0][1]["text_weights"] == {"hybrid": 2.0}
    assert built_queries[0][1]["vector_search_method"] == "KNN"
    assert built_queries[0][1]["knn_ef_runtime"] == 77
    assert built_queries[0][1]["combination_method"] == "LINEAR"
    assert built_queries[0][1]["linear_alpha"] == 0.2
    assert built_queries[0][2].apply_calls == [{"__key": "@__key"}]
    assert response["search_type"] == "hybrid"
    assert response["results"][0]["score_type"] == "hybrid_score"
    assert response["results"][0]["score"] == 2.5


@pytest.mark.asyncio
async def test_search_records_builds_hybrid_query_for_fallback_runtime(monkeypatch):
    server = FakeServer(
        search_type="hybrid",
        search_params={
            "text_scorer": "TFIDF",
            "stopwords": None,
            "text_weights": {"hybrid": 2.0},
            "combination_method": "LINEAR",
            "linear_text_weight": 0.2,
        },
    )
    built_queries = []

    class FakeHybridQuery(FakeQuery):
        def __init__(self, **kwargs):
            built_queries.append(("native", kwargs))
            super().__init__(**kwargs)

    class FakeAggregateHybridQuery(FakeQuery):
        def __init__(self, **kwargs):
            built_queries.append(("fallback", kwargs))
            super().__init__(**kwargs)

    async def fake_query(query):
        server.index.query_calls.append(query)
        return [
            {
                "id": "doc:4",
                "content": "fallback hybrid",
                "hybrid_score": "0.7",
            }
        ]

    monkeypatch.setattr("redisvl.mcp.tools.search.HybridQuery", FakeHybridQuery)
    monkeypatch.setattr(
        "redisvl.mcp.tools.search.AggregateHybridQuery", FakeAggregateHybridQuery
    )
    server.index.query = fake_query

    response = await search_records(server, query="hybrid")

    assert built_queries[0][0] == "fallback"
    assert built_queries[0][1]["text_scorer"] == "TFIDF"
    assert built_queries[0][1]["stopwords"] is None
    assert built_queries[0][1]["text_weights"] == {"hybrid": 2.0}
    assert built_queries[0][1]["alpha"] == pytest.approx(0.8)
    assert built_queries[0][1]["return_fields"] == [
        "__key",
        "content",
        "category",
        "rating",
    ]
    assert response["search_type"] == "hybrid"
    assert response["results"][0]["score"] == 0.7


@pytest.mark.asyncio
async def test_search_records_rejects_native_only_hybrid_runtime_params(monkeypatch):
    server = FakeServer(
        search_type="hybrid",
        search_params={
            "combination_method": "RRF",
            "rrf_window": 50,
        },
    )

    with pytest.raises(ValueError, match="native hybrid search support"):
        server.config.validate_search(supports_native_hybrid_search=False)


def test_register_search_tool_uses_default_and_override_descriptions():
    default_server = FakeServer()
    register_search_tool(default_server)

    assert default_server.registered_tools[0]["name"] == "search-records"
    assert "Search records" in default_server.registered_tools[0]["description"]
    assert "query" in default_server.registered_tools[0]["fn"].__annotations__
    assert "search_type" not in default_server.registered_tools[0]["fn"].__annotations__

    custom_server = FakeServer()
    custom_server.mcp_settings.tool_search_description = "Custom search description"
    register_search_tool(custom_server)

    assert (
        custom_server.registered_tools[0]["description"] == "Custom search description"
    )
