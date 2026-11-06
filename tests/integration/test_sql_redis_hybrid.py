"""Integration tests for FT.HYBRID via SQLQuery's hybrid_vector_search().

These verify that SQLQuery translates hybrid_vector_search(...) into a native
FT.HYBRID command and returns fused (text + vector) results through the normal
index.query() path.

Requires Redis 8.4+ (FT.HYBRID) and a sql-redis build with hybrid support.
Tests skip automatically on older Redis (server version) or redis-py.
"""

import uuid

import numpy as np
import pytest
from packaging.version import Version
from redis import __version__ as redis_version

from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query import SQLQuery
from tests.conftest import (
    skip_if_redis_version_below,
    skip_if_redis_version_below_async,
)

REDIS_HYBRID_AVAILABLE = Version(redis_version) >= Version("7.1.0")
SKIP_REASON = "Requires Redis >= 8.4.0 and redis-py>=7.1.0"


def _hybrid_schema(index_name: str, worker_id: str, unique_id: str) -> dict:
    """Schema with a text, tag, numeric, and vector field for hybrid search."""
    return {
        "index": {
            "name": index_name,
            "prefix": f"hybrid_{worker_id}_{unique_id}",
            "storage_type": "hash",
        },
        "fields": [
            {"name": "title", "type": "text", "attrs": {"sortable": True}},
            {"name": "description", "type": "text"},
            {"name": "genre", "type": "tag", "attrs": {"sortable": True}},
            {"name": "price", "type": "numeric", "attrs": {"sortable": True}},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": 4,
                    "distance_metric": "cosine",
                    "algorithm": "flat",
                    "datatype": "float32",
                },
            },
        ],
    }


# Books carry both descriptive text (for the SEARCH leg) and an embedding
# (for the VSIM leg). The first three are clustered near [0.1..0.4].
_BOOKS = [
    {
        "title": "Dune",
        "description": "epic space opera on a desert planet",
        "genre": "Science Fiction",
        "price": 15,
        "embedding": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes(),
    },
    {
        "title": "Foundation",
        "description": "galactic empire and psychohistory in space",
        "genre": "Science Fiction",
        "price": 18,
        "embedding": np.array([0.15, 0.25, 0.35, 0.45], dtype=np.float32).tobytes(),
    },
    {
        "title": "Neuromancer",
        "description": "cyberpunk hacker and artificial intelligence",
        "genre": "Science Fiction",
        "price": 12,
        "embedding": np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32).tobytes(),
    },
    {
        "title": "The Hobbit",
        "description": "a dragon and a quest adventure",
        "genre": "Fantasy",
        "price": 14,
        "embedding": np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32).tobytes(),
    },
    {
        "title": "1984",
        "description": "totalitarian surveillance dystopia",
        "genre": "Dystopian",
        "price": 25,
        "embedding": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32).tobytes(),
    },
]


@pytest.fixture
def hybrid_index(redis_url, worker_id):
    """Create a books index (text + tag + numeric + vector) for hybrid search."""
    unique_id = str(uuid.uuid4())[:8]
    index_name = f"sql_hybrid_{worker_id}_{unique_id}"

    index = SearchIndex.from_dict(
        _hybrid_schema(index_name, worker_id, unique_id),
        redis_url=redis_url,
    )
    index.create(overwrite=True)
    index.load(_BOOKS)

    yield index

    index.delete(drop=True)


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason=SKIP_REASON)
class TestSQLQueryHybridSearch:
    """SQL hybrid fusion search via hybrid_vector_search() -> FT.HYBRID."""

    def test_hybrid_rrf_returns_fused_rows(self, hybrid_index):
        """RRF fusion returns rows that carry the combined score column."""
        skip_if_redis_version_below(hybrid_index.client, "8.4.0")

        query_vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes()

        sql_query = SQLQuery(
            f"""
            SELECT title, description,
                   hybrid_vector_search(
                       cosine_distance(embedding, :vec),
                       fulltext(description, 'space empire dystopia'),
                       rrf()
                   ) AS hybrid_score
            FROM {hybrid_index.name}
            ORDER BY hybrid_score DESC
            LIMIT 5
            """,
            params={"vec": query_vector},
        )

        results = hybrid_index.query(sql_query)

        assert len(results) > 0
        for result in results:
            assert "title" in result
            assert "hybrid_score" in result
            assert float(result["hybrid_score"]) >= 0

    def test_hybrid_linear_fusion(self, hybrid_index):
        """LINEAR fusion with an alpha weight executes and returns rows."""
        skip_if_redis_version_below(hybrid_index.client, "8.4.0")

        query_vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes()

        sql_query = SQLQuery(
            f"""
            SELECT title,
                   hybrid_vector_search(
                       cosine_distance(embedding, :vec),
                       fulltext(description, 'space'),
                       linear(alpha => 0.3)
                   ) AS hybrid_score
            FROM {hybrid_index.name}
            ORDER BY hybrid_score DESC
            LIMIT 5
            """,
            params={"vec": query_vector},
        )

        results = hybrid_index.query(sql_query)

        assert len(results) > 0
        assert "hybrid_score" in results[0]

    def test_hybrid_with_where_filter(self, hybrid_index):
        """A WHERE clause is applied to both legs of the fusion."""
        skip_if_redis_version_below(hybrid_index.client, "8.4.0")

        query_vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes()

        sql_query = SQLQuery(
            f"""
            SELECT title, genre,
                   hybrid_vector_search(
                       cosine_distance(embedding, :vec),
                       fulltext(description, 'space empire'),
                       rrf()
                   ) AS hybrid_score
            FROM {hybrid_index.name}
            WHERE genre = 'Science Fiction'
            LIMIT 5
            """,
            params={"vec": query_vector},
        )

        results = hybrid_index.query(sql_query)

        assert len(results) > 0
        for result in results:
            assert result["genre"] == "Science Fiction"

    def test_hybrid_rrf_knobs(self, hybrid_index):
        """RRF constant/window knobs are accepted end to end."""
        skip_if_redis_version_below(hybrid_index.client, "8.4.0")

        query_vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes()

        sql_query = SQLQuery(
            f"""
            SELECT title,
                   hybrid_vector_search(
                       cosine_distance(embedding, :vec),
                       fulltext(description, 'space'),
                       rrf(constant => 60, window => 20)
                   ) AS hybrid_score
            FROM {hybrid_index.name}
            LIMIT 5
            """,
            params={"vec": query_vector},
        )

        results = hybrid_index.query(sql_query)

        assert len(results) > 0

    def test_hybrid_redis_query_string(self, hybrid_index, redis_url):
        """redis_query_string() renders an FT.HYBRID command for inspection."""
        skip_if_redis_version_below(hybrid_index.client, "8.4.0")

        query_vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes()

        sql_query = SQLQuery(
            f"""
            SELECT title,
                   hybrid_vector_search(
                       cosine_distance(embedding, :vec),
                       fulltext(description, 'space'),
                       rrf()
                   ) AS hybrid_score
            FROM {hybrid_index.name}
            LIMIT 5
            """,
            params={"vec": query_vector},
        )

        command = sql_query.redis_query_string(redis_url=redis_url)

        assert command.startswith("FT.HYBRID")
        assert "SEARCH" in command
        assert "VSIM" in command
        assert "COMBINE" in command
        # FT.HYBRID rejects an explicit DIALECT argument.
        assert "DIALECT" not in command


@pytest.fixture
async def async_hybrid_index(redis_url, worker_id):
    """Async books index (text + tag + numeric + vector) for hybrid search."""
    unique_id = str(uuid.uuid4())[:8]
    index_name = f"async_sql_hybrid_{worker_id}_{unique_id}"

    index = AsyncSearchIndex.from_dict(
        _hybrid_schema(index_name, worker_id, unique_id),
        redis_url=redis_url,
    )
    await index.create(overwrite=True)
    await index.load(_BOOKS)

    yield index

    await index.delete(drop=True)


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason=SKIP_REASON)
class TestAsyncSQLQueryHybridSearch:
    """Async hybrid fusion search via AsyncSearchIndex."""

    @pytest.mark.asyncio
    async def test_async_hybrid_rrf(self, async_hybrid_index):
        """Async hybrid fusion returns rows with the combined score column."""
        await skip_if_redis_version_below_async(async_hybrid_index.client, "8.4.0")

        query_vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes()

        sql_query = SQLQuery(
            f"""
            SELECT title,
                   hybrid_vector_search(
                       cosine_distance(embedding, :vec),
                       fulltext(description, 'space empire'),
                       rrf()
                   ) AS hybrid_score
            FROM {async_hybrid_index.name}
            ORDER BY hybrid_score DESC
            LIMIT 5
            """,
            params={"vec": query_vector},
        )

        results = await async_hybrid_index.query(sql_query)

        assert len(results) > 0
        assert "hybrid_score" in results[0]
