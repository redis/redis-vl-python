import pytest
import redis
from packaging.version import Version
from redis import Redis
from redis import __version__ as redis_version
from redis.asyncio import Redis as AsyncRedis
from redis.commands.search.aggregation import AggregateRequest, Cursor

from redisvl.extensions.router import Route, SemanticRouter
from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query import AggregateHybridQuery, MultiVectorQuery, Vector
from redisvl.query.hybrid import HybridQuery
from redisvl.redis.utils import array_to_buffer
from redisvl.schema import IndexSchema
from redisvl.utils.vectorize.base import BaseVectorizer
from tests.conftest import (
    skip_if_redis_version_below,
    skip_if_redis_version_below_async,
)

REDIS_HYBRID_AVAILABLE = Version(redis_version) >= Version("7.1.0")


def _row_to_dict(row):
    """Convert flat alternating list [k, v, k, v] or dict to normalized dict without type coercion."""
    if isinstance(row, dict):
        return dict(row)
    d = {}
    for i in range(0, len(row) - 1, 2):
        d[row[i]] = row[i + 1]
    return d


def assert_results_equal(res2, res3):
    """Assert protocol 2 and protocol 3 results are equal, ignoring field order and execution_time/total."""
    if isinstance(res2, list) and isinstance(res3, list):
        assert len(res2) == len(res3)
        for r2, r3 in zip(res2, res3):
            d2 = _row_to_dict(r2)
            d3 = _row_to_dict(r3)
            assert d2 == d3
    elif hasattr(res2, "rows") and hasattr(res3, "rows"):
        assert len(res2.rows) == len(res3.rows)
        for r2, r3 in zip(res2.rows, res3.rows):
            assert _row_to_dict(r2) == _row_to_dict(r3)
    elif hasattr(res2, "results") and hasattr(res3, "results"):
        assert len(res2.results) == len(res3.results)
        for r2, r3 in zip(res2.results, res3.results):
            assert _row_to_dict(r2) == _row_to_dict(r3)
    else:
        assert _row_to_dict(res2) == _row_to_dict(res3)


@pytest.fixture
def test_schema(redis_test_name):
    name = redis_test_name("resp3_idx")
    return IndexSchema.from_dict(
        {
            "index": {"name": name, "prefix": f"{name}:", "storage_type": "hash"},
            "fields": [
                {"name": "tag", "type": "tag"},
                {"name": "txt", "type": "text"},
                {"name": "num", "type": "numeric"},
                {
                    "name": "v",
                    "type": "vector",
                    "attrs": {
                        "dims": 3,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                        "datatype": "float32",
                    },
                },
            ],
        }
    )


@pytest.fixture
def test_docs():
    return [
        {
            "tag": "news",
            "txt": "artificial intelligence and machine learning technology",
            "num": 10,
            "v": [0.1, 0.2, 0.3],
        },
        {
            "tag": "tech",
            "txt": "deep learning neural networks in artificial intelligence",
            "num": 20,
            "v": [0.15, 0.25, 0.35],
        },
        {
            "tag": "news",
            "txt": "redis vector search with high performance databases",
            "num": 30,
            "v": [0.2, 0.3, 0.4],
        },
        {
            "tag": "sports",
            "txt": "football match results and sports highlights today",
            "num": 40,
            "v": [0.9, 0.8, 0.7],
        },
    ]


@pytest.fixture
def indices(redis_url, test_schema, test_docs):
    r2 = Redis.from_url(redis_url, protocol=2)
    r3 = Redis.from_url(redis_url, protocol=3)
    idx2 = SearchIndex(schema=test_schema, redis_client=r2)
    idx3 = SearchIndex(schema=test_schema, redis_client=r3)

    def hash_preprocess(item: dict) -> dict:
        return {
            **item,
            "v": array_to_buffer(item["v"], "float32"),
        }

    idx2.create(overwrite=True, drop=True)
    idx2.load(test_docs, preprocess=hash_preprocess)

    yield idx2, idx3
    try:
        idx2.delete(drop=True)
    except Exception:
        pass


@pytest.fixture
async def async_indices(redis_url, test_schema, test_docs):
    ar2 = AsyncRedis.from_url(redis_url, protocol=2)
    ar3 = AsyncRedis.from_url(redis_url, protocol=3)
    aidx2 = AsyncSearchIndex(schema=test_schema, redis_client=ar2)
    aidx3 = AsyncSearchIndex(schema=test_schema, redis_client=ar3)

    def hash_preprocess(item: dict) -> dict:
        return {
            **item,
            "v": array_to_buffer(item["v"], "float32"),
        }

    await aidx2.create(overwrite=True, drop=True)
    await aidx2.load(test_docs, preprocess=hash_preprocess)

    yield aidx2, aidx3
    try:
        await aidx2.delete(drop=True)
    except Exception:
        pass


def test_aggregate_direct(indices):
    idx2, idx3 = indices
    req = AggregateRequest("*").load("@txt").load("@tag")
    res2 = idx2.aggregate(req)
    res3 = idx3.aggregate(req)

    assert hasattr(res3, "rows")
    assert hasattr(res3, "cursor")
    assert_results_equal(res2, res3)


@pytest.mark.asyncio
async def test_aggregate_direct_async(async_indices):
    aidx2, aidx3 = async_indices
    req = AggregateRequest("*").load("@txt").load("@tag")
    res2 = await aidx2.aggregate(req)
    res3 = await aidx3.aggregate(req)

    assert hasattr(res3, "rows")
    assert hasattr(res3, "cursor")
    assert_results_equal(res2, res3)


def test_aggregate_hybrid_query(indices):
    idx2, idx3 = indices
    q = AggregateHybridQuery(
        text="artificial intelligence",
        text_field_name="txt",
        vector=[0.1, 0.2, 0.3],
        vector_field_name="v",
        return_fields=["txt", "tag", "num"],
    )
    res2 = idx2.query(q)
    res3 = idx3.query(q)

    assert len(res3) > 0
    assert_results_equal(res2, res3)


@pytest.mark.asyncio
async def test_aggregate_hybrid_query_async(async_indices):
    aidx2, aidx3 = async_indices
    q = AggregateHybridQuery(
        text="artificial intelligence",
        text_field_name="txt",
        vector=[0.1, 0.2, 0.3],
        vector_field_name="v",
        return_fields=["txt", "tag", "num"],
    )
    res2 = await aidx2.query(q)
    res3 = await aidx3.query(q)

    assert len(res3) > 0
    assert_results_equal(res2, res3)


def test_multi_vector_query(indices):
    idx2, idx3 = indices
    q = MultiVectorQuery(
        vectors=[
            Vector(vector=[0.1, 0.2, 0.3], field_name="v", max_distance=1.0),
        ],
        return_fields=["txt", "num"],
    )
    res2 = idx2.query(q)
    res3 = idx3.query(q)

    assert len(res3) > 0
    assert_results_equal(res2, res3)


@pytest.mark.asyncio
async def test_multi_vector_query_async(async_indices):
    aidx2, aidx3 = async_indices
    q = MultiVectorQuery(
        vectors=[
            Vector(vector=[0.1, 0.2, 0.3], field_name="v", max_distance=1.0),
        ],
        return_fields=["txt", "num"],
    )
    res2 = await aidx2.query(q)
    res3 = await aidx3.query(q)

    assert len(res3) > 0
    assert_results_equal(res2, res3)


def test_iter_keys_by_filter_cursor(indices):
    idx2, idx3 = indices
    from redisvl.query.filter import Tag

    filter_expr = Tag("tag") == "news"

    keys2 = list(idx2._iter_keys_by_filter(filter_expr, batch_size=1))
    keys3 = list(idx3._iter_keys_by_filter(filter_expr, batch_size=1))

    flat_keys2 = [k for page in keys2 for k in page]
    flat_keys3 = [k for page in keys3 for k in page]
    assert len(flat_keys3) == 2
    assert set(flat_keys2) == set(flat_keys3)


@pytest.mark.asyncio
async def test_iter_keys_by_filter_cursor_async(async_indices):
    aidx2, aidx3 = async_indices
    from redisvl.query.filter import Tag

    filter_expr = Tag("tag") == "news"

    keys2 = []
    async for page in aidx2._iter_keys_by_filter(filter_expr, batch_size=1):
        keys2.append(page)

    keys3 = []
    async for page in aidx3._iter_keys_by_filter(filter_expr, batch_size=1):
        keys3.append(page)

    flat_keys2 = [k for page in keys2 for k in page]
    flat_keys3 = [k for page in keys3 for k in page]
    assert len(flat_keys3) == 2
    assert set(flat_keys2) == set(flat_keys3)


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis-py >= 7.1.0")
def test_hybrid_search_query(indices):
    idx2, idx3 = indices
    skip_if_redis_version_below(idx2.client, "8.4.0")

    hq = HybridQuery(
        text="machine learning",
        text_field_name="txt",
        vector=[0.1, 0.2, 0.3],
        vector_field_name="v",
        return_fields=["txt", "tag", "num"],
    )
    res2 = idx2.query(hq)
    res3 = idx3.query(hq)

    assert len(res3) > 0
    assert_results_equal(res2, res3)


@pytest.mark.skipif(not REDIS_HYBRID_AVAILABLE, reason="Requires redis-py >= 7.1.0")
@pytest.mark.asyncio
async def test_hybrid_search_query_async(async_indices):
    aidx2, aidx3 = async_indices
    await skip_if_redis_version_below_async(aidx2.client, "8.4.0")

    hq = HybridQuery(
        text="machine learning",
        text_field_name="txt",
        vector=[0.1, 0.2, 0.3],
        vector_field_name="v",
        return_fields=["txt", "tag", "num"],
    )
    res2 = await aidx2.query(hq)
    res3 = await aidx3.query(hq)

    assert len(res3) > 0
    assert_results_equal(res2, res3)


class MockVectorizer(BaseVectorizer):
    model: str = "mock"
    dims: int = 3

    def _get_vec(self, text: str):
        t = text.lower()
        if any(w in t for w in ["football", "soccer", "basketball", "game", "sports"]):
            return [0.0, 0.0, 1.0]
        return [1.0, 0.0, 0.0]

    def embed(self, text: str, as_buffer: bool = False, **kwargs):
        vec = self._get_vec(text)
        return array_to_buffer(vec, "float32") if as_buffer else vec

    def embed_many(self, texts: list[str], as_buffer: bool = False, **kwargs):
        return [self.embed(t, as_buffer=as_buffer) for t in texts]

    async def aembed(self, text: str, as_buffer: bool = False, **kwargs):
        return self.embed(text, as_buffer=as_buffer)

    async def aembed_many(self, texts: list[str], as_buffer: bool = False, **kwargs):
        return self.embed_many(texts, as_buffer=as_buffer)


def test_semantic_router_route_match(redis_url, redis_test_name):
    skip_if_redis_version_below(Redis.from_url(redis_url), "7.2.0")

    r2 = Redis.from_url(redis_url, protocol=2)
    r3 = Redis.from_url(redis_url, protocol=3)

    r_tech = Route(
        name="tech",
        references=["coding and programming", "computers and software"],
        distance_threshold=0.8,
    )
    r_sports = Route(
        name="sports",
        references=["football and soccer", "basketball and games"],
        distance_threshold=0.8,
    )

    name2 = redis_test_name("router_p2")
    name3 = redis_test_name("router_p3")

    mock_vec = MockVectorizer()

    router2 = SemanticRouter(
        name=name2, routes=[r_tech, r_sports], redis_client=r2, vectorizer=mock_vec
    )
    router3 = SemanticRouter(
        name=name3, routes=[r_tech, r_sports], redis_client=r3, vectorizer=mock_vec
    )

    match2 = router2("software development and computers")
    match3 = router3("software development and computers")

    assert match2 is not None
    assert match3 is not None
    assert match2.name == "tech"
    assert match3.name == "tech"
    assert pytest.approx(match2.distance, rel=1e-3) == match3.distance

    router2.delete()
    router3.delete()


@pytest.mark.parametrize("protocol", [2, 3])
def test_aggregate_cursor_loop(redis_url, test_schema, test_docs, protocol):
    client = Redis.from_url(redis_url, protocol=protocol)
    index = SearchIndex(schema=test_schema, redis_client=client)
    index.create(overwrite=True, drop=True)

    def hash_preprocess(item: dict) -> dict:
        return {**item, "v": array_to_buffer(item["v"], "float32")}

    index.load(test_docs, preprocess=hash_preprocess)

    try:
        req = AggregateRequest("*").load("@txt").cursor(count=1)
        res = index.aggregate(req)
        assert res.cursor is not None
        assert res.cursor.cid != 0
        cursor = res.cursor
        initial_count = cursor.count

        iterations = 0
        while cursor.cid:
            prev_cursor = cursor
            res = index.aggregate(cursor)
            assert res.cursor is cursor
            assert res.cursor is prev_cursor
            assert cursor.count == initial_count
            iterations += 1
            if iterations > 20:
                pytest.fail("Cursor loop did not terminate (infinite loop)")

        assert iterations > 0
        assert cursor.cid == 0
        assert cursor.count == initial_count
    finally:
        index.delete(drop=True)


@pytest.mark.parametrize("protocol", [2, 3])
@pytest.mark.asyncio
async def test_aggregate_cursor_loop_async(redis_url, test_schema, test_docs, protocol):
    aclient = AsyncRedis.from_url(redis_url, protocol=protocol)
    aindex = AsyncSearchIndex(schema=test_schema, redis_client=aclient)
    await aindex.create(overwrite=True, drop=True)

    def hash_preprocess(item: dict) -> dict:
        return {**item, "v": array_to_buffer(item["v"], "float32")}

    await aindex.load(test_docs, preprocess=hash_preprocess)

    try:
        req = AggregateRequest("*").load("@txt").cursor(count=1)
        res = await aindex.aggregate(req)
        assert res.cursor is not None
        assert res.cursor.cid != 0
        cursor = res.cursor
        initial_count = cursor.count

        iterations = 0
        while cursor.cid:
            prev_cursor = cursor
            res = await aindex.aggregate(cursor)
            assert res.cursor is cursor
            assert res.cursor is prev_cursor
            assert cursor.count == initial_count
            iterations += 1
            if iterations > 20:
                pytest.fail("Cursor loop did not terminate (infinite loop)")

        assert iterations > 0
        assert cursor.cid == 0
        assert cursor.count == initial_count
    finally:
        await aindex.delete(drop=True)


def test_update_by_filter_protocol_3(redis_url, redis_test_name):
    name = redis_test_name("upd_flt_p3")
    schema = IndexSchema.from_dict(
        {
            "index": {"name": name, "prefix": f"{name}:", "storage_type": "hash"},
            "fields": [
                {"name": "tag", "type": "tag"},
                {"name": "txt", "type": "text"},
                {"name": "num", "type": "numeric"},
            ],
        }
    )
    r3 = Redis.from_url(redis_url, protocol=3)
    idx = SearchIndex(schema=schema, redis_client=r3)
    idx.create(overwrite=True, drop=True)

    docs = [
        {"id": "doc1", "tag": "news", "txt": "first news", "num": 10},
        {"id": "doc2", "tag": "news", "txt": "second news", "num": 20},
        {"id": "doc3", "tag": "sports", "txt": "sports news", "num": 30},
        {"id": "doc4", "tag": "tech", "txt": "tech article", "num": 40},
    ]
    idx.load(docs, id_field="id")

    from redisvl.query.filter import Tag

    try:
        filter_expr = Tag("tag") == "news"
        res = idx.update_by_filter(filter_expr, values={"num": 999}, batch_size=1)
        assert res.matched == 2
        assert res.processed == 2
        assert res.completed is True

        d1 = idx.fetch("doc1")
        d2 = idx.fetch("doc2")
        assert d1 is not None and int(d1["num"]) == 999
        assert d2 is not None and int(d2["num"]) == 999

        d3 = idx.fetch("doc3")
        d4 = idx.fetch("doc4")
        assert (
            d3 is not None
            and int(d3["num"]) == 30
            and (d3["tag"] in ("sports", b"sports"))
        )
        assert (
            d4 is not None and int(d4["num"]) == 40 and (d4["tag"] in ("tech", b"tech"))
        )
    finally:
        idx.delete(drop=True)


def test_drop_by_filter_protocol_3(redis_url, redis_test_name):
    name = redis_test_name("drop_flt_p3")
    schema = IndexSchema.from_dict(
        {
            "index": {"name": name, "prefix": f"{name}:", "storage_type": "hash"},
            "fields": [
                {"name": "tag", "type": "tag"},
                {"name": "txt", "type": "text"},
                {"name": "num", "type": "numeric"},
            ],
        }
    )
    r3 = Redis.from_url(redis_url, protocol=3)
    idx = SearchIndex(schema=schema, redis_client=r3)
    idx.create(overwrite=True, drop=True)

    docs = [
        {"id": "doc1", "tag": "news", "txt": "first news", "num": 10},
        {"id": "doc2", "tag": "news", "txt": "second news", "num": 20},
        {"id": "doc3", "tag": "sports", "txt": "sports news", "num": 30},
        {"id": "doc4", "tag": "tech", "txt": "tech article", "num": 40},
    ]
    idx.load(docs, id_field="id")

    from redisvl.query.filter import Tag

    try:
        filter_expr = Tag("tag") == "news"
        res = idx.drop_by_filter(filter_expr, batch_size=1)
        assert res.matched == 2
        assert res.processed == 2
        assert res.completed is True

        # Assert matching documents were actually deleted
        assert not idx.fetch("doc1")
        assert not idx.fetch("doc2")
        assert idx.client.exists(idx.key("doc1")) == 0
        assert idx.client.exists(idx.key("doc2")) == 0

        d3 = idx.fetch("doc3")
        d4 = idx.fetch("doc4")
        assert d3 is not None and int(d3["num"]) == 30
        assert d4 is not None and int(d4["num"]) == 40
    finally:
        idx.delete(drop=True)


@pytest.mark.asyncio
async def test_update_by_filter_protocol_3_async(redis_url, redis_test_name):
    name = redis_test_name("upd_flt_p3_a")
    schema = IndexSchema.from_dict(
        {
            "index": {"name": name, "prefix": f"{name}:", "storage_type": "hash"},
            "fields": [
                {"name": "tag", "type": "tag"},
                {"name": "txt", "type": "text"},
                {"name": "num", "type": "numeric"},
            ],
        }
    )
    ar3 = AsyncRedis.from_url(redis_url, protocol=3)
    aidx = AsyncSearchIndex(schema=schema, redis_client=ar3)
    await aidx.create(overwrite=True, drop=True)

    docs = [
        {"id": "doc1", "tag": "news", "txt": "first news", "num": 10},
        {"id": "doc2", "tag": "news", "txt": "second news", "num": 20},
        {"id": "doc3", "tag": "sports", "txt": "sports news", "num": 30},
        {"id": "doc4", "tag": "tech", "txt": "tech article", "num": 40},
    ]
    await aidx.load(docs, id_field="id")

    from redisvl.query.filter import Tag

    try:
        filter_expr = Tag("tag") == "news"
        res = await aidx.update_by_filter(
            filter_expr, values={"num": 999}, batch_size=1
        )
        assert res.matched == 2
        assert res.processed == 2
        assert res.completed is True

        d1 = await aidx.fetch("doc1")
        d2 = await aidx.fetch("doc2")
        assert d1 is not None and int(d1["num"]) == 999
        assert d2 is not None and int(d2["num"]) == 999

        d3 = await aidx.fetch("doc3")
        d4 = await aidx.fetch("doc4")
        assert (
            d3 is not None
            and int(d3["num"]) == 30
            and (d3["tag"] in ("sports", b"sports"))
        )
        assert (
            d4 is not None and int(d4["num"]) == 40 and (d4["tag"] in ("tech", b"tech"))
        )
    finally:
        await aidx.delete(drop=True)


@pytest.mark.asyncio
async def test_drop_by_filter_protocol_3_async(redis_url, redis_test_name):
    name = redis_test_name("drop_flt_p3_a")
    schema = IndexSchema.from_dict(
        {
            "index": {"name": name, "prefix": f"{name}:", "storage_type": "hash"},
            "fields": [
                {"name": "tag", "type": "tag"},
                {"name": "txt", "type": "text"},
                {"name": "num", "type": "numeric"},
            ],
        }
    )
    ar3 = AsyncRedis.from_url(redis_url, protocol=3)
    aidx = AsyncSearchIndex(schema=schema, redis_client=ar3)
    await aidx.create(overwrite=True, drop=True)

    docs = [
        {"id": "doc1", "tag": "news", "txt": "first news", "num": 10},
        {"id": "doc2", "tag": "news", "txt": "second news", "num": 20},
        {"id": "doc3", "tag": "sports", "txt": "sports news", "num": 30},
        {"id": "doc4", "tag": "tech", "txt": "tech article", "num": 40},
    ]
    await aidx.load(docs, id_field="id")

    from redisvl.query.filter import Tag

    try:
        filter_expr = Tag("tag") == "news"
        res = await aidx.drop_by_filter(filter_expr, batch_size=1)
        assert res.matched == 2
        assert res.processed == 2
        assert res.completed is True

        # Assert matching documents were actually deleted
        assert not (await aidx.fetch("doc1"))
        assert not (await aidx.fetch("doc2"))
        assert await aidx.client.exists(aidx.key("doc1")) == 0
        assert await aidx.client.exists(aidx.key("doc2")) == 0

        d3 = await aidx.fetch("doc3")
        d4 = await aidx.fetch("doc4")
        assert d3 is not None and int(d3["num"]) == 30
        assert d4 is not None and int(d4["num"]) == 40
    finally:
        await aidx.delete(drop=True)
