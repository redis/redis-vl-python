"""Unit tests for surfacing per-key UNLINK failures on Redis Cluster.

On cluster, ``_unlink_batch`` unlinks keys one at a time and logs a per-key
``RedisError`` instead of raising, so ``drop_keys`` can report fewer deletions
than keys requested. Callers that mirror the keyspace in their own state (most
notably ``SemanticRouter``, which persists a route config) must not record a
reference as removed when its key is still there — otherwise the router keeps
matching references the user believes were deleted.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.cluster import RedisCluster
from redis.exceptions import RedisError

from redisvl.extensions.router.schema import Route, RoutingConfig
from redisvl.extensions.router.semantic import SemanticRouter
from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.redis.utils import hashify
from redisvl.schema import IndexSchema
from redisvl.utils.vectorize.base import BaseVectorizer


def _schema() -> IndexSchema:
    return IndexSchema.from_dict(
        {
            "index": {"name": "unlink_failures", "prefix": "unlink_failures"},
            "fields": [{"name": "id", "type": "tag"}],
        }
    )


def _unlink_failing_on(failing: set[str]):
    """UNLINK stub that removes every key except those in ``failing``."""

    def unlink(*keys):
        assert len(keys) == 1, "cluster path must unlink one key at a time"
        if keys[0] in failing:
            raise RedisError(f"boom: {keys[0]}")
        return 1

    return unlink


KEYS = ["unlink_failures:1", "unlink_failures:2", "unlink_failures:3"]


class TestUnlinkBatchReportsFailures:
    """``_unlink_batch`` returns the keys whose UNLINK raised."""

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis")
    def test_cluster_returns_failed_keys(self, _mock_validate):
        client = Mock(spec=RedisCluster)
        client.unlink.side_effect = _unlink_failing_on({KEYS[1]})
        index = SearchIndex(schema=_schema(), redis_client=client)

        unlinked, failed = index._unlink_batch(KEYS)

        assert unlinked == 2
        assert failed == [KEYS[1]]

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis")
    def test_standalone_reports_no_failures(self, _mock_validate):
        client = MagicMock()
        client.unlink.return_value = 3
        index = SearchIndex(schema=_schema(), redis_client=client)

        unlinked, failed = index._unlink_batch(KEYS)

        assert unlinked == 3
        assert failed == []
        client.unlink.assert_called_once_with(*KEYS)

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis")
    def test_drop_keys_still_returns_a_count(self, _mock_validate):
        """The public ``drop_keys`` contract (an int) is unchanged."""
        client = Mock(spec=RedisCluster)
        client.unlink.side_effect = _unlink_failing_on({KEYS[1]})
        index = SearchIndex(schema=_schema(), redis_client=client)

        assert index.drop_keys(KEYS) == 2

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis")
    def test_drop_keys_aggregates_failures_across_batches(self, _mock_validate):
        client = Mock(spec=RedisCluster)
        client.unlink.side_effect = _unlink_failing_on({KEYS[0], KEYS[2]})
        index = SearchIndex(schema=_schema(), redis_client=client)

        unlinked, failed = index._drop_keys(KEYS, batch_size=2)

        assert unlinked == 1
        assert failed == [KEYS[0], KEYS[2]]


class TestAsyncUnlinkBatchReportsFailures:
    """``AsyncSearchIndex`` mirrors the sync failure reporting."""

    @pytest.mark.asyncio
    async def test_cluster_returns_failed_keys(self):
        client = Mock(spec=AsyncRedisCluster)
        client.unlink = AsyncMock(side_effect=_unlink_failing_on({KEYS[1]}))
        index = AsyncSearchIndex(schema=_schema(), redis_client=client)

        unlinked, failed = await index._unlink_batch(KEYS)

        assert unlinked == 2
        assert failed == [KEYS[1]]

    @pytest.mark.asyncio
    async def test_standalone_reports_no_failures(self):
        client = MagicMock()
        client.unlink = AsyncMock(return_value=3)
        index = AsyncSearchIndex(schema=_schema(), redis_client=client)

        unlinked, failed = await index._unlink_batch(KEYS)

        assert unlinked == 3
        assert failed == []

    @pytest.mark.asyncio
    async def test_drop_keys_still_returns_a_count(self):
        client = Mock(spec=AsyncRedisCluster)
        client.unlink = AsyncMock(side_effect=_unlink_failing_on({KEYS[1]}))
        index = AsyncSearchIndex(schema=_schema(), redis_client=client)

        assert await index.drop_keys(KEYS) == 2

    @pytest.mark.asyncio
    async def test_drop_keys_aggregates_failures_across_batches(self):
        client = Mock(spec=AsyncRedisCluster)
        client.unlink = AsyncMock(side_effect=_unlink_failing_on({KEYS[0], KEYS[2]}))
        index = AsyncSearchIndex(schema=_schema(), redis_client=client)

        unlinked, failed = await index._drop_keys(KEYS, batch_size=2)

        assert unlinked == 1
        assert failed == [KEYS[0], KEYS[2]]


class TestDropByFilterMakesProgressOrStops:
    """A fully failed cluster batch must not spin forever.

    ``drop_by_filter`` re-queries from offset 0 each round and its runaway
    backstop counts *deletions*, so a key that never unlinks keeps coming back
    while the counter stays put.
    """

    @patch("redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis")
    def test_stops_when_no_key_in_the_batch_unlinks(self, _mock_validate):
        client = Mock(spec=RedisCluster)
        client.unlink.side_effect = _unlink_failing_on({KEYS[0]})
        index = SearchIndex(schema=_schema(), redis_client=client)
        index.query = MagicMock(return_value=1)  # CountQuery: 1 matching doc
        index._query = MagicMock(return_value=[{"id": KEYS[0]}])

        result = index.drop_by_filter("@id:{1}")

        assert result.processed == 0
        assert result.completed is False

    @pytest.mark.asyncio
    async def test_async_stops_when_no_key_in_the_batch_unlinks(self):
        client = Mock(spec=AsyncRedisCluster)
        client.unlink = AsyncMock(side_effect=_unlink_failing_on({KEYS[0]}))
        index = AsyncSearchIndex(schema=_schema(), redis_client=client)
        index.query = AsyncMock(return_value=1)
        index._query = AsyncMock(return_value=[{"id": KEYS[0]}])

        result = await index.drop_by_filter("@id:{1}")

        assert result.processed == 0
        assert result.completed is False


def _router_with_cluster_index(references: list[str]):
    """A SemanticRouter over a single route, backed by a fake cluster client.

    Built with ``model_construct`` so no vectorizer model is downloaded and no
    Redis connection is made; only the delete paths are under test.
    """
    with patch("redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis"):
        client = Mock(spec=RedisCluster)
        index = SearchIndex(
            schema=IndexSchema.from_dict(
                {
                    "index": {"name": "rtr", "prefix": "rtr"},
                    "fields": [{"name": "reference_id", "type": "tag"}],
                }
            ),
            redis_client=client,
        )

    vectorizer = Mock(spec=BaseVectorizer)
    vectorizer.type = "hf"
    vectorizer.model = "fake-model"

    router = SemanticRouter.model_construct(
        name="rtr",
        routes=[Route(name="greeting", references=list(references))],
        vectorizer=vectorizer,
        routing_config=RoutingConfig(),
    )
    router._index = index
    return router, client


class TestSemanticRouterReconcilesFailedUnlinks:
    """``SemanticRouter`` must not claim a reference is gone if its key remains."""

    def test_delete_route_references_keeps_failed_reference(self):
        router, client = _router_with_cluster_index(["hello", "hi there"])
        keys = ["rtr:greeting:h1", "rtr:greeting:h2"]
        client.hgetall.side_effect = lambda key: {
            "rtr:greeting:h1": {"reference": "hello"},
            "rtr:greeting:h2": {"reference": "hi there"},
        }[key]
        # The second key's UNLINK fails, so its hash is still in the index.
        client.unlink.side_effect = _unlink_failing_on({keys[1]})

        deleted = router.delete_route_references(keys=keys)

        assert deleted == 1
        route = router.get("greeting")
        assert route is not None
        assert route.references == ["hi there"]
        # The persisted config agrees with what the index can still match.
        persisted = client.json.return_value.set.call_args[0][2]
        assert persisted["routes"][0]["references"] == ["hi there"]

    def test_delete_route_references_removes_all_on_success(self):
        router, client = _router_with_cluster_index(["hello", "hi there"])
        keys = ["rtr:greeting:h1", "rtr:greeting:h2"]
        client.hgetall.side_effect = lambda key: {
            "rtr:greeting:h1": {"reference": "hello"},
            "rtr:greeting:h2": {"reference": "hi there"},
        }[key]
        client.unlink.side_effect = _unlink_failing_on(set())

        deleted = router.delete_route_references(keys=keys)

        assert deleted == 2
        route = router.get("greeting")
        assert route is not None
        assert route.references == []

    def test_remove_route_keeps_route_when_a_key_fails(self):
        router, client = _router_with_cluster_index(["hello", "hi there"])
        failing_key = SemanticRouter._route_ref_key(
            router._index, "greeting", hashify("hi there")
        )
        client.unlink.side_effect = _unlink_failing_on({failing_key})

        router.remove_route("greeting")

        route = router.get("greeting")
        assert route is not None, "route was dropped while a reference key remains"
        assert route.references == ["hi there"]
        persisted = client.json.return_value.set.call_args[0][2]
        assert [r["name"] for r in persisted["routes"]] == ["greeting"]

    def test_remove_route_drops_route_on_success(self):
        router, client = _router_with_cluster_index(["hello", "hi there"])
        client.unlink.side_effect = _unlink_failing_on(set())

        router.remove_route("greeting")

        assert router.get("greeting") is None
