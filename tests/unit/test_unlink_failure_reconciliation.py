"""Unit tests for surfacing per-key delete failures on Redis Cluster.

On cluster, keys are removed one at a time and a per-key error is logged instead
of raised, so a delete can partially succeed. Two consequences are tested here:

* Callers that mirror the keyspace in their own state -- notably
  ``SemanticRouter``, which persists a route config -- must not record a reference
  as removed while its key is still there, or the router keeps matching references
  the user believes were deleted.
* ``clear`` and ``drop_by_filter`` re-query from offset 0 and bound themselves by a
  count of deletions, so a batch that removes nothing must end the run rather than
  loop on it forever.

These cover the failure paths only; the all-succeed paths run against real Redis in
``tests/integration/test_semantic_router.py`` and ``test_bulk_operations.py``.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.cluster import RedisCluster
from redis.exceptions import (
    ClusterDownError,
    ConnectionError,
    RedisError,
    SlotNotCoveredError,
)

from redisvl.exceptions import PartialDeletionError
from redisvl.extensions.router.schema import Route, RoutingConfig
from redisvl.extensions.router.semantic import SemanticRouter
from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.redis.utils import hashify
from redisvl.schema import IndexSchema
from redisvl.utils.vectorize.base import BaseVectorizer

KEYS = ["unlink_failures:1", "unlink_failures:2", "unlink_failures:3"]

# Every failure mode a real cluster raises on a per-key command. RedisClusterException
# and its SlotNotCoveredError subclass do NOT inherit from RedisError, so catching
# RedisError alone would let a routine resharding/failover error abort the batch.
CLUSTER_ERRORS = [RedisError, SlotNotCoveredError, ClusterDownError, ConnectionError]


def _schema(name: str) -> IndexSchema:
    return IndexSchema.from_dict(
        {
            "index": {"name": name, "prefix": name},
            "fields": [{"name": "id", "type": "tag"}],
        }
    )


def _failing_on(failing: set[str], error=RedisError):
    """Per-key stub that removes every key except those in ``failing``."""

    def remove(*keys):
        assert len(keys) == 1, "the cluster path must remove one key at a time"
        if keys[0] in failing:
            raise error(f"boom: {keys[0]}")
        return 1

    return remove


def _cluster_index(name: str = "unlink_failures"):
    """A SearchIndex over a fake cluster client; the caller stubs the delete."""
    client = Mock(spec=RedisCluster)
    with patch("redisvl.redis.connection.RedisConnectionFactory.validate_sync_redis"):
        index = SearchIndex(schema=_schema(name), redis_client=client)
    return index, client


def _async_cluster_index():
    client = Mock(spec=AsyncRedisCluster)
    index = AsyncSearchIndex(schema=_schema("unlink_failures"), redis_client=client)
    return index, client


class TestUnlinkReportsFailures:
    """``_drop_keys`` surfaces the keys whose UNLINK failed, across batches."""

    @pytest.mark.parametrize("error", CLUSTER_ERRORS, ids=lambda e: e.__name__)
    def test_cluster_aggregates_failures_across_batches(self, error):
        index, client = _cluster_index()
        client.unlink.side_effect = _failing_on({KEYS[0], KEYS[2]}, error)

        # Two batches, one failure in each; drop_keys keeps its plain-int contract.
        assert index._drop_keys(KEYS, batch_size=2) == (1, [KEYS[0], KEYS[2]])
        assert index.drop_keys(KEYS) == 1

    @pytest.mark.parametrize("error", CLUSTER_ERRORS, ids=lambda e: e.__name__)
    @pytest.mark.asyncio
    async def test_async_cluster_aggregates_failures_across_batches(self, error):
        index, client = _async_cluster_index()
        client.unlink = AsyncMock(side_effect=_failing_on({KEYS[0], KEYS[2]}, error))

        assert await index._drop_keys(KEYS, batch_size=2) == (1, [KEYS[0], KEYS[2]])
        assert await index.drop_keys(KEYS) == 1


class TestBatchedDeleteStops:
    """A batch that removes nothing twice must end the run, not spin forever.

    ``clear`` and ``drop_by_filter`` re-query from offset 0 and their runaway
    backstops count *deletions*, so a key that never goes away keeps coming back
    while the counter stays put.
    """

    @pytest.mark.parametrize(
        "unlink_fails",
        [pytest.param(True, id="unlink-error"), pytest.param(False, id="key-gone")],
    )
    def test_drop_by_filter_stops_and_reports_incomplete(self, unlink_fails):
        index, client = _cluster_index()
        client.unlink.side_effect = (
            _failing_on({KEYS[0]}) if unlink_fails else lambda *keys: 0
        )
        index.query = MagicMock(return_value=1)  # CountQuery: 1 matching doc
        index._query = MagicMock(return_value=[{"id": KEYS[0]}])

        result = index.drop_by_filter("@id:{1}")

        assert (result.processed, result.completed) == (0, False)

    @pytest.mark.asyncio
    async def test_async_drop_by_filter_stops_and_reports_incomplete(self):
        index, client = _async_cluster_index()
        client.unlink = AsyncMock(side_effect=_failing_on({KEYS[0]}))
        index.query = AsyncMock(return_value=1)
        index._query = AsyncMock(return_value=[{"id": KEYS[0]}])

        result = await index.drop_by_filter("@id:{1}")

        assert (result.processed, result.completed) == (0, False)

    def test_drop_by_filter_finishes_batches_a_race_would_abandon(self):
        """A batch may remove nothing because someone else got there first.

        Bailing out on the first fruitless batch would abandon the rest of the
        match set, so the run continues and only stops once a fruitless batch
        brings back nothing new.
        """
        index, client = _cluster_index()
        client.unlink.side_effect = _failing_on({KEYS[0]})
        index.query = MagicMock(return_value=3)
        # Round 1 is a batch someone else already deleted; it leaves the index, so
        # round 2 returns the rest -- of which one key genuinely will not unlink.
        index._query = MagicMock(
            side_effect=[[{"id": KEYS[0]}], [{"id": KEYS[1]}, {"id": KEYS[2]}], []]
        )

        result = index.drop_by_filter("@id:{1}")

        assert (result.processed, result.completed) == (2, True)
        assert index._query.call_count == 3

    def test_clear_stops_instead_of_looping(self):
        index, client = _cluster_index()
        client.delete.side_effect = _failing_on({KEYS[0]})
        index.query = MagicMock(return_value=1)  # CountQuery: 1 matching doc
        index._query = MagicMock(return_value=[{"id": KEYS[0]}])

        assert index.clear() == 0

    @pytest.mark.asyncio
    async def test_async_clear_stops_instead_of_looping(self):
        index, client = _async_cluster_index()
        client.delete = AsyncMock(side_effect=_failing_on({KEYS[0]}))
        index.query = AsyncMock(return_value=1)
        index._query = AsyncMock(return_value=[{"id": KEYS[0]}])

        assert await index.clear() == 0


def _router(references=("hello", "hi there")):
    """A one-route SemanticRouter over a fake cluster client.

    Built with ``model_construct`` so no vectorizer model is downloaded and no
    Redis connection is made; only the delete paths are under test. It therefore
    skips ``__init__``/``_initialize_index`` -- if those grow another private
    attribute this fails loudly with AttributeError.
    """
    index, client = _cluster_index("rtr")

    vectorizer = Mock(spec=BaseVectorizer)
    vectorizer.type, vectorizer.model = "hf", "fake-model"

    router = SemanticRouter.model_construct(
        name="rtr",
        routes=[Route(name="greeting", references=list(references))],
        vectorizer=vectorizer,
        routing_config=RoutingConfig(),
    )
    router._index = index
    return router, client


def _persisted_routes(client) -> list[dict]:
    """The route config the router last wrote to Redis."""
    return client.json.return_value.set.call_args[0][2]["routes"]


class TestSemanticRouterReconcilesFailedUnlinks:
    """``SemanticRouter`` must not claim a reference is gone if its key remains."""

    def test_delete_route_references_keeps_the_failed_reference(self):
        router, client = _router()
        keys = ["rtr:greeting:h1", "rtr:greeting:h2"]
        client.hgetall.side_effect = lambda key: {
            keys[0]: {"route_name": "greeting", "reference": "hello"},
            keys[1]: {"route_name": "greeting", "reference": "hi there"},
        }[key]
        client.unlink.side_effect = _failing_on({keys[1]})

        with pytest.raises(PartialDeletionError) as excinfo:
            router.delete_route_references(keys=keys)

        assert excinfo.value.failed_keys == [keys[1]]
        assert excinfo.value.deleted == 1
        # "hello" went away and is dropped; "hi there" is still in the index. The
        # config is reconciled and persisted before the raise, so a retry resumes.
        assert router.get("greeting").references == ["hi there"]
        assert _persisted_routes(client)[0]["references"] == ["hi there"]

    def test_delete_route_references_survives_a_route_name_with_a_separator(self):
        """The route name comes from the hash, not from splitting the key.

        Deriving it positionally broke on names containing the separator, and the
        resulting error escaped *after* the keys were already unlinked -- leaving
        the config advertising references whose hashes were gone.
        """
        router, client = _router()
        router.routes = [Route(name="support:billing", references=["refund"])]
        key = "rtr:support:billing:h1"
        client.hgetall.side_effect = lambda k: {
            "route_name": "support:billing",
            "reference": "refund",
        }
        client.unlink.side_effect = _failing_on(set())

        # The reference resolved and was reconciled away; splitting the key would
        # have looked up a route named "billing" and left the config claiming
        # "refund" while its hash was gone. That was its only reference, so the
        # route goes too.
        assert router.delete_route_references(keys=[key]) == 1
        assert router.get("support:billing") is None
        assert _persisted_routes(client) == []

    def test_delete_route_references_persists_when_the_hash_is_already_gone(self):
        """A vanished hash yields no reference to reconcile -- and must not raise.

        The key is deleted by then, so raising would skip the config write.
        """
        router, client = _router()
        client.hgetall.side_effect = lambda key: {}
        client.unlink.side_effect = _failing_on(set())

        assert router.delete_route_references(keys=["rtr:greeting:h1"]) == 1
        assert _persisted_routes(client)[0]["references"] == ["hello", "hi there"]

    def test_remove_route_keeps_the_route_when_a_key_fails(self):
        router, client = _router()
        failing_key = SemanticRouter._route_ref_key(
            router._index, "greeting", hashify("hi there")
        )
        client.unlink.side_effect = _failing_on({failing_key})

        with pytest.raises(PartialDeletionError) as excinfo:
            router.remove_route("greeting")

        assert excinfo.value.failed_keys == [failing_key]
        route = router.get("greeting")
        assert route is not None, "route was dropped while a reference key remains"
        assert route.references == ["hi there"]
        assert [r["name"] for r in _persisted_routes(client)] == ["greeting"]

    def test_remove_route_keeps_duplicate_references(self):
        """Duplicates are allowed in, so a failure must not silently collapse them."""
        router, client = _router(references=("hello", "hello"))
        client.unlink.side_effect = _failing_on(
            {SemanticRouter._route_ref_key(router._index, "greeting", hashify("hello"))}
        )

        with pytest.raises(PartialDeletionError):
            router.remove_route("greeting")

        assert router.get("greeting").references == ["hello", "hello"]


class TestPersistedConfigStaysLoadable:
    """Whatever the delete paths persist must be readable back by ``from_existing``.

    ``Route`` rejects an empty reference list, so a route emptied of references and
    written out produced a config that ``from_existing``/``from_dict`` could not
    parse -- ``ValidationError: References must not be empty`` -- leaving the router
    unloadable with no way to repair it through the API.
    """

    def test_deleting_every_reference_removes_the_route(self):
        router, client = _router()
        keys = ["rtr:greeting:h1", "rtr:greeting:h2"]
        client.hgetall.side_effect = lambda key: {
            keys[0]: {"route_name": "greeting", "reference": "hello"},
            keys[1]: {"route_name": "greeting", "reference": "hi there"},
        }[key]
        client.unlink.side_effect = _failing_on(set())

        assert router.delete_route_references(keys=keys) == 2
        assert router.get("greeting") is None
        assert _persisted_routes(client) == []

    def test_a_route_emptied_by_a_delete_is_never_persisted(self):
        """Every persisted route must round-trip through ``Route`` validation."""
        router, client = _router()
        router.routes = [
            Route(name="greeting", references=["hello"]),
            Route(name="farewell", references=["bye"]),
        ]
        client.hgetall.side_effect = lambda key: {
            "route_name": "greeting",
            "reference": "hello",
        }
        client.unlink.side_effect = _failing_on(set())

        router.delete_route_references(keys=["rtr:greeting:h1"])

        persisted = _persisted_routes(client)
        assert [r["name"] for r in persisted] == ["farewell"]
        for route in persisted:
            Route(**route)  # would raise on references=[]
