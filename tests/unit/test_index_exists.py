"""
Unit tests for how SearchIndex.exists() and AsyncSearchIndex.exists() classify
the Redis response.

exists() runs FT.INFO and has to distinguish three outcomes from a single
command: the index is there, the index is absent, or the check itself failed.
Redis Search offers no structured "not found" signal and phrases the error
differently across versions, so these tests pin all three branches. They drive
the real _info() helper rather than stubbing it, because _info() rewrites the
message into "Error while fetching <name> index info: ..." and the not-found
match has to survive that wrapping.
"""

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.cluster import RedisCluster
from redis.exceptions import ConnectionError, ResponseError

from redisvl.exceptions import RedisSearchError
from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.schema import IndexSchema

INDEX_NAME = "my_index"

# Every wording Redis Search has used for an absent index. 8.8 replaced the
# older prose with an error code, so a match on any single one of these would
# break exists() on some supported server.
MISSING_INDEX_MESSAGES = [
    "Unknown index name",
    f"{INDEX_NAME}: no such index",
    f"SEARCH_INDEX_NOT_FOUND Index not found: {INDEX_NAME}",
    "Index not found",
]

# Failures that must never be read as "the index is absent": reporting False
# here would send callers on to create an index that already exists, and would
# skip the schema-compatibility check the extensions perform first. The two
# exception types matter -- _info() catches bare Exception, so narrowing it to
# ResponseError would let the connection case escape unwrapped.
NON_MISSING_INDEX_ERRORS = [
    ResponseError("User acl_user has no permissions to run the 'FT.INFO' command"),
    ConnectionError("Error 111 connecting to localhost:6379. Connection refused."),
]


def _schema(name=INDEX_NAME):
    return IndexSchema.from_dict(
        {
            "index": {"name": name, "prefix": "test"},
            "fields": [{"name": "t", "type": "text"}],
        }
    )


def _sync_index(info_result=None, info_error=None, name=INDEX_NAME):
    """Build a SearchIndex whose FT.INFO call returns or raises as directed."""
    client = MagicMock(spec=Redis)
    if info_error is not None:
        client.ft.return_value.info.side_effect = info_error
    else:
        client.ft.return_value.info.return_value = info_result
    return SearchIndex(schema=_schema(name), redis_client=client), client


def _async_index(info_result=None, info_error=None):
    """Build an AsyncSearchIndex whose FT.INFO call returns or raises as directed."""
    client = MagicMock(spec=AsyncRedis)
    if info_error is not None:
        client.ft.return_value.info = AsyncMock(side_effect=info_error)
    else:
        client.ft.return_value.info = AsyncMock(return_value=info_result)
    return AsyncSearchIndex(schema=_schema(), redis_client=client), client


class TestExistsClassifiesRedisResponse:
    """exists() must map FT.INFO outcomes to True, False, or an exception."""

    @pytest.mark.parametrize("message", MISSING_INDEX_MESSAGES)
    def test_missing_index_wording_returns_false(self, message):
        index, _ = _sync_index(info_error=ResponseError(message))
        assert index.exists() is False

    @pytest.mark.parametrize("error", NON_MISSING_INDEX_ERRORS)
    def test_other_failures_are_raised_not_reported_as_absent(self, error):
        index, _ = _sync_index(info_error=error)
        with pytest.raises(RedisSearchError):
            index.exists()

    def test_failure_on_index_named_after_a_wording_is_still_raised(self):
        # _info() interpolates the index name into its message, so an index
        # named after one of the wordings makes the *wrapper* look like an
        # absence report. Classification therefore reads the original Redis
        # error via __cause__; reading the wrapper would turn this permission
        # error into exists() == False.
        index, _ = _sync_index(
            name="search_index_not_found",
            info_error=ResponseError(
                "User acl_user has no permissions to run the 'FT.INFO' command"
            ),
        )
        with pytest.raises(RedisSearchError):
            index.exists()

    def test_index_name_resolving_elsewhere_returns_false(self):
        # FT.INFO accepts an alias and answers for its target, so a successful
        # reply naming a different index means this name is an alias, not an
        # index. Reporting True would let create(overwrite=True, drop=True)
        # drop the aliased index and its documents.
        index, _ = _sync_index(info_result={"index_name": "some_other_index"})
        assert index.exists() is False

    # The async twin is a hand-maintained copy, so it needs its own witness for
    # each branch -- but the branch logic itself is shared, so one wording is
    # enough. Live servers cover the current wording via the integration suite;
    # the newest one is only reachable here.
    @pytest.mark.asyncio
    async def test_async_missing_index_wording_returns_false(self):
        index, _ = _async_index(info_error=ResponseError(MISSING_INDEX_MESSAGES[2]))
        assert await index.exists() is False

    @pytest.mark.asyncio
    async def test_async_other_failures_are_raised(self):
        index, _ = _async_index(info_error=NON_MISSING_INDEX_ERRORS[0])
        with pytest.raises(RedisSearchError):
            await index.exists()

    @pytest.mark.asyncio
    async def test_async_index_name_resolving_elsewhere_returns_false(self):
        index, _ = _async_index(info_result={"index_name": "some_other_index"})
        assert await index.exists() is False


class TestExistsOnCluster:
    """On a cluster, exists() must go through _info()'s node-targeted path.

    Index metadata is cluster-wide, so any single node can answer -- but the
    command has to carry target_nodes to get there. These are the only cluster
    checks that run by default; the live cluster suite needs
    --run-cluster-tests. Sync and async _info() each have their own cluster
    branch, and exists() reaches both for the first time with this change.
    """

    def test_ft_info_is_routed_to_a_node(self):
        node = Mock()
        client = MagicMock(spec=RedisCluster)
        client.get_random_node.return_value = node
        client.execute_command.return_value = ["index_name", INDEX_NAME]
        index = SearchIndex(schema=_schema(), redis_client=client)

        assert index.exists() is True
        client.execute_command.assert_called_once_with(
            "FT.INFO", INDEX_NAME, target_nodes=node
        )

    @pytest.mark.asyncio
    async def test_async_ft_info_is_routed_to_a_node(self):
        node = Mock()
        client = MagicMock()
        client.__class__ = AsyncRedisCluster  # isinstance drives the branch
        client.get_random_node.return_value = node
        client.execute_command = AsyncMock(return_value=["index_name", INDEX_NAME])
        index = AsyncSearchIndex(schema=_schema(), redis_client=client)

        assert await index.exists() is True
        client.execute_command.assert_awaited_once_with(
            "FT.INFO", INDEX_NAME, target_nodes=node
        )
