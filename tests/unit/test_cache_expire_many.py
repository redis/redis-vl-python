"""Batched TTL refresh behind SemanticCache.check() and acheck().

Hermetic because cluster tests do not run in CI. The cluster-relevant property
is that the pipeline is non-transactional, so one batch may span hash slots.
"""

from unittest.mock import AsyncMock, MagicMock

from redisvl.extensions.cache.base import BaseCache

KEYS = ["cache:1", "cache:2", "cache:3"]


def test_expire_many_sends_one_non_transactional_pipeline():
    client = MagicMock()
    pipe = client.pipeline.return_value.__enter__.return_value

    BaseCache(name="cache", ttl=60, redis_client=client)._expire_many(KEYS)

    client.pipeline.assert_called_once_with(transaction=False)
    assert [c.args for c in pipe.expire.call_args_list] == [(k, 60) for k in KEYS]
    pipe.execute.assert_called_once_with()


async def test_aexpire_many_sends_one_non_transactional_pipeline():
    client = MagicMock()
    pipe = MagicMock(execute=AsyncMock())
    client.pipeline.return_value.__aenter__.return_value = pipe

    await BaseCache(name="cache", ttl=60, async_redis_client=client)._aexpire_many(KEYS)

    client.pipeline.assert_called_once_with(transaction=False)
    assert [c.args for c in pipe.expire.call_args_list] == [(k, 60) for k in KEYS]
    pipe.execute.assert_awaited_once_with()


async def test_expire_many_makes_no_call_without_a_ttl():
    client = MagicMock()
    cache = BaseCache(name="cache", redis_client=client, async_redis_client=client)

    cache._expire_many(KEYS)
    await cache._aexpire_many(KEYS)

    client.pipeline.assert_not_called()
