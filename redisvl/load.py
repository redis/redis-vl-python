import asyncio
import typing as t

import numpy as np
from redis.asyncio import Redis

# TODO Add conncurrent_store_as_json


async def concurrent_store_as_hash(
    data: t.List[t.Dict[str, t.Any]],  # TODO: be stricter about the type
    concurrency: int,
    key_field: str,
    prefix: str,
    redis_conn: Redis,
):
    """
    Gather and load the hashes into Redis using
    async connections.

    Args:
        concurrency (int): Max number of "concurrent" async connections.
        key_field: name of the key in each dict to use as top level key in Redis.
        prefix (str): Redis key prefix for all hashes in the search index.
        redis_conn (Redis): Redis connection.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def load(d: dict):
        async with semaphore:
            key = prefix + str(d[key_field])
            await redis_conn.hset(key, mapping=d)

    # gather with concurrency
    await asyncio.gather(*[load(d) for d in data])
