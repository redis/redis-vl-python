import asyncio
import numpy as np
import typing as t

from redis.asyncio import Redis

# TODO Add conncurrent_store_as_json

async def concurrent_store_as_hash(
    data: t.List[t.Dict[str, t.Any]], # TODO: be stricter about the type
    n: int,
    vector_field_name: str,
    prefix: str,
    redis_conn: Redis
):
    """
    Gather and load the hashes into Redis using
    async connections.

    Args:
        n (int): Max number of "concurrent" async connections.
        vector_field_name (str): name of the vector field
        prefix (str): Redis key prefix for all hashes in the search index.
        redis_conn (Redis): Redis connection.
    """
    semaphore = asyncio.Semaphore(n)
    async def load(d: dict):
        async with semaphore:
            d[vector_field_name] = np.array(d[vector_field_name], dtype = np.float32).tobytes()
            key = prefix + str(d["id"])
            await redis_conn.hset(key, mapping = d)

    # gather with concurrency
    await asyncio.gather(*[load(d) for d in data])
