import os
from functools import wraps
from typing import Optional

# TODO: handle connection errors.


def get_redis_connection(url: Optional[str] = None, **kwargs):
    from redis import Redis

    if url:
        client = Redis.from_url(url, **kwargs)
    else:
        try:
            client = Redis.from_url(get_address_from_env())
        except ValueError:
            raise ValueError("No Redis URL provided and REDIS_URL env var not set")
    return client


def get_async_redis_connection(url: Optional[str] = None, **kwargs):
    from redis.asyncio import Redis as ARedis

    if url:
        client = ARedis.from_url(url, **kwargs)
    else:
        try:
            client = ARedis.from_url(get_address_from_env())
        except ValueError:
            raise ValueError("No Redis URL provided and REDIS_URL env var not set")
    return client


def get_address_from_env():
    """Get a redis connection from environment variables

    Returns:
        str: Redis URL
    """
    addr = os.getenv("REDIS_URL", None)
    if not addr:
        raise ValueError("REDIS_URL env var not set")
    return addr


def check_connected(client_variable_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if getattr(self, client_variable_name) is None:
                raise ValueError(
                    f"SearchIndex.connect() must be called before calling {func.__name__}"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator
