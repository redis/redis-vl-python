import os
from typing import Optional, Union

from redis import Redis
from redis.asyncio import Redis as ARedis

# TODO: handle connection errors.


def get_address_from_env():
    """Get a redis connection from environment variables.

    Returns:
        str: Redis URL
    """
    addr = os.getenv("REDIS_URL", None)
    if not addr:
        raise ValueError("REDIS_URL env var not set")
    return addr


class RedisConnection:
    _redis_url = None
    _kwargs = None
    client: Optional[Union[Redis, ARedis]] = None

    def connect(
        self,
        redis_url: Optional[str] = None,
        use_async: bool = False,
        **kwargs
    ):
        self._redis_url = redis_url
        self._kwargs = kwargs
        if not use_async:
            self.client = self.get_redis_connection(self._redis_url, **self._kwargs)
        else:
            self.client = self.get_async_redis_connection(
                self._redis_url, **self._kwargs
            )

    def set_client(self, client: Union[Redis, ARedis]):
        if not (isinstance(client, Redis) or isinstance(client, ARedis)):
            raise TypeError("Must provide a valid Redis client instance")
        self.client = client

    @staticmethod
    def get_redis_connection(url: Optional[str] = None, **kwargs) -> Redis:
        from redis import Redis

        if url:
            client = Redis.from_url(url, **kwargs)
        else:
            try:
                client = Redis.from_url(get_address_from_env())
            except ValueError:
                raise ValueError("No Redis URL provided and REDIS_URL env var not set")
        return client

    @staticmethod
    def get_async_redis_connection(url: Optional[str] = None, **kwargs) -> ARedis:
        from redis.asyncio import Redis as ARedis

        if url:
            client = ARedis.from_url(url, **kwargs)
        else:
            try:
                client = ARedis.from_url(get_address_from_env())
            except ValueError:
                raise ValueError("No Redis URL provided and REDIS_URL env var not set")
        return client
