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
    """Manages connections to a Redis database, supporting both synchronous and
    asynchronous clients.

    This class allows for establishing and handling Redis connections using
    either standard Redis or async Redis clients, based on the provided
    configuration.
    """

    def __init__(self):
        self._redis_url = None
        self._kwargs = None
        self.client: Optional[Union[Redis, ARedis]] = None

    def connect(
        self, redis_url: Optional[str] = None, use_async: bool = False, **kwargs
    ) -> None:
        """Establishes a connection to the Redis database.

        This method sets up either a synchronous or asynchronous Redis client
        based on the provided parameters.

        Args:
            redis_url (Optional[str]): The URL of the Redis server to connect
                to. If not provided, the environment variable REDIS_URL is used.
            use_async (bool): If True, an asynchronous client is created.
                Defaults to False.
            **kwargs: Additional keyword arguments to be passed to the Redis
                client constructor.

        Raises:
            ValueError: If redis_url is not provided and REDIS_URL environment
                variable is not set.
        """
        self._redis_url = redis_url
        self._kwargs = kwargs
        if use_async:
            self.client = self.get_async_redis_connection(
                self._redis_url, **self._kwargs
            )
        else:
            self.client = self.get_redis_connection(self._redis_url, **self._kwargs)

    def set_client(self, client: Union[Redis, ARedis]) -> None:
        """Sets the Redis client instance for the connection.

        This method allows setting a pre-configured Redis client, either
        synchronous or asynchronous.

        Args:
            client (Union[Redis, ARedis]): The Redis client instance to be set.

        Raises:
            TypeError: If the provided client is not a valid Redis client
                instance.
        """
        if not (isinstance(client, Redis) or isinstance(client, ARedis)):
            raise TypeError("Must provide a valid Redis client instance")
        self.client = client

    @staticmethod
    def get_redis_connection(url: Optional[str] = None, **kwargs) -> Redis:
        """Creates and returns a synchronous Redis client.

        Args:
            url (Optional[str]): The URL of the Redis server. If not provided,
                the environment variable REDIS_URL is used.
            **kwargs: Additional keyword arguments to be passed to the Redis
                client constructor.

        Returns:
            Redis: A synchronous Redis client instance.

        Raises:
            ValueError: If url is not provided and REDIS_URL environment
                variable is not set.
        """
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
        """Creates and returns an asynchronous Redis client.

        Args:
            url (Optional[str]): The URL of the Redis server. If not provided,
                the environment variable REDIS_URL is used.
            **kwargs: Additional keyword arguments to be passed to the async
                Redis client constructor.

        Returns:
            ARedis: An asynchronous Redis client instance.

        Raises:
            ValueError: If url is not provided and REDIS_URL environment
                variable is not set.
        """
        if url:
            client = ARedis.from_url(url, **kwargs)
        else:
            try:
                client = ARedis.from_url(get_address_from_env())
            except ValueError:
                raise ValueError("No Redis URL provided and REDIS_URL env var not set")
        return client
