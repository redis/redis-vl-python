import asyncio
import os
from threading import Thread
from typing import Optional, Union

from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from redisvl.utils.utils import convert_bytes
from redisvl.redis.constants import REDIS_REQUIRED_MODULES


def get_address_from_env() -> str:
    """Get a redis connection from environment variables.

    Returns:
        str: Redis URL
    """
    if "REDIS_URL" not in os.environ:
        raise ValueError("REDIS_URL env var not set")
    return os.environ["REDIS_URL"]


def run_async(coroutine):
    # def run():
    #     asyncio.run(coroutine)

    # thread = Thread(target=run)
    # thread.start()
    # thread.join()
    asyncio.run(coroutine)


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
        self.client: Optional[Union[Redis, AsyncRedis]] = None

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
        self._redis_url = redis_url or get_address_from_env()
        self._kwargs = kwargs
        connection_func = (
            self.get_async_redis_connection if use_async else self.get_redis_connection
        )
        self.client = connection_func(self._redis_url, **self._kwargs)  # type: ignore

        # Check for required modules
        if use_async:
            assert isinstance(self.client, AsyncRedis)
            run_async(self.check_async_redis_modules_exist(self.client))
        else:
            assert isinstance(self.client, Redis)
            self.check_redis_modules_exist(self.client)

    def set_client(self, client: Union[Redis, AsyncRedis]) -> None:
        """Sets the Redis client instance for the connection.

        This method allows setting a pre-configured Redis client, either
        synchronous or asynchronous.

        Args:
            client (Union[Redis, AsyncRedis]): The Redis client instance to be set.

        Raises:
            TypeError: If the provided client is not a valid Redis client
                instance.
        """
        self.client = client

        # Check for required modules
        if isinstance(client, AsyncRedis):
            run_async(self.check_async_redis_modules_exist(self.client))  # type: ignore
        elif isinstance(client, Redis):
            self.check_redis_modules_exist(self.client)  # type: ignore
        else:
            raise TypeError("Invalid Redis client instance")

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
            return Redis.from_url(url, **kwargs)
        # fallback to env var REDIS_URL
        return Redis.from_url(get_address_from_env(), **kwargs)

    @staticmethod
    def get_async_redis_connection(url: Optional[str] = None, **kwargs) -> AsyncRedis:
        """Creates and returns an asynchronous Redis client.

        Args:
            url (Optional[str]): The URL of the Redis server. If not provided,
                the environment variable REDIS_URL is used.
            **kwargs: Additional keyword arguments to be passed to the async
                Redis client constructor.

        Returns:
            AsyncRedis: An asynchronous Redis client instance.

        Raises:
            ValueError: If url is not provided and REDIS_URL environment
                variable is not set.
        """
        if url:
            return AsyncRedis.from_url(url, **kwargs)
        # fallback to env var REDIS_URL
        return AsyncRedis.from_url(get_address_from_env(), **kwargs)

    @staticmethod
    def check_redis_modules_exist(client: Redis) -> None:
        """Validates if the required Redis modules are installed.

        Args:
            client (Redis): Synchronous Redis client.

        Raises:
            ValueError: If required Redis modules are not installed.
        """
        RedisConnection._validate_redis_modules(convert_bytes(client.module_list()))

    @staticmethod
    async def check_async_redis_modules_exist(client: AsyncRedis) -> None:
        """
        Validates if the required Redis modules are installed.

        Args:
            client (AsyncRedis): Asynchronous Redis client.

        Raises:
            ValueError: If required Redis modules are not installed.
        """
        installed_modules = await client.module_list()
        RedisConnection._validate_redis_modules(convert_bytes(installed_modules))

    @staticmethod
    def _validate_redis_modules(installed_modules) -> None:
        """
        Validates if required Redis modules are installed.

        Args:
            installed_modules: List of installed modules.

        Raises:
            ValueError: If required Redis modules are not installed.
        """
        installed_modules = {module["name"]: module for module in installed_modules}
        for required_module in REDIS_REQUIRED_MODULES:
            if required_module["name"] in installed_modules:
                installed_version = installed_modules[required_module["name"]]["ver"]
                if int(installed_version) >= int(required_module["ver"]):  # type: ignore
                    return

        raise ValueError(
            f"Required Redis database module {required_module['name']} with version >= {required_module['ver']} not installed. "
            "Refer to Redis Stack documentation: https://redis.io/docs/stack/"
        )
