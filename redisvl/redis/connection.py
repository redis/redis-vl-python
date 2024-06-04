import os
from typing import Any, Dict, List, Optional, Type

from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio import SSLConnection as ASSLConnection
from redis.connection import (
    AbstractConnection,
    Connection,
    ConnectionPool,
    SSLConnection,
)

from redisvl.redis.constants import REDIS_REQUIRED_MODULES
from redisvl.redis.utils import convert_bytes
from redisvl.version import __version__


def get_address_from_env() -> str:
    """Get a redis connection from environment variables.

    Returns:
        str: Redis URL
    """
    if "REDIS_URL" not in os.environ:
        raise ValueError("REDIS_URL env var not set")
    return os.environ["REDIS_URL"]


def make_lib_name(*args) -> str:
    """Build the lib name to be reported through the Redis client setinfo
    command.

    Returns:
        str: Redis client library name
    """
    custom_libs = f"redisvl_v{__version__}"
    for arg in args:
        if arg:
            custom_libs += f";{arg}"
    return f"redis-py({custom_libs})"


class RedisConnectionFactory:
    """Builds connections to a Redis database, supporting both synchronous and
    asynchronous clients.

    This class allows for establishing and handling Redis connections using
    either standard Redis or async Redis clients, based on the provided
    configuration.
    """

    @classmethod
    def connect(
        cls, redis_url: Optional[str] = None, use_async: bool = False, **kwargs
    ) -> None:
        """Create a connection to the Redis database based on a URL and some
        connection kwargs.

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
        redis_url = redis_url or get_address_from_env()
        connection_func = (
            cls.get_async_redis_connection if use_async else cls.get_redis_connection
        )
        return connection_func(redis_url, **kwargs)  # type: ignore

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
    def validate_redis(
        client: Redis,
        lib_name: Optional[str] = None,
        redis_required_modules: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Validates if the required Redis modules are installed.

        Args:
            client (Redis): Synchronous Redis client.

        Raises:
            ValueError: If required Redis modules are not installed.
        """
        # set client library name
        client.client_setinfo("LIB-NAME", make_lib_name(lib_name))

        # validate available modules
        RedisConnectionFactory._validate_modules(
            convert_bytes(client.module_list()), redis_required_modules
        )

    @staticmethod
    def validate_async_redis(
        client: AsyncRedis,
        lib_name: Optional[str] = None,
        redis_required_modules: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Validates if the required Redis modules are installed.

        Args:
            client (AsyncRedis): Asynchronous Redis client.

        Raises:
            ValueError: If required Redis modules are not installed.
        """
        # pick the right connection class
        connection_class: Type[AbstractConnection] = (
            SSLConnection
            if client.connection_pool.connection_class == ASSLConnection
            else Connection
        )
        # set up a temp sync client
        temp_client = Redis(
            connection_pool=ConnectionPool(
                connection_class=connection_class,
                **client.connection_pool.connection_kwargs,
            )
        )
        RedisConnectionFactory.validate_redis(
            temp_client, lib_name, redis_required_modules
        )

    @staticmethod
    def _validate_modules(
        installed_modules, redis_required_modules: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Validates if required Redis modules are installed.

        Args:
            installed_modules: List of installed modules.
            redis_required_modules: List of required modules.

        Raises:
            ValueError: If required Redis modules are not installed.
        """
        installed_modules = {module["name"]: module for module in installed_modules}
        redis_required_modules = redis_required_modules or REDIS_REQUIRED_MODULES

        for required_module in redis_required_modules:
            if required_module["name"] in installed_modules:
                installed_version = installed_modules[required_module["name"]]["ver"]
                if int(installed_version) >= int(required_module["ver"]):  # type: ignore
                    return

        raise ValueError(
            f"Required Redis database module {required_module['name']} with version >= {required_module['ver']} not installed. "
            "Refer to Redis Stack documentation: https://redis.io/docs/stack/"
        )
