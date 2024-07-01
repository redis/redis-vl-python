import asyncio
import os
from typing import Any, Dict, List, Optional, Type, Union

from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio import SSLConnection as ASSLConnection
from redis.connection import (
    AbstractConnection,
    Connection,
    ConnectionPool,
    SSLConnection,
)
from redis.exceptions import ResponseError

from redisvl.redis.constants import DEFAULT_REQUIRED_MODULES
from redisvl.redis.utils import convert_bytes
from redisvl.version import __version__


def unpack_redis_modules(module_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Unpack a list of Redis modules pulled from the MODULES LIST command."""
    return {module["name"]: module["ver"] for module in module_list}


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


def convert_index_info_to_schema(index_info: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the output of FT.INFO into a schema-ready dictionary.

    Args:
        index_info (Dict[str, Any]): Output of the Redis FT.INFO command.

    Returns:
        Dict[str, Any]: Schema dictionary.
    """
    index_name = index_info["index_name"]
    prefixes = index_info["index_definition"][3][0]
    storage_type = index_info["index_definition"][1].lower()

    index_fields = index_info["attributes"]

    def parse_vector_attrs(attrs):
        vector_attrs = {attrs[i].lower(): attrs[i + 1] for i in range(6, len(attrs), 2)}
        vector_attrs["dims"] = int(vector_attrs.pop("dim"))
        vector_attrs["distance_metric"] = vector_attrs.pop("distance_metric").lower()
        vector_attrs["algorithm"] = vector_attrs.pop("algorithm").lower()
        vector_attrs["datatype"] = vector_attrs.pop("data_type").lower()
        return vector_attrs

    def parse_attrs(attrs):
        return {attrs[i].lower(): attrs[i + 1] for i in range(6, len(attrs), 2)}

    schema_fields = []

    for field_attrs in index_fields:
        # parse field info
        name = field_attrs[1] if storage_type == "hash" else field_attrs[3]
        field = {"name": name, "type": field_attrs[5].lower()}
        if storage_type == "json":
            field["path"] = field_attrs[1]
        # parse field attrs
        if field_attrs[5] == "VECTOR":
            field["attrs"] = parse_vector_attrs(field_attrs)
        else:
            field["attrs"] = parse_attrs(field_attrs)
        # append field
        schema_fields.append(field)

    return {
        "index": {"name": index_name, "prefix": prefixes, "storage_type": storage_type},
        "fields": schema_fields,
    }


def validate_modules(
    installed_modules: Dict[str, Any],
    required_modules: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Validates if required Redis modules are installed.

    Args:
        installed_modules: List of installed modules.
        required_modules: List of required modules.

    Raises:
        ValueError: If required Redis modules are not installed.
    """
    required_modules = required_modules or DEFAULT_REQUIRED_MODULES

    for required_module in required_modules:
        if required_module["name"] in installed_modules:
            installed_version = installed_modules[required_module["name"]]  # type: ignore
            if int(installed_version) >= int(required_module["ver"]):  # type: ignore
                return

    raise ValueError(
        f"Required Redis database module {required_module['name']} with version >= {required_module['ver']} not installed. "
        "See Redis Stack documentation: https://redis.io/docs/stack/"
    )


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
        client: Union[Redis, AsyncRedis],
        lib_name: Optional[str] = None,
        required_modules: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Validates the Redis connection.

        Args:
            client (Redis or AsyncRedis): Redis client.
            lib_name (str): Library name to set on the Redis client.
            required_modules (List[Dict[str, Any]]): List of required modules and their versions.

        Raises:
            ValueError: If required Redis modules are not installed.
        """
        if isinstance(client, AsyncRedis):
            RedisConnectionFactory._run_async(
                RedisConnectionFactory._validate_async_redis,
                client,
                lib_name,
                required_modules,
            )
        else:
            RedisConnectionFactory._validate_sync_redis(
                client, lib_name, required_modules
            )

    @staticmethod
    def _get_modules(client: Redis) -> Dict[str, Any]:
        return unpack_redis_modules(convert_bytes(client.module_list()))

    @staticmethod
    async def _get_modules_async(client: AsyncRedis) -> Dict[str, Any]:
        return unpack_redis_modules(convert_bytes(await client.module_list()))

    @staticmethod
    def _validate_sync_redis(
        client: Redis,
        lib_name: Optional[str],
        required_modules: Optional[List[Dict[str, Any]]],
    ) -> None:
        """Validates the sync client."""
        # Set client library name
        _lib_name = make_lib_name(lib_name)
        try:
            client.client_setinfo("LIB-NAME", _lib_name)  # type: ignore
        except ResponseError:
            # Fall back to a simple log echo
            client.echo(_lib_name)

        # Get list of modules
        installed_modules = RedisConnectionFactory._get_modules(client)

        # Validate available modules
        validate_modules(installed_modules, required_modules)

    @staticmethod
    async def _validate_async_redis(
        client: AsyncRedis,
        lib_name: Optional[str],
        required_modules: Optional[List[Dict[str, Any]]],
    ) -> None:
        """Validates the async client."""
        # Set client library name
        _lib_name = make_lib_name(lib_name)
        try:
            await client.client_setinfo("LIB-NAME", _lib_name)  # type: ignore
        except ResponseError:
            # Fall back to a simple log echo
            await client.echo(_lib_name)

        # Get list of modules
        installed_modules = await RedisConnectionFactory._get_modules_async(client)

        # Validate available modules
        validate_modules(installed_modules, required_modules)

    @staticmethod
    def _run_async(coro, *args, **kwargs):
        """
        Runs an asynchronous function in the appropriate event loop context.

        This method checks if there is an existing event loop running. If there is,
        it schedules the coroutine to be run within the current loop using `asyncio.ensure_future`.
        If no event loop is running, it creates a new event loop, runs the coroutine,
        and then closes the loop to avoid resource leaks.

        Args:
            coro (coroutine): The coroutine function to be run.
            *args: Positional arguments to pass to the coroutine function.
            **kwargs: Keyword arguments to pass to the coroutine function.

        Returns:
            The result of the coroutine if a new event loop is created,
            otherwise a task object representing the coroutine execution.
        """
        try:
            # Try to get the current running event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:  # No running event loop
            loop = None

        if loop and loop.is_running():
            # If an event loop is running, schedule the coroutine to run in the existing loop
            return asyncio.ensure_future(coro(*args, **kwargs))
        else:
            # No event loop is running, create a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Run the coroutine in the new event loop and wait for it to complete
                return loop.run_until_complete(coro(*args, **kwargs))
            finally:
                # Close the event loop to release resources
                loop.close()
