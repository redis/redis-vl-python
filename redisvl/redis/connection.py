import os
from typing import Any, Dict, List, Optional, Type, Union
from warnings import warn

from redis import Redis
from redis.asyncio import Connection as AsyncConnection
from redis.asyncio import ConnectionPool as AsyncConnectionPool
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio import SSLConnection as AsyncSSLConnection
from redis.connection import AbstractConnection, SSLConnection
from redis.exceptions import ResponseError

from redisvl.exceptions import RedisModuleVersionError
from redisvl.redis.constants import DEFAULT_REQUIRED_MODULES
from redisvl.redis.utils import convert_bytes
from redisvl.utils.utils import deprecated_function
from redisvl.version import __version__


def compare_versions(version1, version2):
    """
    Compare two Redis version strings numerically.

    Parameters:
    version1 (str): The first version string (e.g., "7.2.4").
    version2 (str): The second version string (e.g., "6.2.1").

    Returns:
    int: -1 if version1 < version2, 0 if version1 == version2, 1 if version1 > version2.
    """
    v1_parts = list(map(int, version1.split(".")))
    v2_parts = list(map(int, version2.split(".")))

    for v1, v2 in zip(v1_parts, v2_parts):
        if v1 < v2:
            return False
        elif v1 > v2:
            return True

    # If the versions are equal so far, compare the lengths of the version parts
    if len(v1_parts) < len(v2_parts):
        return False
    elif len(v1_parts) > len(v2_parts):
        return True

    return True


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
        # 'SORTABLE', 'NOSTEM' don't have corresponding values.
        # Their presence indicates boolean True
        # TODO 'WITHSUFFIXTRIE' is another boolean attr, but is not returned by ft.info
        original = attrs.copy()
        parsed_attrs = {}
        if "NOSTEM" in attrs:
            parsed_attrs["no_stem"] = True
            attrs.remove("NOSTEM")
        if "CASESENSITIVE" in attrs:
            parsed_attrs["case_sensitive"] = True
            attrs.remove("CASESENSITIVE")
        if "SORTABLE" in attrs:
            parsed_attrs["sortable"] = True
            attrs.remove("SORTABLE")
            if "UNF" in attrs:
                attrs.remove("UNF")  # UNF present on sortable numeric fields only

        try:
            parsed_attrs.update(
                {attrs[i].lower(): attrs[i + 1] for i in range(6, len(attrs), 2)}
            )
        except IndexError as e:
            raise IndexError(f"Error parsing index attributes {original}, {str(e)}")
        return parsed_attrs

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
        RedisModuleVersionError: If required Redis modules are not installed.
    """
    required_modules = required_modules or DEFAULT_REQUIRED_MODULES

    for required_module in required_modules:
        if required_module["name"] in installed_modules:
            installed_version = installed_modules[required_module["name"]]  # type: ignore
            if int(installed_version) >= int(required_module["ver"]):  # type: ignore
                return

    # Build the error message dynamically
    required_modules_str = " OR ".join(
        [f'{module["name"]} >= {module["ver"]}' for module in required_modules]
    )
    error_message = (
        f"Required Redis db module {required_modules_str} not installed. "
        "See Redis Stack docs at https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/."
    )

    raise RedisModuleVersionError(error_message)


class RedisConnectionFactory:
    """Builds connections to a Redis database, supporting both synchronous and
    asynchronous clients.

    This class allows for establishing and handling Redis connections using
    either standard Redis or async Redis clients, based on the provided
    configuration.
    """

    @classmethod
    @deprecated_function(
        "connect", "Please use `get_redis_connection` or `get_async_redis_connection`."
    )
    def connect(
        cls, redis_url: Optional[str] = None, use_async: bool = False, **kwargs
    ) -> Union[Redis, AsyncRedis]:
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
    def get_redis_connection(
        redis_url: Optional[str] = None,
        required_modules: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Redis:
        """Creates and returns a synchronous Redis client.

        Args:
            url (Optional[str]): The URL of the Redis server. If not provided,
                the environment variable REDIS_URL is used.
            required_modules (Optional[List[Dict[str, Any]]]): List of required
                Redis modules with version requirements.
            **kwargs: Additional keyword arguments to be passed to the Redis
                client constructor.

        Returns:
            Redis: A synchronous Redis client instance.

        Raises:
            ValueError: If url is not provided and REDIS_URL environment
                variable is not set.
            RedisModuleVersionError: If required Redis modules are not installed.
        """
        url = redis_url or get_address_from_env()
        client = Redis.from_url(url, **kwargs)

        RedisConnectionFactory.validate_sync_redis(
            client, required_modules=required_modules
        )

        return client

    @staticmethod
    async def _get_aredis_connection(
        url: Optional[str] = None,
        required_modules: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AsyncRedis:
        """Creates and returns an asynchronous Redis client.

        NOTE: This method is the future form of `get_async_redis_connection` but is
        only used internally by the library now.

        Args:
            url (Optional[str]): The URL of the Redis server. If not provided,
                the environment variable REDIS_URL is used.
            required_modules (Optional[List[Dict[str, Any]]]): List of required
                Redis modules with version requirements.
            **kwargs: Additional keyword arguments to be passed to the async
                Redis client constructor.

        Returns:
            AsyncRedis: An asynchronous Redis client instance.

        Raises:
            ValueError: If url is not provided and REDIS_URL environment
                variable is not set.
            RedisModuleVersionError: If required Redis modules are not installed.
        """
        url = url or get_address_from_env()
        client = AsyncRedis.from_url(url, **kwargs)

        await RedisConnectionFactory.validate_async_redis(
            client, required_modules=required_modules
        )
        return client

    @staticmethod
    def get_async_redis_connection(
        url: Optional[str] = None,
        **kwargs,
    ) -> AsyncRedis:
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
        warn(
            "get_async_redis_connection will become async in the next major release.",
            DeprecationWarning,
        )
        url = url or get_address_from_env()
        return AsyncRedis.from_url(url, **kwargs)

    @staticmethod
    def sync_to_async_redis(redis_client: Redis) -> AsyncRedis:
        """Convert a synchronous Redis client to an asynchronous one."""
        # pick the right connection class
        connection_class: Type[AbstractConnection] = (
            AsyncSSLConnection
            if redis_client.connection_pool.connection_class == SSLConnection
            else AsyncConnection
        )  # type: ignore
        # make async client
        return AsyncRedis.from_pool(  # type: ignore
            AsyncConnectionPool(
                connection_class=connection_class,
                **redis_client.connection_pool.connection_kwargs,
            )
        )

    @staticmethod
    def get_modules(client: Redis) -> Dict[str, Any]:
        return unpack_redis_modules(convert_bytes(client.module_list()))

    @staticmethod
    async def get_modules_async(client: AsyncRedis) -> Dict[str, Any]:
        return unpack_redis_modules(convert_bytes(await client.module_list()))

    @staticmethod
    def validate_sync_redis(
        redis_client: Redis,
        lib_name: Optional[str] = None,
        required_modules: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Validates the sync Redis client."""
        if not isinstance(redis_client, Redis):
            raise TypeError("Invalid Redis client instance")

        # Set client library name
        _lib_name = make_lib_name(lib_name)
        try:
            redis_client.client_setinfo("LIB-NAME", _lib_name)  # type: ignore
        except ResponseError:
            # Fall back to a simple log echo
            redis_client.echo(_lib_name)

        # Get list of modules
        installed_modules = RedisConnectionFactory.get_modules(redis_client)

        # Validate available modules
        validate_modules(installed_modules, required_modules)

    @staticmethod
    async def validate_async_redis(
        redis_client: AsyncRedis,
        lib_name: Optional[str] = None,
        required_modules: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Validates the async Redis client."""
        # Set client library name
        _lib_name = make_lib_name(lib_name)
        try:
            await redis_client.client_setinfo("LIB-NAME", _lib_name)  # type: ignore
        except ResponseError:
            # Fall back to a simple log echo
            await redis_client.echo(_lib_name)

        # Get list of modules
        installed_modules = await RedisConnectionFactory.get_modules_async(redis_client)

        # Validate available modules
        validate_modules(installed_modules, required_modules)
