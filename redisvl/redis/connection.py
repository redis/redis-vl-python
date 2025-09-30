import os
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, overload
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from warnings import warn

from redis import Redis, RedisCluster
from redis.asyncio import ConnectionPool as AsyncConnectionPool
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.asyncio.connection import AbstractConnection as AsyncAbstractConnection
from redis.asyncio.connection import Connection as AsyncConnection
from redis.asyncio.connection import SSLConnection as AsyncSSLConnection
from redis.connection import SSLConnection
from redis.exceptions import ResponseError
from redis.sentinel import Sentinel

from redisvl import __version__
from redisvl.redis.constants import REDIS_URL_ENV_VAR
from redisvl.redis.utils import convert_bytes, is_cluster_url
from redisvl.types import AsyncRedisClient, RedisClient, SyncRedisClient
from redisvl.utils.utils import deprecated_function


def _strip_cluster_from_url_and_kwargs(
    url: str, **kwargs
) -> Tuple[str, Dict[str, Any]]:
    """Remove 'cluster' parameter from URL query string and kwargs.

    AsyncRedisCluster doesn't accept 'cluster' parameter, but it might be
    present in the URL or kwargs for compatibility with other Redis clients.

    Args:
        url: Redis URL that might contain cluster parameter
        **kwargs: Keyword arguments that might contain cluster parameter

    Returns:
        Tuple of (cleaned_url, cleaned_kwargs)
    """
    # Parse the URL
    parsed = urlparse(url)

    # Parse query parameters
    query_params = parse_qs(parsed.query)

    # Remove 'cluster' parameter if present
    query_params.pop("cluster", None)

    # Reconstruct the query string
    new_query = urlencode(query_params, doseq=True)

    # Reconstruct the URL
    cleaned_url = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment,
        )
    )

    # Remove 'cluster' from kwargs if present
    cleaned_kwargs = kwargs.copy()
    cleaned_kwargs.pop("cluster", None)

    return cleaned_url, cleaned_kwargs


def compare_versions(version1: str, version2: str):
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
    """Get Redis URL from environment variable."""
    redis_url = os.getenv(REDIS_URL_ENV_VAR)
    if not redis_url:
        raise ValueError(f"{REDIS_URL_ENV_VAR} environment variable not set.")
    return redis_url


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
    prefixes = index_info["index_definition"][3]
    # Normalize single-element prefix lists to string for backward compatibility
    if isinstance(prefixes, list) and len(prefixes) == 1:
        prefixes = prefixes[0]
    storage_type = index_info["index_definition"][1].lower()

    index_fields = index_info["attributes"]

    def parse_vector_attrs(attrs):
        # Parse vector attributes from Redis FT.INFO output
        # Format varies significantly between Redis versions:
        # - Redis 6.2.6-v9: [... "VECTOR"] - no params returned by FT.INFO
        # - Redis 6.2.x: [... "VECTOR", "FLAT", "6", "TYPE", "FLOAT32", "DIM", "3", ...]
        #   Position 6: algorithm value (e.g., "FLAT" or "HNSW")
        #   Position 7: param count
        #   Position 8+: key-value pairs
        # - Redis 7.x+: [... "VECTOR", "ALGORITHM", "FLAT", "TYPE", "FLOAT32", "DIM", "3", ...]
        #   Position 6+: all key-value pairs

        # Check if we have any attributes beyond the type declaration
        if len(attrs) <= 6:
            # Redis 6.2.6-v9 or similar: no vector params in FT.INFO
            # Return None to signal we can't parse this field properly
            return None

        vector_attrs = {}
        start_pos = 6

        # Detect format: if position 6 looks like an algorithm value (not a key),
        # we're dealing with the older format
        if len(attrs) > 6:
            pos6_str = str(attrs[6]).upper()
            # Check if position 6 is an algorithm value (FLAT, HNSW) vs a key (ALGORITHM, TYPE, DIM)
            if pos6_str in ("FLAT", "HNSW"):
                # Old format (Redis 6.2.x): position 6 is algorithm value, position 7 is param count
                # Store the algorithm
                vector_attrs["algorithm"] = pos6_str
                # Skip to position 8 where key-value pairs start
                start_pos = 8

        try:
            for i in range(start_pos, len(attrs), 2):
                if i + 1 < len(attrs):
                    key = str(attrs[i]).lower()
                    vector_attrs[key] = attrs[i + 1]
        except (IndexError, TypeError, ValueError):
            # Silently continue - we'll validate required fields below
            pass

        # Normalize to expected field names
        normalized = {}

        # Handle dims/dim field - REQUIRED for vector fields
        if "dim" in vector_attrs:
            normalized["dims"] = int(vector_attrs.pop("dim"))
        elif "dims" in vector_attrs:
            normalized["dims"] = int(vector_attrs["dims"])
        else:
            # If dims is missing from normal parsing, try scanning the raw attrs
            # This handles edge cases where the format is unexpected
            for i in range(6, len(attrs) - 1):
                if str(attrs[i]).upper() in ("DIM", "DIMS"):
                    try:
                        normalized["dims"] = int(attrs[i + 1])
                        break
                    except (ValueError, IndexError):
                        pass

        # Handle distance_metric field
        if "distance_metric" in vector_attrs:
            normalized["distance_metric"] = vector_attrs["distance_metric"].lower()
        else:
            # Default to cosine if missing
            normalized["distance_metric"] = "cosine"

        # Handle algorithm field
        if "algorithm" in vector_attrs:
            normalized["algorithm"] = vector_attrs["algorithm"].lower()
        else:
            # Default to flat if missing
            normalized["algorithm"] = "flat"

        # Handle datatype field
        if "data_type" in vector_attrs:
            normalized["datatype"] = vector_attrs["data_type"].lower()
        elif "datatype" in vector_attrs:
            normalized["datatype"] = vector_attrs["datatype"].lower()
        elif "type" in vector_attrs:
            # Sometimes it's just "type" instead of "data_type"
            normalized["datatype"] = vector_attrs["type"].lower()
        else:
            # Default to float32 if missing
            normalized["datatype"] = "float32"

        # Validate that we have required dims
        if "dims" not in normalized:
            # Could not parse dims - this field is not properly supported
            return None

        return normalized

    def parse_attrs(attrs, field_type=None):
        # 'SORTABLE', 'NOSTEM' don't have corresponding values.
        # Their presence indicates boolean True
        # TODO 'WITHSUFFIXTRIE' is another boolean attr, but is not returned by ft.info
        original = attrs.copy()
        parsed_attrs = {}

        # Handle all boolean attributes first, regardless of position
        boolean_attrs = {
            "NOSTEM": "no_stem",
            "CASESENSITIVE": "case_sensitive",
            "SORTABLE": "sortable",
            "INDEXMISSING": "index_missing",
            "INDEXEMPTY": "index_empty",
            "NOINDEX": "no_index",
        }

        # Special handling for UNF:
        # - For NUMERIC fields, Redis always adds UNF when SORTABLE is present
        # - For TEXT fields, UNF is only present when explicitly set
        # We only set unf=True for TEXT fields to avoid false positives
        if "UNF" in attrs:
            if field_type == "TEXT":
                parsed_attrs["unf"] = True
            attrs.remove("UNF")

        for redis_attr, python_attr in boolean_attrs.items():
            if redis_attr in attrs:
                parsed_attrs[python_attr] = True
                attrs.remove(redis_attr)

        try:
            # Parse remaining attributes as key-value pairs starting from index 6
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
            attrs = parse_vector_attrs(field_attrs)
            if attrs is None:
                # Vector field attributes cannot be parsed on this Redis version
                # Skip this field - it cannot be properly reconstructed
                continue
            field["attrs"] = attrs
        else:
            field["attrs"] = parse_attrs(field_attrs, field_type=field_attrs[5])
        # append field
        schema_fields.append(field)

    return {
        "index": {"name": index_name, "prefix": prefixes, "storage_type": storage_type},
        "fields": schema_fields,
    }


T = TypeVar("T", Redis, AsyncRedis)


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
    ) -> RedisClient:
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
        **kwargs,
    ) -> SyncRedisClient:
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
        url = redis_url or get_address_from_env()
        if url.startswith("redis+sentinel"):
            client = RedisConnectionFactory._redis_sentinel_client(url, Redis, **kwargs)
        elif is_cluster_url(url, **kwargs):
            client = RedisCluster.from_url(url, **kwargs)
        else:
            client = Redis.from_url(url, **kwargs)
        # Module validation removed - operations will fail naturally if modules are missing
        # Set client library name only
        _lib_name = make_lib_name(kwargs.get("lib_name"))
        try:
            client.client_setinfo("LIB-NAME", _lib_name)
        except ResponseError:
            # Fall back to a simple log echo
            if hasattr(client, "echo"):
                client.echo(_lib_name)
        return client

    @staticmethod
    async def _get_aredis_connection(
        url: Optional[str] = None,
        **kwargs,
    ) -> AsyncRedisClient:
        """Creates and returns an asynchronous Redis client.

        NOTE: This method is the future form of `get_async_redis_connection` but is
        only used internally by the library now.

        Args:
            url (Optional[str]): The URL of the Redis server. If not provided,
                the environment variable REDIS_URL is used.
            **kwargs: Additional keyword arguments to be passed to the async
                Redis client constructor.

        Returns:
            AsyncRedisClient: An asynchronous Redis client instance (either AsyncRedis or AsyncRedisCluster).

        Raises:
            ValueError: If url is not provided and REDIS_URL environment
                variable is not set.
        """
        url = url or get_address_from_env()

        if url.startswith("redis+sentinel"):
            client = RedisConnectionFactory._redis_sentinel_client(
                url, AsyncRedis, **kwargs
            )
        elif is_cluster_url(url, **kwargs):
            # Strip 'cluster' parameter as AsyncRedisCluster doesn't accept it
            cleaned_url, cleaned_kwargs = _strip_cluster_from_url_and_kwargs(
                url, **kwargs
            )
            client = AsyncRedisCluster.from_url(cleaned_url, **cleaned_kwargs)
        else:
            # Also strip cluster parameter for AsyncRedis to avoid connection issues
            cleaned_url, cleaned_kwargs = _strip_cluster_from_url_and_kwargs(
                url, **kwargs
            )
            client = AsyncRedis.from_url(cleaned_url, **cleaned_kwargs)

        # Module validation removed - operations will fail naturally if modules are missing
        # Set client library name only
        _lib_name = make_lib_name(kwargs.get("lib_name"))
        try:
            await client.client_setinfo("LIB-NAME", _lib_name)
        except ResponseError:
            # Fall back to a simple log echo
            if hasattr(client, "echo"):
                await client.echo(_lib_name)
        return client

    @staticmethod
    def get_async_redis_connection(
        url: Optional[str] = None,
        **kwargs,
    ) -> AsyncRedisClient:
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

        if url.startswith("redis+sentinel"):
            return RedisConnectionFactory._redis_sentinel_client(
                url, AsyncRedis, **kwargs
            )
        elif is_cluster_url(url, **kwargs):
            # Strip 'cluster' parameter as AsyncRedisCluster doesn't accept it
            cleaned_url, cleaned_kwargs = _strip_cluster_from_url_and_kwargs(
                url, **kwargs
            )
            return AsyncRedisCluster.from_url(cleaned_url, **cleaned_kwargs)
        else:
            # Also strip cluster parameter for AsyncRedis to avoid connection issues
            cleaned_url, cleaned_kwargs = _strip_cluster_from_url_and_kwargs(
                url, **kwargs
            )
            return AsyncRedis.from_url(cleaned_url, **cleaned_kwargs)

    @staticmethod
    def get_redis_cluster_connection(
        redis_url: Optional[str] = None,
        **kwargs,
    ) -> RedisCluster:
        """Creates and returns a synchronous Redis client for a Redis cluster."""
        url = redis_url or get_address_from_env()
        return RedisCluster.from_url(url, **kwargs)

    @staticmethod
    def get_async_redis_cluster_connection(
        redis_url: Optional[str] = None,
        **kwargs,
    ) -> AsyncRedisCluster:
        """Creates and returns an asynchronous Redis client for a Redis cluster."""
        url = redis_url or get_address_from_env()
        # Strip 'cluster' parameter as AsyncRedisCluster doesn't accept it
        cleaned_url, cleaned_kwargs = _strip_cluster_from_url_and_kwargs(url, **kwargs)
        return AsyncRedisCluster.from_url(cleaned_url, **cleaned_kwargs)

    @staticmethod
    def sync_to_async_redis(
        redis_client: SyncRedisClient,
    ) -> AsyncRedisClient:
        """Convert a synchronous Redis client to an asynchronous one."""
        if isinstance(redis_client, RedisCluster):
            raise ValueError(
                "RedisCluster is not supported for sync-to-async conversion."
            )

        # At this point, redis_client is guaranteed to be Redis type
        assert isinstance(redis_client, Redis)  # Type narrowing for MyPy

        # pick the right connection class
        connection_class: Type[AsyncAbstractConnection] = (
            AsyncSSLConnection
            if redis_client.connection_pool.connection_class == SSLConnection
            else AsyncConnection
        )
        # make async client
        return AsyncRedis.from_pool(
            AsyncConnectionPool(
                connection_class=connection_class,
                **redis_client.connection_pool.connection_kwargs,
            )
        )

    @staticmethod
    def get_modules(client: SyncRedisClient) -> Dict[str, Any]:
        return unpack_redis_modules(convert_bytes(client.module_list()))

    @staticmethod
    async def get_modules_async(client: AsyncRedisClient) -> Dict[str, Any]:
        return unpack_redis_modules(convert_bytes(await client.module_list()))

    @staticmethod
    def validate_sync_redis(
        redis_client: SyncRedisClient,
        lib_name: Optional[str] = None,
    ) -> None:
        """Validates the sync Redis client.

        Note: Module validation has been removed. This method now only validates
        the client type and sets the library name.
        """
        if not issubclass(type(redis_client), (Redis, RedisCluster)):
            raise TypeError(
                "Invalid Redis client instance. Must be Redis or RedisCluster."
            )

        # Set client library name
        _lib_name = make_lib_name(lib_name)
        try:
            redis_client.client_setinfo("LIB-NAME", _lib_name)
        except ResponseError:
            # Fall back to a simple log echo
            # For RedisCluster, echo is not available
            if hasattr(redis_client, "echo"):
                redis_client.echo(_lib_name)

        # Module validation removed - operations will fail naturally if modules are missing

    @staticmethod
    async def validate_async_redis(
        redis_client: AsyncRedisClient,
        lib_name: Optional[str] = None,
    ) -> None:
        """Validates the async Redis client.

        Note: Module validation has been removed. This method now only validates
        the client type and sets the library name.
        """
        if not issubclass(type(redis_client), (AsyncRedis, AsyncRedisCluster)):
            raise TypeError(
                "Invalid async Redis client instance. Must be async Redis or async RedisCluster."
            )
        # Set client library name
        _lib_name = make_lib_name(lib_name)
        try:
            await redis_client.client_setinfo("LIB-NAME", _lib_name)
        except ResponseError:
            # Fall back to a simple log echo
            if hasattr(redis_client, "echo"):
                await redis_client.echo(_lib_name)

        # Module validation removed - operations will fail naturally if modules are missing

    @staticmethod
    @overload
    def _redis_sentinel_client(
        redis_url: str, redis_class: type[Redis], **kwargs: Any
    ) -> Redis: ...

    @staticmethod
    @overload
    def _redis_sentinel_client(
        redis_url: str, redis_class: type[AsyncRedis], **kwargs: Any
    ) -> AsyncRedis: ...

    @staticmethod
    def _redis_sentinel_client(
        redis_url: str, redis_class: Union[type[Redis], type[AsyncRedis]], **kwargs: Any
    ) -> Union[Redis, AsyncRedis]:
        sentinel_list, service_name, db, username, password = (
            RedisConnectionFactory._parse_sentinel_url(redis_url)
        )

        sentinel_kwargs = {}
        if username:
            sentinel_kwargs["username"] = username
            kwargs["username"] = username
        if password:
            sentinel_kwargs["password"] = password
            kwargs["password"] = password
        if db:
            kwargs["db"] = db

        sentinel = Sentinel(sentinel_list, sentinel_kwargs=sentinel_kwargs, **kwargs)
        return sentinel.master_for(service_name, redis_class=redis_class, **kwargs)

    @staticmethod
    def _parse_sentinel_url(url: str) -> tuple:
        parsed_url = urlparse(url)
        hosts_part = parsed_url.netloc.split("@")[-1]
        sentinel_hosts = hosts_part.split(",")

        sentinel_list = []
        for host in sentinel_hosts:
            host_parts = host.split(":")
            if len(host_parts) == 2:
                sentinel_list.append((host_parts[0], int(host_parts[1])))
            else:
                sentinel_list.append((host_parts[0], 26379))

        service_name = "mymaster"
        db = None
        if parsed_url.path:
            path_parts = parsed_url.path.split("/")
            service_name = path_parts[1] or "mymaster"
            if len(path_parts) > 2:
                db = path_parts[2]

        return sentinel_list, service_name, db, parsed_url.username, parsed_url.password
