import os
from typing import Any, Sequence, TypeVar, overload
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from warnings import warn

from redis import Redis, RedisCluster
from redis.asyncio import ConnectionPool as AsyncConnectionPool
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.asyncio.connection import AbstractConnection as AsyncAbstractConnection
from redis.asyncio.connection import Connection as AsyncConnection
from redis.asyncio.connection import SSLConnection as AsyncSSLConnection
from redis.asyncio.sentinel import Sentinel as AsyncSentinel
from redis.connection import SSLConnection
from redis.exceptions import ResponseError
from redis.sentinel import Sentinel

from redisvl import __version__
from redisvl.redis.constants import (
    REDIS_URL_ENV_VAR,
    SVS_MIN_REDIS_VERSION,
    SVS_MIN_SEARCH_VERSION,
)
from redisvl.redis.utils import convert_bytes, is_cluster_url
from redisvl.types import AsyncRedisClient, RedisClient, SyncRedisClient
from redisvl.utils.log import get_logger
from redisvl.utils.utils import deprecated_argument, deprecated_function

logger = get_logger(__name__)


def _split_from_existing_kwargs(
    kwargs: dict[str, Any], *, nested_connection_keys: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    init_kwargs: dict[str, Any] = {}
    connection_kwargs: dict[str, Any] = {}

    for key in ("validate_on_load", "lib_name"):
        if key in kwargs:
            init_kwargs[key] = kwargs.pop(key)

    for key in list(kwargs):
        if key.startswith("_"):
            init_kwargs[key] = kwargs.pop(key)

    for key in nested_connection_keys:
        nested_kwargs = kwargs.pop(key, None)
        if nested_kwargs is not None:
            connection_kwargs.update(nested_kwargs)

    connection_kwargs.update(kwargs)
    return init_kwargs, connection_kwargs


def _strip_cluster_from_url_and_kwargs(
    url: str, **kwargs
) -> tuple[str, dict[str, Any]]:
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


def is_version_gte(version1: str, version2: str) -> bool:
    """
    Check if version1 >= version2.

    Parameters:
        version1 (str): The first version string (e.g., "7.2.4").
        version2 (str): The second version string (e.g., "6.2.1").

    Returns:
        bool: True if version1 >= version2, False otherwise.
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


def unpack_redis_modules(module_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Unpack a list of Redis modules pulled from the MODULES LIST command."""
    return {module["name"]: module["ver"] for module in module_list}


def supports_svs(client: SyncRedisClient) -> bool:
    """Check if Redis server supports SVS-VAMANA.

    Args:
        client: Sync Redis client instance

    Returns:
        True if SVS-VAMANA is supported, False otherwise
    """
    info = client.info("server")  # type: ignore[union-attr]
    redis_version = info.get("redis_version", "0.0.0")  # type: ignore[union-attr]

    modules = RedisConnectionFactory.get_modules(client)
    search_ver = modules.get("search", 0)
    searchlight_ver = modules.get("searchlight", 0)

    # Check if SVS-VAMANA requirements are met
    redis_ok = is_version_gte(redis_version, SVS_MIN_REDIS_VERSION)

    # Check either search or searchlight module (only one is typically installed)
    # Redis Search is the open-source module, SearchLight is the enterprise version
    modules_ok = (
        search_ver >= SVS_MIN_SEARCH_VERSION
        or searchlight_ver >= SVS_MIN_SEARCH_VERSION
    )

    return redis_ok and modules_ok


async def supports_svs_async(client: AsyncRedisClient) -> bool:
    """Async version of _supports_svs.

    Args:
        client: Async Redis client instance

    Returns:
        True if SVS-VAMANA is supported, False otherwise
    """
    info = await client.info("server")  # type: ignore[union-attr]
    redis_version = info.get("redis_version", "0.0.0")  # type: ignore[union-attr]

    modules = await RedisConnectionFactory.get_modules_async(client)
    search_ver = modules.get("search", 0)
    searchlight_ver = modules.get("searchlight", 0)

    # Check if SVS-VAMANA requirements are met
    redis_ok = is_version_gte(redis_version, SVS_MIN_REDIS_VERSION)

    # Check either search or searchlight module (only one is typically installed)
    # Redis Search is the open-source module, SearchLight is the enterprise version
    modules_ok = (
        search_ver >= SVS_MIN_SEARCH_VERSION
        or searchlight_ver >= SVS_MIN_SEARCH_VERSION
    )

    return redis_ok and modules_ok


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


def _identify_client(client: SyncRedisClient, lib_name: str | None = None) -> None:
    """Report RedisVL as the connecting library, tolerating a refusal.

    redis-py sends its own ``CLIENT SETINFO`` during the connection handshake,
    so this call is here to overwrite that with the composed
    ``redis-py(redisvl_v...;<wrapper>)`` name that adoption metrics read. Keep
    it: without it the library and any wrapper above it go unattributed.

    A refusal is ignored, because ``CLIENT SETINFO`` only populates the
    ``lib-name`` field that ``CLIENT LIST`` and ``CLIENT INFO`` display -- its
    own documentation tells client libraries to ignore failures. Two credentials
    hit this: one granting neither ``@connection`` nor the command itself, and
    one on a server predating Redis 7.2, where the command does not exist.

    Only ``ResponseError`` is caught. In the connection-factory path this is the
    first command issued on a freshly created connection, which makes it the
    de-facto connectivity check, so swallowing ``ConnectionError`` would defer a
    genuine failure to some later and more confusing command. Note that on a
    cluster client redis-py routes the command to the default node only, so the
    label reaches one node rather than the whole cluster.
    """
    try:
        client.client_setinfo("LIB-NAME", make_lib_name(lib_name))
    except ResponseError as e:
        logger.debug(f"CLIENT SETINFO was not applied, continuing without it: {e}")


async def _aidentify_client(
    client: AsyncRedisClient, lib_name: str | None = None
) -> None:
    """Async version of :func:`_identify_client`."""
    try:
        await client.client_setinfo("LIB-NAME", make_lib_name(lib_name))
    except ResponseError as e:
        logger.debug(f"CLIENT SETINFO was not applied, continuing without it: {e}")


def convert_index_info_to_schema(index_info: dict[str, Any]) -> dict[str, Any]:
    """Convert the output of FT.INFO into a schema-ready dictionary.

    Args:
        index_info (Dict[str, Any]): Output of the Redis FT.INFO command.

    Returns:
        Dict[str, Any]: Schema dictionary suitable for ``IndexSchema.from_dict()``.
    """
    index_name = index_info["index_name"]
    prefixes = index_info["index_definition"][3]
    # Normalize single-element prefix lists to string for backward compatibility
    if isinstance(prefixes, list) and len(prefixes) == 1:
        prefixes = prefixes[0]
    storage_type = index_info["index_definition"][1].lower()

    # Parse stopwords if present in FT.INFO output
    # stopwords_list is only present when explicitly set (STOPWORDS 0 or custom list)
    # If not present, we use None to indicate default Redis behavior
    stopwords = None
    if "stopwords_list" in index_info:
        # Convert bytes to strings if needed
        stopwords_list = index_info["stopwords_list"]
        stopwords = [
            sw.decode("utf-8") if isinstance(sw, bytes) else sw for sw in stopwords_list
        ]

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

        # Handle HNSW-specific parameters
        if "m" in vector_attrs:
            try:
                normalized["m"] = int(vector_attrs["m"])
            except (ValueError, TypeError):
                pass

        if "ef_construction" in vector_attrs:
            try:
                normalized["ef_construction"] = int(vector_attrs["ef_construction"])
            except (ValueError, TypeError):
                pass

        # Handle SVS-VAMANA specific parameters
        # Compression - Redis uses different internal names, so we need to map them
        if "compression" in vector_attrs:
            compression_value = vector_attrs["compression"]
            # Map Redis internal names to our enum values
            compression_mapping = {
                "GlobalSQ8": "LVQ4x4",  # Default mapping
                "GlobalSQ4": "LVQ4",
                # Add more mappings as we discover them
            }
            # Try to map, otherwise use the value as-is
            normalized["compression"] = compression_mapping.get(
                compression_value, compression_value
            )

        # Dimensionality reduction (reduce parameter)
        if "reduce" in vector_attrs:
            try:
                normalized["reduce"] = int(vector_attrs["reduce"])
            except (ValueError, TypeError):
                pass

        # Graph parameters
        if "graph_max_degree" in vector_attrs:
            try:
                normalized["graph_max_degree"] = int(vector_attrs["graph_max_degree"])
            except (ValueError, TypeError):
                pass

        if "construction_window_size" in vector_attrs:
            try:
                normalized["construction_window_size"] = int(
                    vector_attrs["construction_window_size"]
                )
            except (ValueError, TypeError):
                pass

        if "search_window_size" in vector_attrs:
            try:
                normalized["search_window_size"] = int(
                    vector_attrs["search_window_size"]
                )
            except (ValueError, TypeError):
                pass

        if "epsilon" in vector_attrs:
            try:
                normalized["epsilon"] = float(vector_attrs["epsilon"])
            except (ValueError, TypeError):
                pass

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

    index_dict = {"name": index_name, "prefix": prefixes, "storage_type": storage_type}
    if stopwords is not None:
        index_dict["stopwords"] = stopwords

    return {
        "index": index_dict,
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
        cls, redis_url: str | None = None, use_async: bool = False, **kwargs
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
        redis_url: str | None = None,
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
        client: SyncRedisClient
        if url.startswith("redis+sentinel"):
            client = RedisConnectionFactory._redis_sentinel_client(url, Redis, **kwargs)
        elif is_cluster_url(url, **kwargs):
            client = RedisCluster.from_url(url, **kwargs)
        else:
            client = Redis.from_url(url, **kwargs)
        _identify_client(client, kwargs.get("lib_name"))
        return client

    @staticmethod
    @deprecated_argument("url", "redis_url")
    async def _get_aredis_connection(
        redis_url: str | None = None,
        **kwargs,
    ) -> AsyncRedisClient:
        """Creates and returns an asynchronous Redis client.

        NOTE: This method is the future form of `get_async_redis_connection` but is
        only used internally by the library now.

        Args:
            redis_url (Optional[str]): The URL of the Redis server. If neither
                `redis_url` nor `url` are provided, the environment variable
                REDIS_URL is used.
            url (Optional[str]): Former parameter for the URL of the Redis
                server. Use `redis_url` instead. (Deprecated)
            **kwargs: Additional keyword arguments to be passed to the async
                Redis client constructor.

        Returns:
            AsyncRedisClient: An asynchronous Redis client instance (either AsyncRedis or AsyncRedisCluster).

        Raises:
            ValueError: If url is not provided and REDIS_URL environment
                variable is not set.
        """
        _deprecated_url = kwargs.pop("url", None)
        url = _deprecated_url or redis_url or get_address_from_env()

        client: AsyncRedisClient
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

        await _aidentify_client(client, kwargs.get("lib_name"))
        return client

    @staticmethod
    @deprecated_argument("url", "redis_url")
    def get_async_redis_connection(
        redis_url: str | None = None,
        **kwargs,
    ) -> AsyncRedisClient:
        """Creates and returns an asynchronous Redis client.

        Args:
            redis_url (Optional[str]): The URL of the Redis server. If neither
                `redis_url` nor `url` are provided, the environment variable
                REDIS_URL is used.
            url (Optional[str]): Former parameter for the URL of the Redis
                server. Use `redis_url` instead. (Deprecated)
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
        _deprecated_url = kwargs.pop("url", None)
        url = _deprecated_url or redis_url or get_address_from_env()

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
        redis_url: str | None = None,
        **kwargs,
    ) -> RedisCluster:
        """Creates and returns a synchronous Redis client for a Redis cluster."""
        url = redis_url or get_address_from_env()
        return RedisCluster.from_url(url, **kwargs)

    @staticmethod
    def get_async_redis_cluster_connection(
        redis_url: str | None = None,
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
        connection_class: type[AsyncAbstractConnection] = (
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
    def get_modules(client: SyncRedisClient) -> dict[str, Any]:
        return unpack_redis_modules(convert_bytes(client.module_list()))

    @staticmethod
    async def get_modules_async(client: AsyncRedisClient) -> dict[str, Any]:
        return unpack_redis_modules(convert_bytes(await client.module_list()))

    @staticmethod
    def validate_sync_redis(
        redis_client: SyncRedisClient,
        lib_name: str | None = None,
    ) -> None:
        """Check the client type and report the library name.

        Identification is best effort: a server that refuses ``CLIENT SETINFO``
        is tolerated, so the only failure raised here is a wrong client type.
        (Module validation was removed; a missing module now surfaces when an
        operation needs it.)

        Args:
            redis_client (SyncRedisClient): The client to check.
            lib_name (Optional[str]): Name of a library wrapping RedisVL, to
                report alongside it. Defaults to None.

        Raises:
            TypeError: If the client is not a Redis or RedisCluster instance.
        """
        if not issubclass(type(redis_client), (Redis, RedisCluster)):
            raise TypeError(
                "Invalid Redis client instance. Must be Redis or RedisCluster."
            )

        _identify_client(redis_client, lib_name)

    @staticmethod
    async def validate_async_redis(
        redis_client: AsyncRedisClient,
        lib_name: str | None = None,
    ) -> None:
        """Async version of :meth:`validate_sync_redis`.

        Args:
            redis_client (AsyncRedisClient): The client to check.
            lib_name (Optional[str]): Name of a library wrapping RedisVL, to
                report alongside it. Defaults to None.

        Raises:
            TypeError: If the client is not an async Redis or RedisCluster
                instance.
        """
        if not issubclass(type(redis_client), (AsyncRedis, AsyncRedisCluster)):
            raise TypeError(
                "Invalid async Redis client instance. Must be async Redis or async RedisCluster."
            )

        await _aidentify_client(redis_client, lib_name)

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
        redis_url: str, redis_class: type[Redis] | type[AsyncRedis], **kwargs: Any
    ) -> Redis | AsyncRedis:
        """Create a Redis client connected via Sentinel for high availability.

        Parses a Sentinel URL and creates a Redis client connected to the
        master instance discovered by Sentinel. Supports both sync and async
        clients by using the appropriate Sentinel class.

        Args:
            redis_url: Sentinel URL in the format:
                ``redis+sentinel://[user:pass@]host1:port1[,host2:port2,...][/service][/db]``
                Service name defaults to "mymaster" if not specified.
            redis_class: The Redis client class to use (Redis or AsyncRedis).
            **kwargs: Additional arguments passed to Sentinel and master_for().

        Returns:
            A Redis client (sync or async) connected to the Sentinel-managed master.

        Example:
            >>> client = RedisConnectionFactory._redis_sentinel_client(
            ...     "redis+sentinel://sentinel1:26379,sentinel2:26379/mymaster",
            ...     Redis
            ... )
        """
        sentinel_list, service_name, db, username, password = (
            RedisConnectionFactory._parse_sentinel_url(redis_url)
        )

        sentinel_kwargs: dict[str, Any] = {}
        if username:
            sentinel_kwargs["username"] = username
            kwargs["username"] = username
        if password:
            sentinel_kwargs["password"] = password
            kwargs["password"] = password
        if db:
            kwargs["db"] = db

        # Use AsyncSentinel for async clients, Sentinel for sync clients
        if redis_class == AsyncRedis:
            async_sentinel = AsyncSentinel(
                sentinel_list, sentinel_kwargs=sentinel_kwargs, **kwargs
            )
            return async_sentinel.master_for(
                service_name, redis_class=redis_class, **kwargs  # type: ignore[arg-type]
            )
        else:
            sync_sentinel = Sentinel(
                sentinel_list, sentinel_kwargs=sentinel_kwargs, **kwargs
            )
            return sync_sentinel.master_for(
                service_name, redis_class=redis_class, **kwargs
            )

    @staticmethod
    def _parse_sentinel_url(
        url: str,
    ) -> tuple[list[tuple[str, int]], str, str | None, str | None, str | None]:
        """Parse a Redis Sentinel URL into its components.

        Args:
            url: Sentinel URL in the format:
                ``redis+sentinel://[user:pass@]host1:port1[,host2:port2,...]/service[/db]``

        Returns:
            A tuple containing:
                - sentinel_list: List of (host, port) tuples for Sentinel nodes
                - service_name: The Sentinel service name (defaults to "mymaster")
                - db: The database number (or None if not specified)
                - username: The username for authentication (or None)
                - password: The password for authentication (or None)

        Example:
            >>> RedisConnectionFactory._parse_sentinel_url(
            ...     "redis+sentinel://user:pass@host1:26379,host2:26380/mymaster/0"
            ... )
            ([('host1', 26379), ('host2', 26380)], 'mymaster', '0', 'user', 'pass')
        """
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
