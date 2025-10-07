import asyncio
import json
import threading
import time
import warnings
import weakref
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import redis.exceptions

# Add missing imports
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.cluster import RedisCluster

from redisvl.query.query import VectorQuery
from redisvl.redis.utils import (
    _keys_share_hash_tag,
    async_cluster_create_index,
    async_cluster_search,
    cluster_create_index,
    cluster_search,
    convert_bytes,
    make_dict,
)
from redisvl.types import AsyncRedisClient, SyncRedisClient
from redisvl.utils.utils import deprecated_argument, deprecated_function, sync_wrapper

if TYPE_CHECKING:
    from redis.commands.search.aggregation import AggregateResult
    from redis.commands.search.document import Document
    from redis.commands.search.result import Result
    from redisvl.query.query import BaseQuery

from redis import __version__ as redis_version
from redis.client import NEVER_DECODE

from redisvl.utils.redis_protocol import get_protocol_version

# Redis 5.x compatibility (6 fixed the import path)
if redis_version.startswith("5"):
    from redis.commands.search.indexDefinition import (  # type: ignore[import-untyped]
        IndexDefinition,
    )
else:
    from redis.commands.search.index_definition import (  # type: ignore[no-redef]
        IndexDefinition,
    )

# Need Result outside TYPE_CHECKING for cast
from redis.commands.search.result import Result

from redisvl.exceptions import (
    QueryValidationError,
    RedisSearchError,
    RedisVLError,
    SchemaValidationError,
)
from redisvl.index.storage import BaseStorage, HashStorage, JsonStorage
from redisvl.query import (
    AggregationQuery,
    BaseQuery,
    BaseVectorQuery,
    CountQuery,
    FilterQuery,
)
from redisvl.query.filter import FilterExpression
from redisvl.redis.connection import (
    RedisConnectionFactory,
    convert_index_info_to_schema,
)
from redisvl.schema import IndexSchema, StorageType
from redisvl.schema.fields import (
    VECTOR_NORM_MAP,
    VectorDistanceMetric,
    VectorIndexAlgorithm,
)
from redisvl.utils.log import get_logger

logger = get_logger(__name__)


REQUIRED_MODULES_FOR_INTROSPECTION = [
    {"name": "search", "ver": 20810},
    {"name": "searchlight", "ver": 20810},
]

SearchParams = Union[
    Tuple[
        Union[str, BaseQuery],
        Union[Dict[str, Union[str, int, float, bytes]], None],
    ],
    Union[str, BaseQuery],
]


def process_results(
    results: "Result", query: BaseQuery, schema: IndexSchema
) -> List[Dict[str, Any]]:
    """Convert a list of search Result objects into a list of document
    dictionaries.

    This function processes results from Redis, handling different storage
    types and query types. For JSON storage with empty return fields, it
    unpacks the JSON object while retaining the document ID. The 'payload'
    field is also removed from all resulting documents for consistency.

    Args:
        results (Result): The search results from Redis.
        query (BaseQuery): The query object used for the search.
        storage_type (StorageType): The storage type of the search
            index (json or hash).

    Returns:
        List[Dict[str, Any]]: A list of processed document dictionaries.
    """
    # Handle count queries
    if isinstance(query, CountQuery):
        return results.total

    # Determine if unpacking JSON is needed
    unpack_json = (
        (schema.index.storage_type == StorageType.JSON)
        and isinstance(query, FilterQuery)
        and not query._return_fields  # type: ignore
    )

    if (isinstance(query, BaseVectorQuery)) and query._normalize_vector_distance:
        dist_metric = VectorDistanceMetric(
            schema.fields[query._vector_field_name].attrs.distance_metric.upper()  # type: ignore
        )
        if dist_metric == VectorDistanceMetric.IP:
            warnings.warn(
                "Attempting to normalize inner product distance metric. Use cosine distance instead which is normalized inner product by definition."
            )

        norm_fn = VECTOR_NORM_MAP[dist_metric.value]
    else:
        norm_fn = None

    # Process records
    def _process(doc: "Document") -> Dict[str, Any]:
        doc_dict = doc.__dict__

        # Unpack and Project JSON fields properly
        if unpack_json and "json" in doc_dict:
            json_data = doc_dict.get("json", {})
            if isinstance(json_data, str):
                json_data = json.loads(json_data)
            if isinstance(json_data, dict):
                return {"id": doc_dict.get("id"), **json_data}
            raise ValueError(f"Unable to parse json data from Redis {json_data}")

        if norm_fn:
            # convert float back to string to be consistent
            doc_dict[query.DISTANCE_ID] = str(  # type: ignore
                norm_fn(float(doc_dict[query.DISTANCE_ID]))  # type: ignore
            )

        # Remove 'payload' if present
        doc_dict.pop("payload", None)

        return doc_dict

    return [_process(doc) for doc in results.docs]


def process_aggregate_results(
    results: "AggregateResult", query: AggregationQuery, storage_type: StorageType
) -> List[Dict[str, Any]]:
    """Convert an aggregate result object into a list of document dictionaries.

    This function processes results from Redis, handling different storage
    types and query types. For JSON storage with empty return fields, it
    unpacks the JSON object while retaining the document ID. The 'payload'
    field is also removed from all resulting documents for consistency.

    Args:
        results (AggregateResult): The aggregate results from Redis.
        query (AggregationQuery): The aggregation query object used for the aggregation.
        storage_type (StorageType): The storage type of the search
            index (json or hash).

    Returns:
        List[Dict[str, Any]]: A list of processed document dictionaries.
    """

    def _process(row):
        result = make_dict(convert_bytes(row))
        result.pop("__score", None)
        return result

    return [_process(r) for r in results.rows]


class BaseSearchIndex:
    """Base search engine class"""

    _STORAGE_MAP = {
        StorageType.HASH: HashStorage,
        StorageType.JSON: JsonStorage,
    }

    schema: IndexSchema

    def __init__(*args, **kwargs):
        pass

    @property
    def _storage(self) -> BaseStorage:
        """The storage type for the index schema."""
        return self._STORAGE_MAP[self.schema.index.storage_type](
            index_schema=self.schema
        )

    def _validate_query(self, query: BaseQuery) -> None:
        """Validate a query."""
        if isinstance(query, VectorQuery):
            field = self.schema.fields[query._vector_field_name]
            if query.ef_runtime and field.attrs.algorithm != VectorIndexAlgorithm.HNSW:  # type: ignore
                raise QueryValidationError(
                    "Vector field using 'flat' algorithm does not support EF_RUNTIME query parameter."
                )

    @property
    def name(self) -> str:
        """The name of the Redis search index."""
        return self.schema.index.name

    @property
    def prefix(self) -> str:
        """The optional key prefix that comes before a unique key value in
        forming a Redis key. If multiple prefixes are configured, returns the
        first one."""
        prefix = self.schema.index.prefix
        return prefix[0] if isinstance(prefix, list) else prefix

    @property
    def key_separator(self) -> str:
        """The optional separator between a defined prefix and key value in
        forming a Redis key."""
        return self.schema.index.key_separator

    @property
    def storage_type(self) -> StorageType:
        """The underlying storage type for the search index; either
        hash or json."""
        return self.schema.index.storage_type

    @classmethod
    def from_yaml(cls, schema_path: str, **kwargs):
        """Create a SearchIndex from a YAML schema file.

        Args:
            schema_path (str): Path to the YAML schema file.

        Returns:
            SearchIndex: A RedisVL SearchIndex object.

        .. code-block:: python

            from redisvl.index import SearchIndex

            index = SearchIndex.from_yaml("schemas/schema.yaml", redis_url="redis://localhost:6379")
        """
        schema = IndexSchema.from_yaml(schema_path)
        return cls(schema=schema, **kwargs)

    @classmethod
    def from_dict(cls, schema_dict: Dict[str, Any], **kwargs):
        """Create a SearchIndex from a dictionary.

        Args:
            schema_dict (Dict[str, Any]): A dictionary containing the schema.

        Returns:
            SearchIndex: A RedisVL SearchIndex object.

        .. code-block:: python

            from redisvl.index import SearchIndex

            index = SearchIndex.from_dict({
                "index": {
                    "name": "my-index",
                    "prefix": "rvl",
                    "storage_type": "hash",
                },
                "fields": [
                    {"name": "doc-id", "type": "tag"}
                ]
            }, redis_url="redis://localhost:6379")

        """
        schema = IndexSchema.from_dict(schema_dict)
        return cls(schema=schema, **kwargs)

    def disconnect(self):
        """Disconnect from the Redis database."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def key(self, id: str) -> str:
        """Construct a redis key as a combination of an index key prefix (optional)
        and specified id.

        The id is typically either a unique identifier, or
        derived from some domain-specific metadata combination (like a document
        id or chunk id).

        Args:
            id (str): The specified unique identifier for a particular
                document indexed in Redis.

        Returns:
            str: The full Redis key including key prefix and value as a string.
        """
        return self._storage._key(
            id=id,
            prefix=self.prefix,
            key_separator=self.schema.index.key_separator,
        )


class SearchIndex(BaseSearchIndex):
    """A search index class for interacting with Redis as a vector database.

    The SearchIndex is instantiated with a reference to a Redis database and an
    IndexSchema (YAML path or dictionary object) that describes the various
    settings and field configurations.

    .. code-block:: python

        from redisvl.index import SearchIndex

        # initialize the index object with schema from file
        index = SearchIndex.from_yaml(
            "schemas/schema.yaml",
            redis_url="redis://localhost:6379",
            validate_on_load=True
        )

        # create the index
        index.create(overwrite=True, drop=False)

        # data is an iterable of dictionaries
        index.load(data)

        # delete index and data
        index.delete(drop=True)

    """

    @deprecated_argument("connection_args", "Use connection_kwargs instead.")
    def __init__(
        self,
        schema: IndexSchema,
        redis_client: Optional[SyncRedisClient] = None,
        redis_url: Optional[str] = None,
        connection_kwargs: Optional[Dict[str, Any]] = None,
        validate_on_load: bool = False,
        **kwargs,
    ):
        """Initialize the RedisVL search index with a schema, Redis client
        (or URL string with other connection args), connection_args, and other
        kwargs.

        Args:
            schema (IndexSchema): Index schema object.
            redis_client(Optional[Redis]): An
                instantiated redis client.
            redis_url (Optional[str]): The URL of the Redis server to
                connect to.
            connection_kwargs (Dict[str, Any], optional): Redis client connection
                args.
            validate_on_load (bool, optional): Whether to validate data against schema
                when loading. Defaults to False.
        """
        if "connection_args" in kwargs:
            connection_kwargs = kwargs.pop("connection_args")

        if not isinstance(schema, IndexSchema):
            raise ValueError("Must provide a valid IndexSchema object")

        self.schema = schema
        self._validate_on_load = validate_on_load
        self._lib_name: Optional[str] = kwargs.pop("lib_name", None)

        # Store connection parameters
        self.__redis_client = redis_client
        self._redis_url = redis_url
        self._connection_kwargs = connection_kwargs or {}
        self._lock = threading.Lock()

        self._validated_client = kwargs.pop("_client_validated", False)
        self._owns_redis_client = redis_client is None
        if self._owns_redis_client:
            weakref.finalize(self, self.disconnect)

    def disconnect(self):
        """Disconnect from the Redis database."""
        if self._owns_redis_client is False:
            logger.info("Index does not own client, not disconnecting")
            return
        if self.__redis_client:
            self.__redis_client.close()
        self.__redis_client = None

    @classmethod
    def from_existing(
        cls,
        name: str,
        redis_client: Optional[SyncRedisClient] = None,
        redis_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize from an existing search index in Redis by index name.

        Args:
            name (str): Name of the search index in Redis.
            redis_client(Optional[Redis]): An
                instantiated redis client.
            redis_url (Optional[str]): The URL of the Redis server to
                connect to.

        Raises:
            ValueError: If redis_url or redis_client is not provided.
        """
        if redis_url:
            redis_client = RedisConnectionFactory.get_redis_connection(
                redis_url=redis_url,
                **kwargs,
            )
        elif redis_client:
            # Validate client type and set lib name
            RedisConnectionFactory.validate_sync_redis(redis_client)
            # Mark that client was already validated to avoid duplicate calls
            kwargs["_client_validated"] = True

        if not redis_client:
            raise ValueError("Must provide either a redis_url or redis_client")

        # Fetch index info and convert to schema
        index_info = cls._info(name, redis_client)
        schema_dict = convert_index_info_to_schema(index_info)
        schema = IndexSchema.from_dict(schema_dict)
        return cls(schema, redis_client, **kwargs)

    @property
    def client(self) -> Optional[SyncRedisClient]:
        """The underlying redis-py client object."""
        return self.__redis_client

    @property
    def _redis_client(self) -> SyncRedisClient:
        """
        Get a Redis client instance.

        Lazily creates a Redis client instance if it doesn't exist.
        """
        if self.__redis_client is None:
            with self._lock:
                if self.__redis_client is None:
                    # Pass lib_name to connection factory
                    kwargs = {**self._connection_kwargs}
                    if self._lib_name:
                        kwargs["lib_name"] = self._lib_name
                    self.__redis_client = RedisConnectionFactory.get_redis_connection(
                        redis_url=self._redis_url,
                        **kwargs,
                    )
        if not self._validated_client and self._lib_name:
            # Only set lib name for user-provided clients
            RedisConnectionFactory.validate_sync_redis(
                self.__redis_client,
                self._lib_name,
            )
            self._validated_client = True
        return self.__redis_client

    @deprecated_function("connect", "Pass connection parameters in __init__.")
    def connect(self, redis_url: Optional[str] = None, **kwargs):
        """Connect to a Redis instance using the provided `redis_url`, falling
        back to the `REDIS_URL` environment variable (if available).

        Note: Additional keyword arguments (`**kwargs`) can be used to provide
        extra options specific to the Redis connection.

        Args:
            redis_url (Optional[str], optional): The URL of the Redis server to
                connect to.

        Raises:
            redis.exceptions.ConnectionError: If the connection to the Redis
                server fails.
            ValueError: If the Redis URL is not provided nor accessible
                through the `REDIS_URL` environment variable.
            ModuleNotFoundError: If required Redis modules are not installed.
        """
        self.__redis_client = RedisConnectionFactory.get_redis_connection(
            redis_url=redis_url, **kwargs
        )

    @deprecated_function("set_client", "Pass connection parameters in __init__.")
    def set_client(self, redis_client: SyncRedisClient, **kwargs):
        """Manually set the Redis client to use with the search index.

        This method configures the search index to use a specific Redis or
        Async Redis client. It is useful for cases where an external,
        custom-configured client is preferred instead of creating a new one.

        Args:
            redis_client (Redis): A Redis or Async Redis
                client instance to be used for the connection.

        Raises:
            TypeError: If the provided client is not valid.
        """
        RedisConnectionFactory.validate_sync_redis(redis_client)
        self.__redis_client = redis_client
        return self

    def create(self, overwrite: bool = False, drop: bool = False) -> None:
        """Create an index in Redis with the current schema and properties.

        Args:
            overwrite (bool, optional): Whether to overwrite the index if it
                already exists. Defaults to False.
            drop (bool, optional): Whether to drop all keys associated with the
                index in the case of overwriting. Defaults to False.

        Raises:
            RuntimeError: If the index already exists and 'overwrite' is False.
            ValueError: If no fields are defined for the index.

        .. code-block:: python

            # create an index in Redis; only if one does not exist with given name
            index.create()

            # overwrite an index in Redis without dropping associated data
            index.create(overwrite=True)

            # overwrite an index in Redis; drop associated data (clean slate)
            index.create(overwrite=True, drop=True)
        """
        # Check that fields are defined.
        redis_fields = self.schema.redis_fields
        if not redis_fields:
            raise ValueError("No fields defined for index")
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be of type bool")

        if self.exists():
            if not overwrite:
                logger.info("Index already exists, not overwriting.")
                return None
            logger.info("Index already exists, overwriting.")
            self.delete(drop=drop)

        try:
            definition = IndexDefinition(
                prefix=[self.schema.index.prefix], index_type=self._storage.type
            )
            if isinstance(self._redis_client, RedisCluster):
                cluster_create_index(
                    index_name=self.name,
                    client=self._redis_client,
                    fields=redis_fields,
                    definition=definition,
                )
            else:
                self._redis_client.ft(self.name).create_index(
                    fields=redis_fields,
                    definition=definition,
                )
        except redis.exceptions.RedisError as e:
            raise RedisSearchError(
                f"Failed to create index '{self.name}' on Redis: {str(e)}"
            ) from e
        except Exception as e:
            logger.exception("Error while trying to create the index")
            raise RedisSearchError(
                f"Unexpected error creating index '{self.name}': {str(e)}"
            ) from e

    def delete(self, drop: bool = True):
        """Delete the search index while optionally dropping all keys associated
        with the index.

        Args:
            drop (bool, optional): Delete the key / documents pairs in the
                index. Defaults to True.

        raises:
            redis.exceptions.ResponseError: If the index does not exist.
        """
        try:
            # For Redis Cluster with drop=True, we need to handle key deletion manually
            # to avoid cross-slot errors since we control the keys opaquely
            if drop and isinstance(self._redis_client, RedisCluster):
                # First clear all keys manually (handles cluster compatibility)
                self.clear()
                # Then drop the index without the DD flag
                cmd_args = ["FT.DROPINDEX", self.schema.index.name]
            else:
                # Standard approach for non-cluster or when not dropping keys
                cmd_args = ["FT.DROPINDEX", self.schema.index.name]
                if drop:
                    cmd_args.append("DD")

            if isinstance(self._redis_client, RedisCluster):
                target_nodes = [self._redis_client.get_default_node()]
                self._redis_client.execute_command(*cmd_args, target_nodes=target_nodes)
            else:
                self._redis_client.execute_command(*cmd_args)
        except Exception as e:
            raise RedisSearchError(f"Error while deleting index: {str(e)}") from e

    def clear(self) -> int:
        """Clear all keys in Redis associated with the index, leaving the index
        available and in-place for future insertions or updates.

        NOTE: This method requires custom behavior for Redis Cluster because
        here, we can't easily give control of the keys we're clearing to the
        user so they can separate them based on hash tag.

        Returns:
            int: Count of records deleted from Redis.
        """
        client = cast(SyncRedisClient, self._redis_client)
        total_records_deleted: int = 0

        for batch in self.paginate(
            FilterQuery(FilterExpression("*"), return_fields=["id"]), page_size=500
        ):
            batch_keys = [record["id"] for record in batch]
            if batch_keys:
                is_cluster = isinstance(client, RedisCluster)
                if is_cluster:
                    records_deleted_in_batch = 0
                    for key_to_delete in batch_keys:
                        try:
                            records_deleted_in_batch += cast(
                                int, client.delete(key_to_delete)
                            )
                        except redis.exceptions.RedisError as e:
                            logger.warning(f"Failed to delete key {key_to_delete}: {e}")
                    total_records_deleted += records_deleted_in_batch
                else:
                    record_deleted = cast(int, client.delete(*batch_keys))
                    total_records_deleted += record_deleted

        return total_records_deleted

    def drop_keys(self, keys: Union[str, List[str]]) -> int:
        """Remove a specific entry or entries from the index by it's key ID.

        Args:
            keys (Union[str, List[str]]): The document ID or IDs to remove from the index.

        Returns:
            int: Count of records deleted from Redis.
        """
        if isinstance(keys, List):
            return self._redis_client.delete(*keys)  # type: ignore
        else:
            return self._redis_client.delete(keys)  # type: ignore

    def drop_documents(self, ids: Union[str, List[str]]) -> int:
        """Remove documents from the index by their document IDs.

        This method converts document IDs to Redis keys automatically by applying
        the index's key prefix and separator configuration.

        NOTE: Cluster users will need to incorporate hash tags into their
        document IDs and only call this method with documents from a single hash
        tag at a time.

        Args:
            ids (Union[str, List[str]]): The document ID or IDs to remove from the index.

        Returns:
            int: Count of documents deleted from Redis.
        """
        if isinstance(ids, list):
            if not ids:
                return 0
            keys = [self.key(id) for id in ids]
            # Check for cluster compatibility
            if isinstance(
                self._redis_client, RedisCluster
            ) and not _keys_share_hash_tag(keys):
                raise ValueError(
                    "All keys must share a hash tag when using Redis Cluster."
                )
            return self._redis_client.delete(*keys)  # type: ignore
        else:
            key = self.key(ids)
            return self._redis_client.delete(key)  # type: ignore

    def expire_keys(
        self, keys: Union[str, List[str]], ttl: int
    ) -> Union[int, List[int]]:
        """Set the expiration time for a specific entry or entries in Redis.

        Args:
            keys (Union[str, List[str]]): The entry ID or IDs to set the expiration for.
            ttl (int): The time-to-live in seconds.
        """
        if isinstance(keys, list):
            pipe = self._redis_client.pipeline()  # type: ignore
            for key in keys:
                pipe.expire(key, ttl)
            return pipe.execute()
        else:
            return self._redis_client.expire(keys, ttl)  # type: ignore

    def load(
        self,
        data: Iterable[Any],
        id_field: Optional[str] = None,
        keys: Optional[Iterable[str]] = None,
        ttl: Optional[int] = None,
        preprocess: Optional[Callable] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """Load objects to the Redis database. Returns the list of keys loaded
        to Redis.

        RedisVL automatically handles constructing the object keys, batching,
        optional preprocessing steps, and setting optional expiration
        (TTL policies) on keys.

        Args:
            data (Iterable[Any]): An iterable of objects to store.
            id_field (Optional[str], optional): Specified field used as the id
                portion of the redis key (after the prefix) for each
                object. Defaults to None.
            keys (Optional[Iterable[str]], optional): Optional iterable of keys.
                Must match the length of objects if provided. Defaults to None.
            ttl (Optional[int], optional): Time-to-live in seconds for each key.
                Defaults to None.
            preprocess (Optional[Callable], optional): A function to preprocess
                objects before storage. Defaults to None.
            batch_size (Optional[int], optional): Number of objects to write in
                a single Redis pipeline execution. Defaults to class's
                default batch size.

        Returns:
            List[str]: List of keys loaded to Redis.

        Raises:
            SchemaValidationError: If validation fails when validate_on_load is enabled.
            RedisVLError: If there's an error loading data to Redis.
        """
        try:
            return self._storage.write(
                self._redis_client,
                objects=data,
                id_field=id_field,
                keys=keys,
                ttl=ttl,
                preprocess=preprocess,
                batch_size=batch_size,
                validate=self._validate_on_load,
            )
        except SchemaValidationError as e:
            # Log the detailed validation error with actionable information
            logger.error("Data validation failed during load operation")
            raise
        except Exception as e:
            # Wrap other errors as general RedisVL errors
            logger.exception("Error while loading data to Redis")
            raise RedisVLError(f"Failed to load data: {str(e)}") from e

    def fetch(self, id: str) -> Optional[Dict[str, Any]]:
        """Fetch an object from Redis by id.

        The id is typically either a unique identifier,
        or derived from some domain-specific metadata combination
        (like a document id or chunk id).

        Args:
            id (str): The specified unique identifier for a particular
                document indexed in Redis.

        Returns:
            Dict[str, Any]: The fetched object.
        """
        obj = self._storage.get(self._redis_client, [self.key(id)])
        if obj:
            return convert_bytes(obj[0])
        return None

    def _aggregate(self, aggregation_query: AggregationQuery) -> List[Dict[str, Any]]:
        """Execute an aggregation query and processes the results."""
        results = self.aggregate(
            aggregation_query,
            query_params=aggregation_query.params,  # type: ignore[attr-defined]
        )
        return process_aggregate_results(
            results,
            query=aggregation_query,
            storage_type=self.schema.index.storage_type,
        )

    def aggregate(self, *args, **kwargs) -> "AggregateResult":
        """Perform an aggregation operation against the index.

        Wrapper around the aggregation API that adds the index name
        to the query and passes along the rest of the arguments
        to the redis-py ft().aggregate() method.

        Returns:
            Result: Raw Redis aggregation results.
        """
        try:
            return self._redis_client.ft(self.schema.index.name).aggregate(
                *args, **kwargs
            )
        except redis.exceptions.RedisError as e:
            if "CROSSSLOT" in str(e):
                raise RedisSearchError(
                    "Cross-slot error during aggregation. Ensure consistent hash tags in your keys."
                )
            raise RedisSearchError(f"Error while aggregating: {str(e)}") from e
        except Exception as e:
            raise RedisSearchError(
                f"Unexpected error while aggregating: {str(e)}"
            ) from e

    def batch_search(
        self, queries: List[SearchParams], batch_size: int = 10
    ) -> List["Result"]:
        """Perform a search against the index for multiple queries.

        This method takes a list of queries and optionally query params and
        returns a list of Result objects for each query. Results are
        returned in the same order as the queries.

        NOTE: Cluster users may need to incorporate hash tags into their query
        to avoid cross-slot operations.

        Args:
            queries (List[SearchParams]): The queries to search for.
            batch_size (int, optional): The number of queries to search for at a time.
                Defaults to 10.

        Returns:
            List[Result]: The search results for each query.
        """
        all_results = []
        search = self._redis_client.ft(self.schema.index.name)
        options = {}
        if get_protocol_version(self._redis_client) not in ["3", 3]:
            options[NEVER_DECODE] = True

        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i : i + batch_size]

            # redis-py doesn't support calling `search` in a pipeline,
            # so we need to manually execute each command in a pipeline
            # and parse the results
            with self._redis_client.pipeline(transaction=False) as pipe:
                batch_built_queries = []
                for query in batch_queries:
                    if isinstance(query, tuple):
                        query_args, q = search._mk_query_args(  # type: ignore
                            query[0], query_params=query[1]
                        )
                    else:
                        query_args, q = search._mk_query_args(  # type: ignore
                            query, query_params=None
                        )
                    batch_built_queries.append(q)
                    pipe.execute_command(
                        "FT.SEARCH",
                        *query_args,
                        **options,
                    )

                st = time.time()
                results = pipe.execute()

                # We don't know how long each query took, so we'll use the total time
                # for all queries in the batch as the duration for each query
                duration = (time.time() - st) * 1000.0

                for j, query_results in enumerate(results):
                    _built_query = batch_built_queries[j]
                    parsed_result = search._parse_search(  # type: ignore
                        query_results,
                        query=_built_query,
                        duration=duration,
                    )
                    # Return a parsed Result object for each query
                    all_results.append(parsed_result)
        return all_results

    def search(self, *args, **kwargs) -> "Result":
        """Perform a search against the index.

        Wrapper around the search API that adds the index name
        to the query and passes along the rest of the arguments
        to the redis-py ft().search() method.

        Returns:
            Result: Raw Redis search results.
        """
        try:
            if isinstance(self._redis_client, RedisCluster):
                # Use special cluster search for RedisCluster
                return cluster_search(
                    self._redis_client.ft(self.schema.index.name),
                    *args,
                    **kwargs,  # type: ignore
                )
            else:
                return self._redis_client.ft(self.schema.index.name).search(
                    *args, **kwargs
                )  # type: ignore
        except redis.exceptions.RedisError as e:
            if "CROSSSLOT" in str(e):
                raise RedisSearchError(
                    "Cross-slot error during search. Ensure consistent hash tags in your keys."
                )
            raise RedisSearchError(f"Error while searching: {str(e)}") from e
        except Exception as e:
            raise RedisSearchError(f"Unexpected error while searching: {str(e)}") from e

    def batch_query(
        self, queries: Sequence[BaseQuery], batch_size: int = 10
    ) -> List[List[Dict[str, Any]]]:
        """Execute a batch of queries and process results."""
        results = self.batch_search(
            [(query.query, query.params) for query in queries], batch_size=batch_size
        )
        all_parsed = []
        for query, batch_results in zip(queries, results):
            parsed = process_results(batch_results, query=query, schema=self.schema)
            # Create separate lists of parsed results for each query
            # passed in to the batch_search method, so that callers can
            # access the results for each query individually
            all_parsed.append(parsed)
        return all_parsed

    def _query(self, query: BaseQuery) -> List[Dict[str, Any]]:
        """Execute a query and process results."""
        try:
            self._validate_query(query)
        except QueryValidationError as e:
            raise QueryValidationError(f"Invalid query: {str(e)}") from e
        results = self.search(query.query, query_params=query.params)
        return process_results(results, query=query, schema=self.schema)

    def query(self, query: Union[BaseQuery, AggregationQuery]) -> List[Dict[str, Any]]:
        """Execute a query on the index.

        This method takes a BaseQuery or AggregationQuery object directly, and
        handles post-processing of the search.

        Args:
            query (Union[BaseQuery, AggregateQuery]): The query to run.

        Returns:
            List[Result]: A list of search results.

        .. code-block:: python

            from redisvl.query import VectorQuery

            query = VectorQuery(
                vector=[0.16, -0.34, 0.98, 0.23],
                vector_field_name="embedding",
                num_results=3
            )

            results = index.query(query)

        """
        if isinstance(query, AggregationQuery):
            return self._aggregate(query)
        else:
            return self._query(query)

    def paginate(self, query: BaseQuery, page_size: int = 30) -> Generator:
        """Execute a given query against the index and return results in
        paginated batches.

        This method accepts a RedisVL query instance, enabling pagination of
        results which allows for subsequent processing over each batch with a
        generator.

        Args:
            query (BaseQuery): The search query to be executed.
            page_size (int, optional): The number of results to return in each
                batch. Defaults to 30.

        Yields:
            A generator yielding batches of search results.

        Raises:
            TypeError: If the page_size argument is not of type int.
            ValueError: If the page_size argument is less than or equal to zero.

        .. code-block:: python

            # Iterate over paginated search results in batches of 10
            for result_batch in index.paginate(query, page_size=10):
                # Process each batch of results
                pass

        Note:
            The page_size parameter controls the number of items each result
            batch contains. Adjust this value based on performance
            considerations and the expected volume of search results.

        """
        if not isinstance(page_size, int):
            raise TypeError("page_size must be an integer")

        if page_size <= 0:
            raise ValueError("page_size must be greater than 0")

        offset = 0
        while True:
            query.paging(offset, page_size)
            results = self._query(query)
            if not results:
                break
            yield results
            # Increment the offset for the next batch of pagination
            offset += page_size

    def listall(self) -> List[str]:
        """List all search indices in Redis database.

        Returns:
            List[str]: The list of indices in the database.
        """
        return convert_bytes(self._redis_client.execute_command("FT._LIST"))

    def exists(self) -> bool:
        """Check if the index exists in Redis.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        return self.schema.index.name in self.listall()

    @staticmethod
    def _info(name: str, redis_client: SyncRedisClient) -> Dict[str, Any]:
        """Run FT.INFO to fetch information about the index."""
        try:
            return convert_bytes(redis_client.ft(name).info())  # type: ignore
        except Exception as e:
            raise RedisSearchError(
                f"Error while fetching {name} index info: {str(e)}"
            ) from e

    def info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about the index.

        Args:
            name (str, optional): Index name to fetch info about.
                Defaults to None.

        Returns:
            dict: A dictionary containing the information about the index.
        """
        index_name = name or self.schema.index.name
        return self._info(index_name, self._redis_client)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class AsyncSearchIndex(BaseSearchIndex):
    """A search index class for interacting with Redis as a vector database in
    async-mode.

    The AsyncSearchIndex is instantiated with a reference to a Redis database
    and an IndexSchema (YAML path or dictionary object) that describes the
    various settings and field configurations.

    .. code-block:: python

        from redisvl.index import AsyncSearchIndex

        # initialize the index object with schema from file
        index = AsyncSearchIndex.from_yaml(
            "schemas/schema.yaml",
            redis_url="redis://localhost:6379",
            validate_on_load=True
        )

        # create the index
        await index.create(overwrite=True, drop=False)

        # data is an iterable of dictionaries
        await index.load(data)

        # delete index and data
        await index.delete(drop=True)

    """

    @deprecated_argument("redis_kwargs", "Use connection_kwargs instead.")
    def __init__(
        self,
        schema: IndexSchema,
        *,
        redis_url: Optional[str] = None,
        redis_client: Optional[AsyncRedisClient] = None,
        connection_kwargs: Optional[Dict[str, Any]] = None,
        validate_on_load: bool = False,
        **kwargs,
    ):
        """Initialize the RedisVL async search index with a schema.

        Args:
            schema (IndexSchema): Index schema object.
            redis_url (Optional[str], optional): The URL of the Redis server to
                connect to.
            redis_client (Optional[AsyncRedis]): An
                instantiated redis client.
            connection_kwargs (Optional[Dict[str, Any]]): Redis client connection
                args.
            validate_on_load (bool, optional): Whether to validate data against schema
                when loading. Defaults to False.
        """
        if "redis_kwargs" in kwargs:
            connection_kwargs = kwargs.pop("redis_kwargs")

        # final validation on schema object
        if not isinstance(schema, IndexSchema):
            raise ValueError("Must provide a valid IndexSchema object")

        self.schema = schema
        self._validate_on_load = validate_on_load
        self._lib_name: Optional[str] = kwargs.pop("lib_name", None)

        # Store connection parameters
        self._redis_client = redis_client
        self._redis_url = redis_url
        self._connection_kwargs = connection_kwargs or {}
        self._lock = asyncio.Lock()

        self._validated_client = kwargs.pop("_client_validated", False)
        self._owns_redis_client = redis_client is None
        if self._owns_redis_client:
            weakref.finalize(self, sync_wrapper(self.disconnect))

    @classmethod
    async def from_existing(
        cls,
        name: str,
        redis_client: Optional[AsyncRedisClient] = None,
        redis_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize from an existing search index in Redis by index name.

        Args:
            name (str): Name of the search index in Redis.
            redis_client(Optional[Redis]): An
                instantiated redis client.
            redis_url (Optional[str]): The URL of the Redis server to
                connect to.
        """
        if not redis_url and not redis_client:
            raise ValueError(
                "Must provide either a redis_url or redis_client to fetch Redis index info."
            )

        if redis_url:
            redis_client = await RedisConnectionFactory._get_aredis_connection(
                url=redis_url,
                **kwargs,
            )
        elif redis_client:
            # Validate client type and set lib name
            await RedisConnectionFactory.validate_async_redis(redis_client)
            # Mark that client was already validated to avoid duplicate calls
            kwargs["_client_validated"] = True

        if redis_client is None:
            raise ValueError(
                "Failed to obtain a valid Redis client. "
                "Please provide a valid redis_client or redis_url."
            )

        # Fetch index info and convert to schema
        index_info = await cls._info(name, redis_client)
        schema_dict = convert_index_info_to_schema(index_info)
        schema = IndexSchema.from_dict(schema_dict)
        return cls(schema, redis_client=redis_client, **kwargs)

    @property
    def client(self) -> Optional[AsyncRedisClient]:
        """The underlying redis-py client object."""
        return self._redis_client

    @deprecated_function("connect", "Pass connection parameters in __init__.")
    async def connect(self, redis_url: Optional[str] = None, **kwargs):
        """[DEPRECATED] Connect to a Redis instance. Use connection parameters in __init__."""
        warnings.warn(
            "connect() is deprecated; pass connection parameters in __init__",
            DeprecationWarning,
        )
        client = await RedisConnectionFactory._get_aredis_connection(
            redis_url=redis_url, **kwargs
        )
        await self.set_client(client)

    @deprecated_function("set_client", "Pass connection parameters in __init__.")
    async def set_client(self, redis_client: Union[AsyncRedisClient, SyncRedisClient]):
        """
        [DEPRECATED] Manually set the Redis client to use with the search index.
        This method is deprecated; please provide connection parameters in __init__.
        """
        redis_client = await self._validate_client(redis_client)
        await self.disconnect()
        async with self._lock:
            self._redis_client = redis_client
        return self

    async def _get_client(self) -> AsyncRedisClient:
        """Lazily instantiate and return the async Redis client."""
        if self._redis_client is None:
            async with self._lock:
                # Double-check to protect against concurrent access
                if self._redis_client is None:
                    # Pass lib_name to connection factory
                    kwargs = {**self._connection_kwargs}
                    if self._redis_url:
                        kwargs["url"] = self._redis_url
                    if self._lib_name:
                        kwargs["lib_name"] = self._lib_name
                    self._redis_client = (
                        await RedisConnectionFactory._get_aredis_connection(**kwargs)
                    )
        if not self._validated_client and self._lib_name:
            # Set lib name for user-provided clients
            await RedisConnectionFactory.validate_async_redis(
                self._redis_client,
                self._lib_name,
            )
            self._validated_client = True
        return self._redis_client

    async def _validate_client(
        self, redis_client: Union[AsyncRedisClient, SyncRedisClient]
    ) -> AsyncRedisClient:
        # Handle deprecated sync client conversion
        if isinstance(redis_client, (Redis, RedisCluster)):
            warnings.warn(
                "Passing a sync Redis client to AsyncSearchIndex is deprecated "
                "and will be removed in the next major version. Please use an "
                "async Redis client instead.",
                DeprecationWarning,
            )
            # Use a new variable name
            async_redis_client: AsyncRedisClient = (
                RedisConnectionFactory.sync_to_async_redis(redis_client)
            )
            return async_redis_client  # Return the converted client
        # Check if it's a valid async client (standard or cluster)
        elif not isinstance(redis_client, (AsyncRedis, AsyncRedisCluster)):
            raise ValueError(
                "Invalid async client type: must be AsyncRedis or AsyncRedisCluster"
            )
        # If it passed the elif, it's already an AsyncRedisClient
        return redis_client

    @staticmethod
    async def _info(name: str, redis_client: AsyncRedisClient) -> Dict[str, Any]:
        try:
            return convert_bytes(await redis_client.ft(name).info())
        except Exception as e:
            raise RedisSearchError(
                f"Error while fetching {name} index info: {str(e)}"
            ) from e

    async def create(self, overwrite: bool = False, drop: bool = False) -> None:
        """Asynchronously create an index in Redis with the current schema
            and properties.

        Args:
            overwrite (bool, optional): Whether to overwrite the index if it
                already exists. Defaults to False.
            drop (bool, optional): Whether to drop all keys associated with the
                index in the case of overwriting. Defaults to False.

        Raises:
            RuntimeError: If the index already exists and 'overwrite' is False.
            ValueError: If no fields are defined for the index.

        .. code-block:: python

            # create an index in Redis; only if one does not exist with given name
            await index.create()

            # overwrite an index in Redis without dropping associated data
            await index.create(overwrite=True)

            # overwrite an index in Redis; drop associated data (clean slate)
            await index.create(overwrite=True, drop=True)
        """
        client = await self._get_client()
        redis_fields = self.schema.redis_fields

        if not redis_fields:
            raise ValueError("No fields defined for index")
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be of type bool")

        if await self.exists():
            if not overwrite:
                logger.info("Index already exists, not overwriting.")
                return None
            logger.info("Index already exists, overwriting.")
            await self.delete(drop)

        try:
            definition = IndexDefinition(
                prefix=[self.schema.index.prefix], index_type=self._storage.type
            )
            if isinstance(client, AsyncRedisCluster):
                await async_cluster_create_index(
                    index_name=self.schema.index.name,
                    client=client,
                    fields=redis_fields,
                    definition=definition,
                )
            else:
                await client.ft(self.schema.index.name).create_index(
                    fields=redis_fields,
                    definition=definition,
                )
        except redis.exceptions.RedisError as e:
            raise RedisSearchError(
                f"Failed to create index '{self.name}' on Redis: {str(e)}"
            ) from e
        except Exception as e:
            logger.exception("Error while trying to create the index")
            raise RedisSearchError(
                f"Unexpected error creating index '{self.name}': {str(e)}"
            ) from e

    async def delete(self, drop: bool = True):
        """Delete the search index.

        Args:
            drop (bool, optional): Delete the documents in the index.
                Defaults to True.

        Raises:
            redis.exceptions.ResponseError: If the index does not exist.
        """
        client = await self._get_client()
        try:
            # For Redis Cluster with drop=True, we need to handle key deletion manually
            # to avoid cross-slot errors since we control the keys opaquely
            if drop and isinstance(client, AsyncRedisCluster):
                # First clear all keys manually (handles cluster compatibility)
                await self.clear()
                # Then drop the index without the DD flag
                cmd_args = ["FT.DROPINDEX", self.schema.index.name]
            else:
                # Standard approach for non-cluster or when not dropping keys
                cmd_args = ["FT.DROPINDEX", self.schema.index.name]
                if drop:
                    cmd_args.append("DD")

            if isinstance(client, AsyncRedisCluster):
                target_nodes = [client.get_default_node()]
                await client.execute_command(*cmd_args, target_nodes=target_nodes)
            else:
                await client.execute_command(*cmd_args)
        except Exception as e:
            raise RedisSearchError(f"Error while deleting index: {str(e)}") from e

    async def clear(self) -> int:
        """Clear all keys in Redis associated with the index, leaving the index
        available and in-place for future insertions or updates.

        NOTE: This method requires custom behavior for Redis Cluster because here,
        we can't easily give control of the keys we're clearing to the user so they
        can separate them based on hash tag.

        Returns:
            int: Count of records deleted from Redis.
        """
        client = await self._get_client()
        total_records_deleted: int = 0

        async for batch in self.paginate(
            FilterQuery(FilterExpression("*"), return_fields=["id"]), page_size=500
        ):
            batch_keys = [record["id"] for record in batch]
            if batch_keys:
                is_cluster = isinstance(client, AsyncRedisCluster)
                if is_cluster:
                    records_deleted_in_batch = 0
                    for key_to_delete in batch_keys:
                        try:
                            records_deleted_in_batch += cast(
                                int, await client.delete(key_to_delete)
                            )
                        except redis.exceptions.RedisError as e:
                            logger.warning(f"Failed to delete key {key_to_delete}: {e}")
                    total_records_deleted += records_deleted_in_batch
                else:
                    records_deleted = await client.delete(*batch_keys)
                    total_records_deleted += records_deleted

        return total_records_deleted

    async def drop_keys(self, keys: Union[str, List[str]]) -> int:
        """Remove a specific entry or entries from the index by it's key ID.

        Args:
            keys (Union[str, List[str]]): The document ID or IDs to remove from the index.

        Returns:
            int: Count of records deleted from Redis.
        """
        client = await self._get_client()
        if isinstance(keys, list):
            return await client.delete(*keys)
        else:
            return await client.delete(keys)

    async def drop_documents(self, ids: Union[str, List[str]]) -> int:
        """Remove documents from the index by their document IDs.

        This method converts document IDs to Redis keys automatically by applying
        the index's key prefix and separator configuration.

        NOTE: Cluster users will need to incorporate hash tags into their
        document IDs and only call this method with documents from a single hash
        tag at a time.

        Args:
            ids (Union[str, List[str]]): The document ID or IDs to remove from the index.

        Returns:
            int: Count of documents deleted from Redis.
        """
        client = await self._get_client()
        if isinstance(ids, list):
            if not ids:
                return 0
            keys = [self.key(id) for id in ids]
            # Check for cluster compatibility
            if isinstance(client, AsyncRedisCluster) and not _keys_share_hash_tag(keys):
                raise ValueError(
                    "All keys must share a hash tag when using Redis Cluster."
                )
            return await client.delete(*keys)
        else:
            key = self.key(ids)
            return await client.delete(key)

    async def expire_keys(
        self, keys: Union[str, List[str]], ttl: int
    ) -> Union[int, List[int]]:
        """Set the expiration time for a specific entry or entries in Redis.

        Args:
            keys (Union[str, List[str]]): The entry ID or IDs to set the expiration for.
            ttl (int): The time-to-live in seconds.
        """
        client = await self._get_client()
        if isinstance(keys, list):
            pipe = client.pipeline()
            for key in keys:
                pipe.expire(key, ttl)
            return await pipe.execute()
        else:
            return await client.expire(keys, ttl)

    @deprecated_argument("concurrency", "Use batch_size instead.")
    async def load(
        self,
        data: Iterable[Any],
        id_field: Optional[str] = None,
        keys: Optional[Iterable[str]] = None,
        ttl: Optional[int] = None,
        preprocess: Optional[Callable] = None,
        concurrency: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """Asynchronously load objects to Redis. Returns the list of keys loaded
        to Redis.

        RedisVL automatically handles constructing the object keys, batching,
        optional preprocessing steps, and setting optional expiration
        (TTL policies) on keys.

        Args:
            data (Iterable[Any]): An iterable of objects to store.
            id_field (Optional[str], optional): Specified field used as the id
                portion of the redis key (after the prefix) for each
                object. Defaults to None.
            keys (Optional[Iterable[str]], optional): Optional iterable of keys.
                Must match the length of objects if provided. Defaults to None.
            ttl (Optional[int], optional): Time-to-live in seconds for each key.
                Defaults to None.
            preprocess (Optional[Callable], optional): A function to
                preprocess objects before storage. Defaults to None.
            batch_size (Optional[int], optional): Number of objects to write in
                a single Redis pipeline execution. Defaults to class's
                default batch size.

        Returns:
            List[str]: List of keys loaded to Redis.

        Raises:
            SchemaValidationError: If validation fails when validate_on_load is enabled.
            RedisVLError: If there's an error loading data to Redis.

        .. code-block:: python

            data = [{"test": "foo"}, {"test": "bar"}]

            # simple case
            keys = await index.load(data)

            # set 360 second ttl policy on data
            keys = await index.load(data, ttl=360)

            # load data with predefined keys
            keys = await index.load(data, keys=["rvl:foo", "rvl:bar"])

            # load data with preprocessing step
            def add_field(d):
                d["new_field"] = 123
                return d
            keys = await index.load(data, preprocess=add_field)

        """
        client = await self._get_client()
        try:
            return await self._storage.awrite(
                client,
                objects=data,
                id_field=id_field,
                keys=keys,
                ttl=ttl,
                preprocess=preprocess,
                batch_size=batch_size,
                validate=self._validate_on_load,
            )
        except SchemaValidationError as e:
            # Log the detailed validation error with actionable information
            logger.error("Data validation failed during load operation")
            raise
        except Exception as e:
            # Wrap other errors as general RedisVL errors
            logger.exception("Error while loading data to Redis")
            raise RedisVLError(f"Failed to load data: {str(e)}") from e

    async def fetch(self, id: str) -> Optional[Dict[str, Any]]:
        """Asynchronously etch an object from Redis by id. The id is typically
        either a unique identifier, or derived from some domain-specific
        metadata combination (like a document id or chunk id).

        Args:
            id (str): The specified unique identifier for a particular
                document indexed in Redis.

        Returns:
            Dict[str, Any]: The fetched object.
        """
        client = await self._get_client()
        obj = await self._storage.aget(client, [self.key(id)])
        if obj:
            return convert_bytes(obj[0])
        return None

    async def _aggregate(
        self, aggregation_query: AggregationQuery
    ) -> List[Dict[str, Any]]:
        """Execute an aggregation query and processes the results."""
        results = await self.aggregate(
            aggregation_query,
            query_params=aggregation_query.params,  # type: ignore[attr-defined]
        )
        return process_aggregate_results(
            results,
            query=aggregation_query,
            storage_type=self.schema.index.storage_type,
        )

    async def aggregate(self, *args, **kwargs) -> "AggregateResult":
        """Perform an aggregation operation against the index.

        Wrapper around the aggregation API that adds the index name
        to the query and passes along the rest of the arguments
        to the redis-py ft().aggregate() method.

        Returns:
            Result: Raw Redis aggregation results.
        """
        client = await self._get_client()
        try:
            return await client.ft(self.schema.index.name).aggregate(*args, **kwargs)
        except redis.exceptions.RedisError as e:
            if "CROSSSLOT" in str(e):
                raise RedisSearchError(
                    "Cross-slot error during aggregation. Ensure consistent hash tags in your keys."
                )
            raise RedisSearchError(f"Error while aggregating: {str(e)}") from e
        except Exception as e:
            raise RedisSearchError(
                f"Unexpected error while aggregating: {str(e)}"
            ) from e

    async def batch_search(
        self, queries: List[SearchParams], batch_size: int = 10
    ) -> List["Result"]:
        """Asynchronously execute a batch of search queries.

        This method takes a list of search queries and executes them in batches
        to improve performance when dealing with multiple queries.

        NOTE: Cluster users may need to incorporate hash tags into their query
        to avoid cross-slot operations.

        Args:
            queries (List[SearchParams]): A list of search queries to execute.
                Each query can be either a string or a tuple of (query, params).
            batch_size (int, optional): The number of queries to execute in each
                batch. Defaults to 10.

        Returns:
            List[Result]: A list of search results corresponding to each query.

        .. code-block:: python

            queries = [
                "hello world",
                ("goodbye world", {"num_results": 5}),
            ]

            results = await index.batch_search(queries)
        """
        all_results = []
        client = await self._get_client()
        search = client.ft(self.schema.index.name)
        options = {}
        if get_protocol_version(client) not in ["3", 3]:
            options[NEVER_DECODE] = True

        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i : i + batch_size]

            # redis-py doesn't support calling `search` in an async pipeline,
            # so we need to manually execute each command in a pipeline
            # and parse the results
            async with client.pipeline(transaction=False) as pipe:
                batch_built_queries = []
                for query in batch_queries:
                    if isinstance(query, tuple):
                        query_args, q = search._mk_query_args(  # type: ignore
                            query[0], query_params=query[1]
                        )
                    else:
                        query_args, q = search._mk_query_args(  # type: ignore
                            query, query_params=None
                        )
                    batch_built_queries.append(q)
                    pipe.execute_command(
                        "FT.SEARCH",
                        *query_args,
                        **options,
                    )

                st = time.time()
                results = await pipe.execute()

                # We don't know how long each query took, so we'll use the total time
                # for all queries in the batch as the duration for each query
                duration = (time.time() - st) * 1000.0

                for j, query_results in enumerate(results):
                    _built_query = batch_built_queries[j]
                    parsed_result = search._parse_search(  # type: ignore
                        query_results,
                        query=_built_query,
                        duration=duration,
                    )
                    # Return a parsed Result object for each query
                    all_results.append(parsed_result)
        return all_results

    async def search(self, *args, **kwargs) -> "Result":
        """Perform an async search against the index.

        Wrapper around the search API that adds the index name
        to the query and passes along the rest of the arguments
        to the redis-py ft().search() method.

        Returns:
            Result: Raw Redis search results.
        """
        try:
            client = await self._get_client()
            if isinstance(client, AsyncRedisCluster):
                # Use special cluster search for AsyncRedisCluster
                return await async_cluster_search(
                    client.ft(self.schema.index.name),
                    *args,
                    **kwargs,  # type: ignore
                )
            else:
                return await client.ft(self.schema.index.name).search(*args, **kwargs)  # type: ignore
        except redis.exceptions.RedisError as e:
            if "CROSSSLOT" in str(e):
                raise RedisSearchError(
                    "Cross-slot error during search. Ensure consistent hash tags in your keys."
                )
            raise RedisSearchError(f"Error while searching: {str(e)}") from e
        except Exception as e:
            raise RedisSearchError(f"Unexpected error while searching: {str(e)}") from e

    async def batch_query(
        self, queries: List[BaseQuery], batch_size: int = 10
    ) -> List[List[Dict[str, Any]]]:
        """Asynchronously execute a batch of queries and process results."""
        results = await self.batch_search(
            [(query.query, query.params) for query in queries], batch_size=batch_size
        )
        all_parsed = []
        for query, batch_results in zip(queries, results):
            parsed = process_results(
                batch_results,
                query=query,
                schema=self.schema,
            )
            # Create separate lists of parsed results for each query
            # passed in to the batch_search method, so that callers can
            # access the results for each query individually
            all_parsed.append(parsed)

        return all_parsed

    async def _query(self, query: BaseQuery) -> List[Dict[str, Any]]:
        """Asynchronously execute a query and process results."""
        try:
            self._validate_query(query)
        except QueryValidationError as e:
            raise QueryValidationError(f"Invalid query: {str(e)}") from e
        results = await self.search(query.query, query_params=query.params)
        return process_results(results, query=query, schema=self.schema)

    async def query(
        self, query: Union[BaseQuery, AggregationQuery]
    ) -> List[Dict[str, Any]]:
        """Asynchronously execute a query on the index.

        This method takes a BaseQuery or AggregationQuery object directly, runs
        the search, and handles post-processing of the search.

        Args:
            query (Union[BaseQuery, AggregateQuery]): The query to run.

        Returns:
            List[Result]: A list of search results.

        .. code-block:: python

            from redisvl.query import VectorQuery

            query = VectorQuery(
                vector=[0.16, -0.34, 0.98, 0.23],
                vector_field_name="embedding",
                num_results=3
            )

            results = await index.query(query)
        """
        if isinstance(query, AggregationQuery):
            return await self._aggregate(query)
        else:
            return await self._query(query)

    async def paginate(self, query: BaseQuery, page_size: int = 30) -> AsyncGenerator:
        """Execute a given query against the index and return results in
        paginated batches.

        This method accepts a RedisVL query instance, enabling async pagination
        of results which allows for subsequent processing over each batch with a
        generator.

        Args:
            query (BaseQuery): The search query to be executed.
            page_size (int, optional): The number of results to return in each
                batch. Defaults to 30.

        Yields:
            An async generator yielding batches of search results.

        Raises:
            TypeError: If the page_size argument is not of type int.
            ValueError: If the page_size argument is less than or equal to zero.

        .. code-block:: python

            # Iterate over paginated search results in batches of 10
            async for result_batch in index.paginate(query, page_size=10):
                # Process each batch of results
                pass

        Note:
            The page_size parameter controls the number of items each result
            batch contains. Adjust this value based on performance
            considerations and the expected volume of search results.

        """
        if not isinstance(page_size, int):
            raise TypeError("page_size must be of type int")

        if page_size <= 0:
            raise ValueError("page_size must be greater than 0")

        first = 0
        while True:
            query.paging(first, page_size)
            results = await self._query(query)
            if not results:
                break
            yield results
            first += page_size

    async def listall(self) -> List[str]:
        """List all search indices in Redis database.

        Returns:
            List[str]: The list of indices in the database.
        """
        client = await self._get_client()
        if isinstance(client, AsyncRedisCluster):
            target_nodes = client.get_random_node()
            return convert_bytes(await target_nodes.execute_command("FT._LIST"))
        else:
            return convert_bytes(await client.execute_command("FT._LIST"))

    async def exists(self) -> bool:
        """Check if the index exists in Redis.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        return self.schema.index.name in await self.listall()

    async def info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about the index.

        Args:
            name (str, optional): Index name to fetch info about.
                Defaults to None.

        Returns:
            dict: A dictionary containing the information about the index.
        """
        client = await self._get_client()
        index_name = name or self.schema.index.name
        return await self._info(index_name, client)

    async def disconnect(self):
        if self._owns_redis_client is False:
            return
        if self._redis_client is not None:
            await self._redis_client.aclose()
        self._redis_client = None

    def disconnect_sync(self):
        if self._redis_client is None or self._owns_redis_client is False:
            return
        sync_wrapper(self.disconnect)()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
