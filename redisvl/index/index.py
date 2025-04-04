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
    Tuple,
    Union,
)

from redisvl.redis.utils import convert_bytes, make_dict
from redisvl.utils.utils import deprecated_argument, deprecated_function, sync_wrapper

if TYPE_CHECKING:
    from redis.commands.search.aggregation import AggregateResult
    from redis.commands.search.document import Document
    from redis.commands.search.result import Result
    from redisvl.query.query import BaseQuery

import redis
import redis.asyncio as aredis
from redis.client import NEVER_DECODE
from redis.commands.helpers import get_protocol_version  # type: ignore
from redis.commands.search.indexDefinition import IndexDefinition

from redisvl.exceptions import (
    RedisModuleVersionError,
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
    HybridQuery,
)
from redisvl.query.filter import FilterExpression
from redisvl.redis.connection import (
    RedisConnectionFactory,
    convert_index_info_to_schema,
)
from redisvl.redis.utils import convert_bytes
from redisvl.schema import IndexSchema, StorageType
from redisvl.schema.fields import VECTOR_NORM_MAP, VectorDistanceMetric
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
    """Convert an aggregate reslt object into a list of document dictionaries.

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

    @property
    def name(self) -> str:
        """The name of the Redis search index."""
        return self.schema.index.name

    @property
    def prefix(self) -> str:
        """The optional key prefix that comes before a unique key value in
        forming a Redis key."""
        return self.schema.index.prefix

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
            prefix=self.schema.index.prefix,
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
        redis_client: Optional[redis.Redis] = None,
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
            redis_client(Optional[redis.Redis]): An
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

        self._validated_client = False
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
        redis_client: Optional[redis.Redis] = None,
        redis_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize from an existing search index in Redis by index name.

        Args:
            name (str): Name of the search index in Redis.
            redis_client(Optional[redis.Redis]): An
                instantiated redis client.
            redis_url (Optional[str]): The URL of the Redis server to
                connect to.

        Raises:
            ValueError: If redis_url or redis_client is not provided.
            RedisModuleVersionError: If required Redis modules are not installed.
        """
        try:
            if redis_url:
                redis_client = RedisConnectionFactory.get_redis_connection(
                    redis_url=redis_url,
                    required_modules=REQUIRED_MODULES_FOR_INTROSPECTION,
                    **kwargs,
                )
            elif redis_client:
                RedisConnectionFactory.validate_sync_redis(
                    redis_client, required_modules=REQUIRED_MODULES_FOR_INTROSPECTION
                )
        except RedisModuleVersionError as e:
            raise RedisModuleVersionError(
                f"Loading from existing index failed. {str(e)}"
            )

        if not redis_client:
            raise ValueError("Must provide either a redis_url or redis_client")

        # Fetch index info and convert to schema
        index_info = cls._info(name, redis_client)
        schema_dict = convert_index_info_to_schema(index_info)
        schema = IndexSchema.from_dict(schema_dict)
        return cls(schema, redis_client, **kwargs)

    @property
    def client(self) -> Optional[redis.Redis]:
        """The underlying redis-py client object."""
        return self.__redis_client

    @property
    def _redis_client(self) -> redis.Redis:
        """
        Get a Redis client instance.

        Lazily creates a Redis client instance if it doesn't exist.
        """
        if self.__redis_client is None:
            with self._lock:
                if self.__redis_client is None:
                    self.__redis_client = RedisConnectionFactory.get_redis_connection(
                        redis_url=self._redis_url,
                        **self._connection_kwargs,
                    )
        if not self._validated_client:
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
    def set_client(self, redis_client: redis.Redis, **kwargs):
        """Manually set the Redis client to use with the search index.

        This method configures the search index to use a specific Redis or
        Async Redis client. It is useful for cases where an external,
        custom-configured client is preferred instead of creating a new one.

        Args:
            redis_client (redis.Redis): A Redis or Async Redis
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
            self._redis_client.ft(self.name).create_index(  # type: ignore
                fields=redis_fields,
                definition=IndexDefinition(
                    prefix=[self.schema.index.prefix], index_type=self._storage.type
                ),
            )
        except:
            logger.exception("Error while trying to create the index")
            raise

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
            self._redis_client.ft(self.schema.index.name).dropindex(  # type: ignore
                delete_documents=drop
            )
        except Exception as e:
            raise RedisSearchError(f"Error while deleting index: {str(e)}") from e

    def clear(self) -> int:
        """Clear all keys in Redis associated with the index, leaving the index
        available and in-place for future insertions or updates.

        Returns:
            int: Count of records deleted from Redis.
        """
        # Track deleted records
        total_records_deleted: int = 0

        # Paginate using queries and delete in batches
        for batch in self.paginate(
            FilterQuery(FilterExpression("*"), return_fields=["id"]), page_size=500
        ):
            batch_keys = [record["id"] for record in batch]
            record_deleted = self._redis_client.delete(*batch_keys)  # type: ignore
            total_records_deleted += record_deleted  # type: ignore

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
                self._redis_client,  # type: ignore
                objects=data,
                id_field=id_field,
                keys=keys,
                ttl=ttl,
                preprocess=preprocess,
                batch_size=batch_size,
                validate=self._validate_on_load,
            )
        except SchemaValidationError:
            # Pass through validation errors directly
            logger.exception("Schema validation error while loading data")
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
        obj = self._storage.get(self._redis_client, [self.key(id)])  # type: ignore
        if obj:
            return convert_bytes(obj[0])
        return None

    def _aggregate(self, aggregation_query: AggregationQuery) -> List[Dict[str, Any]]:
        """Execute an aggretation query and processes the results."""
        results = self.aggregate(
            aggregation_query, query_params=aggregation_query.params  # type: ignore[attr-defined]
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
            return self._redis_client.ft(self.schema.index.name).aggregate(  # type: ignore
                *args, **kwargs
            )
        except Exception as e:
            raise RedisSearchError(f"Error while aggregating: {str(e)}") from e

    def batch_search(
        self,
        queries: List[SearchParams],
        batch_size: int = 10,
    ) -> List["Result"]:
        """Perform a search against the index for multiple queries.

        This method takes a list of queries and optionally query params and
        returns a list of Result objects for each query. Results are
        returned in the same order as the queries.

        Args:
            queries (List[SearchParams]): The queries to search for. batch_size
            (int, optional): The number of queries to search for at a time.
                Defaults to 10.

        Returns:
            List[Result]: The search results for each query.
        """
        all_parsed = []
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

                for i, query_results in enumerate(results):
                    _built_query = batch_built_queries[i]
                    parsed_result = search._parse_search(  # type: ignore
                        query_results,
                        query=_built_query,
                        duration=duration,
                    )
                    # Return a parsed Result object for each query
                    all_parsed.append(parsed_result)
        return all_parsed

    def search(self, *args, **kwargs) -> "Result":
        """Perform a search against the index.

        Wrapper around the search API that adds the index name
        to the query and passes along the rest of the arguments
        to the redis-py ft().search() method.

        Returns:
            Result: Raw Redis search results.
        """
        try:
            return self._redis_client.ft(self.schema.index.name).search(  # type: ignore
                *args, **kwargs
            )
        except Exception as e:
            raise RedisSearchError(f"Error while searching: {str(e)}") from e

    def batch_query(
        self, queries: List[BaseQuery], batch_size: int = 10
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
        return convert_bytes(self._redis_client.execute_command("FT._LIST"))  # type: ignore

    def exists(self) -> bool:
        """Check if the index exists in Redis.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        return self.schema.index.name in self.listall()

    @staticmethod
    def _info(name: str, redis_client: redis.Redis) -> Dict[str, Any]:
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
        return self._info(index_name, self._redis_client)  # type: ignore

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
        redis_client: Optional[aredis.Redis] = None,
        connection_kwargs: Optional[Dict[str, Any]] = None,
        validate_on_load: bool = False,
        **kwargs,
    ):
        """Initialize the RedisVL async search index with a schema.

        Args:
            schema (IndexSchema): Index schema object.
            redis_url (Optional[str], optional): The URL of the Redis server to
                connect to.
            redis_client (Optional[aredis.Redis]): An
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

        self._validated_client = False
        self._owns_redis_client = redis_client is None
        if self._owns_redis_client:
            weakref.finalize(self, sync_wrapper(self.disconnect))

    @classmethod
    async def from_existing(
        cls,
        name: str,
        redis_client: Optional[aredis.Redis] = None,
        redis_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize from an existing search index in Redis by index name.

        Args:
            name (str): Name of the search index in Redis.
            redis_client(Optional[redis.Redis]): An
                instantiated redis client.
            redis_url (Optional[str]): The URL of the Redis server to
                connect to.
        """
        if not redis_url and not redis_client:
            raise ValueError(
                "Must provide either a redis_url or redis_client to fetch Redis index info."
            )

        try:
            if redis_url:
                redis_client = await RedisConnectionFactory._get_aredis_connection(
                    url=redis_url,
                    required_modules=REQUIRED_MODULES_FOR_INTROSPECTION,
                    **kwargs,
                )
            elif redis_client:
                await RedisConnectionFactory.validate_async_redis(
                    redis_client, required_modules=REQUIRED_MODULES_FOR_INTROSPECTION
                )
        except RedisModuleVersionError as e:
            raise RedisModuleVersionError(
                f"Loading from existing index failed. {str(e)}"
            ) from e

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
    def client(self) -> Optional[aredis.Redis]:
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
    async def set_client(self, redis_client: Union[aredis.Redis, redis.Redis]):
        """
        [DEPRECATED] Manually set the Redis client to use with the search index.
        This method is deprecated; please provide connection parameters in __init__.
        """
        redis_client = await self._validate_client(redis_client)
        await self.disconnect()
        async with self._lock:
            self._redis_client = redis_client
        return self

    async def _get_client(self) -> aredis.Redis:
        """Lazily instantiate and return the async Redis client."""
        if self._redis_client is None:
            async with self._lock:
                # Double-check to protect against concurrent access
                if self._redis_client is None:
                    kwargs = self._connection_kwargs
                    if self._redis_url:
                        kwargs["url"] = self._redis_url
                    self._redis_client = (
                        await RedisConnectionFactory._get_aredis_connection(**kwargs)
                    )
        if not self._validated_client:
            await RedisConnectionFactory.validate_async_redis(
                self._redis_client,
                self._lib_name,
            )
            self._validated_client = True
        return self._redis_client

    async def _validate_client(
        self, redis_client: Union[aredis.Redis, redis.Redis]
    ) -> aredis.Redis:
        if isinstance(redis_client, redis.Redis):
            warnings.warn(
                "Converting sync Redis client to async client is deprecated "
                "and will be removed in the next major version. Please use an "
                "async Redis client instead.",
                DeprecationWarning,
            )
            redis_client = RedisConnectionFactory.sync_to_async_redis(redis_client)
        elif not isinstance(redis_client, aredis.Redis):
            raise ValueError("Invalid client type: must be redis.asyncio.Redis")
        return redis_client

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
            await client.ft(self.schema.index.name).create_index(
                fields=redis_fields,
                definition=IndexDefinition(
                    prefix=[self.schema.index.prefix], index_type=self._storage.type
                ),
            )
        except:
            logger.exception("Error while trying to create the index")
            raise

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
            await client.ft(self.schema.index.name).dropindex(delete_documents=drop)
        except Exception as e:
            raise RedisSearchError(f"Error while deleting index: {str(e)}") from e

    async def clear(self) -> int:
        """Clear all keys in Redis associated with the index, leaving the index
        available and in-place for future insertions or updates.

        Returns:
            int: Count of records deleted from Redis.
        """
        client = await self._get_client()
        total_records_deleted: int = 0

        async for batch in self.paginate(
            FilterQuery(FilterExpression("*"), return_fields=["id"]), page_size=500
        ):
            batch_keys = [record["id"] for record in batch]
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
        except SchemaValidationError:
            # Pass through validation errors directly
            logger.exception("Schema validation error while loading data")
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
        """Execute an aggretation query and processes the results."""
        results = await self.aggregate(
            aggregation_query, query_params=aggregation_query.params  # type: ignore[attr-defined]
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
            return client.ft(self.schema.index.name).aggregate(*args, **kwargs)
        except Exception as e:
            raise RedisSearchError(f"Error while aggregating: {str(e)}") from e

    async def batch_search(
        self, queries: List[SearchParams], batch_size: int = 10
    ) -> List["Result"]:
        """Perform a search against the index for multiple queries.

        This method takes a list of queries and returns a list of Result objects
        for each query. Results are returned in the same order as the queries.

        Args:
            queries (List[SearchParams]): The queries to search for. batch_size
            (int, optional): The number of queries to search for at a time.
                Defaults to 10.

        Returns:
            List[Result]: The search results for each query.
        """
        all_results = []
        client = await self._get_client()
        search = client.ft(self.schema.index.name)
        options = {}
        if get_protocol_version(client) not in ["3", 3]:
            options[NEVER_DECODE] = True

        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i : i + batch_size]

            # redis-py doesn't support calling `search` in a pipeline,
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

                for i, query_results in enumerate(results):
                    _built_query = batch_built_queries[i]
                    parsed_result = search._parse_search(  # type: ignore
                        query_results,
                        query=_built_query,
                        duration=duration,
                    )
                    # Return a parsed Result object for each query
                    all_results.append(parsed_result)
        return all_results

    async def search(self, *args, **kwargs) -> "Result":
        """Perform a search on this index.

        Wrapper around redis.search.Search that adds the index name
        to the search query and passes along the rest of the arguments
        to the redis-py ft.search() method.

        Returns:
            Result: Raw Redis search results.
        """
        client = await self._get_client()
        try:
            return await client.ft(self.schema.index.name).search(*args, **kwargs)  # type: ignore
        except Exception as e:
            raise RedisSearchError(f"Error while searching: {str(e)}") from e

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
        return await type(self)._info(index_name, client)

    @staticmethod
    async def _info(name: str, redis_client: aredis.Redis) -> Dict[str, Any]:
        try:
            return convert_bytes(await redis_client.ft(name).info())  # type: ignore
        except Exception as e:
            raise RedisSearchError(
                f"Error while fetching {name} index info: {str(e)}"
            ) from e

    async def disconnect(self):
        if self._owns_redis_client is False:
            return
        if self._redis_client is not None:
            await self._redis_client.aclose()  # type: ignore
        self._redis_client = None

    def disconnect_sync(self):
        if self._redis_client is None or self._owns_redis_client is False:
            return
        sync_wrapper(self.disconnect)()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
