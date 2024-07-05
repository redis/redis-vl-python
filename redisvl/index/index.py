import json
from functools import wraps
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
    Union,
)

if TYPE_CHECKING:
    from redis.commands.search.document import Document
    from redis.commands.search.result import Result
    from redisvl.query.query import BaseQuery

import redis
import redis.asyncio as aredis
from redis.commands.search.indexDefinition import IndexDefinition

from redisvl.index.storage import HashStorage, JsonStorage
from redisvl.query.query import BaseQuery, CountQuery, FilterQuery
from redisvl.redis.connection import (
    RedisConnectionFactory,
    convert_index_info_to_schema,
    validate_modules,
)
from redisvl.redis.utils import convert_bytes
from redisvl.schema import IndexSchema, StorageType
from redisvl.utils.log import get_logger

logger = get_logger(__name__)


def process_results(
    results: "Result", query: BaseQuery, storage_type: StorageType
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
        (storage_type == StorageType.JSON)
        and isinstance(query, FilterQuery)
        and not query._return_fields
    )

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

        # Remove 'payload' if present
        doc_dict.pop("payload", None)

        return doc_dict

    return [_process(doc) for doc in results.docs]


def setup_redis():
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            RedisConnectionFactory.validate_redis(self._redis_client, self._lib_name)
            return result

        return wrapper

    return decorator


def check_index_exists():
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.exists():
                raise RuntimeError(
                    f"Index has not been created. Must be created before calling {func.__name__}"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def check_async_index_exists():
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not await self.exists():
                raise ValueError(
                    f"Index has not been created. Must be created before calling {func.__name__}"
                )
            return await func(self, *args, **kwargs)

        return wrapper

    return decorator


class BaseSearchIndex:
    """Base search engine class"""

    _STORAGE_MAP = {
        StorageType.HASH: HashStorage,
        StorageType.JSON: JsonStorage,
    }

    def __init__(
        self,
        schema: IndexSchema,
        redis_client: Optional[Union[redis.Redis, aredis.Redis]] = None,
        redis_url: Optional[str] = None,
        connection_args: Dict[str, Any] = {},
        **kwargs,
    ):
        """Initialize the RedisVL search index with a schema, Redis client
        (or URL string with other connection args), connection_args, and other
        kwargs.

        Args:
            schema (IndexSchema): Index schema object.
            redis_client(Union[redis.Redis, aredis.Redis], optional): An
                instantiated redis client.
            redis_url (str, optional): The URL of the Redis server to
                connect to.
            connection_args (Dict[str, Any], optional): Redis client connection
                args.
        """
        # final validation on schema object
        if not isinstance(schema, IndexSchema):
            raise ValueError("Must provide a valid IndexSchema object")

        self.schema = schema

        self._lib_name: Optional[str] = kwargs.pop("lib_name", None)

        # set up redis connection
        self._redis_client: Optional[Union[redis.Redis, aredis.Redis]] = None
        if redis_client is not None:
            self.set_client(redis_client)
        elif redis_url is not None:
            self.connect(redis_url, **connection_args)

        # set up index storage layer
        self._storage = self._STORAGE_MAP[self.schema.index.storage_type](
            prefix=self.schema.index.prefix,
            key_separator=self.schema.index.key_separator,
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

    @property
    def client(self) -> Optional[Union[redis.Redis, aredis.Redis]]:
        """The underlying redis-py client object."""
        return self._redis_client

    @classmethod
    def from_yaml(cls, schema_path: str, **kwargs):
        """Create a SearchIndex from a YAML schema file.

        Args:
            schema_path (str): Path to the YAML schema file.

        Returns:
            SearchIndex: A RedisVL SearchIndex object.

        .. code-block:: python

            from redisvl.index import SearchIndex

            index = SearchIndex.from_yaml("schemas/schema.yaml")
        """
        schema = IndexSchema.from_yaml(schema_path)
        return cls(schema=schema, **kwargs)

    @classmethod
    def from_dict(cls, schema_dict: Dict[str, Any], **kwargs):
        """Create a SearchIndex from a dictionary.

        Args:
            schema_dict (Dict[str, Any]): A dictionary containing the schema.
            connection_args (Dict[str, Any], optional): Redis client connection
                args.

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
            })

        """
        schema = IndexSchema.from_dict(schema_dict)
        return cls(schema=schema, **kwargs)

    def connect(self, redis_url: Optional[str] = None, **kwargs):
        """Connect to Redis at a given URL."""
        raise NotImplementedError

    def set_client(self, client: Union[redis.Redis, aredis.Redis]):
        """Manually set the Redis client to use with the search index."""
        raise NotImplementedError

    def disconnect(self):
        """Disconnect from the Redis database."""
        self._redis_client = None
        return self

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
        index = SearchIndex.from_yaml("schemas/schema.yaml")
        index.connect(redis_url="redis://localhost:6379")

        # create the index
        index.create(overwrite=True)

        # data is an iterable of dictionaries
        index.load(data)

        # delete index and data
        index.delete(drop=True)

    """

    @classmethod
    def from_existing(
        cls,
        name: str,
        redis_client: Optional[redis.Redis] = None,
        redis_url: Optional[str] = None,
        **kwargs,
    ):
        # Handle redis instance
        if redis_url:
            redis_client = RedisConnectionFactory.connect(
                redis_url=redis_url, use_async=False, **kwargs
            )
        if not redis_client:
            raise ValueError(
                "Must provide either a redis_url or redis_client to fetch Redis index info."
            )

        # Validate modules
        installed_modules = RedisConnectionFactory._get_modules(redis_client)
        validate_modules(installed_modules, [{"name": "search", "ver": 20810}])

        # Fetch index info and convert to schema
        index_info = cls._info(name, redis_client)
        schema_dict = convert_index_info_to_schema(index_info)
        schema = IndexSchema.from_dict(schema_dict)
        return cls(schema, redis_client, **kwargs)

    def connect(self, redis_url: Optional[str] = None, **kwargs):
        """Connect to a Redis instance using the provided `redis_url`, falling
        back to the `REDIS_URL` environment variable (if available).

        Note: Additional keyword arguments (`**kwargs`) can be used to provide
        extra options specific to the Redis connection.

        Args:
            redis_url (Optional[str], optional): The URL of the Redis server to
                connect to. If not provided, the method defaults to using the
                `REDIS_URL` environment variable.

        Raises:
            redis.exceptions.ConnectionError: If the connection to the Redis
                server fails.
            ValueError: If the Redis URL is not provided nor accessible
                through the `REDIS_URL` environment variable.

        .. code-block:: python

            index.connect(redis_url="redis://localhost:6379")

        """
        client = RedisConnectionFactory.connect(
            redis_url=redis_url, use_async=False, **kwargs
        )
        return self.set_client(client)

    @setup_redis()
    def set_client(self, client: redis.Redis, **kwargs):
        """Manually set the Redis client to use with the search index.

        This method configures the search index to use a specific Redis or
        Async Redis client. It is useful for cases where an external,
        custom-configured client is preferred instead of creating a new one.

        Args:
            client (redis.Redis): A Redis or Async Redis
                client instance to be used for the connection.

        Raises:
            TypeError: If the provided client is not valid.

        .. code-block:: python

            import redis
            from redisvl.index import SearchIndex

            client = redis.Redis.from_url("redis://localhost:6379")
            index = SearchIndex.from_yaml("schemas/schema.yaml")
            index.set_client(client)

        """
        if not isinstance(client, redis.Redis):
            raise TypeError("Invalid Redis client instance")

        self._redis_client = client

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

    @check_index_exists()
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
        except:
            logger.exception("Error while deleting index")

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
            ValueError: If the length of provided keys does not match the length
                of objects.

        .. code-block:: python

            data = [{"test": "foo"}, {"test": "bar"}]

            # simple case
            keys = index.load(data)

            # set 360 second ttl policy on data
            keys = index.load(data, ttl=360)

            # load data with predefined keys
            keys = index.load(data, keys=["rvl:foo", "rvl:bar"])

            # load data with preprocessing step
            def add_field(d):
                d["new_field"] = 123
                return d
            keys = index.load(data, preprocess=add_field)
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
            )
        except:
            logger.exception("Error while loading data to Redis")
            raise

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

    @check_index_exists()
    def search(self, *args, **kwargs) -> "Result":
        """Perform a search against the index.

        Wrapper around redis.search.Search that adds the index name
        to the search query and passes along the rest of the arguments
        to the redis-py ft.search() method.

        Returns:
            Result: Raw Redis search results.
        """
        try:
            return self._redis_client.ft(self.schema.index.name).search(  # type: ignore
                *args, **kwargs
            )
        except:
            logger.exception("Error while searching")
            raise

    def _query(self, query: BaseQuery) -> List[Dict[str, Any]]:
        """Execute a query and process results."""
        results = self.search(query.query, query_params=query.params)
        return process_results(
            results, query=query, storage_type=self.schema.index.storage_type
        )

    def query(self, query: BaseQuery) -> List[Dict[str, Any]]:
        """Execute a query on the index.

        This method takes a BaseQuery object directly, runs the search, and
        handles post-processing of the search.

        Args:
            query (BaseQuery): The query to run.

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
            query.set_paging(offset, page_size)
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
        except:
            logger.exception(f"Error while fetching {name} index info")
            raise

    @check_index_exists()
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


class AsyncSearchIndex(BaseSearchIndex):
    """A search index class for interacting with Redis as a vector database in
    async-mode.

    The AsyncSearchIndex is instantiated with a reference to a Redis database
    and an IndexSchema (YAML path or dictionary object) that describes the
    various settings and field configurations.

    .. code-block:: python

        from redisvl.index import AsyncSearchIndex

        # initialize the index object with schema from file
        index = AsyncSearchIndex.from_yaml("schemas/schema.yaml")
        index.connect(redis_url="redis://localhost:6379")

        # create the index
        await index.create(overwrite=True)

        # data is an iterable of dictionaries
        await index.load(data)

        # delete index and data
        await index.delete(drop=True)

    """

    @classmethod
    async def from_existing(
        cls,
        name: str,
        redis_client: Optional[aredis.Redis] = None,
        redis_url: Optional[str] = None,
        **kwargs,
    ):
        if redis_url:
            redis_client = RedisConnectionFactory.connect(
                redis_url=redis_url, use_async=True, **kwargs
            )

        if not redis_client:
            raise ValueError(
                "Must provide either a redis_url or redis_client to fetch Redis index info."
            )

        # Validate modules
        installed_modules = await RedisConnectionFactory._get_modules_async(
            redis_client
        )
        validate_modules(installed_modules, [{"name": "search", "ver": 20810}])

        # Fetch index info and convert to schema
        index_info = await cls._info(name, redis_client)
        schema_dict = convert_index_info_to_schema(index_info)
        schema = IndexSchema.from_dict(schema_dict)
        return cls(schema, redis_client, **kwargs)

    def connect(self, redis_url: Optional[str] = None, **kwargs):
        """Connect to a Redis instance using the provided `redis_url`, falling
        back to the `REDIS_URL` environment variable (if available).

        Note: Additional keyword arguments (`**kwargs`) can be used to provide
        extra options specific to the Redis connection.

        Args:
            redis_url (Optional[str], optional): The URL of the Redis server to
                connect to. If not provided, the method defaults to using the
                `REDIS_URL` environment variable.

        Raises:
            redis.exceptions.ConnectionError: If the connection to the Redis
                server fails.
            ValueError: If the Redis URL is not provided nor accessible
                through the `REDIS_URL` environment variable.

        .. code-block:: python

            index.connect(redis_url="redis://localhost:6379")

        """
        client = RedisConnectionFactory.connect(
            redis_url=redis_url, use_async=True, **kwargs
        )
        return self.set_client(client)

    @setup_redis()
    def set_client(self, client: aredis.Redis):
        """Manually set the Redis client to use with the search index.

        This method configures the search index to use a specific
        Async Redis client. It is useful for cases where an external,
        custom-configured client is preferred instead of creating a new one.

        Args:
            client (aredis.Redis): An Async Redis
                client instance to be used for the connection.

        Raises:
            TypeError: If the provided client is not valid.

        .. code-block:: python

            import redis.asyncio as aredis
            from redisvl.index import AsyncSearchIndex

            # async Redis client and index
            client = aredis.Redis.from_url("redis://localhost:6379")
            index = AsyncSearchIndex.from_yaml("schemas/schema.yaml")
            index.set_client(client)

        """
        if not isinstance(client, aredis.Redis):
            raise TypeError("Invalid Redis client instance")

        self._redis_client = client

        return self

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
            await self._redis_client.ft(self.schema.index.name).create_index(  # type: ignore
                fields=redis_fields,
                definition=IndexDefinition(
                    prefix=[self.schema.index.prefix], index_type=self._storage.type
                ),
            )
        except:
            logger.exception("Error while trying to create the index")
            raise

    @check_async_index_exists()
    async def delete(self, drop: bool = True):
        """Delete the search index.

        Args:
            drop (bool, optional): Delete the documents in the index.
                Defaults to True.

        Raises:
            redis.exceptions.ResponseError: If the index does not exist.
        """
        try:
            await self._redis_client.ft(self.schema.index.name).dropindex(  # type: ignore
                delete_documents=drop
            )
        except:
            logger.exception("Error while deleting index")
            raise

    async def load(
        self,
        data: Iterable[Any],
        id_field: Optional[str] = None,
        keys: Optional[Iterable[str]] = None,
        ttl: Optional[int] = None,
        preprocess: Optional[Callable] = None,
        concurrency: Optional[int] = None,
    ) -> List[str]:
        """Asynchronously load objects to Redis with concurrency control.
        Returns the list of keys loaded to Redis.

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
            preprocess (Optional[Callable], optional): An async function to
                preprocess objects before storage. Defaults to None.
            concurrency (Optional[int], optional): The maximum number of
                concurrent write operations. Defaults to class's default
                concurrency level.

        Returns:
            List[str]: List of keys loaded to Redis.

        Raises:
            ValueError: If the length of provided keys does not match the
                length of objects.

        .. code-block:: python

            data = [{"test": "foo"}, {"test": "bar"}]

            # simple case
            keys = await index.load(data)

            # set 360 second ttl policy on data
            keys = await index.load(data, ttl=360)

            # load data with predefined keys
            keys = await index.load(data, keys=["rvl:foo", "rvl:bar"])

            # load data with preprocessing step
            async def add_field(d):
                d["new_field"] = 123
                return d
            keys = await index.load(data, preprocess=add_field)

        """
        try:
            return await self._storage.awrite(
                self._redis_client,  # type: ignore
                objects=data,
                id_field=id_field,
                keys=keys,
                ttl=ttl,
                preprocess=preprocess,
                concurrency=concurrency,
            )
        except:
            logger.exception("Error while loading data to Redis")
            raise

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
        obj = await self._storage.aget(self._redis_client, [self.key(id)])  # type: ignore
        if obj:
            return convert_bytes(obj[0])
        return None

    @check_async_index_exists()
    async def search(self, *args, **kwargs) -> "Result":
        """Perform a search on this index.

        Wrapper around redis.search.Search that adds the index name
        to the search query and passes along the rest of the arguments
        to the redis-py ft.search() method.

        Returns:
            Result: Raw Redis search results.
        """
        try:
            return await self._redis_client.ft(self.schema.index.name).search(  # type: ignore
                *args, **kwargs
            )
        except:
            logger.exception("Error while searching")
            raise

    async def _query(self, query: BaseQuery) -> List[Dict[str, Any]]:
        """Asynchronously execute a query and process results."""
        results = await self.search(query.query, query_params=query.params)
        return process_results(
            results, query=query, storage_type=self.schema.index.storage_type
        )

    async def query(self, query: BaseQuery) -> List[Dict[str, Any]]:
        """Asynchronously execute a query on the index.

        This method takes a BaseQuery object directly, runs the search, and
        handles post-processing of the search.

        Args:
            query (BaseQuery): The query to run.

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
            raise TypeError("page_size must be an integer")

        if page_size <= 0:
            raise ValueError("page_size must be greater than 0")

        first = 0
        while True:
            query.set_paging(first, page_size)
            results = await self._query(query)
            if not results:
                break
            yield results
            # increment the pagination tracker
            first += page_size

    async def listall(self) -> List[str]:
        """List all search indices in Redis database.

        Returns:
            List[str]: The list of indices in the database.
        """
        return convert_bytes(
            await self._redis_client.execute_command("FT._LIST")  # type: ignore
        )

    async def exists(self) -> bool:
        """Check if the index exists in Redis.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        return self.schema.index.name in await self.listall()

    @staticmethod
    async def _info(name: str, redis_client: aredis.Redis) -> Dict[str, Any]:
        try:
            return convert_bytes(await redis_client.ft(name).info())  # type: ignore
        except:
            logger.exception(f"Error while fetching {name} index info")
            raise

    @check_async_index_exists()
    async def info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about the index.

        Args:
            name (str, optional): Index name to fetch info about.
                Defaults to None.

        Returns:
            dict: A dictionary containing the information about the index.
        """
        index_name = name or self.schema.index.name
        return await self._info(index_name, self._redis_client)  # type: ignore
