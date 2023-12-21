import json
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Union

if TYPE_CHECKING:
    from redis.commands.search.document import Document
    from redis.commands.search.result import Result
    from redisvl.query.query import BaseQuery

import redis
import redis.asyncio as aredis
from redis.commands.search.indexDefinition import IndexDefinition

from redisvl.query.query import BaseQuery, CountQuery, FilterQuery
from redisvl.schema import IndexSchema, StorageType
from redisvl.storage import HashStorage, JsonStorage
from redisvl.utils.connection import RedisConnection
from redisvl.utils.utils import (
    check_async_redis_modules_exist,
    check_redis_modules_exist,
    convert_bytes,
)


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


def check_modules_present(client_variable_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            _redis_conn = getattr(self, client_variable_name)
            check_redis_modules_exist(_redis_conn.client)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def check_async_modules_present(client_variable_name: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            _redis_conn = getattr(self, client_variable_name)
            await check_async_redis_modules_exist(_redis_conn.client)
            return await func(self, *args, **kwargs)

        return wrapper

    return decorator


def check_index_exists():
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.exists():
                raise ValueError(
                    f"Index has not been created. Must be created before calling {func.__name__}"
                )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def check_async_index_exists():
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not await self.aexists():
                raise ValueError(
                    f"Index has not been created. Must be created before calling {func.__name__}"
                )
            return await func(self, *args, **kwargs)

        return wrapper

    return decorator


class SearchIndex:
    """A class for interacting with Redis as a vector database.

    This class is a wrapper around the redis-py client that provides
    purpose-built methods for interacting with Redis as a vector database.

    Example:
        >>> from redisvl.index import SearchIndex
        >>> index = SearchIndex.from_yaml("schema.yaml", redis_url="redis://localhost:6379")
        >>> index.create(overwrite=True)
        >>> index.load(data) # data is an iterable of dictionaries
        >>>
        >>> # Use an async connection
        >>> index = SearchIndex.from_yaml("schema.yaml", redis_url="redis://localhost:6379", use_async=True)
        >>> await index.acreate(overwrite=True)
        >>> await index.aload(data)
    """

    _STORAGE_MAP = {
        StorageType.HASH: HashStorage,
        StorageType.JSON: JsonStorage,
    }

    def __init__(
        self,
        schema: IndexSchema,
        redis_url: Optional[str] = None,
        redis_client: Optional[Union[redis.Redis, aredis.Redis]] = None,
        connection_args: Dict[str, Any] = {},
        **kwargs,
    ):
        """Initialize the RedisVL search index class with a schema,
        redis_url, connection_args, and other kwargs.
        """
        # final validation on schema object
        if not schema or not isinstance(schema, IndexSchema):
            raise ValueError("Must provide a valid schema object")

        # set up redis connection
        self._redis_conn = RedisConnection()
        if redis_client is not None:
            self.set_client(redis_client)
        elif redis_url is not None:
            self.connect(redis_url, **kwargs, **connection_args)

        self.schema = schema

        self._storage = self._STORAGE_MAP[self.schema.storage_type](
            self.schema.prefix, self.schema.key_separator
        )

    @property
    def name(self) -> str:
        """The name of the Redis search index."""
        return self.schema.name

    @property
    def prefix(self) -> str:
        """The optional key prefix that comes before a unique key value in
        forming a Redis key."""
        return self.schema.prefix

    @property
    def key_separator(self) -> str:
        """The optional separator between a defined prefix and key value in
        forming a Redis key."""
        return self.schema.key_separator

    @property
    def storage_type(self) -> StorageType:
        """The underlying storage type for the search index: hash or json."""
        return self.schema.storage_type

    @property
    def client(self) -> Optional[Union[redis.Redis, aredis.Redis]]:
        """The underlying redis-py client object."""
        return self._redis_conn.client

    @classmethod
    def from_existing(cls):
        raise DeprecationWarning(
            "This method is deprecated since 0.0.5. Use the from_yaml or from_dict constructors with an IndexSchema instead."
        )

    @classmethod
    def from_yaml(
        cls, schema_path: str, connection_args: Dict[str, Any] = {}, **kwargs
    ):
        """Create a SearchIndex from a YAML schema file.

        Args:
            schema_path (str): Path to the YAML schema file.
            connection_args (Dict[str, Any], optional): Redis client connection
                args.

        Example:
            >>> from redisvl.index import SearchIndex
            >>> index = SearchIndex.from_yaml("schema.yaml", redis_url="redis://localhost:6379")
            >>> index.create(overwrite=True)

        Returns:
            SearchIndex: A RedisVL SearchIndex object.
        """
        schema = IndexSchema.from_yaml(schema_path)
        return cls(schema=schema, connection_args=connection_args, **kwargs)

    @classmethod
    def from_dict(
        cls, schema_dict: Dict[str, Any], connection_args: Dict[str, Any] = {}, **kwargs
    ):
        """Create a SearchIndex from a dictionary.

        Args:
            schema_dict (Dict[str, Any]): A dictionary containing the schema.
            connection_args (Dict[str, Any], optional): Redis client connection
                args.

        Example:
            >>> from redisvl.index import SearchIndex
            >>> index = SearchIndex.from_dict({
            >>>     "index": {
            >>>         "name": "my-index",
            >>>         "prefix": "rvl",
            >>>         "storage_type": "hash",
            >>>     },
            >>>     "fields": {
            >>>         "tag": [{"name": "doc-id"}]
            >>>     }
            >>> }, redis_url="redis://localhost:6379")
            >>> index.create(overwrite=True)

        Returns:
            SearchIndex: A RedisVL SearchIndex object.
        """
        schema = IndexSchema.from_dict(schema_dict)
        return cls(schema=schema, connection_args=connection_args, **kwargs)

    def connect(
        self, redis_url: Optional[str] = None, use_async: bool = False, **kwargs
    ):
        """Connect to a Redis instance.

        This method establishes a connection to a Redis server. If `redis_url`
        is provided, it will be used as the connection endpoint. Otherwise, the
        method attempts to use the `REDIS_URL` environment variable as the
        connection URL. The `use_async` parameter determines whether the
        connection should be asynchronous.

        Note: Additional keyword arguments (`**kwargs`) can be used to provide
        extra options specific to the Redis connection.

        Args:
            redis_url (Optional[str], optional): The URL of the Redis server to
                connect to. If not provided, the method defaults to using the
                `REDIS_URL` environment variable.
            use_async (bool): If `True`, establishes a connection with an async
                Redis client. Defaults to `False`.

        Example:
            >>> # standard sync Redis connection
            >>> index.connect(redis_url="redis://localhost:6379")
            >>> # async Redis connection
            >>> index.connect(redis_url="redis://localhost:6379", use_async=True)

        Raises:
            redis.exceptions.ConnectionError: If the connection to the Redis
                server fails.
            ValueError: If the Redis URL is not provided nor accessible
                through the `REDIS_URL` environment variable.
        """
        self._redis_conn.connect(redis_url, use_async, **kwargs)
        return self

    def disconnect(self):
        """Reset the Redis connection."""
        self._redis_conn = RedisConnection()
        return self

    def set_client(self, client: Union[redis.Redis, aredis.Redis]):
        """Manually set the Redis client to use with the search index.

        This method configures the search index to use a specific Redis or
        Async Redis client. It is useful for cases where an external,
        custom-configured client is preferred instead of creating a new one.

        Args:
            client (Union[redis.Redis, aredis.Redis]): A Redis or Async Redis
                client instance to be used for the connection.

        Example:
            >>> import redis
            >>> r = redis.Redis.from_url("redis://localhost:6379")
            >>> index.set_client(r)
            >>> # async Redis client
            >>> import redis.asyncio as aredis
            >>> r = aredis.Redis.from_url("redis://localhost:6379")
            >>> index.set_client(r)


        Raises:
            TypeError: If the provided client is not valid.
        """
        self._redis_conn.set_client(client)
        return self

    def key(self, id: str) -> str:
        """Create a redis key as a combination of an index key prefix (optional)
        and specified id. The id is typically either a unique identifier, or
        derived from some domain-specific metadata combination (like a document
        id or chunk id).

        Args:
            id (str): The specified unique identifier for a particular
                document indexed in Redis.

        Returns:
            str: The full Redis key including key prefix and value as a string.
        """
        return self._storage._key(id, self.schema.prefix, self.schema.key_separator)

    @check_modules_present("_redis_conn")
    def create(self, overwrite: bool = False) -> None:
        """Create an index in Redis from this SearchIndex object.

        Args:
            overwrite (bool, optional): Whether to overwrite the index if it
                already exists. Defaults to False.

        Raises:
            RuntimeError: If the index already exists and 'overwrite' is False.
            ValueError: If no fields are defined for the index.
        """
        # Check that fields are defined.
        redis_fields = self.schema.redis_fields
        if not redis_fields:
            raise ValueError("No fields defined for index")
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be of type bool")

        if self.exists():
            if not overwrite:
                print("Index already exists, not overwriting.")
                return None
            print("Index already exists, overwriting.")
            self.delete()

        # Create the index with the specified fields and settings.
        self._redis_conn.client.ft(self.name).create_index(  # type: ignore
            fields=redis_fields,
            definition=IndexDefinition(
                prefix=[self.prefix], index_type=self._storage.type
            ),
        )

    @check_modules_present("_redis_conn")
    @check_index_exists()
    def delete(self, drop: bool = True):
        """Delete the search index.

        Args:
            drop (bool, optional): Delete the documents in the index.
                Defaults to True.

        raises:
            redis.exceptions.ResponseError: If the index does not exist.
        """
        # Delete the search index
        self._redis_conn.client.ft(self.name).dropindex(delete_documents=drop)  # type: ignore

    @check_modules_present("_redis_conn")
    def load(
        self,
        data: Iterable[Any],
        key_field: Optional[str] = None,
        keys: Optional[Iterable[str]] = None,
        ttl: Optional[int] = None,
        preprocess: Optional[Callable] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """
        Load a batch of objects to Redis. Returns the list of keys loaded
        to Redis.

        Args:
            data (Iterable[Any]): An iterable of objects to store.
            key_field (Optional[str], optional): Field used as the key for each
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

        Example:
            >>> keys = index.load([{"test": "foo"}, {"test": "bar"}])
        """
        return self._storage.write(
            self._redis_conn.client,  # type: ignore
            objects=data,
            key_field=key_field,
            keys=keys,
            ttl=ttl,
            preprocess=preprocess,
            batch_size=batch_size,
        )

    def fetch(self, id: str) -> Dict[str, Any]:
        """
        Fetch an object from Redis by id. The id is typically either a
        unique identifier, or derived from some domain-specific metadata
        combination (like a document id or chunk id).

        Args:
            id (str): The specified unique identifier for a particular
                document indexed in Redis.

        Returns:
            Dict[str, Any]: The fetched object.
        """
        return convert_bytes(self._redis_conn.client.hgetall(self.key(id)))  # type: ignore

    @check_modules_present("_redis_conn")
    @check_index_exists()
    def search(self, *args, **kwargs) -> Union["Result", Any]:
        """Perform a search on this index.

        Wrapper around redis.search.Search that adds the index name
        to the search query and passes along the rest of the arguments
        to the redis-py ft.search() method.

        Returns:
            Union["Result", Any]: Search results.
        """
        results = self._redis_conn.client.ft(self.name).search(  # type: ignore
            *args, **kwargs
        )
        return results

    @check_modules_present("_redis_conn")
    @check_index_exists()
    def query(self, query: "BaseQuery") -> List[Dict[str, Any]]:
        """Run a query on this index.

        This is similar to the search method, but takes a BaseQuery
        object directly (does not allow for the usage of a raw
        redis query string) and post-processes results of the search.

        Args:
            query (BaseQuery): The query to run.

        Returns:
            List[Result]: A list of search results.
        """
        results = self.search(query.query, query_params=query.params)
        # post process the results
        return process_results(
            results, query=query, storage_type=self.schema.storage_type
        )

    @check_modules_present("_redis_conn")
    @check_index_exists()
    def query_all(self, query: "BaseQuery", batch_size: int = 100):
        """Fetch all results for a given query in batches.

        Args:
            query (BaseQuery): The query to run.
            batch_size (int): Batch size for fetching results.

        Yields:
            List[Dict[str, Any]]: A batch of search results.
        """
        first = 0
        while True:
            query.set_paging(first, batch_size)
            batch_results = self.query(query)
            if not batch_results:
                break
            yield batch_results
            # increment the pagination tracker
            first += batch_size

    @check_modules_present("_redis_conn")
    def exists(self) -> bool:
        """Check if the index exists in Redis.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        indices = convert_bytes(self._redis_conn.client.execute_command("FT._LIST"))  # type: ignore
        return self.name in indices

    @check_modules_present("_redis_conn")
    @check_index_exists()
    def info(self) -> Dict[str, Any]:
        """Get information about the index.

        Returns:
            dict: A dictionary containing the information about the index.
        """
        return convert_bytes(
            self._redis_conn.client.ft(self.name).info()  # type: ignore
        )

    @check_async_modules_present("_redis_conn")
    async def acreate(self, overwrite: bool = False) -> None:
        """Asynchronously create an index in Redis from this SearchIndex object.

        Args:
            overwrite (bool, optional): Whether to overwrite the index if it
                already exists. Defaults to False.

        Raises:
            RuntimeError: If the index already exists and 'overwrite' is False.
        """
        redis_fields = self.schema.redis_fields
        if not redis_fields:
            raise ValueError("No fields defined for index")
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be of type bool")

        if await self.aexists():
            if not overwrite:
                print("Index already exists, not overwriting.")
                return None
            print("Index already exists, overwriting.")
            await self.adelete()

        # Create Index with proper IndexType
        await self._redis_conn.client.ft(self.name).create_index(  # type: ignore
            fields=redis_fields,
            definition=IndexDefinition(
                prefix=[self.prefix], index_type=self._storage.type
            ),
        )

    @check_async_modules_present("_redis_conn")
    @check_async_index_exists()
    async def adelete(self, drop: bool = True):
        """Delete the search index.

        Args:
            drop (bool, optional): Delete the documents in the index.
                Defaults to True.

        Raises:
            redis.exceptions.ResponseError: If the index does not exist.
        """
        # Delete the search index
        await self._redis_conn.client.ft(self.name).dropindex(delete_documents=drop)  # type: ignore

    @check_async_modules_present("_redis_conn")
    async def aload(
        self,
        data: Iterable[Any],
        key_field: Optional[str] = None,
        keys: Optional[Iterable[str]] = None,
        ttl: Optional[int] = None,
        preprocess: Optional[Callable] = None,
        concurrency: Optional[int] = None,
    ) -> List[str]:
        """
        Asynchronously load objects to Redis with concurrency control. Returns
        the list of keys loaded to Redis.

        Args:
            data (Iterable[Any]): An iterable of objects to store.
            key_field (Optional[str], optional): Field used as the key for each
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

        Example:
            >>> keys = await index.aload([{"test": "foo"}, {"test": "bar"}])
        """
        return await self._storage.awrite(
            self._redis_conn.client,  # type: ignore
            objects=data,
            key_field=key_field,
            keys=keys,
            ttl=ttl,
            preprocess=preprocess,
            concurrency=concurrency,
        )

    async def afetch(self, id: str) -> Dict[str, Any]:
        """
        Asynchronously etch an object from Redis by id. The id is typically
        either a unique identifier, or derived from some domain-specific
        metadata
        combination (like a document id or chunk id).

        Args:
            id (str): The specified unique identifier for a particular
                document indexed in Redis.

        Returns:
            Dict[str, Any]: The fetched object.
        """
        return convert_bytes(await self._redis_conn.client.hgetall(self.key(id)))  # type: ignore

    @check_async_modules_present("_redis_conn")
    @check_async_index_exists()
    async def asearch(self, *args, **kwargs) -> Union["Result", Any]:
        """Perform a search on this index.

        Wrapper around redis.search.Search that adds the index name
        to the search query and passes along the rest of the arguments
        to the redis-py ft.search() method.

        Returns:
            Union["Result", Any]: Search results.
        """
        results = await self._redis_conn.client.ft(self.name).search(  # type: ignore
            *args, **kwargs
        )
        return results

    @check_async_modules_present("_redis_conn")
    @check_async_index_exists()
    async def aquery(self, query: "BaseQuery") -> List[Dict[str, Any]]:
        """Run a query on this index.

        This is similar to the search method, but takes a BaseQuery
        object directly (does not allow for the usage of a raw
        redis query string) and post-processes results of the search.

        Args:
            query (BaseQuery): The query to run.

        Returns:
            List[Result]: A list of search results.
        """
        results = await self.asearch(query.query, query_params=query.params)
        # post process the results
        return process_results(
            results, query=query, storage_type=self.schema.storage_type
        )

    @check_async_modules_present("_redis_conn")
    async def aexists(self) -> bool:
        """Check if the index exists in Redis.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        indices = await self._redis_conn.client.execute_command("FT._LIST")  # type: ignore
        return self.name in convert_bytes(indices)

    @check_async_modules_present("_redis_conn")
    @check_async_index_exists()
    async def ainfo(self) -> Dict[str, Any]:
        """Get information about the index.

        Returns:
            dict: A dictionary containing the information about the index.
        """
        return convert_bytes(
            await self._redis_conn.client.ft(self.name).info()  # type: ignore
        )
