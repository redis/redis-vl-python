from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Union

if TYPE_CHECKING:
    from redis.commands.search.field import Field
    from redis.commands.search.result import Result
    from redisvl.query.query import BaseQuery

import redis
from redis.commands.search.indexDefinition import IndexDefinition

from redisvl.query.query import CountQuery
from redisvl.schema import SchemaModel, IndexModel, FieldsModel, StorageType, read_schema
from redisvl.storage import HashStorage, JsonStorage, BaseStorage
from redisvl.utils.connection import (
    check_connected,
    get_async_redis_connection,
    get_redis_connection,
)
from redisvl.utils.utils import (
    check_redis_modules_exist,
    convert_bytes,
    make_dict,
    process_results,
)


class SearchIndexBase:
    def __init__(
        self,
        name: str,
        prefix: str,
        storage_type: str,
        key_separator: str,
        fields: Optional[List["Field"]] = None,
    ):
        self._redis_conn: Optional[redis.Redis] = None
        # configure index and storage specs
        self._index = IndexModel(
            name=name,
            prefix=prefix,
            storage_type=storage_type,
            key_separator=key_separator
        )
        self._storage = self._create_storage(self._index)
        self._fields = fields

    @staticmethod
    def _create_storage(index: IndexModel) -> BaseStorage:
        """Create and return a storage instance based on the provided storage type."""
        if index.storage_type == StorageType.HASH:
            return HashStorage.from_index_model(index)
        elif index.storage_type == StorageType.JSON:
            return JsonStorage.from_index_model(index)
        else:
            raise ValueError(f"Invalid storage type provided: {index.storage_type}")

    def set_client(self, client: redis.Redis):
        self._redis_conn = client

    @property
    def storage(self) -> BaseStorage:
        return self._storage

    @property
    def storage_type(self) -> str:
        return str(self._index.storage_type)

    @property
    def name(self) -> str:
        return self._index.name

    @property
    def prefix(self) -> str:
        return self._index.prefix

    @property
    def key_separator(self) -> str:
        return self._index.key_separator

    @property
    @check_connected("_redis_conn")
    def client(self) -> redis.Redis:
        """The redis-py client object.

        Returns:
            redis.Redis: The redis-py client object
        """
        return self._redis_conn  # type: ignore

    @classmethod
    def from_yaml(cls, schema_path: str):
        """Create a SearchIndex from a YAML schema file.

        Args:
            schema_path (str): Path to the YAML schema file.

        Returns:
            SearchIndex: A SearchIndex object.
        """
        schema = read_schema(schema_path)
        return cls(fields=schema.index_fields, **schema.index.dict())

    @classmethod
    def from_dict(cls, schema_dict: Dict[str, Any]):
        """Create a SearchIndex from a dictionary.

        Args:
            schema_dict (Dict[str, Any]): A dictionary containing the schema.

        Returns:
            SearchIndex: A SearchIndex object.
        """
        schema = SchemaModel(**schema_dict)
        return cls(fields=schema.index_fields, **schema.index.dict())

    @classmethod
    def from_existing(
        cls,
        name: str,
        url: Optional[str] = None,
        fields: Optional[List["Field"]] = None,
        **kwargs,
    ):
        raise NotImplementedError

    @check_connected("_redis_conn")
    def search(self, *args, **kwargs) -> Union["Result", Any]:
        raise NotImplementedError

    @check_connected("_redis_conn")
    def query(self, query: "BaseQuery") -> List[Dict[str, Any]]:
        raise NotImplementedError

    def connect(self, url: str, **kwargs):
        """Connect to a Redis instance."""
        raise NotImplementedError

    def disconnect(self):
        """Disconnect from the Redis instance."""
        self._redis_conn = None
        return self

    def key(self, key_value: str) -> str:
        """
        Create a redis key as a combination of an index key prefix (optional) and specified key value.
        The key value is typically a unique identifier, created at random, or derived from
        some specified metadata.

        Args:
            key_value (str): The specified unique identifier for a particular document
                             indexed in Redis.

        Returns:
            str: The full Redis key including key prefix and value as a string.
        """
        return self._storage._key(key_value)

    @check_connected("_redis_conn")
    def info(self) -> Dict[str, Any]:
        raise NotImplementedError

    def create(self, overwrite: Optional[bool] = False):
        raise NotImplementedError

    def delete(self, drop: bool = True):
        raise NotImplementedError

    def load(
        self,
        data: Iterable[Dict[str, Any]],
        key_field: Optional[str] = None,
        preprocess: Optional[Callable] = None,
        **kwargs,
    ):
        raise NotImplementedError


class SearchIndex(SearchIndexBase):
    """A class for interacting with Redis as a vector database.

    This class is a wrapper around the redis-py client that provides
    purpose-built methods for interacting with Redis as a vector database.

    Example:
        >>> from redisvl.index import SearchIndex
        >>> index = SearchIndex.from_yaml("schema.yaml")
        >>> index.create(overwrite=True)
        >>> index.load(data) # data is an iterable of dictionaries
    """

    def __init__(
        self,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
        storage_type: Optional[str] = None,
        key_separator: Optional[str] = None,
        fields: Optional[List["Field"]] = None,
    ):
        super().__init__(name, prefix, storage_type, key_separator, fields)

    @classmethod
    def from_existing(
        cls,
        name: str,
        url: Optional[str] = None,
        fields: Optional[List["Field"]] = None,
        key_separator: Optional[str] = None,
        **kwargs,
    ):
        """Create a SearchIndex from an existing index in Redis.

        Args:
            name (str): Index name.
            url (Optional[str], optional): Redis URL. REDIS_URL env var
                is used if not provided. Defaults to None.
            fields (Optional[List[Field]], optional): List of Redis search
                fields to include in the schema. Defaults to None.

        Returns:
            SearchIndex: A SearchIndex object.

        Raises:
            redis.exceptions.ResponseError: If the index does not exist.
            ValueError: If the REDIS_URL env var is not set and url is not provided.

        """
        client = get_redis_connection(url, **kwargs)
        info = convert_bytes(client.ft(name).info())
        index_definition = make_dict(info["index_definition"])
        storage_type = index_definition["key_type"].lower()
        prefix = index_definition["prefixes"][0]
        instance = cls(
            name=name,
            storage_type=storage_type,
            prefix=prefix,
            key_separator=key_separator,
            fields=fields,
        )
        instance.set_client(client)
        return instance

    def connect(self, url: Optional[str] = None, **kwargs):
        """Connect to a Redis instance.

        Args:
            url (str): Redis URL. REDIS_URL env var is used if not provided.

        Raises:
            redis.exceptions.ConnectionError: If the connection to Redis fails.
            ValueError: If the REDIS_URL env var is not set and url is not provided.
        """
        self._redis_conn = get_redis_connection(url, **kwargs)
        return self

    @check_connected("_redis_conn")
    def create(self, overwrite: Optional[bool] = False) -> None:
        """
        Create an index in Redis from this SearchIndex object.

        Args:
            overwrite: Whether to overwrite the index if it already exists. Defaults to False.

        Raises:
            RuntimeError: If the index already exists and 'overwrite' is False.
            ValueError: If no fields are defined for the index.
        """
        # Ensure that the Redis connection has the necessary modules.
        check_redis_modules_exist(self._redis_conn)

        # Check that fields are defined.
        if not self._fields:
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
        self._redis_conn.ft(self._index.name).create_index(  # type: ignore
            fields=self._fields,
            definition=IndexDefinition(
                prefix=[self._index.prefix],
                index_type=self._storage.type
            ),
        )

    @check_connected("_redis_conn")
    def delete(self, drop: bool = True):
        """Delete the search index.

        Args:
            drop (bool, optional): Delete the documents in the index. Defaults to True.

        raises:
            redis.exceptions.ResponseError: If the index does not exist.
        """
        # Delete the search index
        self._redis_conn.ft(self._index.name).dropindex(delete_documents=drop)  # type: ignore

    @check_connected("_redis_conn")
    def load(
        self,
        data: Iterable[Any],
        key_field: Optional[str] = None,
        keys: Optional[Iterable[str]] = None,
        ttl: Optional[int] = None,
        preprocess: Optional[Callable] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        """
        Load a batch of objects to Redis.

        Args:
            data (Iterable[Any]): An iterable of objects to store.
            key_field (Optional[str]): Field used as the key for each object. Defaults to None.
            keys (Optional[Iterable[str]]): Optional iterable of keys, must match the length of objects if provided.
            ttl (Optional[int]): Time-to-live in seconds for each key. Defaults to None.
            preprocess (Optional[Callable]): A function to preprocess objects before storage. Defaults to None.
            batch_size (Optional[int]): Number of objects to write in a single Redis pipeline execution. Defaults to class's default batch size.

        Raises:
            ValueError: If the length of provided keys does not match the length of objects.

        Example:
            >>> data = [{"foo": "bar"}, {"test": "values"}]
            >>> async def func(record: dict): record["new"] = "value"; return record
            >>> index.load(data, preprocess=func)
        """
        self._storage.write(
            self.client,
            objects=data,
            key_field=key_field,
            keys=keys,
            ttl=ttl,
            preprocess=preprocess,
            batch_size=batch_size,
        )

    @check_connected("_redis_conn")
    def search(self, *args, **kwargs) -> Union["Result", Any]:
        """Perform a search on this index.

        Wrapper around redis.search.Search that adds the index name
        to the search query and passes along the rest of the arguments
        to the redis-py ft.search() method.

        Returns:
            Union["Result", Any]: Search results.
        """
        results = self._redis_conn.ft(self._index.name).search(  # type: ignore
            *args, **kwargs
        )
        return results

    @check_connected("_redis_conn")
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
        if isinstance(query, CountQuery):
            return results.total
        return process_results(results)

    @check_connected("_redis_conn")
    def exists(self) -> bool:
        """Check if the index exists in Redis.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        indices = convert_bytes(self._redis_conn.execute_command("FT._LIST"))  # type: ignore
        return self._index.name in indices

    @check_connected("_redis_conn")
    def info(self) -> Dict[str, Any]:
        """Get information about the index.

        Returns:
            dict: A dictionary containing the information about the index.
        """
        return convert_bytes(
            self._redis_conn.ft(self._index.name).info() # type: ignore
        )


class AsyncSearchIndex(SearchIndexBase):
    """A class for interacting with Redis as a vector database asynchronously.

    This class is a wrapper around the redis-py client that provides
    purpose-built methods for interacting with Redis as a vector database.

    Example:
        >>> from redisvl.index import AsyncSearchIndex
        >>> index = AsyncSearchIndex.from_yaml("schema.yaml")
        >>> await index.create(overwrite=True)
        >>> await index.load(data) # data is an iterable of dictionaries
    """

    def __init__(
        self,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
        storage_type: Optional[str] = None,
        key_separator: Optional[str] = None,
        fields: Optional[List["Field"]] = None,
    ):
        super().__init__(name, prefix, storage_type, key_separator, fields)

    @classmethod
    async def from_existing(
        cls,
        name: str,
        url: Optional[str] = None,
        fields: Optional[List["Field"]] = None,
        key_separator: Optional[str] = None,
        **kwargs,
    ):
        """Create a SearchIndex from an existing index in Redis.

        Args:
            name (str): Index name.
            url (Optional[str], optional): Redis URL. REDIS_URL env var
                is used if not provided. Defaults to None.
            fields (Optional[List[Field]], optional): List of Redis search
                fields to include in the schema. Defaults to None.

        Returns:
            SearchIndex: A SearchIndex object.

        Raises:
            redis.exceptions.ResponseError: If the index does not exist.
            ValueError: If the REDIS_URL env var is not set and url is not provided.

        """
        client = get_async_redis_connection(url, **kwargs)
        info = convert_bytes(await client.ft(name).info())
        index_definition = make_dict(info["index_definition"])
        storage_type = index_definition["key_type"].lower()
        prefix = index_definition["prefixes"][0]
        instance = cls(
            name=name,
            storage_type=storage_type,
            prefix=prefix,
            key_separator=key_separator,
            fields=fields,
        )
        instance.set_client(client)
        return instance

    def connect(self, url: Optional[str] = None, **kwargs):
        """Connect to a Redis instance.

        Args:
            url (str): Redis URL. REDIS_URL env var is used if not provided.

        Raises:
            redis.exceptions.ConnectionError: If the connection to Redis fails.
            ValueError: If no Redis URL is provided and REDIS_URL env var is not set.
        """
        self._redis_conn = get_async_redis_connection(url, **kwargs)
        return self

    @check_connected("_redis_conn")
    async def create(self, overwrite: Optional[bool] = False) -> None:
        """
        Asynchronously create an index in Redis from this SearchIndex object.

        Args:
            overwrite: Whether to overwrite the index if it already exists. Defaults to False.

        Raises:
            RuntimeError: If the index already exists and 'overwrite' is False.
        """
        # TODO - enable async version of this
        # check_redis_modules_exist(self._redis_conn)

        if not self._fields:
            raise ValueError("No fields defined for index")
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be of type bool")

        if await self.exists():
            if not overwrite:
                print("Index already exists, not overwriting.")
                return None
            print("Index already exists, overwriting.")
            await self.delete()

        # Create Index with proper IndexType
        await self._redis_conn.ft(self._index.name).create_index(  # type: ignore
            fields=self._fields,
            definition=IndexDefinition(
                prefix=[self._index.prefix],
                index_type=self._storage.type
            ),
        )

    @check_connected("_redis_conn")
    async def delete(self, drop: bool = True):
        """Delete the search index.

        Args:
            drop (bool, optional): Delete the documents in the index. Defaults to True.

        Raises:
            redis.exceptions.ResponseError: If the index does not exist.
        """
        # Delete the search index
        await self._redis_conn.ft(self._index.name).dropindex(delete_documents=drop)  # type: ignore

    @check_connected("_redis_conn")
    async def load(
        self,
        data: Iterable[Any],
        key_field: Optional[str] = None,
        keys: Optional[Iterable[str]] = None,
        ttl: Optional[int] = None,
        preprocess: Optional[Callable] = None,
        concurrency: Optional[int] = None,
        **kwargs
    ):
        """
        Asynchronously load objects to Redis with concurrency control.

        Args:
            redis_client (AsyncRedis): An asynchronous Redis client used for writing data.
            objects (Iterable[Any]): An iterable of objects to store.
            key_field (Optional[str]): Field used as the key for each object. Defaults to None.
            keys (Optional[Iterable[str]]): Optional iterable of keys, must match the length of objects if provided.
            ttl (Optional[int]): Time-to-live in seconds for each key. Defaults to None.
            preprocess (Optional[Callable]): An async function to preprocess objects before storage. Defaults to None.
            concurrency (Optional[int]): The maximum number of concurrent write operations. Defaults to class's default concurrency level.

        Raises:
            ValueError: If the length of provided keys does not match the length of objects.

        Example:
            >>> data = [{"foo": "bar"}, {"test": "values"}]
            >>> async def func(record: dict): record["new"] = "value"; return record
            >>> await index.load(data, preprocess=func)
        """
        await self._storage.awrite(
            self.client,
            objects=data,
            key_field=key_field,
            keys=keys,
            ttl=ttl,
            preprocess=preprocess,
            concurrency=concurrency
        )

    @check_connected("_redis_conn")
    async def search(self, *args, **kwargs) -> Union["Result", Any]:
        """Perform a search on this index.

        Wrapper around redis.search.Search that adds the index name
        to the search query and passes along the rest of the arguments
        to the redis-py ft.search() method.

        Returns:
            Union["Result", Any]: Search results.
        """
        results = await self._redis_conn.ft(self._index.name).search(  # type: ignore
            *args, **kwargs
        )
        return results

    async def query(self, query: "BaseQuery") -> List[Dict[str, Any]]:
        """Run a query on this index.

        This is similar to the search method, but takes a BaseQuery
        object directly (does not allow for the usage of a raw
        redis query string) and post-processes results of the search.

        Args:
            query (BaseQuery): The query to run.

        Returns:
            List[Result]: A list of search results.
        """
        results = await self.search(query.query, query_params=query.params)
        if isinstance(query, CountQuery):
            return results.total
        return process_results(results)

    @check_connected("_redis_conn")
    async def exists(self) -> bool:
        """Check if the index exists in Redis.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        indices = await self._redis_conn.execute_command("FT._LIST")  # type: ignore
        return self._index.name in convert_bytes(indices)

    @check_connected("_redis_conn")
    async def info(self) -> Dict[str, Any]:
        """Get information about the index.

        Returns:
            dict: A dictionary containing the information about the index.
        """
        return convert_bytes(
            await self._redis_conn.ft(self._index.name).info() # type: ignore
        )
