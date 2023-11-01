import asyncio
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Union
from uuid import uuid4

if TYPE_CHECKING:
    from redis.commands.search.field import Field
    from redis.commands.search.result import Result
    from redisvl.query.query import BaseQuery

import redis
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from redisvl.query.query import CountQuery
from redisvl.schema import SchemaModel, Storage, read_schema
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
        prefix: str = "rvl",
        storage_type: str = "hash",
        fields: Optional[List["Field"]] = None,
    ):
        self._name = name
        self._prefix = prefix
        self._fields = fields
        self._storage = self._validate_and_convert_storage(storage_type)
        self._redis_conn: Optional[redis.Redis] = None


    def _validate_and_convert_storage(self, storage: str) -> Storage:
        try:
            return Storage(storage.lower())
        except ValueError as e:
            raise e(f"Invalid storage type provided: {storage}. Allowed values are: 'hash', 'json'.")


    def set_client(self, client: redis.Redis):
        self._redis_conn = client

    @property
    @check_connected("_redis_conn")
    def client(self) -> redis.Redis:
        """The redis-py client object.

        Returns:
            redis.Redis: The redis-py client object
        """
        return self._redis_conn  # type: ignore

    @check_connected("_redis_conn")
    def search(self, *args, **kwargs) -> Union["Result", Any]:
        """Perform a search on this index.

        Wrapper around redis.search.Search that adds the index name
        to the search query and passes along the rest of the arguments
        to the redis-py ft.search() method.

        Returns:
            Union["Result", Any]: Search results.
        """
        results = self._redis_conn.ft(self._name).search(  # type: ignore
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
        raise NotImplementedError

    def connect(self, url: str, **kwargs):
        """Connect to a Redis instance.

        Args:
            url (str): Redis URL. REDIS_URL env var is used if not provided.
        """
        raise NotImplementedError

    def disconnect(self):
        """Disconnect from the Redis instance"""
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
        return f"{self._prefix}:{key_value}" if self._prefix else key_value

    def _create_key(
        self, record: Dict[str, Any], key_field: Optional[str] = None
    ) -> str:
        """Construct the Redis HASH top level key.

        Args:
            record (Dict[str, Any]): A dictionary containing the record to be indexed.
            key_field (Optional[str], optional): A field within the record
                to use in the Redis hash key.

        Returns:
            str: The key to be used for a given record in Redis.

        Raises:
            ValueError: If the key field is not found in the record.
        """
        if key_field is None:
            key_value = uuid4().hex
        else:
            try:
                key_value = record[key_field]  # type: ignore
            except KeyError:
                raise ValueError(f"Key field {key_field} not found in record {record}")
        return self.key(key_value)

    @check_connected("_redis_conn")
    def info(self) -> Dict[str, Any]:
        """Get information about the index.

        Returns:
            dict: A dictionary containing the information about the index.
        """
        return convert_bytes(self._redis_conn.ft(self._name).info())  # type: ignore

    def create(self, overwrite: Optional[bool] = False):
        """Create an index in Redis from this SearchIndex object.

        Args:
            overwrite (bool, optional): Overwrite the index if it already exists. Defaults to False.

        Raises:
            redis.exceptions.ResponseError: If the index already exists.
        """
        raise NotImplementedError

    def delete(self, drop: bool = True):
        """Delete the search index.

        Args:
            drop (bool, optional): Delete the documents in the index. Defaults to True.

        Raises:
            redis.exceptions.ResponseError: If the index does not exist.
        """
        raise NotImplementedError

    def load(
        self,
        data: Iterable[Dict[str, Any]],
        key_field: Optional[str] = None,
        preprocess: Optional[Callable] = None,
        **kwargs,
    ):
        """Load data into Redis and index using this SearchIndex object.

        Args:
            data (Iterable[Dict[str, Any]]): An iterable of dictionaries
                containing the data to be indexed.
            key_field (Optional[str], optional): A field within the record
                to use in the Redis hash key.
            preprocess (Optional[Callabl], optional): An optional preprocessor function
                that mutates the individual record before writing to redis.

        Raises:
            redis.exceptions.ResponseError: If the index does not exist.
        """
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
        name: str,
        prefix: str = "rvl",
        storage_type: str = "hash",
        fields: Optional[List["Field"]] = None,
    ):
        super().__init__(name, prefix, storage_type, fields)

    @classmethod
    def from_existing(
        cls,
        name: str,
        url: Optional[str] = None,
        fields: Optional[List["Field"]] = None,
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

        # Translate the internal storage type to the appropriate index type.
        index_type = IndexType.JSON if self._storage == Storage.JSON else IndexType.HASH

        # Create the index with the specified fields and settings.
        self._redis_conn.ft(self._name).create_index(  # type: ignore
            fields=self._fields,
            definition=IndexDefinition(prefix=[self._prefix], index_type=index_type),
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
        self._redis_conn.ft(self._name).dropindex(delete_documents=drop)  # type: ignore

    def _set(self, pipe, key: str, record: dict, ttl: Optional[int]) -> None:
        """
        Set the record in Redis using the appropriate storage mechanism.

        Args:
            pipe: Redis pipeline object.
            key (str): Key under which the record is stored.
            record (dict): The record to store.
            ttl (Optional[int]): Time to live for the key.

        Raises:
            ValueError: If an unexpected storage type is encountered.
        """
        if self._storage == Storage.HASH:
            pipe.hset(key, mapping=record)
        elif self._storage == Storage.JSON:
            pipe.json.set(key, path="$", obj=record)
        else:
            raise ValueError(f"Unexpected storage type: {self._storage}. Invalid storage type.")

        if ttl:
            pipe.expire(key, ttl)


    def _preprocess(self, preprocess: Callable, record: dict) -> dict:
        """
        Preprocess the record using a custom function.

        Args:
            preprocess (Callable): The function to apply to the record.
            record (dict): The record to preprocess.

        Returns:
            dict: The preprocessed record.

        Raises:
            RuntimeError: If an error occurs during preprocessing.
            TypeError: If the preprocessed record is not a dictionary.
        """
        try:
            record = preprocess(record)
        except Exception as e:
            raise RuntimeError("Error while preprocessing records on load") from e

        if not isinstance(record, dict):
            raise TypeError(f"Preprocessed records must be of type dict, got {type(record)}")

        return record

    @check_connected("_redis_conn")
    def load(
        self,
        data: Iterable[Dict[str, Any]],
        key_field: Optional[str] = None,
        preprocess: Optional[Callable] = None,
        batch_size: int = 300,
        **kwargs,
    ) -> None:
        """
        Load data into Redis and index using this SearchIndex object.

        Args:
            data (Iterable[Dict[str, Any]]): An iterable of dictionaries to be indexed.
            key_field (Optional[str], optional): A field within the record to use as the Redis hash key.
            preprocess (Optional[Callable], optional): An optional function to modify records before writing to Redis.
            batch_size (int, optional): The size of batches to use for writes to Redis db.

        Raises:
            TypeError: If data is not a non-empty iterable or does not contain dictionaries.
            redis.exceptions.ResponseError: If the index does not exist.

        Example:
            >>> data = [{"foo": "bar"}, {"test": "values"}]
            >>> def func(record: dict): record["new"] = "value"; return record
            >>> index.load(data, preprocess=func)
        """
        if not isinstance(data, Iterable) or not data:
            raise TypeError("data must be a non-empty iterable")
        if not isinstance(next(iter(data)), dict):
            raise TypeError("data must contain dictionaries")

        ttl = kwargs.get("ttl")

        with self._redis_conn.pipeline(transaction=False) as pipe:
            for i, record in enumerate(data, start=1):
                key = self._create_key(record, key_field)
                if preprocess:
                    record = self._preprocess(preprocess, record)
                self._set(pipe, key, record, ttl)

                # execute mini batches
                if i % batch_size == 0:
                    pipe.execute()
            # final batch cleanup
            pipe.execute()

    @check_connected("_redis_conn")
    def exists(self) -> bool:
        """Check if the index exists in Redis.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        indices = convert_bytes(self._redis_conn.execute_command("FT._LIST"))  # type: ignore
        return self._name in indices


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
        name: str,
        prefix: str = "rvl",
        storage_type: str = "hash",
        fields: Optional[List["Field"]] = None,
    ):
        super().__init__(name, prefix, storage_type, fields)

    @classmethod
    async def from_existing(
        cls,
        name: str,
        url: Optional[str] = None,
        fields: Optional[List["Field"]] = None,
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

        # Translate the internal storage type to the appropriate index type.
        index_type = IndexType.JSON if self._storage == Storage.JSON else IndexType.HASH

        # Create Index with proper IndexType
        await self._redis_conn.ft(self._name).create_index(  # type: ignore
            fields=self._fields,
            definition=IndexDefinition(prefix=[self._prefix], index_type=index_type),
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
        await self._redis_conn.ft(self._name).dropindex(delete_documents=drop)  # type: ignore

    async def _set(self, key: str, record: dict, ttl: Optional[int]) -> None:
        """
        Asynchronously set the record in Redis using the appropriate storage mechanism.

        Args:
            key (str): Key under which the record is stored.
            record (dict): The record to store.
            ttl (Optional[int]): Time to live for the key.

        Raises:
            ValueError: If an unexpected storage type is encountered.
        """
        if self._storage == Storage.HASH:
            await self._redis_conn.hset(key, mapping=record)
        elif self._storage == Storage.JSON:
            await self._redis_conn.json.set(key, path="$", obj=record)
        else:
            raise ValueError(f"Unexpected storage type: {self._storage}. Invalid storage type.")

        if ttl:
            await self._redis_conn.expire(key, ttl)

    async def _preprocess(self, preprocess: Callable, record: dict) -> dict:
        """
        Asynchronously preprocess the record using a custom function.

        Args:
            preprocess (Callable): The function to apply to the record.
            record (dict): The record to preprocess.

        Returns:
            dict: The preprocessed record.

        Raises:
            RuntimeError: If an error occurs during preprocessing.
            TypeError: If the preprocessed record is not a dictionary.
        """
        try:
            record = preprocess(record)
        except Exception as e:
            raise RuntimeError("Error while preprocessing records on load") from e

        if not isinstance(record, dict):
            raise TypeError(f"Preprocessed records must be of type dict, got {type(record)}")

        return record

    @check_connected("_redis_conn")
    async def load(
        self,
        data: Iterable[Dict[str, Any]],
        concurrency: int = 10,
        key_field: Optional[str] = None,
        preprocess: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        """
        Asynchronously load data into Redis and index using this SearchIndex object.

        Args:
            data (Iterable[Dict[str, Any]]): An iterable of dictionaries to be indexed.
            concurrency (int, optional): Number of concurrent tasks to run. Defaults to 10.
            key_field (Optional[str], optional): A field within the record to use as the Redis hash key.
            preprocess (Optional[Callable], optional): An optional function to modify records before writing to Redis.

        Raises:
            TypeError: If data is not a non-empty iterable or does not contain dictionaries.
            redis.exceptions.ResponseError: If the index does not exist.

        Example:
            >>> data = [{"foo": "bar"}, {"test": "values"}]
            >>> async def func(record: dict): record["new"] = "value"; return record
            >>> await index.load(data, preprocess=func)
        """
        if not isinstance(data, Iterable) or not data:
            raise TypeError("data must be a non-empty iterable")
        if not isinstance(next(iter(data)), dict):
            raise TypeError("data must contain dictionaries")

        ttl = kwargs.get("ttl")
        semaphore = asyncio.Semaphore(concurrency)

        async def _load(record: Dict[str, Any]) -> None:
            async with semaphore:
                key = self._create_key(record, key_field)
                if preprocess:
                    record = await self._preprocess(preprocess, record)
                await self._set(key, record, ttl)

        tasks = [_load(record) for record in data]

        if tasks:
            await asyncio.gather(*tasks)

    @check_connected("_redis_conn")
    async def search(self, *args, **kwargs) -> Union["Result", Any]:
        """Perform a search on this index.

        Wrapper around redis.search.Search that adds the index name
        to the search query and passes along the rest of the arguments
        to the redis-py ft.search() method.

        Returns:
            Union["Result", Any]: Search results.
        """
        results = await self._redis_conn.ft(self._name).search(  # type: ignore
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
        return self._name in convert_bytes(indices)
