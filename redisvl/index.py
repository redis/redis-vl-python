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
from redisvl.schema import SchemaModel, read_schema
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
        self._storage = storage_type
        self._fields = fields
        self._redis_conn: Optional[redis.Redis] = None

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

    @check_connected("_redis_conn")
    def create(self, overwrite: Optional[bool] = False):
        """Create an index in Redis from this SearchIndex object.

        Args:
            overwrite (bool, optional): Overwrite the index if it already exists. Defaults to False.

        Raises:
            redis.exceptions.ResponseError: If the index already exists.
        """
        check_redis_modules_exist(self._redis_conn)

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

        # set storage_type, default to hash
        storage_type = IndexType.HASH
        # TODO - enable JSON support
        # if self._storage.lower() == "json":
        #     storage_type = IndexType.JSON

        # Create Index
        # will raise correct response error if index already exists
        self._redis_conn.ft(self._name).create_index(  # type: ignore
            fields=self._fields,
            definition=IndexDefinition(prefix=[self._prefix], index_type=storage_type),
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

    @check_connected("_redis_conn")
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
            key_field (Optional[str], optional): A field within the record to
                use in the Redis hash key.
            preprocess (Optional[Callable], optional): An optional preprocessor function
                that mutates the individual record before writing to redis.

        raises:
            redis.exceptions.ResponseError: If the index does not exist.

        Example:
            >>> data = [{"foo": "bar"}, {"test": "values"}]
            >>> def func(record: dict): record["new"]="value";return record
            >>> index.load(data, preprocess=func)
        """
        # TODO -- should we return a count of the upserts? or some kind of metadata?
        if data:
            if not isinstance(data, Iterable):
                if not isinstance(data[0], dict):
                    raise TypeError("data must be an iterable of dictionaries")

            # Check if outer interface passes in TTL on load
            ttl = kwargs.get("ttl")
            with self._redis_conn.pipeline(transaction=False) as pipe:  # type: ignore
                for record in data:
                    key = self._create_key(record, key_field)
                    # Optionally preprocess the record and validate type
                    if preprocess:
                        try:
                            record = preprocess(record)
                        except Exception as e:
                            raise RuntimeError(
                                "Error while preprocessing records on load"
                            ) from e
                    if not isinstance(record, dict):
                        raise TypeError(
                            f"Individual records must be of type dict, got type {type(record)}"
                        )
                    # Write the record to Redis
                    pipe.hset(key, mapping=record)  # type: ignore
                    if ttl:
                        pipe.expire(key, ttl)
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

    @check_connected("_redis_conn")
    async def create(self, overwrite: Optional[bool] = False):
        """Create an index in Redis from this SearchIndex object.

        Args:
            overwrite (bool, optional): Overwrite the index if it already exists. Defaults to False.

        Raises:
            redis.exceptions.ResponseError: If the index already exists.
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

        # set storage_type, default to hash
        storage_type = IndexType.HASH
        # TODO - enable JSON support
        # if self._storage.lower() == "json":
        #     storage_type = IndexType.JSON

        # Create Index
        await self._redis_conn.ft(self._name).create_index(  # type: ignore
            fields=self._fields,
            definition=IndexDefinition(prefix=[self._prefix], index_type=storage_type),
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

    @check_connected("_redis_conn")
    async def load(
        self,
        data: Iterable[Dict[str, Any]],
        concurrency: int = 10,
        key_field: Optional[str] = None,
        preprocess: Optional[Callable] = None,
        **kwargs,
    ):
        """Load data into Redis and index using this SearchIndex object.

        Args:
            data (Iterable[Dict[str, Any]]): An iterable of dictionaries
                containing the data to be indexed.
            concurrency (int, optional): Number of concurrent tasks to run. Defaults to 10.
            key_field (Optional[str], optional): A field within the record to
                use in the Redis hash key.
            preprocess (Optional[Callable], optional): An optional preprocessor function
                that mutates the individual record before writing to redis.

        Raises:
            redis.exceptions.ResponseError: If the index does not exist.

        Example:
            >>> data = [{"foo": "bar"}, {"test": "values"}]
            >>> def func(record: dict): record["new"]="value";return record
            >>> await index.load(data, preprocess=func)
        """
        ttl = kwargs.get("ttl")
        semaphore = asyncio.Semaphore(concurrency)

        async def _load(record: dict):
            async with semaphore:
                key = self._create_key(record, key_field)
                # Optionally preprocess the record and validate type
                if preprocess:
                    try:
                        record = preprocess(record)
                    except Exception as e:
                        raise RuntimeError(
                            "Error while preprocessing records on load"
                        ) from e
                if not isinstance(record, dict):
                    raise TypeError(
                        f"Individual records must be of type dict, got type {type(record)}"
                    )
                # Write the record to Redis
                await self._redis_conn.hset(key, mapping=record)  # type: ignore
                if ttl:
                    await self._redis_conn.expire(key, ttl)  # type: ignore

        # Gather with concurrency
        await asyncio.gather(*[_load(record) for record in data])

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
