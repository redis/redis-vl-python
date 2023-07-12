import asyncio
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

if TYPE_CHECKING:
    from redis.commands.search.field import Field

import redis
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from redisvl.schema import SchemaModel, read_schema
from redisvl.utils.connection import (
    check_connected,
    get_async_redis_connection,
    get_redis_connection,
)
from redisvl.utils.utils import convert_bytes, make_dict


class SearchIndexBase:
    def __init__(
        self,
        name: str,
        storage_type: str = "hash",
        key_field: str = "id",
        prefix: str = "",
        fields: Optional[List["Field"]] = None,
    ):
        self._name = name
        self._key_field = key_field
        self._storage = storage_type
        self._prefix = prefix
        self._fields = fields
        self._redis_conn: Optional[redis.Redis] = None

    def set_client(self, client: redis.Redis):
        self._redis_conn = client

    @check_connected("_redis_conn")
    def get_client(self) -> redis.Redis:
        return self._redis_conn  # type: ignore

    @check_connected("_redis_conn")
    def search(self, *args, **kwargs):
        """Perform a search on this index

        Wrapper around redis.search.Search that adds the index name
        to the search query. This is a convenience method to avoid
        having to specify the index name in every search query.

        Returns:
            Search: A search object for this index
        """
        return self._redis_conn.ft(self._name).search(*args, **kwargs)  # type: ignore

    @classmethod
    def from_yaml(cls, schema_path: str):
        """Create a SearchIndex from a YAML schema file

        Args:
            schema_path (str): Path to the YAML schema file

        Returns:
            SearchIndex: A SearchIndex object
        """
        schema = read_schema(schema_path)
        return cls(fields=schema.index_fields, **schema.index.model_dump())

    @classmethod
    def from_dict(cls, schema_dict: Dict[str, Any]):
        """Create a SearchIndex from a dictionary

        Args:
            schema_dict (t.Dict[str, t.Any]): A dictionary containing the schema

        Returns:
            SearchIndex: A SearchIndex object
        """
        schema = SchemaModel(**schema_dict)
        return cls(fields=schema.index_fields, **schema.index.model_dump())

    @classmethod
    def from_existing(cls, client: redis.Redis, index_name: str):
        """Create a SearchIndex from an existing index in Redis"""
        # TODO assert client connected
        # TODO try/except
        info = convert_bytes(client.ft(index_name).info())  # TODO catch response error
        index_definition = make_dict(info["index_definition"])
        storage_type = index_definition["key_type"].lower()
        prefix = index_definition["prefixes"][0]
        fields = None  # TODO figure out if we need to parse fields
        instance = cls(
            index_name,
            key_field="",  # TODO check key field on load again
            storage_type=storage_type,
            prefix=prefix,
            fields=fields,
        )
        instance.set_client(client)
        return instance

    def connect(self, url: str, **kwargs):
        """Connect to a Redis instance

        Args:
            url (str): Redis URL. Defaults to "redis://localhost:6379".
        """
        raise NotImplementedError

    def disconnect(self):
        """Disconnect from the Redis instance"""
        self._redis_conn = None

    @check_connected("_redis_conn")
    def info(self) -> Dict[str, Any]:
        """Get information about the index

        Returns:
            dict: A dictionary containing the information about the index
        """
        return convert_bytes(self._redis_conn.ft(self._name).info())  # type: ignore

    def create(self):
        """Create an index in Redis from this SearchIndex object

        Raises:
            redis.exceptions.ResponseError: If the index already exists
        """
        raise NotImplementedError

    def delete(self, drop: bool = True):
        """Delete the search index

        Args:
            drop (bool, optional): Delete the documents in the index. Defaults to True.

        raises:
            redis.exceptions.ResponseError: If the index does not exist
        """
        raise NotImplementedError

    def load(self, data: Iterable[Dict[str, Any]], **kwargs):
        """Load data into Redis and index using this SearchIndex object

        Args:
            data (Iterable[Dict[str, Any]]): An iterable of dictionaries
                containing the data to be indexed
        raises:
            redis.exceptions.ResponseError: If the index does not exist
        """
        # TODO consider adding key_field
        raise NotImplementedError


class SearchIndex(SearchIndexBase):
    def __init__(
        self,
        name: str,
        storage_type: str = "hash",
        key_field: str = "id",
        prefix: str = "",
        fields: Optional[List["Field"]] = None,
    ):
        super().__init__(name, storage_type, key_field, prefix, fields)

    def connect(self, url: Optional[str] = None, **kwargs):
        """Connect to a Redis instance

        Args:
            url (str): Redis URL. REDIS_ADDRESS env var is used if not provided.


        Raises:
            redis.exceptions.ConnectionError: If the connection to Redis fails
            ValueError: If the REDIS_ADDRESS env var is not set and url is not provided
        """
        self._redis_conn = get_redis_connection(url, **kwargs)

    @check_connected("_redis_conn")
    def create(self):
        """Create an index in Redis from this SearchIndex object

        Raises:
            redis.exceptions.ResponseError: If the index already exists
        """
        # set storage_type, default to hash
        storage_type = IndexType.HASH
        if self._storage.lower() == "json":
            self._storage = IndexType.JSON

        # Create Index
        self._redis_conn.ft(self._name).create_index(
            fields=self._fields,
            definition=IndexDefinition(prefix=[self._prefix], index_type=storage_type),
        )  # type: ignore

    @check_connected("_redis_conn")
    def delete(self, drop: bool = True):
        """Delete the search index

        Args:
            drop (bool, optional): Delete the documents in the index. Defaults to True.

        raises:
            redis.exceptions.ResponseError: If the index does not exist
        """
        # Delete the search index
        self._redis_conn.ft(self._name).dropindex(delete_documents=drop)  # type: ignore

    @check_connected("_redis_conn")
    def load(self, data: Iterable[Dict[str, Any]], **kwargs):
        """Load data into Redis and index using this SearchIndex object

        Args:
            data (Iterable[Dict[str, Any]]): An iterable of dictionaries
                containing the data to be indexed
        raises:
            redis.exceptions.ResponseError: If the index does not exist
        """

        for record in data:
            key = f"{self._prefix}:{str(record[self._key_field])}"
            self._redis_conn.hset(key, mapping=record)  # type: ignore


class AsyncSearchIndex(SearchIndexBase):
    def __init__(
        self,
        name: str,
        storage_type: str = "hash",
        key_field: str = "id",
        prefix: str = "",
        fields: Optional[List["Field"]] = None,
    ):
        super().__init__(name, storage_type, key_field, prefix, fields)

    def connect(self, url: Optional[str] = None, **kwargs):
        """Connect to a Redis instance

        Args:
            url (str): Redis URL. REDIS_ADDRESS env var is used if not provided.

        Raises:
            redis.exceptions.ConnectionError: If the connection to Redis fails
            ValueError: If no Redis URL is provided and REDIS_ADDRESS env var is not set
        """
        self._redis_conn = get_async_redis_connection(url, **kwargs)

    @check_connected("_redis_conn")
    async def create(self):
        """Create an index in Redis from this SearchIndex object

        Raises:
            redis.exceptions.ResponseError: If the index already exists
        """
        # set storage_type, default to hash
        storage_type = IndexType.HASH
        if self._storage.lower() == "json":
            self._storage = IndexType.JSON

        # Create Index
        await self._redis_conn.ft(self._name).create_index(
            fields=self._fields,
            definition=IndexDefinition(prefix=[self._prefix], index_type=storage_type),
        )  # type: ignore

    @check_connected("_redis_conn")
    async def delete(self, drop: bool = True):
        """Delete the search index

        Args:
            drop (bool, optional): Delete the documents in the index. Defaults to True.

        raises:
            redis.exceptions.ResponseError: If the index does not exist
        """
        # Delete the search index
        await self._redis_conn.ft(self._name).dropindex(delete_documents=drop)  # type: ignore

    @check_connected("_redis_conn")
    async def load(self, data: Iterable[Dict[str, Any]], concurrency: int = 10):
        """Load data into Redis and index using this SearchIndex object

        Args:
            data (Iterable[Dict[str, Any]]): An iterable of dictionaries
            concurrency (int, optional): Number of concurrent tasks to run. Defaults to 10.

        raises:
            redis.exceptions.ResponseError: If the index does not exist
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def load(d: dict):
            async with semaphore:
                key = self._prefix + str(d[self._key_field])
                await self._redis_conn.hset(key, mapping=d)  # type: ignore

        # gather with concurrency
        await asyncio.gather(*[load(d) for d in data])
