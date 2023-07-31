import asyncio
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional
from uuid import uuid4

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
from redisvl.utils.utils import check_redis_modules_exist, convert_bytes, make_dict


class SearchIndexBase:
    def __init__(
        self,
        name: str,
        prefix: str = "rvl",
        storage_type: Optional[str] = "hash",
        key_field: Optional[str] = None,
        fields: Optional[List["Field"]] = None,
    ):
        self._name = name
        self._prefix = prefix
        self._storage = storage_type
        self._fields = fields
        self._redis_conn: Optional[redis.Redis] = None
        self._key_field = key_field

    def set_client(self, client: redis.Redis):
        self._redis_conn = client

    @property
    @check_connected("_redis_conn")
    def client(self) -> redis.Redis:
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

    def _get_key_field(self, record: Dict[str, Any]):
        """Get the key field for this index

        Args:
            record (Dict[str, Any]): A dictionary containing the record to be indexed

        Returns:
            str: The key to be used for a given record

        Raises:
            ValueError: If the key field is not found in the record
        """
        if self._key_field is None:
            return uuid4().hex
        else:
            try:
                return record[self._key_field]  # type: ignore
            except KeyError:
                raise ValueError(
                    f"Key field {self._key_field} not found in record {record}"
                )

    @check_connected("_redis_conn")
    def info(self) -> Dict[str, Any]:
        """Get information about the index

        Returns:
            dict: A dictionary containing the information about the index
        """
        return convert_bytes(self._redis_conn.ft(self._name).info())  # type: ignore

    def create(self, overwrite: Optional[bool] = False):
        """Create an index in Redis from this SearchIndex object

        Args:
            overwrite (bool, optional): Overwrite the index if it already exists. Defaults to False.

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
        prefix: str = "rvl",
        storage_type: Optional[str] = "hash",
        key_field: Optional[str] = None,
        fields: Optional[List["Field"]] = None,
    ):
        super().__init__(name, prefix, storage_type, key_field, fields)

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
    def create(self, overwrite: Optional[bool] = False):
        """Create an index in Redis from this SearchIndex object

        Args:
            overwrite (bool, optional): Overwrite the index if it already exists. Defaults to False.

        Raises:
            redis.exceptions.ResponseError: If the index already exists
        """
        check_redis_modules_exist(self._redis_conn)

        if not self._fields:
            raise ValueError("No fields defined for index")

        if self.exists() and overwrite:
            self.delete()

        # set storage_type, default to hash
        storage_type = IndexType.HASH
        if self._storage.lower() == "json":
            self._storage = IndexType.JSON

        # Create Index
        # will raise correct response error if index already exists
        self._redis_conn.ft(self._name).create_index(  # type: ignore
            fields=self._fields,
            definition=IndexDefinition(prefix=[self._prefix], index_type=storage_type),
        )

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
        # TODO -- should we return a count of the upserts? or some kind of metadata?
        """
        if data:
            if not isinstance(data, Iterable):
                if not isinstance(data[0], dict):
                    raise TypeError("data must be an iterable of dictionaries")

            pipe = self._redis_conn.pipeline(transaction=False)
            for record in data:
                key = f"{self._prefix}:{self._get_key_field(record)}"
                pipe.hset(key, mapping=record)  # type: ignore
            pipe.execute()

    @check_connected("_redis_conn")
    def exists(self) -> bool:
        """Check if the index exists in Redis

        Returns:
            bool: True if the index exists, False otherwise
        """
        indices = convert_bytes(self._redis_conn.execute_command("FT._LIST"))  # type: ignore
        return self._name in indices


class AsyncSearchIndex(SearchIndexBase):
    def __init__(
        self,
        name: str,
        prefix: str = "rvl",
        storage_type: Optional[str] = "hash",
        key_field: Optional[str] = None,
        fields: Optional[List["Field"]] = None,
    ):
        super().__init__(name, prefix, storage_type, key_field, fields)

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
    async def create(self, overwrite: Optional[bool] = False):
        """Create an index in Redis from this SearchIndex object

        Args:
            overwrite (bool, optional): Overwrite the index if it already exists. Defaults to False.

        Raises:
            redis.exceptions.ResponseError: If the index already exists
        """
        exists = await self.exists()
        if exists and overwrite:
            await self.delete()

        # set storage_type, default to hash
        storage_type = IndexType.HASH
        if self._storage.lower() == "json":
            self._storage = IndexType.JSON

        # Create Index
        await self._redis_conn.ft(self._name).create_index(  # type: ignore
            fields=self._fields,
            definition=IndexDefinition(prefix=[self._prefix], index_type=storage_type),
        )

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

        async def _load(d: dict):
            async with semaphore:
                key = f"{self._prefix}:{self._get_key_field(d)}"
                await self._redis_conn.hset(key, mapping=d)  # type: ignore

        # gather with concurrency
        await asyncio.gather(*[_load(d) for d in data])

    @check_connected("_redis_conn")
    async def exists(self) -> bool:
        """Check if the index exists in Redis

        Returns:
            bool: True if the index exists, False otherwise
        """
        indices = await self._redis_conn.execute_command("FT._LIST")  # type: ignore
        return self._name in convert_bytes(indices)
