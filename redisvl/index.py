import asyncio
import typing as t

from redis.commands.search.field import Field
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from redisvl.schema import read_field_spec, read_schema
from redisvl.utils.connection import get_async_redis_connection, get_redis_connection
from redisvl.utils.utils import convert_bytes


class SearchIndexBase:
    def __init__(
        self,
        name: str,
        storage_type: str = "hash",
        key_field: str = "id",
        prefix: str = "",
        fields: t.List[Field] = None,
    ):
        self.index_name = name
        self.key_field = key_field
        self.storage_type = storage_type
        self.prefix = prefix
        self.fields = fields
        self.redis = None

    def search(self, *args, **kwargs):
        """Perform a search on this index

        Wrapper around redis.search.Search that adds the index name
        to the search query. This is a convenience method to avoid
        having to specify the index name in every search query.

        Returns:
            Search: A search object for this index
        """
        return self.redis.ft(self.index_name).search(*args, **kwargs)

    @classmethod
    def from_yaml(cls, schema_path: str):
        """Create a SearchIndex from a YAML schema file

        Args:
            schema_path (str): Path to the YAML schema file

        Returns:
            SearchIndex: A SearchIndex object
        """
        index_attrs, fields = read_schema(schema_path)
        return cls(fields=fields, **index_attrs)

    @classmethod
    def from_dict(cls, schema_dict: t.Dict[str, t.Any]):
        """Create a SearchIndex from a dictionary

        Args:
            schema_dict (t.Dict[str, t.Any]): A dictionary containing the schema

        Returns:
            SearchIndex: A SearchIndex object
        """
        fields = read_field_spec(schema_dict["fields"])
        index_attrs = schema_dict["index"]
        return cls(fields=fields, **index_attrs)

    @classmethod
    def from_existing(cls):
        """Create a SearchIndex from an existing index in Redis
        """
        raise NotImplementedError

    def connect(self, host="localhost", port=6379, username=None, password=None):
        """Connect to a Redis instance

        Args:
            host (str, optional): Redis host. Defaults to "localhost".
            port (int, optional): Redis port. Defaults to 6379.
            username (str, optional): Redis username. Defaults to None.
            password (str, optional): Redis password. Defaults to None.
        """
        raise NotImplementedError

    def disconnect(self):
        """Disconnect from the Redis instance"""
        self.redis = None

    def info(self) -> t.Dict[str, t.Any]:
        """Get information about the index

        Returns:
            dict: A dictionary containing the information about the index
        """
        return convert_bytes(self.redis.ft(self.index_name).info())

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

    def load(self, data: t.Iterable[t.Dict[str, t.Any]], **kwargs):
        """Load data into Redis and index using this SearchIndex object

        Args:
            data (t.Iterable[t.Dict[str, t.Any]]): An iterable of dictionaries
                containing the data to be indexed
        raises:
            redis.exceptions.ResponseError: If the index does not exist
        """
        raise NotImplementedError


class SearchIndex(SearchIndexBase):
    def __init__(
        self,
        name: str,
        storage_type: str = "hash",
        key_field: str = "id",
        prefix: str = "",
        fields: t.List[Field] = None,
    ):
        super().__init__(name, storage_type, key_field, prefix, fields)

    def connect(self, host="localhost", port=6379, username=None, password=None, **kwargs):
        """Connect to a Redis instance

        Args:
            host (str, optional): Redis host. Defaults to "localhost".
            port (int, optional): Redis port. Defaults to 6379.
            username (str, optional): Redis username. Defaults to None.
            password (str, optional): Redis password. Defaults to None.
        """
        self.redis = get_redis_connection(host, port, username, password, **kwargs)

    def create(self):
        """Create an index in Redis from this SearchIndex object

        Raises:
            redis.exceptions.ResponseError: If the index already exists
        """
        # set storage_type, default to hash
        storage_type = IndexType.HASH
        if self.storage_type.lower() == "json":
            self.storage_type = IndexType.JSON

        # Create Index
        self.redis.ft(self.index_name).create_index(
            fields=self.fields,
            definition=IndexDefinition(prefix=[self.prefix], index_type=storage_type),
        )

    def delete(self, drop: bool = True):
        """Delete the search index

        Args:
            drop (bool, optional): Delete the documents in the index. Defaults to True.

        raises:
            redis.exceptions.ResponseError: If the index does not exist
        """
        # Delete the search index
        self.redis.ft(self.index_name).dropindex(delete_documents=drop)

    def load(self, data: t.Iterable[t.Dict[str, t.Any]], **kwargs):
        """Load data into Redis and index using this SearchIndex object

        Args:
            data (t.Iterable[t.Dict[str, t.Any]]): An iterable of dictionaries
                containing the data to be indexed
        raises:
            redis.exceptions.ResponseError: If the index does not exist
        """

        for record in data:
            key = self.prefix + str(record[self.key_field])
            self.redis.hset(key, mapping=record)


class AsyncSearchIndex(SearchIndexBase):
    def __init__(
        self,
        name: str,
        storage_type: str = "hash",
        key_field: str = "id",
        prefix: str = "",
        fields: t.List[Field] = None,
    ):
        super().__init__(name, storage_type, key_field, prefix, fields)

    def connect(self, host="localhost", port=6379, username=None, password=None, **kwargs):
        """Connect to a Redis instance

        Args:
            host (str, optional): Redis host. Defaults to "localhost".
            port (int, optional): Redis port. Defaults to 6379.
            username (str, optional): Redis username. Defaults to None.
            password (str, optional): Redis password. Defaults to None.
        """
        self.redis = get_async_redis_connection(host, port, username, password)

    async def create(self):
        """Create an index in Redis from this SearchIndex object

        Raises:
            redis.exceptions.ResponseError: If the index already exists
        """
        # set storage_type, default to hash
        storage_type = IndexType.HASH
        if self.storage_type.lower() == "json":
            self.storage_type = IndexType.JSON

        # Create Index
        await self.redis.ft(self.index_name).create_index(
            fields=self.fields,
            definition=IndexDefinition(prefix=[self.prefix], index_type=storage_type),
        )

    async def delete(self, drop: bool = True):
        """Delete the search index

        Args:
            drop (bool, optional): Delete the documents in the index. Defaults to True.

        raises:
            redis.exceptions.ResponseError: If the index does not exist
        """
        # Delete the search index
        await self.redis.ft(self.index_name).dropindex(delete_documents=drop)

    async def load(self, data: t.Iterable[t.Dict[str, t.Any]], concurrency: int = 10):
        """Load data into Redis and index using this SearchIndex object

        Args:
            data (t.Iterable[t.Dict[str, t.Any]]): An iterable of dictionaries
            concurrency (int, optional): Number of concurrent tasks to run. Defaults to 10.

        raises:
            redis.exceptions.ResponseError: If the index does not exist
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def load(d: dict):
            async with semaphore:
                key = self.prefix + str(d[self.key_field])
                await self.redis.hset(key, mapping=d)

        # gather with concurrency
        await asyncio.gather(*[load(d) for d in data])
