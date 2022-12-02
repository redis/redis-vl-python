import asyncio
import typing as t

from redis.commands.search.field import Field
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from redisvl.schema import read_field_spec, read_schema
from redisvl.utils.connection import get_async_redis_connection, get_redis_connection


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

    @property
    def search(self):
        return self.redis.ft(self.index_name).search

    @classmethod
    def from_yaml(cls, schema_path: str):
        index_attrs, fields = read_schema(schema_path)
        return cls(fields=fields, **index_attrs)

    @classmethod
    def from_dict(cls, schema_dict: t.Dict[str, t.Any]):
        fields = read_field_spec(schema_dict["fields"])
        index_attrs = schema_dict["index"]
        return cls(fields=fields, **index_attrs)

    @classmethod
    def from_existing(cls):
        raise NotImplementedError

    def connect(self, host="localhost", port=6379, username=None, password=None):
        raise NotImplementedError

    def disconnect(self):
        self.redis = None

    def create(self):
        raise NotImplementedError

    def delete(self, drop: bool = True):
        raise NotImplementedError

    def load(self, reader, **kwargs):
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

    def connect(self, host="localhost", port=6379, username=None, password=None):
        # TODO error handling
        self.redis = get_redis_connection(host, port, username, password)

    def create(self):
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
        # Delete the search index
        self.redis.ft(self.index_name).dropindex(delete_documents=drop)

    def load(self, data: t.Iterable[t.Dict[str, t.Any]], **kwargs):
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

    def connect(self, host="localhost", port=6379, username=None, password=None):
        # TODO error handling
        self.redis = get_async_redis_connection(host, port, password)

    async def create(self):
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
        # Delete the search index
        await self.redis.ft(self.index_name).dropindex(delete_documents=drop)

    async def load(self, data: t.Iterable[t.Dict[str, t.Any]], concurrency: int = 10):
        """
        Gather and load the hashes into Redis using
        async connections.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def load(d: dict):
            async with semaphore:
                key = self.prefix + str(d[self.key_field])
                await self.redis.hset(key, mapping=d)

        # gather with concurrency
        await asyncio.gather(*[load(d) for d in data])
