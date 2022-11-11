import re
import typing as t
from typing import Optional, Pattern

from redis.asyncio import Redis
from redis.commands.search.field import Field
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from redisvl.schema import read_field_spec, read_schema


class TokenEscaper:
    """
    Escape punctuation within an input string. Taken from RedisOM Python.
    """

    # Characters that RediSearch requires us to escape during queries.
    # Source: https://redis.io/docs/stack/search/reference/escaping/#the-rules-of-text-field-tokenization
    DEFAULT_ESCAPED_CHARS = r"[,.<>{}\[\]\\\"\':;!@#$%^&*()\-+=~\/ ]"

    def __init__(self, escape_chars_re: Optional[Pattern] = None):
        if escape_chars_re:
            self.escaped_chars_re = escape_chars_re
        else:
            self.escaped_chars_re = re.compile(self.DEFAULT_ESCAPED_CHARS)

    def escape(self, value: str) -> str:
        def escape_symbol(match):
            value = match.group(0)
            return f"\\{value}"

        return self.escaped_chars_re.sub(escape_symbol, value)


class SearchIndex:
    """
    SearchIndex is used to wrap and capture all information
    and actions applied to a RediSearch index including creation,
    manegement, and query construction.
    """

    escaper = TokenEscaper()

    # TODO think about, should this have a redis connection? SearchIndexManupulator?
    def __init__(
        self,
        redis_conn: Redis,
        name: str,
        storage_type: str = "hash",
        key_field: str = "id",
        prefix: str = "",
        fields: t.List[Field] = None,
    ):
        self.index_name = name
        self.key_field = key_field
        self.redis_conn = redis_conn
        self.storage_type = storage_type
        self.prefix = prefix
        self.fields = fields

    @classmethod
    def from_yaml(cls, redis_conn: Redis, schema_path: str):
        index_attrs, fields = read_schema(schema_path)
        return cls(redis_conn, fields=fields, **index_attrs)

    @classmethod
    def from_dict(cls, redis_conn: Redis, schema_dict: t.Dict[str, t.Any]):
        # TODO error handling
        fields = read_field_spec(schema_dict["fields"])
        index_attrs = schema_dict["index"]
        return cls(redis_conn, fields=fields, **index_attrs)

    async def create(
        self,
    ):
        # set storage_type, default to hash
        storage_type = IndexType.HASH
        if self.storage_type.lower() == "json":
            self.storage_type = IndexType.JSON

        # Create Index
        await self.redis_conn.ft(self.index_name).create_index(
            fields=self.fields,
            definition=IndexDefinition(prefix=[self.prefix], index_type=storage_type),
        )

    async def delete(self, dd: bool = True):
        # Delete the search index
        await self.redis_conn.ft(self.index_name).dropindex(delete_documents=dd)
