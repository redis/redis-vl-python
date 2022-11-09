import re

from redis.asyncio import Redis
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from typing import Optional, Pattern


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
    def __init__(self, redis_conn: Redis, name: str,  storage_type: str = "hash", prefix: str = "vector"):
        self.index_name = name
        self.redis_conn = redis_conn

        if storage_type == "hash":
            self.storage_type = IndexType.HASH
        else:
            # TODO: add support for other storage types (i.e. JSON)
            self.storage_type = IndexType.JSON
        self.prefix = prefix

    async def create(
        self,
        fields,
    ):
        # Create Index
        await self.redis_conn.ft(self.index_name).create_index(
            fields = fields,
            definition= IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH)
        )

    async def delete(self):
        await self.redis_conn.ft(self.index_name).dropindex(delete_documents=True)

