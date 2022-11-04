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

    def __init__(self, index_name: str, redis_conn: Redis):
        self.index_name = index_name
        self.redis_conn = redis_conn

    async def create(
        self,
        *fields,
        prefix: str
    ):
        # Create Index
        await self.redis_conn.ft(self.index_name).create_index(
            fields = fields,
            definition= IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)
        )

    async def delete(self):
        await self.redis_conn.ft(self.index_name).dropindex(delete_documents=True)

    def process_tags(self, categories: list, years: list) -> str:
        """
        Helper function to process tags data. TODO - factor this
        out so it's agnostic to the name of the field.

        Args:
            categories (list): List of categories.
            years (list): List of years.

        Returns:
            str: RediSearch tag query string.
        """
        tag = "("
        if years:
            years = "|".join([self.escaper.escape(year) for year in years])
            tag += f"(@year:{{{years}}})"
        if categories:
            categories = "|".join([self.escaper.escape(cat) for cat in categories])
            if tag:
                tag += f" (@categories:{{{categories}}})"
            else:
                tag += f"(@categories:{{{categories}}})"
        tag += ")"
        # if no tags are selected
        if len(tag) < 3:
            tag = "*"
        return tag

    def vector_query(
        self,
        categories: list,
        years: list,
        search_type: str="KNN",
        number_of_results: int=20
    ) -> Query:
        """
        Create a RediSearch query to perform hybrid vector and tag based searches.


        Args:
            categories (list): List of categories.
            years (list): List of years.
            search_type (str, optional): Style of search. Defaults to "KNN".
            number_of_results (int, optional): How many results to fetch. Defaults to 20.

        Returns:
            Query: RediSearch Query

        """
        # Parse tags to create query
        tag_query = self.process_tags(categories, years)
        base_query = f'{tag_query}=>[{search_type} {number_of_results} @vector $vec_param AS vector_score]'
        return Query(base_query)\
            .sort_by("vector_score")\
            .paging(0, number_of_results)\
            .return_fields("paper_id", "paper_pk", "vector_score")\
            .dialect(2)

    def count_query(
        self,
        years: list,
        categories: list
    ) -> Query:
        """
        Create a RediSearch query to count available documents.

        Args:
            categories (list): List of categories.
            years (list): List of years.

        Returns:
            Query: RediSearch Query
        """
        # Parse tags to create query
        tag_query = self.process_tags(categories, years)
        return Query(f'{tag_query}')\
            .no_content()\
            .dialect(2)
