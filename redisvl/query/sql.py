"""SQL Query class for executing SQL-like queries against Redis."""

from typing import Any, Dict, Optional


class SQLQuery:
    """A query class that translates SQL-like syntax into Redis queries.

    This class allows users to write SQL SELECT statements that are
    automatically translated into Redis FT.SEARCH or FT.AGGREGATE commands.

    .. code-block:: python

        from redisvl.query import SQLQuery
        from redisvl.index import SearchIndex

        index = SearchIndex.from_existing("products", redis_url="redis://localhost:6379")

        sql_query = SQLQuery('''
            SELECT title, price, category
            FROM products
            WHERE category = 'electronics' AND price < 100
        ''')

        results = index.query(sql_query)

    Note:
        Requires the optional `sql-redis` package. Install with:
        ``pip install redisvl[sql]``
    """

    def __init__(self, sql: str, params: Optional[Dict[str, Any]] = None):
        """Initialize a SQLQuery.

        Args:
            sql: The SQL SELECT statement to execute.
            params: Optional dictionary of parameters for parameterized queries.
                   Useful for passing vector data for similarity searches.
        """
        self.sql = sql
        self.params = params or {}
