"""SQL Query class for executing SQL-like queries against Redis."""

import re
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

    def _substitute_params(self, sql: str, params: Dict[str, Any]) -> str:
        """Substitute parameter placeholders in SQL with actual values.

        Uses token-based approach: splits SQL on :param patterns, then rebuilds
        with substituted values. This prevents partial matching (e.g., :id
        won't match inside :product_id) and is faster than regex at scale.

        Args:
            sql: The SQL string with :param placeholders.
            params: Dictionary mapping parameter names to values.

        Returns:
            SQL string with parameters substituted.

        Note:
            - String values are wrapped in single quotes with proper escaping
            - Numeric values are converted to strings
            - Bytes values (e.g., vectors) are NOT substituted here
        """
        if not params:
            return sql

        # Split SQL on :param patterns, keeping the delimiters
        # Pattern matches : followed by valid identifier (letter/underscore, then alphanumeric/underscore)
        tokens = re.split(r"(:[a-zA-Z_][a-zA-Z0-9_]*)", sql)

        result = []
        for token in tokens:
            if token.startswith(":"):
                key = token[1:]  # Remove leading :
                if key in params:
                    value = params[key]
                    if isinstance(value, (int, float)):
                        result.append(str(value))
                    elif isinstance(value, str):
                        # Escape single quotes using SQL standard: ' -> ''
                        escaped = value.replace("'", "''")
                        result.append(f"'{escaped}'")
                    else:
                        # Keep placeholder for bytes (vectors handled by Executor)
                        result.append(token)
                else:
                    # Keep unmatched placeholders as-is
                    result.append(token)
            else:
                result.append(token)

        return "".join(result)

    def redis_query_string(
        self,
        redis_client: Optional[Any] = None,
        redis_url: str = "redis://localhost:6379",
    ) -> str:
        """Translate the SQL query to a Redis command string.

        This method uses the sql-redis translator to convert the SQL statement
        into the equivalent Redis FT.SEARCH or FT.AGGREGATE command.

        Args:
            redis_client: A Redis client connection used to load index schemas.
                If not provided, a connection will be created using redis_url.
            redis_url: The Redis URL to connect to if redis_client is not provided.
                Defaults to "redis://localhost:6379".

        Returns:
            The Redis command string (e.g., 'FT.SEARCH products "@category:{electronics}"').

        Raises:
            ImportError: If sql-redis package is not installed.

        Example:
            .. code-block:: python

                from redisvl.query import SQLQuery

                sql_query = SQLQuery("SELECT * FROM products WHERE category = 'electronics'")

                # Using redis_url
                redis_cmd = sql_query.redis_query_string(redis_url="redis://localhost:6379")

                # Or using an existing client
                from redis import Redis
                client = Redis()
                redis_cmd = sql_query.redis_query_string(redis_client=client)

                print(redis_cmd)
                # Output: FT.SEARCH products "@category:{electronics}"
        """
        try:
            from sql_redis.schema import SchemaRegistry
            from sql_redis.translator import Translator
        except ImportError:
            raise ImportError(
                "sql-redis is required for SQL query support. "
                "Install it with: pip install redisvl[sql]"
            )

        # Get or create Redis client
        if redis_client is None:
            from redis import Redis

            redis_client = Redis.from_url(redis_url)

        # Load schemas from Redis
        registry = SchemaRegistry(redis_client)
        registry.load_all()

        # Translate SQL to Redis command
        translator = Translator(registry)

        # Substitute non-bytes params in SQL before translation
        sql = self._substitute_params(self.sql, self.params)

        translated = translator.translate(sql)
        return translated.to_command_string()
