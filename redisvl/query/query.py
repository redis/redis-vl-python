from typing import Any, Dict, List, Optional, Union

from redis.commands.search.query import Query as RedisQuery

from redisvl.query.filter import FilterExpression
from redisvl.redis.utils import array_to_buffer


class BaseQuery(RedisQuery):
    """Base query class used to subclass many query types."""

    _params: Dict[str, Any] = {}
    _filter_expression: Union[str, FilterExpression] = FilterExpression("*")

    def __init__(self, query_string: str = "*"):
        """
        Initialize the BaseQuery class.

        Args:
            query_string (str, optional): The query string to use. Defaults to '*'.
        """
        super().__init__(query_string)

    def __str__(self) -> str:
        """Return the string representation of the query."""
        return " ".join([str(x) for x in self.get_args()])

    def _build_query_string(self) -> str:
        """Build the full Redis query string."""
        raise NotImplementedError("Must be implemented by subclasses")

    def set_filter(
        self, filter_expression: Optional[Union[str, FilterExpression]] = None
    ):
        """Set the filter expression for the query.

        Args:
            filter_expression (Optional[Union[str, FilterExpression]], optional): The filter
                expression or query string to use on the query.

        Raises:
            TypeError: If filter_expression is not a valid FilterExpression or string.
        """
        if filter_expression is None:
            # Default filter to match everything
            self._filter_expression = FilterExpression("*")
        elif isinstance(filter_expression, (FilterExpression, str)):
            self._filter_expression = filter_expression
        else:
            raise TypeError(
                "filter_expression must be of type FilterExpression or string or None"
            )

        # Reset the query string
        self._query_string = self._build_query_string()

    @property
    def filter(self) -> Union[str, FilterExpression]:
        """The filter expression for the query."""
        return self._filter_expression

    @property
    def query(self) -> "BaseQuery":
        """Return self as the query object."""
        return self

    @property
    def params(self) -> Dict[str, Any]:
        """Return the query parameters."""
        return self._params


class FilterQuery(BaseQuery):
    def __init__(
        self,
        filter_expression: Optional[Union[str, FilterExpression]] = None,
        return_fields: Optional[List[str]] = None,
        num_results: int = 10,
        dialect: int = 2,
        sort_by: Optional[str] = None,
        in_order: bool = False,
        params: Optional[Dict[str, Any]] = None,
    ):
        """A query for running a filtered search with a filter expression.

        Args:
            filter_expression (Optional[Union[str, FilterExpression]]): The optional filter
                expression to query with. Defaults to '*'.
            return_fields (Optional[List[str]], optional): The fields to return.
            num_results (Optional[int], optional): The number of results to return. Defaults to 10.
            dialect (int, optional): The query dialect. Defaults to 2.
            sort_by (Optional[str], optional): The field to order the results by. Defaults to None.
            in_order (bool, optional): Requires the terms in the field to have the same order as the terms in the query filter. Defaults to False.
            params (Optional[Dict[str, Any]], optional): The parameters for the query. Defaults to None.

        Raises:
            TypeError: If filter_expression is not of type redisvl.query.FilterExpression
        """
        self.set_filter(filter_expression)

        if params:
            self._params = params

        self._num_results = num_results

        # Initialize the base query with the full query string constructed from the filter expression
        query_string = self._build_query_string()
        super().__init__(query_string)

        # Handle query settings
        if return_fields:
            self.return_fields(*return_fields)
        self.paging(0, self._num_results).dialect(dialect)

        if sort_by:
            self.sort_by(sort_by)

        if in_order:
            self.in_order()

    def _build_query_string(self) -> str:
        """Build the full query string based on the filter and other components."""
        if isinstance(self._filter_expression, FilterExpression):
            return str(self._filter_expression)
        return self._filter_expression


class CountQuery(BaseQuery):
    def __init__(
        self,
        filter_expression: Optional[Union[str, FilterExpression]] = None,
        dialect: int = 2,
        params: Optional[Dict[str, Any]] = None,
    ):
        """A query for a simple count operation provided some filter expression.

        Args:
            filter_expression (Optional[Union[str, FilterExpression]]): The filter expression to query with. Defaults to None.
            params (Optional[Dict[str, Any]], optional): The parameters for the query. Defaults to None.

        Raises:
            TypeError: If filter_expression is not of type redisvl.query.FilterExpression

        .. code-block:: python

            from redisvl.query import CountQuery
            from redisvl.query.filter import Tag

            t = Tag("brand") == "Nike"
            query = CountQuery(filter_expression=t)

            count = index.query(query)
        """
        self.set_filter(filter_expression)

        if params:
            self._params = params

        # Initialize the base query with the full query string constructed from the filter expression
        query_string = self._build_query_string()
        super().__init__(query_string)

        # Query specific modifications
        self.no_content().paging(0, 0).dialect(dialect)

    def _build_query_string(self) -> str:
        """Build the full query string based on the filter and other components."""
        if isinstance(self._filter_expression, FilterExpression):
            return str(self._filter_expression)
        return self._filter_expression


class BaseVectorQuery:
    DISTANCE_ID: str = "vector_distance"
    VECTOR_PARAM: str = "vector"


class VectorQuery(BaseVectorQuery, BaseQuery):
    def __init__(
        self,
        vector: Union[List[float], bytes],
        vector_field_name: str,
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[Union[str, FilterExpression]] = None,
        dtype: str = "float32",
        num_results: int = 10,
        return_score: bool = True,
        dialect: int = 2,
        sort_by: Optional[str] = None,
        in_order: bool = False,
    ):
        """A query for running a vector search along with an optional filter
        expression.

        Args:
            vector (List[float]): The vector to perform the vector search with.
            vector_field_name (str): The name of the vector field to search
                against in the database.
            return_fields (List[str]): The declared fields to return with search
                results.
            filter_expression (Union[str, FilterExpression], optional): A filter to apply
                along with the vector search. Defaults to None.
            dtype (str, optional): The dtype of the vector. Defaults to
                "float32".
            num_results (int, optional): The top k results to return from the
                vector search. Defaults to 10.
            return_score (bool, optional): Whether to return the vector
                distance. Defaults to True.
            dialect (int, optional): The RediSearch query dialect.
                Defaults to 2.
            sort_by (Optional[str]): The field to order the results by. Defaults
                to None. Results will be ordered by vector distance.
            in_order (bool): Requires the terms in the field to have
                the same order as the terms in the query filter, regardless of
                the offsets between them. Defaults to False.

        Raises:
            TypeError: If filter_expression is not of type redisvl.query.FilterExpression

        Note:
            Learn more about vector queries in Redis: https://redis.io/docs/interact/search-and-query/search/vectors/#knn-search
        """
        self._vector = vector
        self._vector_field_name = vector_field_name
        self._dtype = dtype
        self._num_results = num_results
        self.set_filter(filter_expression)
        query_string = self._build_query_string()

        super().__init__(query_string)

        # Handle query modifiers
        if return_fields:
            self.return_fields(*return_fields)

        self.paging(0, self._num_results).dialect(dialect)

        if return_score:
            self.return_fields(self.DISTANCE_ID)

        if sort_by:
            self.sort_by(sort_by)
        else:
            self.sort_by(self.DISTANCE_ID)

        if in_order:
            self.in_order()

    def _build_query_string(self) -> str:
        """Build the full query string for vector search with optional filtering."""
        filter_expression = self._filter_expression
        if isinstance(filter_expression, FilterExpression):
            filter_expression = str(filter_expression)
        return f"{filter_expression}=>[KNN {self._num_results} @{self._vector_field_name} ${self.VECTOR_PARAM} AS {self.DISTANCE_ID}]"

    @property
    def params(self) -> Dict[str, Any]:
        """Return the parameters for the query.

        Returns:
            Dict[str, Any]: The parameters for the query.
        """
        if isinstance(self._vector, bytes):
            vector = self._vector
        else:
            vector = array_to_buffer(self._vector, dtype=self._dtype)

        return {self.VECTOR_PARAM: vector}


class VectorRangeQuery(BaseVectorQuery, BaseQuery):
    DISTANCE_THRESHOLD_PARAM: str = "distance_threshold"

    def __init__(
        self,
        vector: Union[List[float], bytes],
        vector_field_name: str,
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[Union[str, FilterExpression]] = None,
        dtype: str = "float32",
        distance_threshold: float = 0.2,
        num_results: int = 10,
        return_score: bool = True,
        dialect: int = 2,
        sort_by: Optional[str] = None,
        in_order: bool = False,
    ):
        """A query for running a filtered vector search based on semantic
        distance threshold.

        Args:
            vector (List[float]): The vector to perform the range query with.
            vector_field_name (str): The name of the vector field to search
                against in the database.
            return_fields (List[str]): The declared fields to return with search
                results.
            filter_expression (Union[str, FilterExpression], optional): A filter to apply
                along with the range query. Defaults to None.
            dtype (str, optional): The dtype of the vector. Defaults to
                "float32".
            distance_threshold (str, float): The threshold for vector distance.
                A smaller threshold indicates a stricter semantic search.
                Defaults to 0.2.
            num_results (int): The MAX number of results to return.
                Defaults to 10.
            return_score (bool, optional): Whether to return the vector
                distance. Defaults to True.
            dialect (int, optional): The RediSearch query dialect.
                Defaults to 2.
            sort_by (Optional[str]): The field to order the results by. Defaults
                to None. Results will be ordered by vector distance.
            in_order (bool): Requires the terms in the field to have
                the same order as the terms in the query filter, regardless of
                the offsets between them. Defaults to False.

        Raises:
            TypeError: If filter_expression is not of type redisvl.query.FilterExpression

        Note:
            Learn more about vector range queries: https://redis.io/docs/interact/search-and-query/search/vectors/#range-query

        """
        self._vector = vector
        self._vector_field_name = vector_field_name
        self._dtype = dtype
        self._num_results = num_results
        self.set_distance_threshold(distance_threshold)
        self.set_filter(filter_expression)
        query_string = self._build_query_string()

        super().__init__(query_string)

        # Handle query modifiers
        if return_fields:
            self.return_fields(*return_fields)

        self.paging(0, self._num_results).dialect(dialect)

        if return_score:
            self.return_fields(self.DISTANCE_ID)

        if sort_by:
            self.sort_by(sort_by)
        else:
            self.sort_by(self.DISTANCE_ID)

        if in_order:
            self.in_order()

    def _build_query_string(self) -> str:
        """Build the full query string for vector range queries with optional filtering"""
        base_query = f"@{self._vector_field_name}:[VECTOR_RANGE ${self.DISTANCE_THRESHOLD_PARAM} ${self.VECTOR_PARAM}]"

        filter_expression = self._filter_expression
        if isinstance(filter_expression, FilterExpression):
            filter_expression = str(filter_expression)

        if filter_expression == "*":
            return f"{base_query}=>{{$yield_distance_as: {self.DISTANCE_ID}}}"
        return f"({base_query}=>{{$yield_distance_as: {self.DISTANCE_ID}}} {filter_expression})"

    def set_distance_threshold(self, distance_threshold: float):
        """Set the distance threshold for the query.

        Args:
            distance_threshold (float): vector distance
        """
        if not isinstance(distance_threshold, (float, int)):
            raise TypeError("distance_threshold must be of type int or float")
        self._distance_threshold = distance_threshold

    @property
    def distance_threshold(self) -> float:
        """Return the distance threshold for the query.

        Returns:
            float: The distance threshold for the query.
        """
        return self._distance_threshold

    @property
    def params(self) -> Dict[str, Any]:
        """Return the parameters for the query.

        Returns:
            Dict[str, Any]: The parameters for the query.
        """
        if isinstance(self._vector, bytes):
            vector_param = self._vector
        else:
            vector_param = array_to_buffer(self._vector, dtype=self._dtype)

        return {
            self.VECTOR_PARAM: vector_param,
            self.DISTANCE_THRESHOLD_PARAM: self._distance_threshold,
        }


class RangeQuery(VectorRangeQuery):
    # keep for backwards compatibility
    pass
