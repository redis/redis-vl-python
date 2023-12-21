from typing import Any, Dict, List, Optional, Union

import numpy as np
from redis.commands.search.query import Query

from redisvl.query.filter import FilterExpression
from redisvl.utils.utils import array_to_buffer


class BaseQuery:
    def __init__(
        self,
        return_fields: Optional[List[str]] = None,
        num_results: int = 10,
        dialect: int = 2,
    ):
        """Base query class used to subclass many query types."""
        self._return_fields = return_fields if return_fields is not None else []
        self._num_results = num_results
        self._dialect = dialect
        self._first = 0
        self._limit = num_results

    def __str__(self) -> str:
        return " ".join([str(x) for x in self.query.get_args()])

    def set_filter(self, filter_expression: Optional[FilterExpression] = None):
        """Set the filter for the query.

        Args:
            filter_expression (Optional[FilterExpression], optional): The filter
                to apply to the query.

        Raises:
            TypeError: If filter_expression is not of type redisvl.query.FilterExpression
        """
        if filter_expression is None:
            # Default filter to match everything
            self._filter = FilterExpression("*")
        elif isinstance(filter_expression, FilterExpression):
            self._filter = filter_expression
        else:
            raise TypeError(
                "filter_expression must be of type FilterExpression or None"
            )

    def get_filter(self) -> FilterExpression:
        """Get the filter for the query.

        Returns:
            FilterExpression: The filter for the query.
        """
        return self._filter

    def set_paging(self, first: int, limit: int):
        """
        Set the paging parameters for the query to limit the results between
        fist and num_results.

        Args:
            first (int): The zero-indexed offset for which to fetch query results
            limit (int): _description_

        Raises:
            TypeError: _description_
            TypeError: _description_
        """
        if not isinstance(first, int):
            raise TypeError("first must be of type int")
        if not isinstance(limit, int):
            raise TypeError("limit must be of type int")
        self._first = first
        self._limit = limit

    @property
    def query(self) -> Query:
        raise NotImplementedError

    @property
    def params(self) -> Dict[str, Any]:
        return {}


class CountQuery(BaseQuery):
    def __init__(
        self,
        filter_expression: FilterExpression,
        dialect: int = 2,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Query for a simple count operation provided some filter expression.

        Args:
            filter_expression (FilterExpression): The filter expression to query for.
            params (Optional[Dict[str, Any]], optional): The parameters for the query. Defaults to None.

        Raises:
            TypeError: If filter_expression is not of type redisvl.query.FilterExpression

        Examples:
            >>> from redisvl.query import CountQuery
            >>> from redisvl.query.filter import Tag
            >>> t = Tag("brand") == "Nike"
            >>> q = CountQuery(filter_expression=t)
            >>> count = index.query(q)
        """
        super().__init__(num_results=0, dialect=dialect)
        self.set_filter(filter_expression)
        self._params = params or {}

    @property
    def query(self) -> Query:
        """The loaded Redis-Py query.

        Returns:
            redis.commands.search.query.Query: The query object.
        """
        base_query = str(self._filter)
        query = Query(base_query).no_content().paging(0, 0).dialect(self._dialect)
        return query

    @property
    def params(self) -> Dict[str, Any]:
        """The parameters for the query.

        Returns:
            Dict[str, Any]: The parameters for the query.
        """
        return self._params


class FilterQuery(BaseQuery):
    def __init__(
        self,
        filter_expression: FilterExpression,
        return_fields: Optional[List[str]] = None,
        num_results: int = 10,
        dialect: int = 2,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Query for a filter expression.

        Args:
            filter_expression (FilterExpression): The filter expression to
                query for.
            return_fields (Optional[List[str]], optional): The fields to return.
            num_results (Optional[int], optional): The number of results to
                return. Defaults to 10.
            params (Optional[Dict[str, Any]], optional): The parameters for the
                query. Defaults to None.

        Raises:
            TypeError: If filter_expression is not of type redisvl.query.FilterExpression

        Examples:
            >>> from redisvl.query import FilterQuery
            >>> from redisvl.query.filter import Tag
            >>> t = Tag("brand") == "Nike"
            >>> q = FilterQuery(return_fields=["brand", "price"], filter_expression=t)
        """
        super().__init__(return_fields, num_results, dialect)
        self.set_filter(filter_expression)
        self._params = params or {}

    @property
    def query(self) -> Query:
        """Return a Redis-Py Query object representing the query.

        Returns:
            redis.commands.search.query.Query: The query object.
        """
        base_query = str(self._filter)
        query = (
            Query(base_query)
            .return_fields(*self._return_fields)
            .paging(self._first, self._limit)
            .dialect(self._dialect)
        )
        return query


class BaseVectorQuery(BaseQuery):
    DTYPES = {
        "float32": np.float32,
        "float64": np.float64,
    }
    DISTANCE_ID = "vector_distance"
    VECTOR_PARAM = "vector"

    def __init__(
        self,
        vector: Union[List[float], bytes],
        vector_field_name: str,
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[FilterExpression] = None,
        dtype: str = "float32",
        num_results: int = 10,
        return_score: bool = True,
        dialect: int = 2,
    ):
        super().__init__(return_fields, num_results, dialect)
        self.set_filter(filter_expression)
        self._vector = vector
        self._field = vector_field_name
        self._dtype = dtype.lower()

        if return_score:
            self._return_fields.append(self.DISTANCE_ID)


class VectorQuery(BaseVectorQuery):
    def __init__(
        self,
        vector: Union[List[float], bytes],
        vector_field_name: str,
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[FilterExpression] = None,
        dtype: str = "float32",
        num_results: int = 10,
        return_score: bool = True,
        dialect: int = 2,
    ):
        """Query for vector fields.

        Read more: https://redis.io/docs/interact/search-and-query/search/vectors/#knn-search

        Args:
            vector (List[float]): The vector to query for.
            vector_field_name (str): The name of the vector field.
            return_fields (List[str]): The fields to return.
            filter_expression (FilterExpression, optional): A filter to apply to the query. Defaults to None.
            dtype (str, optional): The dtype of the vector. Defaults to "float32".
            num_results (Optional[int], optional): The number of results to return. Defaults to 10.
            return_score (bool, optional): Whether to return the score. Defaults to True.

        Raises:
            TypeError: If filter_expression is not of type redisvl.query.FilterExpression
        """
        super().__init__(
            vector,
            vector_field_name,
            return_fields,
            filter_expression,
            dtype,
            num_results,
            return_score,
            dialect,
        )

    @property
    def query(self) -> Query:
        """Return a Redis-Py Query object representing the query.

        Returns:
            redis.commands.search.query.Query: The query object.
        """
        base_query = f"{str(self._filter)}=>[KNN {self._num_results} @{self._field} ${self.VECTOR_PARAM} AS {self.DISTANCE_ID}]"
        query = (
            Query(base_query)
            .return_fields(*self._return_fields)
            .sort_by(self.DISTANCE_ID)
            .paging(self._first, self._limit)
            .dialect(self._dialect)
        )
        return query

    @property
    def params(self) -> Dict[str, Any]:
        """Return the parameters for the query.

        Returns:
            Dict[str, Any]: The parameters for the query.
        """
        if isinstance(self._vector, bytes):
            vector_param = self._vector
        else:
            vector_param = array_to_buffer(self._vector, dtype=self.DTYPES[self._dtype])

        return {self.VECTOR_PARAM: vector_param}


class RangeQuery(BaseVectorQuery):
    DISTANCE_THRESHOLD_PARAM = "distance_threshold"

    def __init__(
        self,
        vector: Union[List[float], bytes],
        vector_field_name: str,
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[FilterExpression] = None,
        dtype: str = "float32",
        distance_threshold: float = 0.2,
        num_results: int = 10,
        return_score: bool = True,
        dialect: int = 2,
    ):
        """Vector query by distance range.

        Range queries are for filtering vector search results
        by the distance between a vector field value and a query
        vector, in terms of the index distance metric.

        Read more: https://redis.io/docs/interact/search-and-query/search/vectors/#range-query

        Args:
            vector (List[float]): The vector to query for.
            vector_field_name (str): The name of the vector field.
            return_fields (List[str]): The fields to return.
            filter_expression (FilterExpression, optional): A filter to apply to the query. Defaults to None.
            dtype (str, optional): The dtype of the vector. Defaults to "float32".
            distance_threshold (str, float): The threshold for vector distance. Defaults to 0.2.
            num_results (int): The MAX number of results to return. defaults to 10.
            return_score (bool, optional): Whether to return the score. Defaults to True.

        Raises:
            TypeError: If filter_expression is not of type redisvl.query.FilterExpression
        """
        super().__init__(
            vector,
            vector_field_name,
            return_fields,
            filter_expression,
            dtype,
            num_results,
            return_score,
            dialect,
        )
        self.set_distance_threshold(distance_threshold)

    def set_distance_threshold(self, distance_threshold: float):
        """Set the distance treshold for the query.

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
    def query(self) -> Query:
        """Return a Redis-Py Query object representing the query.

        Returns:
            redis.commands.search.query.Query: The query object.
        """
        base_query = f"@{self._field}:[VECTOR_RANGE ${self.DISTANCE_THRESHOLD_PARAM} ${self.VECTOR_PARAM}]"

        _filter = str(self._filter)

        if _filter != "*":
            base_query = (
                f"({base_query}=>{{$yield_distance_as: {self.DISTANCE_ID}}} {_filter})"
            )
        else:
            base_query = f"{base_query}=>{{$yield_distance_as: {self.DISTANCE_ID}}}"

        query = (
            Query(base_query)
            .return_fields(*self._return_fields)
            .sort_by(self.DISTANCE_ID)
            .paging(self._first, self._limit)
            .dialect(self._dialect)
        )
        return query

    @property
    def params(self) -> Dict[str, Any]:
        """Return the parameters for the query.

        Returns:
            Dict[str, Any]: The parameters for the query.
        """
        if isinstance(self._vector, bytes):
            vector_param = self._vector
        else:
            vector_param = array_to_buffer(self._vector, dtype=self.DTYPES[self._dtype])

        return {
            self.VECTOR_PARAM: vector_param,
            self.DISTANCE_THRESHOLD_PARAM: self._distance_threshold,
        }
