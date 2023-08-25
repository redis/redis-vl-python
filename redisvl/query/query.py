from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from redis.commands.search.query import Query

from redisvl.query.filter import FilterExpression
from redisvl.utils.utils import array_to_buffer


class BaseQuery:
    def __init__(
        self, return_fields: Optional[List[str]] = None, num_results: Optional[int] = 10
    ):
        self._return_fields = return_fields
        self._num_results = num_results

    @property
    def query(self) -> "Query":
        pass

    @property
    def params(self) -> Dict[str, Any]:
        pass


class FilterQuery(BaseQuery):
    def __init__(
        self,
        return_fields: List[str],
        filter_expression: FilterExpression,
        num_results: Optional[int] = 10,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Query for a filter expression.

        Args:
            return_fields (List[str]): The fields to return.
            filter_expression (FilterExpression): The filter expression to query for.
            num_results (Optional[int], optional): The number of results to return. Defaults to 10.
            params (Optional[Dict[str, Any]], optional): The parameters for the query. Defaults to None.

        Raises:
            TypeError: If filter_expression is not of type redisvl.query.FilterExpression

        Examples:
            >>> from redisvl.query import FilterQuery
            >>> from redisvl.query.filter import Tag
            >>> t = Tag("brand") == "Nike"
            >>> q = FilterQuery(return_fields=["brand", "price"], filter_expression=t)
        """

        super().__init__(return_fields, num_results)
        self.set_filter(filter_expression)
        self._params = params

    def __str__(self) -> str:
        return " ".join([str(x) for x in self.query.get_args()])

    def set_filter(self, filter_expression: FilterExpression):
        """Set the filter for the query.

        Args:
            filter_expression (FilterExpression): The filter to apply to the query.

        Raises:
            TypeError: If filter_expression is not of type redisvl.query.FilterExpression
        """
        if not isinstance(filter_expression, FilterExpression):
            raise TypeError(
                "filter_expression must be of type redisvl.query.FilterExpression"
            )
        self._filter = str(filter_expression)

    def get_filter(self) -> FilterExpression:
        """Get the filter for the query.

        Returns:
            FilterExpression: The filter for the query.
        """
        return self._filter

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
            .paging(0, self._num_results)
            .dialect(2)
        )
        return query

    @property
    def params(self) -> Dict[str, Any]:
        """Return the parameters for the query.

        Returns:
            Dict[str, Any]: The parameters for the query.
        """
        return self._params


class VectorQuery(BaseQuery):
    dtypes = {
        "float32": np.float32,
        "float64": np.float64,
    }

    DISTANCE_ID = "vector_distance"

    def __init__(
        self,
        vector: List[float],
        vector_field_name: str,
        return_fields: List[str],
        filter_expression: FilterExpression = None,
        dtype: str = "float32",
        num_results: Optional[int] = 10,
        return_score: bool = True,
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
        super().__init__(return_fields, num_results)
        self._vector = vector
        self._field = vector_field_name
        self._dtype = dtype.lower()
        if filter_expression:
            self.set_filter(filter_expression)
        else:
            self._filter = "*"

        if return_score:
            self._return_fields.append(self.DISTANCE_ID)

    def set_filter(self, filter_expression: FilterExpression):
        """Set the filter for the query.

        Args:
            filter_expression (FilterExpression): The filter to apply to the query.
        """
        if not isinstance(filter_expression, FilterExpression):
            raise TypeError(
                "filter_expression must be of type redisvl.query.FilterExpression"
            )
        self._filter = str(filter_expression)

    def get_filter(self) -> FilterExpression:
        """Get the filter for the query.

        Returns:
            FilterExpression: The filter for the query.
        """
        return self._filter

    def __str__(self):
        return " ".join([str(x) for x in self.query.get_args()])

    @property
    def query(self) -> Query:
        """Return a Redis-Py Query object representing the query.

        Returns:
            redis.commands.search.query.Query: The query object.
        """
        base_query = f"{self._filter}=>[KNN {self._num_results} @{self._field} $vector AS {self.DISTANCE_ID}]"
        query = (
            Query(base_query)
            .return_fields(*self._return_fields)
            .sort_by(self.DISTANCE_ID)
            .paging(0, self._num_results)
            .dialect(2)
        )
        return query

    @property
    def params(self) -> Dict[str, Any]:
        """Return the parameters for the query.

        Returns:
            Dict[str, Any]: The parameters for the query.
        """
        return {"vector": array_to_buffer(self._vector, dtype=self.dtypes[self._dtype])}



class RangeQuery(BaseQuery):
    dtypes = {
        "float32": np.float32,
        "float64": np.float64,
    }

    DISTANCE_ID = "vector_distance"
    VECTOR_PARAM = "vector"
    DISTANCE_THRESHOLD_PARAM = "distance_threshold"

    def __init__(
        self,
        vector: List[float],
        vector_field_name: str,
        return_fields: List[str],
        filter_expression: FilterExpression = None,
        dtype: str = "float32",
        distance_threshold: float = 0.2,
        num_results: Optional[int] = 10,
        return_score: bool = True,
    ):
        """Range query for vector fields. Range queries are for filtering vector search results
        by the distance between a vector field value and a query vector, in terms of the index distance metric.

        Read more: https://redis.io/docs/interact/search-and-query/search/vectors/#range-query

        Args:
            vector (List[float]): The vector to query for.
            vector_field_name (str): The name of the vector field.
            return_fields (List[str]): The fields to return.
            filter_expression (FilterExpression, optional): A filter to apply to the query. Defaults to None.
            dtype (str, optional): The dtype of the vector. Defaults to "float32".
            distance_threshold (str, float): The thresh
            num_results (Optional[int], optional): The MAX number of results to return. Defaults to 10.
            return_score (bool, optional): Whether to return the score. Defaults to True.

        Raises:
            TypeError: If filter_expression is not of type redisvl.query.FilterExpression

        """
        super().__init__(return_fields, num_results)
        self._vector = vector
        self._field = vector_field_name
        self._dtype = dtype.lower()

        self.set_distance_threshold(distance_threshold)

        if filter_expression:
            self.set_filter(filter_expression)
        else:
            self._filter = "*"

        if return_score:
            self._return_fields.append(self.DISTANCE_ID)

    def set_distance_threshold(self, distance_threshold: float):
        """_summary_

        Args:
            distance_threshold (float): _description_
        """
        if not isinstance(distance_threshold, (float, int)):
            raise TypeError(
                "distance_threshold must be of type int or float"
            )
        self._distance_threshold = distance_threshold

    def get_distance_threshold(self) -> Union[float, int]:
        """Get the distance threhold for the query.

        Returns:
            Union[float, int]: The distance threshold for the query.
        """
        return self._distance_threshold

    def set_filter(self, filter_expression: FilterExpression):
        """Set the filter for the query.

        Args:
            filter_expression (FilterExpression): The filter to apply to the query.
        """
        if not isinstance(filter_expression, FilterExpression):
            raise TypeError(
                "filter_expression must be of type redisvl.query.FilterExpression"
            )
        self._filter = str(filter_expression)

    def get_filter(self) -> FilterExpression:
        """Get the filter for the query.

        Returns:
            FilterExpression: The filter for the query.
        """
        return self._filter

    def __str__(self):
        return " ".join([str(x) for x in self.query.get_args()])

    @property
    def query(self) -> Query:
        """Return a Redis-Py Query object representing the query.

        Returns:
            redis.commands.search.query.Query: The query object.
        """
        base_query = f"@{self._field}:[VECTOR_RANGE ${self.DISTANCE_THRESHOLD_PARAM} ${self.VECTOR_PARAM}]"

        if len(self._filter) > 1:
            base_query = "(" + base_query + " " + self._filter + ")"

        base_query += f"=>{{$yield_distance_as: {self.DISTANCE_ID}}}"
        query = (
            Query(base_query)
            .return_fields(*self._return_fields)
            .sort_by(self.DISTANCE_ID)
            # TODO -- upper bound for num results, optional?
            .paging(0, self._num_results)
            .dialect(2)
        )
        return query

    @property
    def params(self) -> Dict[str, Any]:
        """Return the parameters for the query.

        Returns:
            Dict[str, Any]: The parameters for the query.
        """
        return {
            self.VECTOR_PARAM: array_to_buffer(self._vector, dtype=self.dtypes[self._dtype]),
            self.DISTANCE_THRESHOLD_PARAM: self._distance_threshold
        }
