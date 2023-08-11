from typing import TYPE_CHECKING, Any, Dict, List, Optional

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
        hybrid_filter: FilterExpression = None,
        dtype: str = "float32",
        num_results: Optional[int] = 10,
        return_score: bool = True,
    ):
        """Query for vector fields

        Args:
            vector (List[float]): The vector to query for.
            vector_field_name (str): The name of the vector field
            return_fields (List[str]): The fields to return.
            hybrid_filter (FilterExpression, optional): A filter to apply to the query. Defaults to None.
            dtype (str, optional): The dtype of the vector. Defaults to "float32".
            num_results (Optional[int], optional): The number of results to return. Defaults to 10.
            return_score (bool, optional): Whether to return the score. Defaults to True.

        Raises:
            TypeError: If hybrid_filter is not of type redisvl.query.FilterExpression

        """
        super().__init__(return_fields, num_results)
        self._vector = vector
        self._field = vector_field_name
        self._dtype = dtype.lower()
        if hybrid_filter:
            self.set_filter(hybrid_filter)
        else:
            self._filter = "*"

        if return_score:
            self._return_fields.append(self.DISTANCE_ID)

    def set_filter(self, hybrid_filter: FilterExpression):
        """Set the filter for the query.

        Args:
            hybrid_filter (FilterExpression): The filter to apply to the query.
        """
        if not isinstance(hybrid_filter, FilterExpression):
            raise TypeError(
                "hybrid_filter must be of type redisvl.query.FilterExpression"
            )
        self._filter = str(hybrid_filter)

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
