from enum import Enum
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


class HybridPolicy(str, Enum):
    """Enum for valid hybrid policy options in vector queries."""

    BATCHES = "BATCHES"
    ADHOC_BF = "ADHOC_BF"


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
        hybrid_policy: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize_cosine_distance: bool = False,
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
            hybrid_policy (Optional[str]): Controls how filters are applied during vector search.
                Options are "BATCHES" (paginates through small batches of nearest neighbors) or
                "ADHOC_BF" (computes scores for all vectors passing the filter).
                "BATCHES" mode is typically faster for queries with selective filters.
                "ADHOC_BF" mode is better when filters match a large portion of the dataset.
                Defaults to None, which lets Redis auto-select the optimal policy.
            batch_size (Optional[int]): When hybrid_policy is "BATCHES", controls the number
                of vectors to fetch in each batch. Larger values may improve performance
                at the cost of memory usage. Only applies when hybrid_policy="BATCHES".
                Defaults to None, which lets Redis auto-select an appropriate batch size.
            normalize_cosine_distance (bool): by default Redis returns cosine distance as a value
                between 0 and 2 where 0 is the best match. If set to True, the cosine distance will be
                converted to cosine similarity with a value between 0 and 1 where 1 is the best match.

        Raises:
            TypeError: If filter_expression is not of type redisvl.query.FilterExpression

        Note:
            Learn more about vector queries in Redis: https://redis.io/docs/interact/search-and-query/search/vectors/#knn-search
        """
        self._vector = vector
        self._vector_field_name = vector_field_name
        self._dtype = dtype
        self._num_results = num_results
        self._hybrid_policy: Optional[HybridPolicy] = None
        self._batch_size: Optional[int] = None
        self._normalize_cosine_distance = normalize_cosine_distance
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

        if hybrid_policy is not None:
            self.set_hybrid_policy(hybrid_policy)

        if batch_size is not None:
            self.set_batch_size(batch_size)

    def _build_query_string(self) -> str:
        """Build the full query string for vector search with optional filtering."""
        filter_expression = self._filter_expression
        if isinstance(filter_expression, FilterExpression):
            filter_expression = str(filter_expression)

        # Base KNN query
        knn_query = (
            f"KNN {self._num_results} @{self._vector_field_name} ${self.VECTOR_PARAM}"
        )

        # Add hybrid policy parameters if specified
        if self._hybrid_policy:
            knn_query += f" HYBRID_POLICY {self._hybrid_policy.value}"

            # Add batch size if specified and using BATCHES policy
            if self._hybrid_policy == HybridPolicy.BATCHES and self._batch_size:
                knn_query += f" BATCH_SIZE {self._batch_size}"

        # Add distance field alias
        knn_query += f" AS {self.DISTANCE_ID}"

        return f"{filter_expression}=>[{knn_query}]"

    def set_hybrid_policy(self, hybrid_policy: str):
        """Set the hybrid policy for the query.

        Args:
            hybrid_policy (str): The hybrid policy to use. Options are "BATCHES"
                                or "ADHOC_BF".

        Raises:
            ValueError: If hybrid_policy is not one of the valid options
        """
        try:
            self._hybrid_policy = HybridPolicy(hybrid_policy)
        except ValueError:
            raise ValueError(
                f"hybrid_policy must be one of {', '.join([p.value for p in HybridPolicy])}"
            )

        # Reset the query string
        self._query_string = self._build_query_string()

    def set_batch_size(self, batch_size: int):
        """Set the batch size for the query.

        Args:
            batch_size (int): The batch size to use when hybrid_policy is "BATCHES".

        Raises:
            TypeError: If batch_size is not an integer
            ValueError: If batch_size is not positive
        """
        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._batch_size = batch_size

        # Reset the query string
        self._query_string = self._build_query_string()

    @property
    def hybrid_policy(self) -> Optional[str]:
        """Return the hybrid policy for the query.

        Returns:
            Optional[str]: The hybrid policy for the query.
        """
        return self._hybrid_policy.value if self._hybrid_policy else None

    @property
    def batch_size(self) -> Optional[int]:
        """Return the batch size for the query.

        Returns:
            Optional[int]: The batch size for the query.
        """
        return self._batch_size

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

        params = {self.VECTOR_PARAM: vector}

        return params


class VectorRangeQuery(BaseVectorQuery, BaseQuery):
    DISTANCE_THRESHOLD_PARAM: str = "distance_threshold"
    EPSILON_PARAM: str = "EPSILON"  # Parameter name for epsilon
    HYBRID_POLICY_PARAM: str = "HYBRID_POLICY"  # Parameter name for hybrid policy
    BATCH_SIZE_PARAM: str = "BATCH_SIZE"  # Parameter name for batch size

    def __init__(
        self,
        vector: Union[List[float], bytes],
        vector_field_name: str,
        return_fields: Optional[List[str]] = None,
        filter_expression: Optional[Union[str, FilterExpression]] = None,
        dtype: str = "float32",
        distance_threshold: float = 0.2,
        epsilon: Optional[float] = None,
        num_results: int = 10,
        return_score: bool = True,
        dialect: int = 2,
        sort_by: Optional[str] = None,
        in_order: bool = False,
        hybrid_policy: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize_cosine_distance: bool = False,
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
            distance_threshold (float): The threshold for vector distance.
                A smaller threshold indicates a stricter semantic search.
                Defaults to 0.2.
            epsilon (Optional[float]): The relative factor for vector range queries,
                setting boundaries for candidates within radius * (1 + epsilon).
                This controls how extensive the search is beyond the specified radius.
                Higher values increase recall at the expense of performance.
                Defaults to None, which uses the index-defined epsilon (typically 0.01).
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
            hybrid_policy (Optional[str]): Controls how filters are applied during vector search.
                Options are "BATCHES" (paginates through small batches of nearest neighbors) or
                "ADHOC_BF" (computes scores for all vectors passing the filter).
                "BATCHES" mode is typically faster for queries with selective filters.
                "ADHOC_BF" mode is better when filters match a large portion of the dataset.
                Defaults to None, which lets Redis auto-select the optimal policy.
            batch_size (Optional[int]): When hybrid_policy is "BATCHES", controls the number
                of vectors to fetch in each batch. Larger values may improve performance
                at the cost of memory usage. Only applies when hybrid_policy="BATCHES".
                Defaults to None, which lets Redis auto-select an appropriate batch size.
            normalize_cosine_distance (bool): by default Redis returns cosine distance as a value
                between 0 and 2 where 0 is the best match. If set to True, the cosine distance will be
                converted to cosine similarity with a value between 0 and 1 where 1 is the best match.

        Raises:
            TypeError: If filter_expression is not of type redisvl.query.FilterExpression

        Note:
            Learn more about vector range queries: https://redis.io/docs/interact/search-and-query/search/vectors/#range-query

        """
        self._vector = vector
        self._vector_field_name = vector_field_name
        self._dtype = dtype
        self._num_results = num_results
        self._distance_threshold: float = 0.2  # Initialize with default
        self._epsilon: Optional[float] = None
        self._hybrid_policy: Optional[HybridPolicy] = None
        self._batch_size: Optional[int] = None

        if epsilon is not None:
            self.set_epsilon(epsilon)

        if hybrid_policy is not None:
            self.set_hybrid_policy(hybrid_policy)

        if batch_size is not None:
            self.set_batch_size(batch_size)

        self._normalize_cosine_distance = normalize_cosine_distance
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

    def set_distance_threshold(self, distance_threshold: float):
        """Set the distance threshold for the query.

        Args:
            distance_threshold (float): Vector distance threshold.

        Raises:
            TypeError: If distance_threshold is not a float or int
            ValueError: If distance_threshold is negative
        """
        if not isinstance(distance_threshold, (float, int)):
            raise TypeError("distance_threshold must be of type float or int")
        if distance_threshold < 0:
            raise ValueError("distance_threshold must be non-negative")
        self._distance_threshold = distance_threshold

        # Reset the query string
        self._query_string = self._build_query_string()

    def set_epsilon(self, epsilon: float):
        """Set the epsilon parameter for the range query.

        Args:
            epsilon (float): The relative factor for vector range queries,
                setting boundaries for candidates within radius * (1 + epsilon).

        Raises:
            TypeError: If epsilon is not a float or int
            ValueError: If epsilon is negative
        """
        if not isinstance(epsilon, (float, int)):
            raise TypeError("epsilon must be of type float or int")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        self._epsilon = epsilon

        # Reset the query string
        self._query_string = self._build_query_string()

    def set_hybrid_policy(self, hybrid_policy: str):
        """Set the hybrid policy for the query.

        Args:
            hybrid_policy (str): The hybrid policy to use. Options are "BATCHES"
                                or "ADHOC_BF".

        Raises:
            ValueError: If hybrid_policy is not one of the valid options
        """
        try:
            self._hybrid_policy = HybridPolicy(hybrid_policy)
        except ValueError:
            raise ValueError(
                f"hybrid_policy must be one of {', '.join([p.value for p in HybridPolicy])}"
            )

        # Reset the query string
        self._query_string = self._build_query_string()

    def set_batch_size(self, batch_size: int):
        """Set the batch size for the query.

        Args:
            batch_size (int): The batch size to use when hybrid_policy is "BATCHES".

        Raises:
            TypeError: If batch_size is not an integer
            ValueError: If batch_size is not positive
        """
        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._batch_size = batch_size

        # Reset the query string
        self._query_string = self._build_query_string()

    def _build_query_string(self) -> str:
        """Build the full query string for vector range queries with optional filtering"""
        # Build base query with vector range only
        base_query = f"@{self._vector_field_name}:[VECTOR_RANGE ${self.DISTANCE_THRESHOLD_PARAM} ${self.VECTOR_PARAM}]"

        # Build query attributes section
        attr_parts = []
        attr_parts.append(f"$YIELD_DISTANCE_AS: {self.DISTANCE_ID}")

        if self._epsilon is not None:
            attr_parts.append(f"$EPSILON: {self._epsilon}")

        # Add query attributes section
        attr_section = f"=>{{{'; '.join(attr_parts)}}}"

        # Add filter expression if present
        filter_expression = self._filter_expression
        if isinstance(filter_expression, FilterExpression):
            filter_expression = str(filter_expression)

        if filter_expression == "*":
            return f"{base_query}{attr_section}"
        return f"({base_query}{attr_section} {filter_expression})"

    @property
    def distance_threshold(self) -> float:
        """Return the distance threshold for the query.

        Returns:
            float: The distance threshold for the query.
        """
        return self._distance_threshold

    @property
    def epsilon(self) -> Optional[float]:
        """Return the epsilon for the query.

        Returns:
            Optional[float]: The epsilon for the query, or None if not set.
        """
        return self._epsilon

    @property
    def hybrid_policy(self) -> Optional[str]:
        """Return the hybrid policy for the query.

        Returns:
            Optional[str]: The hybrid policy for the query.
        """
        return self._hybrid_policy.value if self._hybrid_policy else None

    @property
    def batch_size(self) -> Optional[int]:
        """Return the batch size for the query.

        Returns:
            Optional[int]: The batch size for the query.
        """
        return self._batch_size

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

        params = {
            self.VECTOR_PARAM: vector_param,
            self.DISTANCE_THRESHOLD_PARAM: self._distance_threshold,
        }

        # Add hybrid policy and batch size as query parameters (not in query string)
        if self._hybrid_policy:
            params[self.HYBRID_POLICY_PARAM] = self._hybrid_policy.value

            if self._hybrid_policy == HybridPolicy.BATCHES and self._batch_size:
                params[self.BATCH_SIZE_PARAM] = self._batch_size

        return params


class RangeQuery(VectorRangeQuery):
    # keep for backwards compatibility
    pass
