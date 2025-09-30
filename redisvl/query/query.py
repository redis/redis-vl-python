from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from redis.commands.search.query import Query as RedisQuery

from redisvl.query.filter import FilterExpression
from redisvl.redis.utils import array_to_buffer
from redisvl.utils.log import get_logger
from redisvl.utils.token_escaper import TokenEscaper
from redisvl.utils.utils import denorm_cosine_distance, lazy_import

logger = get_logger(__name__)

nltk = lazy_import("nltk")
nltk_stopwords = lazy_import("nltk.corpus.stopwords")

# Type alias for sort specification
# Can be:
# - str: single field name (ASC by default)
# - Tuple[str, str]: (field_name, direction)
# - List: list of field names or tuples
SortSpec = Union[str, Tuple[str, str], List[Union[str, Tuple[str, str]]]]


class BaseQuery(RedisQuery):
    """
    Base query class used to subclass many query types.

    NOTE: In the base class, the `_query_string` field is set once on
    initialization, and afterward, redis-py expects to be able to read it. By
    contrast, our query subclasses allow users to call methods that alter the
    query string at runtime. To avoid having to rebuild `_query_string` every
    time one of these methods is called, we lazily build the query string when a
    user calls `query()` or accesses the property `_query_string`, when the
    underlying `_built_query_string` field is None. Any method that alters the query
    string should set `_built_query_string` to None so that the next time the query
    string is accessed, it is rebuilt.
    """

    _params: Dict[str, Any] = {}
    _filter_expression: Union[str, FilterExpression] = FilterExpression("*")
    _built_query_string: Optional[str] = None

    def __init__(self, query_string: str = "*"):
        """
        Initialize the BaseQuery class.

        Args:
            query_string (str, optional): The query string to use. Defaults to '*'.
        """
        # The parent class expects a query string, so we pass it in, but we'll
        # actually manage building it dynamically.
        super().__init__(query_string)

        # This is a private field that we use to track whether the query string
        # has been built, and we set it to None here to indicate that the field
        # has not been built yet.
        self._built_query_string = None

        # Initialize skip_decode_fields set
        self._skip_decode_fields: Set[str] = set()

    def __str__(self) -> str:
        """Return the string representation of the query."""
        return " ".join([str(x) for x in self.get_args()])

    def _build_query_string(self) -> str:
        """Build the full Redis query string."""
        raise NotImplementedError("Must be implemented by subclasses")

    @staticmethod
    def _parse_sort_spec(sort_spec: Optional[SortSpec]) -> List[Tuple[str, bool]]:
        """Parse sort specification into list of (field, ascending) tuples.

        Args:
            sort_spec: Sort specification in various formats:
                - str: single field name (defaults to ASC)
                - Tuple[str, str]: (field_name, "ASC"|"DESC")
                - List: list of strings or tuples

        Returns:
            List of (field_name, ascending) tuples where ascending is a boolean.

        Raises:
            TypeError: If sort_spec is not a valid type.
            ValueError: If direction is not "ASC" or "DESC".

        Examples:
            >>> BaseQuery._parse_sort_spec("price")
            [("price", True)]
            >>> BaseQuery._parse_sort_spec(("price", "DESC"))
            [("price", False)]
            >>> BaseQuery._parse_sort_spec(["price", ("rating", "DESC")])
            [("price", True), ("rating", False)]
        """
        if sort_spec is None or sort_spec == []:
            return []

        result: List[Tuple[str, bool]] = []

        # Single field as string
        if isinstance(sort_spec, str):
            result.append((sort_spec, True))  # Default to ASC

        # Single field as tuple
        elif isinstance(sort_spec, tuple):
            if len(sort_spec) != 2:
                raise ValueError(
                    f"Sort tuple must have exactly 2 elements (field, direction), got {len(sort_spec)}"
                )
            field, direction = sort_spec
            if not isinstance(field, str):
                raise TypeError(f"Field name must be a string, got {type(field)}")
            if not isinstance(direction, str):
                raise TypeError(f"Direction must be a string, got {type(direction)}")

            direction_upper = direction.upper()
            if direction_upper not in ("ASC", "DESC"):
                raise ValueError(
                    f"Sort direction must be 'ASC' or 'DESC', got '{direction}'"
                )

            result.append((field, direction_upper == "ASC"))

        # Multiple fields as list
        elif isinstance(sort_spec, list):
            for item in sort_spec:
                # Recursively parse each item
                parsed = BaseQuery._parse_sort_spec(item)
                result.extend(parsed)

        else:
            raise TypeError(
                f"sort_by must be a string, tuple, or list, got {type(sort_spec)}"
            )

        return result

    def sort_by(
        self, sort_spec: Optional[SortSpec] = None, asc: bool = True
    ) -> "BaseQuery":
        """Set the sort order for query results.

        This method supports sorting by single or multiple fields. Note that Redis Search
        natively supports only a single SORTBY field. When multiple fields are specified,
        only the FIRST field is used for the Redis SORTBY clause.

        Args:
            sort_spec: Sort specification in various formats:
                - str: single field name
                - Tuple[str, str]: (field_name, "ASC"|"DESC")
                - List: list of field names or tuples
            asc: Default sort direction when not specified (only used when sort_spec is a string).
                Defaults to True (ascending).

        Returns:
            self: Returns the query object for method chaining.

        Raises:
            TypeError: If sort_spec is not a valid type.
            ValueError: If direction is not "ASC" or "DESC".

        Examples:
            >>> query.sort_by("price")  # Single field, ascending
            >>> query.sort_by(("price", "DESC"))  # Single field, descending
            >>> query.sort_by(["price", "rating"])  # Multiple fields (only first used)
            >>> query.sort_by([("price", "DESC"), ("rating", "ASC")])

        Note:
            When multiple fields are specified, only the first field is used for sorting
            in Redis. Future versions may support multi-field sorting through post-query
            sorting in Python.
        """
        if sort_spec is None or sort_spec == []:
            # No sorting
            self._sortby = None
            return self

        # Handle backward compatibility: if sort_spec is a string and asc is specified
        # treat it as the old (field, asc) format
        parsed: List[Tuple[str, bool]]
        if isinstance(sort_spec, str) and asc is not True:
            # Old API: query.sort_by("field", asc=False)
            parsed = [(sort_spec, asc)]
        else:
            # New API: parse the sort_spec
            parsed = self._parse_sort_spec(sort_spec)

        if not parsed:
            self._sortby = None
            return self

        # Use the first field for Redis SORTBY
        first_field, first_asc = parsed[0]

        # Log warning if multiple fields specified
        if len(parsed) > 1:
            logger.warning(
                f"Multiple sort fields specified: {[f[0] for f in parsed]}. "
                f"Redis Search only supports single-field sorting. Using first field: '{first_field}'. "
                "Additional fields are ignored."
            )

        # Call parent's sort_by with the first field
        super().sort_by(first_field, asc=first_asc)

        return self

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

        # Invalidate the query string
        self._built_query_string = None

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

    @property
    def _query_string(self) -> str:
        """Maintains compatibility with parent class while providing lazy loading."""
        if self._built_query_string is None:
            self._built_query_string = self._build_query_string()
        return self._built_query_string

    @_query_string.setter
    def _query_string(self, value: Optional[str]):
        """Setter for _query_string to maintain compatibility with parent class."""
        self._built_query_string = value

    def return_fields(
        self, *fields, skip_decode: Optional[Union[str, List[str]]] = None
    ):
        """
        Set the fields to return with search results.

        Args:
            *fields: Variable number of field names to return.
            skip_decode: Optional field name or list of field names that should not be
                decoded. Useful for binary data like embeddings.

        Returns:
            self: Returns the query object for method chaining.

        Raises:
            TypeError: If skip_decode is not a string, list, or None.
        """
        # Only clear fields when skip_decode is provided (indicating user is explicitly setting fields)
        # This preserves backward compatibility when return_fields is called multiple times
        if skip_decode is not None:
            # Clear existing fields to provide replacement behavior
            self._return_fields = []
            self._return_fields_decode_as = {}

            # Process skip_decode parameter to prepare decode settings
            if isinstance(skip_decode, str):
                skip_decode_set = {skip_decode}
                self._skip_decode_fields = {skip_decode}
            elif isinstance(skip_decode, list):
                skip_decode_set = set(skip_decode)
                self._skip_decode_fields = set(skip_decode)
            else:
                raise TypeError("skip_decode must be a string or list of strings")

            # Add fields using parent's return_field method with proper decode settings
            for field in fields:
                if field in skip_decode_set:
                    # Use return_field with decode_field=False for skip_decode fields
                    super().return_field(field, decode_field=False)
                else:
                    # Use normal return_field for other fields
                    super().return_field(field)
        else:
            # Standard additive behavior (backward compatible)
            super().return_fields(*fields)

            # Initialize skip_decode_fields if not already set
            if not hasattr(self, "_skip_decode_fields"):
                self._skip_decode_fields = set()

        return self


class FilterQuery(BaseQuery):
    def __init__(
        self,
        filter_expression: Optional[Union[str, FilterExpression]] = None,
        return_fields: Optional[List[str]] = None,
        num_results: int = 10,
        dialect: int = 2,
        sort_by: Optional[SortSpec] = None,
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
            sort_by (Optional[SortSpec], optional): The field(s) to order the results by. Can be:
                - str: single field name (e.g., "price")
                - Tuple[str, str]: (field_name, "ASC"|"DESC") (e.g., ("price", "DESC"))
                - List: list of fields or tuples (e.g., ["price", ("rating", "DESC")])
                Note: Redis Search only supports single-field sorting, so only the first field is used.
                Defaults to None.
            in_order (bool, optional): Requires the terms in the field to have the same order as the
                terms in the query filter. Defaults to False.
            params (Optional[Dict[str, Any]], optional): The parameters for the query. Defaults to None.

        Raises:
            TypeError: If filter_expression is not of type redisvl.query.FilterExpression
        """
        self.set_filter(filter_expression)

        if params:
            self._params = params

        self._num_results = num_results

        # Initialize the base query with the query string from the property
        super().__init__("*")
        self._built_query_string = None  # Ensure it's invalidated after initialization

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
            filter_expression (Optional[Union[str, FilterExpression]]): The filter expression to
                query with. Defaults to None.
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

        # Initialize the base query with the query string from the property
        super().__init__("*")
        self._built_query_string = None

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
    EF_RUNTIME: str = "EF_RUNTIME"
    EF_RUNTIME_PARAM: str = "EF"

    _normalize_vector_distance: bool = False


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
        sort_by: Optional[SortSpec] = None,
        in_order: bool = False,
        hybrid_policy: Optional[str] = None,
        batch_size: Optional[int] = None,
        ef_runtime: Optional[int] = None,
        normalize_vector_distance: bool = False,
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
            sort_by (Optional[SortSpec]): The field(s) to order the results by. Can be:
                - str: single field name
                - Tuple[str, str]: (field_name, "ASC"|"DESC")
                - List: list of fields or tuples
                Note: Only the first field is used for Redis sorting.
                Defaults to None. Results will be ordered by vector distance.
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
            ef_runtime (Optional[int]): Controls the size of the dynamic candidate list for HNSW
                algorithm at query time. Higher values improve recall at the expense of
                slower search performance. Defaults to None, which uses the index-defined value.
            normalize_vector_distance (bool): Redis supports 3 distance metrics: L2 (euclidean),
                IP (inner product), and COSINE. By default, L2 distance returns an unbounded value.
                COSINE distance returns a value between 0 and 2. IP returns a value determined by
                the magnitude of the vector. Setting this flag to true converts COSINE and L2 distance
                to a similarity score between 0 and 1. Note: setting this flag to true for IP will
                throw a warning since by definition COSINE similarity is normalized IP.

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
        self._ef_runtime: Optional[int] = None
        self._normalize_vector_distance = normalize_vector_distance
        self.set_filter(filter_expression)

        # Initialize the base query
        super().__init__("*")
        self._built_query_string = None

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

        if ef_runtime is not None:
            self.set_ef_runtime(ef_runtime)

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

        # Add EF_RUNTIME parameter if specified
        if self._ef_runtime:
            knn_query += f" {self.EF_RUNTIME} ${self.EF_RUNTIME_PARAM}"

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

        # Invalidate the query string
        self._built_query_string = None

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

        # Invalidate the query string
        self._built_query_string = None

    def set_ef_runtime(self, ef_runtime: int):
        """Set the EF_RUNTIME parameter for the query.

        Args:
            ef_runtime (int): The EF_RUNTIME value to use for HNSW algorithm.
                Higher values improve recall at the expense of slower search.

        Raises:
            TypeError: If ef_runtime is not an integer
            ValueError: If ef_runtime is not positive
        """
        if not isinstance(ef_runtime, int):
            raise TypeError("ef_runtime must be an integer")
        if ef_runtime <= 0:
            raise ValueError("ef_runtime must be positive")
        self._ef_runtime = ef_runtime

        # Invalidate the query string
        self._built_query_string = None

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
    def ef_runtime(self) -> Optional[int]:
        """Return the EF_RUNTIME parameter for the query.

        Returns:
            Optional[int]: The EF_RUNTIME value for the query.
        """
        return self._ef_runtime

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

        params: Dict[str, Any] = {self.VECTOR_PARAM: vector}

        # Add EF_RUNTIME parameter if specified
        if self._ef_runtime is not None:
            params[self.EF_RUNTIME_PARAM] = self._ef_runtime

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
        sort_by: Optional[SortSpec] = None,
        in_order: bool = False,
        hybrid_policy: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize_vector_distance: bool = False,
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
            sort_by (Optional[SortSpec]): The field(s) to order the results by. Can be:
                - str: single field name
                - Tuple[str, str]: (field_name, "ASC"|"DESC")
                - List: list of fields or tuples
                Note: Only the first field is used for Redis sorting.
                Defaults to None. Results will be ordered by vector distance.
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
            normalize_vector_distance (bool): Redis supports 3 distance metrics: L2 (euclidean),
                IP (inner product), and COSINE. By default, L2 distance returns an unbounded value.
                COSINE distance returns a value between 0 and 2. IP returns a value determined by
                the magnitude of the vector. Setting this flag to true converts COSINE and L2 distance
                to a similarity score between 0 and 1. Note: setting this flag to true for IP will
                throw a warning since by definition COSINE similarity is normalized IP.

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
        self._normalize_vector_distance = normalize_vector_distance

        # Initialize the base query
        super().__init__("*")
        self._built_query_string = None

        if epsilon is not None:
            self.set_epsilon(epsilon)

        if hybrid_policy is not None:
            self.set_hybrid_policy(hybrid_policy)

        if batch_size is not None:
            self.set_batch_size(batch_size)

        self.set_distance_threshold(distance_threshold)
        self.set_filter(filter_expression)

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
        if self._normalize_vector_distance:
            if distance_threshold > 1:
                raise ValueError(
                    "distance_threshold must be between 0 and 1 when normalize_vector_distance is set to True"
                )

            # User sets normalized value 0-1 denormalize for use in DB
            distance_threshold = denorm_cosine_distance(distance_threshold)
        self._distance_threshold = distance_threshold

        # Invalidate the query string
        self._built_query_string = None

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

        # Invalidate the query string
        self._built_query_string = None

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

        # Invalidate the query string
        self._built_query_string = None

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

        # Invalidate the query string
        self._built_query_string = None

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
            vector = self._vector
        else:
            vector = array_to_buffer(self._vector, dtype=self._dtype)

        params = {
            self.VECTOR_PARAM: vector,
            self.DISTANCE_THRESHOLD_PARAM: self._distance_threshold,
        }

        # Add hybrid policy and batch size as query parameters (not in query string)
        if self._hybrid_policy is not None:
            params[self.HYBRID_POLICY_PARAM] = self._hybrid_policy.value
            if (
                self._hybrid_policy == HybridPolicy.BATCHES
                and self._batch_size is not None
            ):
                params[self.BATCH_SIZE_PARAM] = self._batch_size

        return params


class RangeQuery(VectorRangeQuery):
    # keep for backwards compatibility
    pass


class TextQuery(BaseQuery):
    """
    TextQuery is a query for running a full text search, along with an optional filter expression.

    .. code-block:: python

        from redisvl.query import TextQuery
        from redisvl.index import SearchIndex

        index = SearchIndex.from_yaml(index.yaml)

        query = TextQuery(
            text="example text",
            text_field_name="text_field",
            text_scorer="BM25STD",
            filter_expression=None,
            num_results=10,
            return_fields=["field1", "field2"],
            stopwords="english",
            dialect=2,
        )

        results = index.query(query)
    """

    def __init__(
        self,
        text: str,
        text_field_name: Union[str, Dict[str, float]],
        text_scorer: str = "BM25STD",
        filter_expression: Optional[Union[str, FilterExpression]] = None,
        return_fields: Optional[List[str]] = None,
        num_results: int = 10,
        return_score: bool = True,
        dialect: int = 2,
        sort_by: Optional[SortSpec] = None,
        in_order: bool = False,
        params: Optional[Dict[str, Any]] = None,
        stopwords: Optional[Union[str, Set[str]]] = "english",
    ):
        """A query for running a full text search, along with an optional filter expression.

        Args:
            text (str): The text string to perform the text search with.
            text_field_name (Union[str, Dict[str, float]]): The name of the document field to perform
                text search on, or a dictionary mapping field names to their weights.
            text_scorer (str, optional): The text scoring algorithm to use.
                Defaults to BM25STD. Options are {TFIDF, BM25STD, BM25, TFIDF.DOCNORM, DISMAX, DOCSCORE}.
                See https://redis.io/docs/latest/develop/interact/search-and-query/advanced-concepts/scoring/
            filter_expression (Union[str, FilterExpression], optional): A filter to apply
                along with the text search. Defaults to None.
            return_fields (List[str]): The declared fields to return with search
                results.
            num_results (int, optional): The top k results to return from the
                search. Defaults to 10.
            return_score (bool, optional): Whether to return the text score.
                Defaults to True.
            dialect (int, optional): The RediSearch query dialect.
                Defaults to 2.
            sort_by (Optional[SortSpec]): The field(s) to order the results by. Can be:
                - str: single field name
                - Tuple[str, str]: (field_name, "ASC"|"DESC")
                - List: list of fields or tuples
                Note: Only the first field is used for Redis sorting.
                Defaults to None. Results will be ordered by text score.
            in_order (bool): Requires the terms in the field to have
                the same order as the terms in the query filter, regardless of
                the offsets between them. Defaults to False.
            params (Optional[Dict[str, Any]], optional): The parameters for the query.
                Defaults to None.
            stopwords (Optional[Union[str, Set[str]]): The set of stop words to remove
                from the query text. If a language like 'english' or 'spanish' is provided
                a default set of stopwords for that language will be used. Users may specify
                their own stop words by providing a List or Set of words. if set to None,
                then no words will be removed. Defaults to 'english'.

        Raises:
            ValueError: if stopwords language string cannot be loaded.
            TypeError: If stopwords is not a valid iterable set of strings.
        """
        self._text = text
        self._field_weights = self._parse_field_weights(text_field_name)
        self._num_results = num_results

        self._set_stopwords(stopwords)
        self.set_filter(filter_expression)

        if params:
            self._params = params

        # Initialize the base query
        super().__init__("*")
        self._built_query_string = None

        # handle query settings
        self.scorer(text_scorer)

        if return_fields:
            self.return_fields(*return_fields)
        self.paging(0, self._num_results).dialect(dialect)

        if sort_by:
            self.sort_by(sort_by)

        if in_order:
            self.in_order()

        if return_score:
            self.with_scores()

    @property
    def stopwords(self):
        return self._stopwords

    def _set_stopwords(self, stopwords: Optional[Union[str, Set[str]]] = "english"):
        """Set the stopwords to use in the query.
        Args:
            stopwords (Optional[Union[str, Set[str]]]): The stopwords to use. If a string
                such as "english" "german" is provided then a default set of stopwords for that
                language will be used. if a list, set, or tuple of strings is provided then those
                will be used as stopwords. Defaults to "english". if set to "None" then no stopwords
                will be removed.
        Raises:
            TypeError: If the stopwords are not a set, list, or tuple of strings.
        """
        if not stopwords:
            self._stopwords = set()
        elif isinstance(stopwords, str):
            try:
                nltk.download("stopwords", quiet=True)
                self._stopwords = set(nltk_stopwords.words(stopwords))
            except ImportError:
                raise ValueError(
                    f"Loading stopwords for {stopwords} failed: nltk is not installed."
                )
            except Exception as e:
                raise ValueError(f"Error trying to load {stopwords} from nltk. {e}")
        elif isinstance(stopwords, (Set, List, Tuple)) and all(  # type: ignore
            isinstance(word, str) for word in stopwords
        ):
            self._stopwords = set(stopwords)
        else:
            raise TypeError("stopwords must be a set, list, or tuple of strings")

    def _tokenize_and_escape_query(self, user_query: str) -> str:
        """Convert a raw user query to a redis full text query joined by ORs
        Args:
            user_query (str): The user query to tokenize and escape.

        Returns:
            str: The tokenized and escaped query string.
        Raises:
            ValueError: If the text string becomes empty after stopwords are removed.
        """
        escaper = TokenEscaper()

        tokens = [
            escaper.escape(
                token.strip().strip(",").replace("“", "").replace("”", "").lower()
            )
            for token in user_query.split()
        ]
        return " | ".join(
            [token for token in tokens if token and token not in self._stopwords]
        )

    def _parse_field_weights(
        self, field_spec: Union[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Parse the field specification into a weights dictionary.

        Args:
            field_spec: Either a single field name or dictionary of field:weight mappings

        Returns:
            Dictionary mapping field names to their weights
        """
        if isinstance(field_spec, str):
            return {field_spec: 1.0}
        elif isinstance(field_spec, dict):
            # Validate all weights are numeric and positive
            for field, weight in field_spec.items():
                if not isinstance(field, str):
                    raise TypeError(f"Field name must be a string, got {type(field)}")
                if not isinstance(weight, (int, float)):
                    raise TypeError(
                        f"Weight for field '{field}' must be numeric, got {type(weight)}"
                    )
                if weight <= 0:
                    raise ValueError(
                        f"Weight for field '{field}' must be positive, got {weight}"
                    )
            return field_spec
        else:
            raise TypeError(
                "text_field_name must be a string or dictionary of field:weight mappings"
            )

    def set_field_weights(self, field_weights: Union[str, Dict[str, float]]):
        """Set or update the field weights for the query.

        Args:
            field_weights: Either a single field name or dictionary of field:weight mappings
        """
        self._field_weights = self._parse_field_weights(field_weights)
        # Invalidate the query string
        self._built_query_string = None

    @property
    def field_weights(self) -> Dict[str, float]:
        """Get the field weights for the query.

        Returns:
            Dictionary mapping field names to their weights
        """
        return self._field_weights.copy()

    @property
    def text_field_name(self) -> Union[str, Dict[str, float]]:
        """Get the text field name(s) - for backward compatibility.

        Returns:
            Either a single field name string (if only one field with weight 1.0)
            or a dictionary of field:weight mappings.
        """
        if len(self._field_weights) == 1:
            field, weight = next(iter(self._field_weights.items()))
            if weight == 1.0:
                return field
        return self._field_weights.copy()

    def _build_query_string(self) -> str:
        """Build the full query string for text search with optional filtering."""
        filter_expression = self._filter_expression
        if isinstance(filter_expression, FilterExpression):
            filter_expression = str(filter_expression)

        escaped_query = self._tokenize_and_escape_query(self._text)

        # Build query parts for each field with its weight
        field_queries = []
        for field, weight in self._field_weights.items():
            if weight == 1.0:
                # Default weight doesn't need explicit weight syntax
                field_queries.append(f"@{field}:({escaped_query})")
            else:
                # Use Redis weight syntax for non-default weights
                field_queries.append(
                    f"@{field}:({escaped_query}) => {{ $weight: {weight} }}"
                )

        # Join multiple field queries with OR operator
        if len(field_queries) == 1:
            text = field_queries[0]
        else:
            text = "(" + " | ".join(field_queries) + ")"

        if filter_expression and filter_expression != "*":
            text += f" AND {filter_expression}"
        return text
