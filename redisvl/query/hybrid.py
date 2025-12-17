from typing import Any, Dict, List, Literal, Optional, Set, Union

from redis.commands.search.query import Filter

from redisvl.query.filter import FilterExpression
from redisvl.redis.utils import array_to_buffer
from redisvl.utils.full_text_query_helper import FullTextQueryHelper

_IMPORT_ERROR_MESSAGE = "Hybrid queries require Redis >= 8.4.0 and redis-py>=7.1.0"


class HybridQuery:
    """
    A hybrid search query that combines text search and vector similarity, with configurable fusion methods.

    .. code-block:: python

        from redisvl.query import HybridQuery
        from redisvl.index import SearchIndex

        index = SearchIndex.from_yaml("path/to/index.yaml")

        query = HybridQuery(
            text="example text",
            text_field_name="text_field",
            vector=[0.1, 0.2, 0.3],
            vector_field_name="vector_field",
            text_scorer="BM25STD",
            yield_text_score_as="text_score",
            yield_vsim_score_as="vector_similarity",
            combination_method="LINEAR",
            linear_alpha=0.3,
            yield_combined_score_as="hybrid_score",
            num_results=10,
            return_fields=["field1", "field2"],
            stopwords="english",
        )

        results = index.query(query)

    See Also:
        - `FT.HYBRID command documentation <https://redis.io/docs/latest/commands/ft.hybrid>`_
        - `redis-py hybrid_search documentation <https://redis.readthedocs.io/en/stable/redismodules.html#redis.commands.search.commands.SearchCommands.hybrid_search>`_
    """

    def __init__(
        self,
        text: str,
        text_field_name: str,
        vector: Union[bytes, List[float]],
        vector_field_name: str,
        vector_param_name: str = "vector",
        text_scorer: str = "BM25STD",
        yield_text_score_as: Optional[str] = None,
        vector_search_method: Optional[Literal["KNN", "RANGE"]] = None,
        knn_ef_runtime: int = 10,
        range_radius: Optional[float] = None,
        range_epsilon: float = 0.01,
        yield_vsim_score_as: Optional[str] = None,
        filter_expression: Optional[Union[str, FilterExpression]] = None,
        combination_method: Optional[Literal["RRF", "LINEAR"]] = None,
        rrf_window: int = 20,
        rrf_constant: int = 60,
        linear_alpha: float = 0.3,
        yield_combined_score_as: Optional[str] = None,
        dtype: str = "float32",
        num_results: Optional[int] = 10,
        return_fields: Optional[List[str]] = None,
        stopwords: Optional[Union[str, Set[str]]] = "english",
        text_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Instantiates a HybridQuery object.

        Args:
            text: The text to search for.
            text_field_name: The text field name to search in.
            vector: The vector to perform vector similarity search.
            vector_field_name: The vector field name to search in.
            vector_param_name: The name of the parameter substitution containing the vector blob.
            text_scorer: The text scorer to use. Options are {TFIDF, TFIDF.DOCNORM,
                BM25STD, BM25STD.NORM, BM25STD.TANH, DISMAX, DOCSCORE, HAMMING}. Defaults to "BM25STD". For more
                information about supported scoring algorithms,
                see https://redis.io/docs/latest/develop/ai/search-and-query/advanced-concepts/scoring/
            yield_text_score_as: The name of the field to yield the text score as.
            vector_search_method: The vector search method to use. Options are {KNN, RANGE}. Defaults to None.
            knn_ef_runtime: The exploration factor parameter for HNSW, optional if `vector_search_method` is "KNN".
            range_radius: The search radius to use, required if `vector_search_method` is "RANGE".
            range_epsilon: The epsilon value to use, optional if `vector_search_method` is "RANGE"; defines the
                accuracy of the search.
            yield_vsim_score_as: The name of the field to yield the vector similarity score as.
            filter_expression: The filter expression to use for both the text and vector searches. Defaults to None.
            combination_method: The combination method to use. Options are {RRF, LINEAR}. If not specified, the server
                defaults to RRF. If "RRF" is specified, then at least one of `rrf_window` or `rrf_constant` must be
                provided. If "LINEAR" is specified, then at least one of `linear_alpha` or `linear_beta` must be
                provided.
            rrf_window: The window size to use for the reciprocal rank fusion (RRF) combination method. Limits
                fusion scope.
            rrf_constant: The constant to use for the reciprocal rank fusion (RRF) combination method. Controls decay
                of rank influence.
            linear_alpha: The weight of the text query for the linear combination method (LINEAR).
            yield_combined_score_as: The name of the field to yield the combined score as.
            dtype: The data type of the vector. Defaults to "float32".
            num_results: The number of results to return.
            return_fields: The fields to return. Defaults to None.
            stopwords (Optional[Union[str, Set[str]]], optional): The stopwords to remove from the
                provided text prior to search-use. If a string such as "english" "german" is
                provided then a default set of stopwords for that language will be used. if a list,
                set, or tuple of strings is provided then those will be used as stopwords.
                Defaults to "english". if set to "None" then no stopwords will be removed.

                Note: This parameter controls query-time stopword filtering (client-side).
                For index-level stopwords configuration (server-side), see IndexInfo.stopwords.
                Using query-time stopwords with index-level STOPWORDS 0 is counterproductive.
            text_weights (Optional[Dict[str, float]]): The importance weighting of individual words
                within the query text. Defaults to None, as no modifications will be made to the
                text_scorer score.

        Raises:
            ImportError: If redis-py>=7.1.0 is not installed.
            TypeError: If the stopwords are not a set, list, or tuple of strings.
            ValueError: If the text string is empty, or if the text string becomes empty after
                stopwords are removed.
            ValueError: If `vector_search_method` is defined and isn't one of {KNN, RANGE}.
            ValueError: If `vector_search_method` is "KNN" and `knn_k` is not provided.
            ValueError: If `vector_search_method` is "RANGE" and `range_radius` is not provided.
        """
        try:
            from redis.commands.search.hybrid_query import (
                CombineResultsMethod,
                HybridPostProcessingConfig,
            )
        except (ImportError, ModuleNotFoundError):
            raise ImportError(_IMPORT_ERROR_MESSAGE)

        self.postprocessing_config = HybridPostProcessingConfig()
        if num_results:
            self.postprocessing_config.limit(offset=0, num=num_results)
        if return_fields:
            self.postprocessing_config.load(*(f"@{f}" for f in return_fields))

        self._ft_helper = FullTextQueryHelper(
            stopwords=stopwords,
            text_weights=text_weights,
        )

        query_string = self._ft_helper.build_query_string(
            text, text_field_name, filter_expression
        )

        if isinstance(vector, bytes):
            vector_data = vector
        else:
            vector_data = array_to_buffer(vector, dtype)

        self.params = {
            vector_param_name: vector_data,
        }

        self.query = build_base_query(
            text_query=query_string,
            vector_param_name=vector_param_name,
            vector_field_name=vector_field_name,
            text_scorer=text_scorer,
            yield_text_score_as=yield_text_score_as,
            vector_search_method=vector_search_method,
            num_results=num_results,
            knn_ef_runtime=knn_ef_runtime,
            range_radius=range_radius,
            range_epsilon=range_epsilon,
            yield_vsim_score_as=yield_vsim_score_as,
            filter_expression=filter_expression,
        )

        if combination_method:
            self.combination_method: Optional[CombineResultsMethod] = (
                build_combination_method(
                    combination_method=combination_method,
                    rrf_window=rrf_window,
                    rrf_constant=rrf_constant,
                    linear_alpha=linear_alpha,
                    yield_score_as=yield_combined_score_as,
                )
            )
        else:
            self.combination_method = None


def build_base_query(
    text_query: str,
    vector_param_name: str,
    vector_field_name: str,
    text_scorer: str = "BM25STD",
    yield_text_score_as: Optional[str] = None,
    vector_search_method: Optional[Literal["KNN", "RANGE"]] = None,
    num_results: Optional[int] = None,
    knn_ef_runtime: Optional[int] = None,
    range_radius: Optional[float] = None,
    range_epsilon: Optional[float] = None,
    yield_vsim_score_as: Optional[str] = None,
    filter_expression: Optional[Union[str, FilterExpression]] = None,
):
    """Build a Redis HybridQuery for performing hybrid search.

    Args:
        text_query: The query for the text search.
        vector_param_name: The name of the parameter substitution containing the vector blob.
        vector_field_name: The vector field name to search in.
        text_scorer: The text scorer to use. Options are {TFIDF, TFIDF.DOCNORM,
            BM25STD, BM25STD.NORM, BM25STD.TANH, DISMAX, DOCSCORE, HAMMING}. Defaults to "BM25STD". For more
            information about supported scroring algorithms,
            see https://redis.io/docs/latest/develop/ai/search-and-query/advanced-concepts/scoring/
        yield_text_score_as: The name of the field to yield the text score as.
        vector_search_method: The vector search method to use. Options are {KNN, RANGE}. Defaults to None.
        num_results: The number of nearest neighbors to return, required if `vector_search_method` is "KNN".
        knn_ef_runtime: The exploration factor parameter for HNSW, optional if `vector_search_method` is "KNN".
        range_radius: The search radius to use, required if `vector_search_method` is "RANGE".
        range_epsilon: The epsilon value to use, optional if `vector_search_method` is "RANGE"; defines the
            accuracy of the search.
        yield_vsim_score_as: The name of the field to yield the vector similarity score as.
        filter_expression: The filter expression to use for the vector similarity search. Defaults to None.

    Notes:
        If RRF combination method is used, then at least one of `rrf_window` or `rrf_constant` must be provided.
        If LINEAR combination method is used, then at least one of `linear_alpha` or `linear_beta` must be provided.

    Raises:
        ImportError: If redis-py>=7.1.0 is not installed.
        ValueError: If `vector_search_method` is defined and isn't one of {KNN, RANGE}.
        ValueError: If `vector_search_method` is "KNN" and `knn_k` is not provided.
        ValueError: If `vector_search_method` is "RANGE" and `range_radius` is not provided.

    Returns:
        A Redis HybridQuery object that defines the text and vector searches to be performed.
    """
    try:
        from redis.commands.search.hybrid_query import HybridQuery as RedisHybridQuery
        from redis.commands.search.hybrid_query import (
            HybridSearchQuery,
            HybridVsimQuery,
            VectorSearchMethods,
        )
    except (ImportError, ModuleNotFoundError):
        raise ImportError(_IMPORT_ERROR_MESSAGE)

    # Serialize the full-text search query
    search_query = HybridSearchQuery(
        query_string=text_query,
        scorer=text_scorer,
        yield_score_as=yield_text_score_as,
    )

    # Serialize vector similarity search method and params, if specified
    vsim_search_method: Optional[VectorSearchMethods] = None
    vsim_search_method_params: Dict[str, Any] = {}
    if vector_search_method == "KNN":
        vsim_search_method = VectorSearchMethods.KNN
        if not num_results:
            raise ValueError(
                "Must provide `num_results` if vector_search_method is KNN"
            )

        vsim_search_method_params["K"] = num_results
        if knn_ef_runtime:
            vsim_search_method_params["EF_RUNTIME"] = knn_ef_runtime

    elif vector_search_method == "RANGE":
        vsim_search_method = VectorSearchMethods.RANGE
        if not range_radius:
            raise ValueError("Must provide RADIUS if vector_search_method is RANGE")

        vsim_search_method_params["RADIUS"] = range_radius
        if range_epsilon:
            vsim_search_method_params["EPSILON"] = range_epsilon

    elif vector_search_method is not None:
        raise ValueError(f"Unknown vector search method: {vector_search_method}")

    if isinstance(filter_expression, FilterExpression):
        filter_expression = str(filter_expression)

    if filter_expression and filter_expression != "*":
        vsim_filter = Filter("FILTER", str(filter_expression))
    else:
        vsim_filter = None

    # Serialize the vector similarity query
    vsim_query = HybridVsimQuery(
        vector_field_name="@" + vector_field_name,
        vector_data="$" + vector_param_name,
        vsim_search_method=vsim_search_method,
        vsim_search_method_params=vsim_search_method_params,
        filter=vsim_filter,
        yield_score_as=yield_vsim_score_as,
    )

    return RedisHybridQuery(
        search_query=search_query,
        vector_similarity_query=vsim_query,
    )


def build_combination_method(
    combination_method: Literal["RRF", "LINEAR"],
    rrf_window: Optional[int] = None,
    rrf_constant: Optional[float] = None,
    linear_alpha: Optional[float] = None,
    yield_score_as: Optional[str] = None,
):
    """Build a configuration for combining hybrid search scores.

    Args:
        combination_method: The combination method to use. Options are {RRF, LINEAR}.
        rrf_window: The window size to use for the reciprocal rank fusion (RRF) combination method. Limits
            fusion scope.
        rrf_constant: The constant to use for the reciprocal rank fusion (RRF) combination method. Controls decay
            of rank influence.
        linear_alpha: The weight of the first query for the linear combination method (LINEAR).
        yield_score_as: The name of the field to yield the combined score as.

    Raises:
        ImportError: If redis-py>=7.1.0 is not installed.
        ValueError: If `combination_method` is defined and isn't one of {RRF, LINEAR}.
        ValueError: If `combination_method` is "RRF" and neither `rrf_window` nor `rrf_constant` is provided.
        ValueError: If `combination_method` is "LINEAR" and neither `linear_alpha` nor `linear_beta` is provided.

    Returns:
        A CombineResultsMethod object that defines how the text and vector scores should be combined.
    """
    try:
        from redis.commands.search.hybrid_query import (
            CombinationMethods,
            CombineResultsMethod,
        )
    except (ImportError, ModuleNotFoundError):
        raise ImportError(_IMPORT_ERROR_MESSAGE)

    method_params: Dict[str, Any] = {}
    if combination_method == "RRF":
        method = CombinationMethods.RRF
        if rrf_window:
            method_params["WINDOW"] = rrf_window
        if rrf_constant:
            method_params["CONSTANT"] = rrf_constant

    elif combination_method == "LINEAR":
        method = CombinationMethods.LINEAR
        if linear_alpha:
            method_params["ALPHA"] = linear_alpha
            method_params["BETA"] = 1 - linear_alpha

    else:
        raise ValueError(f"Unknown combination method: {combination_method}")

    if yield_score_as:
        method_params["YIELD_SCORE_AS"] = yield_score_as

    if not method_params:
        raise ValueError(
            "No parameters provided for combination method - must provide at least one parameter."
        )

    return CombineResultsMethod(
        method=method,
        **method_params,
    )
