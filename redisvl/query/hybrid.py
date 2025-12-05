from typing import Any, Dict, List, Literal, Optional, Set, Union

from redis.commands.search.query import Filter

from redisvl.utils.full_text_query_helper import FullTextQueryHelper

try:
    from redis.commands.search.hybrid_query import (
        CombinationMethods,
        CombineResultsMethod,
        HybridPostProcessingConfig,
        HybridQuery as RedisHybridQuery,
        HybridSearchQuery,
        HybridVsimQuery,
        VectorSearchMethods,
    )
except ImportError:
    raise ImportError("Hybrid queries require redis>=7.1.0")

from redisvl.query.filter import FilterExpression


class HybridQuery:
    """
    A hybrid search query that combines text search and vector similarity, with configurable fusion methods.
    """

    def __init__(
        self,
        text: str,
        text_field_name: str,
        vector: Union[bytes, List[float]],
        vector_field_name: str,
        text_scorer: str = "BM25STD",
        text_filter_expression: Optional[Union[str, FilterExpression]] = None,
        yield_text_score_as: Optional[str] = None,
        vector_search_method: Optional[Literal["KNN", "RANGE"]] = None,
        knn_k: Optional[int] = None,
        knn_ef_runtime: Optional[int] = None,
        range_radius: Optional[int] = None,
        range_epsilon: Optional[float] = None,
        yield_vsim_score_as: Optional[str] = None,
        vector_filter_expression: Optional[Union[str, FilterExpression]] = None,
        stopwords: Optional[Union[str, Set[str]]] = "english",
        text_weights: Optional[Dict[str, float]] = None,
        combination_method: Optional[Literal["RRF", "LINEAR"]] = None,
        rrf_window: Optional[int] = None,
        rrf_constant: Optional[float] = None,
        linear_alpha: Optional[float] = None,
        linear_beta: Optional[float] = None,
        yield_combined_score_as: Optional[str] = None,
    ):
        """
        Instantiates a HybridQuery object.

        Args:
            text: The text to search for.
            text_field_name: The text field name to search in.
            vector: The vector to perform vector similarity search.
            vector_field_name: The vector field name to search in.
            text_scorer: The text scorer to use. Options are {TFIDF, TFIDF.DOCNORM,
                BM25STD, BM25STD.NORM, BM25STD.TANH, DISMAX, DOCSCORE, HAMMING}. Defaults to "BM25STD". For more
                information about supported scroring algorithms,
                see https://redis.io/docs/latest/develop/ai/search-and-query/advanced-concepts/scoring/
            text_filter_expression: The filter expression to use for the text search. Defaults to None.
            yield_text_score_as: The name of the field to yield the text score as.
            vector_search_method: The vector search method to use. Options are {KNN, RANGE}. Defaults to None.
            knn_k: The number of nearest neighbors to return, required if `vector_search_method` is "KNN".
            knn_ef_runtime: The exploration factor parameter for HNSW, optional if `vector_search_method` is "KNN".
            range_radius: The search radius to use, required if `vector_search_method` is "RANGE".
            range_epsilon: The epsilon value to use, optional if `vector_search_method` is "RANGE"; defines the
                accuracy of the search.
            yield_vsim_score_as: The name of the field to yield the vector similarity score as.
            vector_filter_expression: The filter expression to use for the vector similarity search. Defaults to None.
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
            combination_method: The combination method to use. Options are {RRF, LINEAR}. Defaults to None.
            rrf_window: The window size to use for the reciprocal rank fusion (RRF) combination method. Limits
                fusion scope.
            rrf_constant: The constant to use for the reciprocal rank fusion (RRF) combination method. Controls decay
                of rank influence.
            linear_alpha: The weight of the first query for the linear combination method (LINEAR).
            linear_beta: The weight of the second query for the linear combination method (LINEAR).
            yield_combined_score_as: The name of the field to yield the combined score as.

        Notes:
            If RRF combination method is used, then at least one of `rrf_window` or `rrf_constant` must be provided.
            If LINEAR combination method is used, then at least one of `linear_alpha` or `linear_beta` must be provided.

        Raises:
            TypeError: If the stopwords are not a set, list, or tuple of strings.
            ValueError: If the text string is empty, or if the text string becomes empty after
                stopwords are removed.
            ValueError: If `vector_search_method` is not one of {KNN, RANGE} (or None).
            ValueError: If `vector_search_method` is "KNN" and `knn_k` is not provided.
            ValueError: If `vector_search_method` is "RANGE" and `range_radius` is not provided.
        """
        self.postprocessing_config = HybridPostProcessingConfig()
        self._ft_helper = FullTextQueryHelper(
            stopwords=stopwords,
            text_weights=text_weights,
        )

        query_string = self._ft_helper.build_query_string(
            text, text_field_name, text_filter_expression
        )

        self.query = self.build_query(
            text_query=query_string,
            vector=vector,
            vector_field_name=vector_field_name,
            text_scorer=text_scorer,
            yield_text_score_as=yield_text_score_as,
            vector_search_method=vector_search_method,
            knn_k=knn_k,
            knn_ef_runtime=knn_ef_runtime,
            range_radius=range_radius,
            range_epsilon=range_epsilon,
            yield_vsim_score_as=yield_vsim_score_as,
            vector_filter_expression=vector_filter_expression,
        )

        if combination_method:
            self.combination_method: Optional[CombineResultsMethod] = (
                self.build_combination_method(
                    combination_method=combination_method,
                    rrf_window=rrf_window,
                    rrf_constant=rrf_constant,
                    linear_alpha=linear_alpha,
                    linear_beta=linear_beta,
                    yield_score_as=yield_combined_score_as,
                )
            )
        else:
            self.combination_method = None

    @staticmethod
    def build_query(
        text_query: str,
        vector: Union[bytes, List[float]],
        vector_field_name: str,
        text_scorer: str = "BM25STD",
        yield_text_score_as: Optional[str] = None,
        vector_search_method: Optional[Literal["KNN", "RANGE"]] = None,
        knn_k: Optional[int] = None,
        knn_ef_runtime: Optional[int] = None,
        range_radius: Optional[int] = None,
        range_epsilon: Optional[float] = None,
        yield_vsim_score_as: Optional[str] = None,
        vector_filter_expression: Optional[Union[str, FilterExpression]] = None,
    ) -> RedisHybridQuery:
        """Build a Redis HybridQuery for the hybrid search."""

        # Serialize the full-text search query
        search_query = HybridSearchQuery(
            query_string=text_query,
            scorer=text_scorer,
            yield_score_as=yield_text_score_as,
        )

        # If the vector isn't already bytes, it needs to be represented as a string
        if not isinstance(vector, bytes):
            vector_data: Union[str, bytes] = str(vector)
        else:
            vector_data = vector

        # Serialize vector similarity search method and params, if specified
        vsim_search_method: Optional[VectorSearchMethods] = None
        vsim_search_method_params: Dict[str, Any] = {}
        if vector_search_method == "KNN":
            vsim_search_method = VectorSearchMethods.KNN
            if not knn_k:
                raise ValueError("Must provide K if vector_search_method is KNN")

            vsim_search_method_params["K"] = knn_k
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

        if vector_filter_expression:
            vsim_filter = Filter("FILTER", str(vector_filter_expression))
        else:
            vsim_filter = None

        # Serialize the vector similarity query
        vsim_query = HybridVsimQuery(
            vector_field_name=vector_field_name,
            vector_data=vector_data,
            vsim_search_method=vsim_search_method,
            vsim_search_method_params=vsim_search_method_params,
            filter=vsim_filter,
            yield_score_as=yield_vsim_score_as,
        )

        return RedisHybridQuery(
            search_query=search_query,
            vector_similarity_query=vsim_query,
        )

    @staticmethod
    def build_combination_method(
        combination_method: Literal["RRF", "LINEAR"],
        rrf_window: Optional[int] = None,
        rrf_constant: Optional[float] = None,
        linear_alpha: Optional[float] = None,
        linear_beta: Optional[float] = None,
        yield_score_as: Optional[str] = None,
    ) -> CombineResultsMethod:
        """Build a configuration for combining hybrid search scores."""
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
            if linear_beta:
                method_params["BETA"] = linear_beta

        else:
            raise ValueError(f"Unknown combination method: {combination_method}")

        if yield_score_as:
            method_params["YIELD_SCORE_AS"] = yield_score_as

        return CombineResultsMethod(
            method=method,
            **method_params,
        )
