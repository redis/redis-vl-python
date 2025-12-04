from typing import Any, Dict, List, Literal, Optional, Set, Union

from redis.commands.search.query import Filter

from redisvl.utils.full_text_query_helper import FullTextQueryHelper

try:
    from redis.commands.search.hybrid_query import HybridQuery as _HybridQuery
    from redis.commands.search.hybrid_query import (
        HybridSearchQuery,
        HybridVsimQuery,
        VectorSearchMethods,
    )
except ImportError:
    raise ImportError("Hybrid queries require redis>=7.1.0")

from redisvl.query.filter import FilterExpression


class HybridQuery(_HybridQuery):
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
        range_radius: Optional[float] = None,
        range_epsilon: Optional[float] = None,
        yield_vsim_score_as: Optional[str] = None,
        vector_filter_expression: Optional[Union[str, FilterExpression]] = None,
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
            text_scorer: The text scorer to use. Options are {TFIDF, TFIDF.DOCNORM,
                BM25, DISMAX, DOCSCORE, BM25STD}. Defaults to "BM25STD".
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

        Raises:
            TypeError: If the stopwords are not a set, list, or tuple of strings.
            ValueError: If the text string is empty, or if the text string becomes empty after
                stopwords are removed.
            ValueError: If `vector_search_method` is not one of {KNN, RANGE} (or None).
            ValueError: If `vector_search_method` is "KNN" and `knn_k` is not provided.
            ValueError: If `vector_search_method` is "RANGE" and `range_radius` is not provided.
        """
        self._ft_helper = FullTextQueryHelper(
            stopwords=stopwords,
            text_weights=text_weights,
        )

        # Serialize the full-text search query
        search_query = HybridSearchQuery(
            query_string=self._ft_helper.build_query_string(
                text=text,
                text_field_name=text_field_name,
                filter_expression=text_filter_expression,
            ),
            scorer=text_scorer,
            yield_score_as=yield_text_score_as,
        )

        # If the vector isn't already bytes, it needs to be represented as a string
        if not isinstance(vector, bytes):
            vector_data: Union[str, bytes] = str(vector)
        else:
            vector_data = vector

        # Serialize vector similarity search method and params, if specified
        vsim_search_method = None
        vsim_search_method_params = {}
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

        # Serialize the vector similarity query
        vsim_query = HybridVsimQuery(
            vector_field_name=vector_field_name,
            vector_data=vector_data,
            vsim_search_method=vsim_search_method,
            vsim_search_method_params=vsim_search_method_params,
            filter=vector_filter_expression and Filter("FILTER", str(vector_filter_expression)),
            yield_score_as=yield_vsim_score_as,
        )

        # Initialize the base HybridQuery
        super().__init__(
            search_query=search_query,
            vector_similarity_query=vsim_query,
        )
